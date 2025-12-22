#!/usr/bin/env python3
"""Generate and benchmark quantized CoreML model variants for Sortformer."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

import coremltools as ct
import coremltools.optimize.coreml as cto
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import librosa

# Use Agg backend
matplotlib.use("Agg")

# Constants matching Export
SAMPLE_RATE = 16_000
CHUNK_LEN = 4  # Output frames
SUBSAMPLING = 8
INPUT_CHUNK_FRAMES = CHUNK_LEN * SUBSAMPLING  # 32
FEAT_DIM = 128
SPKCACHE_LEN = 63
FIFO_LEN = 63
FC_D_MODEL = 512
WARMUP_RUNS = 2

# CoreML compute units
_COMPUTE_UNIT_ORDER = ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]

def _default_compute_units() -> list[ct.ComputeUnit]:
    units: list[ct.ComputeUnit] = []
    for name in _COMPUTE_UNIT_ORDER:
        if hasattr(ct.ComputeUnit, name):
            units.append(getattr(ct.ComputeUnit, name))
    # Fallback if "ALL" not found (some versions)
    if not units and hasattr(ct.ComputeUnit, "ALL"):
        units.append(ct.ComputeUnit.ALL)
    elif not units:
        # Minimal fallback
        units.append(ct.ComputeUnit.CPU_ONLY)
    return units

DEFAULT_COMPUTE_UNITS = _default_compute_units()

@dataclass
class VariantSpec:
    name: str
    suffix: str
    description: str
    transform: str = "quantize"

MODEL_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(name="int8-linear", suffix="-int8-linear", description="INT8 linear quantization"),
    VariantSpec(name="int8-lut", suffix="-int8-lut", description="INT8 LUT quantization (palettization)", transform="palettize"),
    VariantSpec(name="int16", suffix="-int16", description="Float16 to Int16 quantization (weights only)"),
)

@dataclass
class ComparisonResult:
    mse: float
    mae: float
    max_abs: float
    corr: float

def compute_error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = reference.reshape(-1)
    cand = candidate.reshape(-1)
    diff = cand - ref
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    ref_std = float(np.std(ref))
    cand_std = float(np.std(cand))
    corr = float(np.corrcoef(ref, cand)[0, 1]) if (ref_std > 1e-9 and cand_std > 1e-9) else float("nan")
    return {"mse": mse, "mae": mae, "max_abs": max_abs, "corr": corr}

def generate_variant(base_model_path: Path, variant_spec: VariantSpec, output_dir: Path) -> Path:
    """Generate a single optimized variant."""
    print(f"  Generating {variant_spec.name}...")
    base_model = ct.models.MLModel(str(base_model_path))
    
    if variant_spec.transform == "quantize":
        if "int16" in variant_spec.name:
             config = cto.OptimizationConfig(
                 global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype=np.int8, granularity="per_tensor")
             )
        else:
             # Standard int8
             config = cto.OptimizationConfig(
                 global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype=np.int8, granularity="per_channel")
             )
        variant_model = cto.linear_quantize_weights(base_model, config=config)
        
    elif variant_spec.transform == "palettize":
        # LUT Quantization (Palettization)
        config = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=8, granularity="per_channel")
        )
        variant_model = cto.palettize_weights(base_model, config=config)
    else:
        raise ValueError(f"Unknown transform: {variant_spec.transform}")

    output_path = output_dir / f"{base_model_path.stem}{variant_spec.suffix}.mlpackage"
    variant_model.save(str(output_path))
    print(f"  Saved to {output_path}")
    return output_path

def collect_calibration_data(audio_path: Path, model_name: str, max_chunks=20) -> Tuple[List[Dict], List[np.ndarray]]:
    """
    Run PyTorch inference to collect real inputs (features + state) and outputs (preds) for benchmarking.
    """
    print(f"Collecting calibration data from {audio_path} using {model_name}...")
    from nemo.collections.asr.models import SortformerEncLabelModel
    
    model = SortformerEncLabelModel.from_pretrained(model_name, map_location="cpu")
    model.eval()
    
    # Configure for streaming
    modules = model.sortformer_modules
    modules.chunk_len = CHUNK_LEN
    modules.fifo_len = FIFO_LEN
    modules.spkcache_len = SPKCACHE_LEN
    
    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, duration=30.0) # Limit to 30s
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    
    # Parameters
    stride = (CHUNK_LEN) * 160 # 640
    # Exact sample count for 32 frames: (32-1)*160 + 400 = 5360
    # However, preprocessor might output more if we simply feed 5360 due to padding/alignment.
    # We will slice the features to be safe.
    chunk_size_samples = 5360 
    
    inputs_list = []
    outputs_list = []
    
    state = modules.init_streaming_state(batch_size=1, device='cpu')
    
    num_chunks = min(max_chunks, (len(audio) - chunk_size_samples) // stride)
    if num_chunks < 1:
        print("Audio too short for benchmarking.")
        return [], []
    
    print(f"Processing {num_chunks} chunks...")
    for i in range(num_chunks):
        start = i * stride
        end = start + chunk_size_samples
        if end > len(audio): break
        
        audio_chunk = audio_tensor[:, start:end]
        length_tensor = torch.tensor([audio_chunk.shape[1]], dtype=torch.long)
        
        # Preprocess
        with torch.no_grad():
            feats, feat_lens = model.preprocessor(input_signal=audio_chunk, length=length_tensor)
            # feats: [B, D, T] -> [1, 128, T]
            
            # FORCE SLICE to 32 frames (INPUT_CHUNK_FRAMES)
            if feats.shape[2] > INPUT_CHUNK_FRAMES:
                feats = feats[:, :, :INPUT_CHUNK_FRAMES]
            elif feats.shape[2] < INPUT_CHUNK_FRAMES:
                # Pad if too short (should not happen with 5360 samples)
                pad = torch.zeros(1, 128, INPUT_CHUNK_FRAMES - feats.shape[2])
                feats = torch.cat([feats, pad], dim=2)
                
            # Transpose for Main Model: [B, T, D]
            chunk_in = feats.transpose(1, 2)
        
        # Prepare inputs dict
        curr_spk_len = state.spkcache.shape[1]
        curr_fifo_len = state.fifo.shape[1]
        
        # Min-1 logic match
        spk_in_len = max(1, curr_spk_len)
        fifo_in_len = max(1, curr_fifo_len)
        
        # Pad states to Fixed Size for CoreML
        if curr_spk_len < SPKCACHE_LEN:
            spk_pad = torch.nn.functional.pad(state.spkcache, (0,0,0, SPKCACHE_LEN - curr_spk_len))
        else:
            spk_pad = state.spkcache[:, :SPKCACHE_LEN, :]
            
        if curr_fifo_len < FIFO_LEN:
            fifo_pad = torch.nn.functional.pad(state.fifo, (0,0,0, FIFO_LEN - curr_fifo_len))
        else:
            fifo_pad = state.fifo[:, :FIFO_LEN, :]

        coreml_input = {
            "chunk": chunk_in.numpy().astype(np.float32),
            "chunk_lengths": np.array([INPUT_CHUNK_FRAMES], dtype=np.int32),
            "spkcache": spk_pad.numpy().astype(np.float32),
            "spkcache_lengths": np.array([spk_in_len], dtype=np.int32),
            "fifo": fifo_pad.numpy().astype(np.float32),
            "fifo_lengths": np.array([fifo_in_len], dtype=np.int32)
        }
        
        # Run Baseline (PyTorch)
        with torch.no_grad():
             t_preds, t_chunk_embs, _ = model.forward_for_export(
                 chunk_in, 
                 torch.tensor([INPUT_CHUNK_FRAMES], dtype=torch.int32),
                 spk_pad, 
                 torch.tensor([spk_in_len], dtype=torch.int32),
                 fifo_pad, 
                 torch.tensor([fifo_in_len], dtype=torch.int32)
             )
        
        inputs_list.append(coreml_input)
        outputs_list.append(t_preds.numpy())
        
        # Update State 
        with torch.no_grad():
            state, _ = modules.streaming_update(state, t_chunk_embs, t_preds)
            
    return inputs_list, outputs_list

def benchmark_variant(model_path: Path, inputs: List[Dict], references: List[np.ndarray], compute_units: List[ct.ComputeUnit]) -> Dict:
    results = {}
    
    for cu in compute_units:
        cu_name = cu.name
        try:
            print(f"  Benchmarking on {cu_name}...")
            # Load
            start = time.perf_counter()
            model = ct.models.MLModel(str(model_path), compute_units=cu)
            load_time = time.perf_counter() - start
            
            # Predict
            latencies = []
            predictions = []
            
            # Warmup
            if inputs:
                for _ in range(WARMUP_RUNS):
                    model.predict(inputs[0])
            
            for i, inp in enumerate(inputs):
                t0 = time.perf_counter()
                out = model.predict(inp) # returns dict {'preds': ...}
                latencies.append(time.perf_counter() - t0)
                predictions.append(out['preds']) # Key 'preds' from model output
                
            # Metrics
            if not predictions:
                print("    No predictions made.")
                continue

            refs_stack = np.concatenate(references, axis=0) # [N, 4, 4]
            preds_stack = np.concatenate(predictions, axis=0)
            
            error_metrics = compute_error_metrics(refs_stack, preds_stack)
            
            results[cu_name] = {
                "load_time": load_time,
                "latency_mean": np.mean(latencies),
                "metrics": error_metrics
            }
            print(f"    MSE: {error_metrics['mse']:.6f}, Latency: {np.mean(latencies)*1000:.2f}ms")
            
        except Exception as e:
            print(f"    Failed on {cu_name}: {e}")
            results[cu_name] = {"error": str(e)}
            
    return results

def plot_results(benchmark_data: Dict, output_path: Path):
    """Plot latency vs MSE tradeoff."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data for "ALL" compute unit (usually Neural Engine)
    labels = []
    mses = []
    latencies = []
    
    for variant, data in benchmark_data.items():
        if "ALL" in data and "metrics" in data["ALL"]:
            labels.append(variant)
            mses.append(data["ALL"]["metrics"]["mse"])
            latencies.append(data["ALL"]["latency_mean"] * 1000)
        elif "CPU_AND_GPU" in data and "metrics" in data["CPU_AND_GPU"]: # Fallback
            labels.append(variant)
            mses.append(data["CPU_AND_GPU"]["metrics"]["mse"])
            latencies.append(data["CPU_AND_GPU"]["latency_mean"] * 1000)
    
    if not labels:
        print("No valid data to plot.")
        return

    ax.scatter(latencies, mses, color='blue', s=100)
    
    for i, label in enumerate(labels):
        ax.annotate(label, (latencies[i], mses[i]), xytext=(5, 5), textcoords='offset points')
        
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("MSE (vs FP32)")
    ax.set_title("Sortformer Quantization Tradeoff")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(str(output_path))
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coreml-dir", type=Path, default=Path("coreml_models"))
    parser.add_argument("--audio-path", type=Path, default=Path("../multispeaker.wav"))
    parser.add_argument("--model-name", type=str, default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    parser.add_argument("--skip-gen", action="store_true")
    args = parser.parse_args()
    
    base_model_path = args.coreml_dir / "Sortformer.mlpackage"
    if not base_model_path.exists():
        # Fallback to parent dir check if script is deep
        base_model_path = args.coreml_dir / "Sortformer.mlpackage"
        if not base_model_path.exists():
              raise FileNotFoundError(f"Base model not found: {base_model_path}")

    # 1. Generate Variants
    if not args.skip_gen:
        print("Generating quantized variants...")
        for spec in MODEL_VARIANTS:
            try:
                generate_variant(base_model_path, spec, args.coreml_dir)
            except Exception as e:
                print(f"Skipping {spec.name}: {e}")
    
    # 2. Collect Calibration Data
    inputs, baselines = collect_calibration_data(args.audio_path, args.model_name)
    
    # 3. Benchmark
    print("\nBenchmarking...")
    all_results = {}
    
    # Benchmark Baseline (FP32/FP16 Original)
    print("Benchmarking Original...")
    all_results["Original"] = benchmark_variant(base_model_path, inputs, baselines, DEFAULT_COMPUTE_UNITS)
    
    # Benchmark Variants
    for spec in MODEL_VARIANTS:
        path = args.coreml_dir / f"Sortformer{spec.suffix}.mlpackage"
        if path.exists():
            print(f"Benchmarking {spec.name}...")
            all_results[spec.name] = benchmark_variant(path, inputs, baselines, DEFAULT_COMPUTE_UNITS)
            
    # 4. Save/Plot
    plot_results(all_results, args.coreml_dir / "quantization_tradeoff.png")

if __name__ == "__main__":
    main()

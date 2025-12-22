import torch
import torch.nn as nn
import coremltools as ct
import argparse
import os
import sys
import numpy as np
import types

# Ensure we use the right environment for imports
# (User's environment has 'nemo' installed)
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeaturesTA

# --- Monkey-Patching for CoreML Tracing ---
def patched_concat_and_pad(embs, lengths):
    # Optimized for Batch Size 1 (CoreML)
    # embs: list of tensors [spkcache, fifo, chunk]
    # lengths: list of tensors [tail constraints]
    
    # NOTE: logic must be unconditional to preserve dynamic slicing in Trace.
    # We cannot use .item() or if statements based on values.
    
    to_concat = []
    total_len_list = []
    
    # Iterate over the three parts (spkcache, fifo, chunk)
    # We assume fixed structure of 3 inputs as per Sortformer logic
    for i in range(len(embs)):
        val = embs[i]
        length_t = lengths[i]
        
        # Dynamic Slice using Tensor index
        # val: (B, T_max, D) or (B, D, T_max)? 
        # In convert(), we determined inputs are (B, T, D).
        # concat acts on dim 1 (Time).
        
        limit = length_t[0] # Tensor Scalar
        
        # Slice: val[:, :limit, :]
        # Note: CoreML slicing with dynamic index works if inputs are tracked.
        part = val[:, :limit, :]
        
        to_concat.append(part)
        total_len_list.append(limit)
        
    # Concatenate on Time dimension (1)
    # Even if some parts are empty (0-dim on axis 1), torch.cat handles it.
    total_embs = torch.cat(to_concat, dim=1)
    
    # Compute Total Length
    total_length = torch.tensor(0, dtype=torch.int32)
    for l in total_len_list:
        total_length = total_length + l
        
    return total_embs, total_length.unsqueeze(0)

# Apply patch
# SortformerModules.concat_and_pad = staticmethod(patched_concat_and_pad)
# We handle patching on the instance in export_model instead.

class WrapperBase(nn.Module):
    """
    Base class for CoreML wrappers to handle dtype conversions.
    Ensures inputs are cast to internal dtype (e.g. FP16) and outputs back to FP32.
    """
    def __init__(self, model, internal_dtype=torch.float32):
        super().__init__()
        self.model = model
        self.internal_dtype = internal_dtype

    def forward(self, *args):
        # Cast float inputs to internal_dtype
        casted_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.is_floating_point():
                casted_args.append(arg.to(self.internal_dtype))
            else:
                casted_args.append(arg)
        
        # Run model
        # Unpack args? No, model(*casted_args) works.
        outputs = self.model(*casted_args)
        
        # Cast outputs back to float32
        if isinstance(outputs, tuple):
            return tuple(o.to(torch.float32) if isinstance(o, torch.Tensor) and o.is_floating_point() else o for o in outputs)
        elif isinstance(outputs, torch.Tensor) and outputs.is_floating_point():
            return outputs.to(torch.float32)
        return outputs

class PreprocessorWrapper(nn.Module):
    """
    Wraps the NeMo preprocessor (FilterbankFeaturesTA) for CoreML export.
    We need to ensure it takes (audio, length) and returns (features, length).
    """
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def forward(self, audio_signal, length):
        # NeMo preprocessor returns (features, length)
        # features shape: [B, D, T]
        return self.preprocessor(input_signal=audio_signal, length=length)

class SortformerCoreMLWrapper(nn.Module):
    """
    Wraps the entire Sortformer pipeline (Encoder + Streaming Logic for Export)
    The 'forward_for_export' method in the model is the target.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
        return self.model.forward_for_export(
             chunk=chunk,
             chunk_lengths=chunk_lengths,
             spkcache=spkcache,
             spkcache_lengths=spkcache_lengths,
             fifo=fifo,
             fifo_lengths=fifo_lengths
        )

def export_model(model_name, output_dir, use_fp16=True):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model: {model_name}")
    # Load model on CPU to avoid map_location issues if CUDA is not avail
    if os.path.exists(model_name):
        model = SortformerEncLabelModel.restore_from(model_name, map_location="cpu")
    else:
        model = SortformerEncLabelModel.from_pretrained(model_name, map_location="cpu")
    model.eval()
    print(f"preprocessor: {model._cfg['preprocessor']}")
    
    # --- Override Config for Streaming (Match single_file.py) ---
    print("Overriding Model Config for Low-Latency Streaming (chunk_len=4)...")
    model.sortformer_modules.chunk_len = 4
    model.sortformer_modules.chunk_right_context = 1
    model.sortformer_modules.fifo_len = 63
    model.sortformer_modules.spkcache_len = 63
    model.sortformer_modules.spkcache_update_period = 50
    
    print(f"Sortformer Config: chunk_len={model.sortformer_modules.chunk_len}, "
          f"fifo_len={model.sortformer_modules.fifo_len}, "
          f"spkcache_len={model.sortformer_modules.spkcache_len}")
          
    # --- Monkey-Patch model instance ---
    print("Exporting Preprocessor...")
    preprocessor = model.preprocessor
    
    # Disable pad_to to avoid dynamic padding issues in CoreML with RangeDim
    if hasattr(preprocessor, 'pad_to'):
        print(f"Disabling preprocessor pad_to (was {preprocessor.pad_to}) used for flexible input support.")
        preprocessor.pad_to = 0
    
    # --- Monkey-Patch model instance ---
    # Ensure forward_for_export uses our patched version
    # It calls self.concat_and_pad_script(...)
    # We assign our function directly to the instance.
    model.concat_and_pad_script = patched_concat_and_pad
    print("Patched model.concat_and_pad_script with custom implementation.")

    # Create wrapper
    preproc_wrapper = PreprocessorWrapper(preprocessor)
    precision = ct.precision.FLOAT16 if use_fp16 else ct.precision.FLOAT32
    
    # Calculate input info for Main Model first to sync preprocessor size
    modules = model.sortformer_modules
    chunk_len = modules.chunk_len
    # Calculate input size based on subsampling
    input_chunk_time = chunk_len * modules.subsampling_factor
    print(f"Calculated Input Chunk Time (Frames): {input_chunk_time}")
    
    # Calculate Audio Samples for Preprocessor
    # window=25ms (400), stride=10ms (160)
    # frames = (samples - window) // stride + 1
    # samples ~= (frames - 1) * stride + window
    stride = 160
    window = 400
    audio_samples = (input_chunk_time - 1) * stride + window
    print(f"Calculated Audio Samples for Chunk: {audio_samples} (matches {input_chunk_time} frames)")

    # Trace preprocessor
    dummy_wav = torch.randn(1, audio_samples)
    dummy_len = torch.tensor([audio_samples], dtype=torch.long)
    
    # We trace on CPU.
    traced_preproc = torch.jit.trace(preproc_wrapper, (dummy_wav, dummy_len))
    
    # Convert to CoreML
    preproc_mlmodel = ct.convert(
        traced_preproc,
        inputs=[
            ct.TensorType(name="audio_signal", shape=dummy_wav.shape, dtype=np.float32),
            ct.TensorType(name="length", shape=dummy_len.shape, dtype=np.int32)
        ],
        outputs=[
            ct.TensorType(name="features", dtype=np.float32),
            ct.TensorType(name="feature_lengths", dtype=np.int32)
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=precision
    )
    preproc_mlmodel.save(os.path.join(output_dir, "SortformerPreprocessor.mlpackage"))
    print("Saved SortformerPreprocessor.mlpackage")

    # --- Export Main Model ---
    print("Exporting Sortformer Main Model...")
    
    # Debug Config
    modules = model.sortformer_modules
    print(f"Sortformer Config: chunk_len={modules.chunk_len}, fifo_len={modules.fifo_len}, spkcache_len={modules.spkcache_len}")
    print(f"Subsampling Factor: {modules.subsampling_factor}")
    print(f"FC D_Model: {modules.fc_d_model}")
    
    # Encoder config
    if hasattr(model, 'encoder'):
        print(f"Encoder info: {model.encoder}")
        # Try to find input dim
        if hasattr(model.encoder, '_feat_in'):
            print(f"Encoder feat_in: {model.encoder._feat_in}")
    
    # Preprocessor info
    feat_dim = 80
    if hasattr(preprocessor, '_featurizer'):
        # FilterbankFeaturesTA has 'n_filt' in config usually
        pass
    if hasattr(model._cfg.preprocessor, 'features'):
        feat_dim = model._cfg.preprocessor.features
    elif hasattr(model._cfg.preprocessor, 'n_filt'):
        feat_dim = model._cfg.preprocessor.n_filt
    print(f"Feature Dim: {feat_dim}")

    fc_d_model = modules.fc_d_model # 512 normally
    spkcache_len = modules.spkcache_len
    fifo_len = modules.fifo_len
    chunk_len = modules.chunk_len
    
    # Calculate input size based on subsampling
    input_chunk_time = chunk_len * modules.subsampling_factor
    print(f"Calculated Input Chunk Time (Frames): {input_chunk_time}")

    # Use Encoder's feat_in for dimension
    if hasattr(model, 'encoder') and hasattr(model.encoder, '_feat_in'):
        feat_dim = model.encoder._feat_in
    else:
        # Fallback
        if hasattr(model._cfg.preprocessor, 'features'):
            feat_dim = model._cfg.preprocessor.features
    print(f"Using Feature Dim: {feat_dim}")

    wrapper = SortformerCoreMLWrapper(model)
    wrapper.eval()
    
    # Dummy inputs
    # Important: sortformer_diar_models.py forward_for_export calls self.encoder.pre_encode(chunk) directly.
    # ConformerEncoder.forward transposes (B, D, T) -> (B, T, D) before calling pre_encode.
    # Since forward_for_export skips this transpose, we must provide (B, T, D).
    input_chunk = torch.randn(1, input_chunk_time, feat_dim) # [B, T, D]
    input_chunk_len = torch.tensor([input_chunk_time], dtype=torch.long) # Use input time, not output chunk_len for the length tensor
    
    input_spkcache = torch.randn(1, spkcache_len, fc_d_model)
    input_spkcache_len = torch.tensor([spkcache_len], dtype=torch.long)
    
    input_fifo = torch.randn(1, fifo_len, fc_d_model)
    input_fifo_len = torch.tensor([fifo_len], dtype=torch.long)
    
    # Trace
    traced_model = torch.jit.trace(wrapper, (
        input_chunk, input_chunk_len,
        input_spkcache, input_spkcache_len,
        input_fifo, input_fifo_len
    ))

    print(f"Converting Main Model with precision={precision}...")
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="chunk", shape=input_chunk.shape),
            ct.TensorType(name="chunk_lengths", shape=input_chunk_len.shape, dtype=np.int32),
            ct.TensorType(name="spkcache", shape=input_spkcache.shape),
            ct.TensorType(name="spkcache_lengths", shape=input_spkcache_len.shape, dtype=np.int32),
            ct.TensorType(name="fifo", shape=input_fifo.shape),
            ct.TensorType(name="fifo_lengths", shape=input_fifo_len.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="preds"),
            ct.TensorType(name="chunk_embs"),
            ct.TensorType(name="chunk_emb_lengths", dtype=np.int32),
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=precision
    )
    
    mlmodel.save(os.path.join(output_dir, "Sortformer.mlpackage"))
    print("Saved Sortformer.mlpackage")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="nvidia/diar_streaming_sortformer_4spk-v2.1", help="NeMo model name or path")
    parser.add_argument("--output_dir", default="coreml_models", help="Output directory")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for CoreML")
    args = parser.parse_args()
    
    # Default to FP16 if not specified? Or should we force it per requirements?
    # Requirement: "FP16 internal compute"
    # So we should default to True or just pass True.
    # If args.fp16 is set, use it. But let's default to True (or pass explicit).
    # Actually, let's just default True in the function.
    
    print(f"CoreMLTools Version: {ct.__version__}")
    export_model(args.model_name, args.output_dir, use_fp16=True)

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
from coreml_wrappers import *


def export_model(model_name, output_dir, use_fp16=True, use_custom_concat=False):
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
    model.sortformer_modules.chunk_len = 6
    model.sortformer_modules.chunk_right_context = 1
    model.sortformer_modules.chunk_left_context = 1
    model.sortformer_modules.fifo_len = 40
    model.sortformer_modules.spkcache_len = 120
    model.sortformer_modules.spkcache_update_period = 42

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

    # Create wrapper
    preproc_wrapper = PreprocessorWrapper(preprocessor)
    precision = ct.precision.FLOAT16 if use_fp16 else ct.precision.FLOAT32

    # Calculate input info for Main Model first to sync preprocessor size
    modules = model.sortformer_modules
    chunk_len = modules.chunk_len
    # Calculate input size based on subsampling
    input_chunk_time = (
                                   chunk_len + modules.chunk_left_context + modules.chunk_right_context) * modules.subsampling_factor
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
            ct.TensorType(name="audio_signal", shape=dummy_wav.shape),
            ct.TensorType(name="length", shape=dummy_len.shape, dtype=np.int32)
        ],
        outputs=[
            ct.TensorType(name="features", dtype=np.float32),
            ct.TensorType(name="feature_lengths", dtype=np.int32)
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL
    )
    preproc_mlmodel.save(os.path.join(output_dir, "SortformerPreprocessor.mlpackage"))
    print("Saved SortformerPreprocessor.mlpackage")

    # --- Export Main Model ---
    print("Exporting Sortformer Main Model...")

    # Debug Config
    modules = model.sortformer_modules
    print(
        f"Sortformer Config: chunk_len={modules.chunk_len}, fifo_len={modules.fifo_len}, spkcache_len={modules.spkcache_len}")
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

    fc_d_model = modules.fc_d_model  # 512 normally
    spkcache_len = modules.spkcache_len
    fifo_len = modules.fifo_len
    chunk_len = modules.chunk_len

    # Calculate input size based on subsampling (including context)
    input_chunk_time = (
                                   chunk_len + modules.chunk_left_context + modules.chunk_right_context) * modules.subsampling_factor
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

    conformer_encoder_wrapper = ConformerEncoderWrapper(model)
    transformer_encoder_wrapper = SortformerEncoderWrapper(model)
    wrapper.eval()

    conformer_encoder_wrapper.eval()
    transformer_encoder_wrapper.eval()

    # Dummy inputs
    # Important: sortformer_diar_models.py forward_for_export calls self.encoder.pre_encode(chunk) directly.
    # ConformerEncoder.forward transposes (B, D, T) -> (B, T, D) before calling pre_encode.
    # Since forward_for_export skips this transpose, we must provide (B, T, D).
    input_chunk = torch.randn(1, input_chunk_time, feat_dim)  # [B, T, D]
    input_chunk_len = torch.tensor([input_chunk_time],
                                   dtype=torch.long)  # Use input time, not output chunk_len for the length tensor

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
            ct.TensorType(name="chunk", shape=input_chunk.shape, dtype=np.float32),
            ct.TensorType(name="chunk_lengths", shape=input_chunk_len.shape, dtype=np.int32),
            ct.TensorType(name="spkcache", shape=input_spkcache.shape),
            ct.TensorType(name="spkcache_lengths", shape=input_spkcache_len.shape, dtype=np.int32),
            ct.TensorType(name="fifo", shape=input_fifo.shape),
            ct.TensorType(name="fifo_lengths", shape=input_fifo_len.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="speaker_preds"),
            ct.TensorType(name="chunk_pre_encoder_embs"),
            ct.TensorType(name="chunk_pre_encoder_lengths", dtype=np.int32),
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL
    )

    mlmodel.save(os.path.join(output_dir, f"Sortformer{'16' if use_fp16 else '32'}.mlpackage"))
    print("Saved Sortformer.mlpackage")


def convert_pre_encoder(
        model: SortformerEncLabelModel,
        precision,
        name: str,
        input_chunk, input_chunk_len,
        input_spkcache, input_spkcache_len,
        input_fifo, input_fifo_len
):
    wrapper = PreEncoderWrapper(model)
    wrapper.eval()

    traced_model = torch.jit.trace(wrapper, (
        input_chunk, input_chunk_len,
        input_spkcache, input_spkcache_len,
        input_fifo, input_fifo_len
    ))

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="chunk", shape=input_chunk.shape, dtype=np.float32),
            ct.TensorType(name="chunk_lengths", shape=input_chunk_len.shape, dtype=np.int32),
            ct.TensorType(name="spkcache", shape=input_spkcache.shape),
            ct.TensorType(name="spkcache_lengths", shape=input_spkcache_len.shape, dtype=np.int32),
            ct.TensorType(name="fifo", shape=input_fifo.shape),
            ct.TensorType(name="fifo_lengths", shape=input_fifo_len.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="pre_encoder_embs", dtype=np.float32),
            ct.TensorType(name="pre_encoder_lengths", dtype=np.int32),
            ct.TensorType(name="chunk_pre_encoder_embs", dtype=np.float32),
            ct.TensorType(name="chunk_pre_encoder_lengths", dtype=np.int32),
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY
    )

    mlmodel.save(name)
    return mlmodel, traced_model


def convert_head(
        model: SortformerEncLabelModel,
        precision,
        name: str,
        pre_encoder_embs, pre_encoder_lengths,
        chunk_pre_encoder_embs, chunk_pre_encoder_lengths,
):
    wrapper = SortformerHeadWrapper(model)
    wrapper.eval()

    traced_model = torch.jit.trace(wrapper, (
        pre_encoder_embs, pre_encoder_lengths,
        chunk_pre_encoder_embs, chunk_pre_encoder_lengths,
    ))

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="pre_encoder_embs", shape=pre_encoder_embs.shape, dtype=np.float32),
            ct.TensorType(name="pre_encoder_lengths", shape=pre_encoder_lengths.shape, dtype=np.int32),
            ct.TensorType(name="chunk_pre_encoder_embs", shape=chunk_pre_encoder_embs.shape, dtype=np.float32),
            ct.TensorType(name="chunk_pre_encoder_lengths", shape=chunk_pre_encoder_lengths.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="speaker_preds", dtype=np.float32),
            ct.TensorType(name="chunk_pre_encoder_embs"),
            ct.TensorType(name="chunk_pre_encoder_lengths")
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL
    )

    mlmodel.save(name)
    return mlmodel, traced_model


def export_pipeline(
        model_name: str,
        output_dir: str,
        preproc_precision: str = "fp32",
        pre_encoder_precision: str = "fp32",
        head_precision: str = "fp16",
        skip_modules: bool = False,
        verify: bool = False
):
    """
    Export the Sortformer model as a pipeline of separate CoreML models.
    Each component can have different precision.
    
    Components:
    1. Preprocessor (audio -> mel features)
    2. Pre-encoder (features -> pre-encoded embeddings + concat with spkcache/fifo)
    3. Conformer Encoder (pre-encoded -> encoder embeddings)
    4. Transformer Encoder (encoder embeddings -> predictions)
    
    Args:
        :param model_name: NeMo model name or path
        :param output_dir: Output directory for mlpackage files
        :param preproc_precision: Precision for preprocessor ("fp16" or "fp32")
        :param pre_encoder_precision: Precision for pre-encoder ("fp16" or "fp32")
        :param head_precision: Precision for head module (conformer + transformer) ("fp16" or "fp32")
        :param skip_modules: Whether to skip the individual modules
    """
    os.makedirs(output_dir, exist_ok=True)

    def get_precision(s):
        return ct.precision.FLOAT16 if s.lower() == "fp16" else ct.precision.FLOAT32

    print("=" * 70)
    print("Exporting Sortformer Pipeline")
    print("=" * 70)
    print(f"Preprocessor:   {preproc_precision}")
    print(f"Pre-encoder:    {pre_encoder_precision}")
    print(f"Head:      {head_precision}")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {model_name}")
    if os.path.exists(model_name):
        model = SortformerEncLabelModel.restore_from(model_name, map_location="cpu")
    else:
        model = SortformerEncLabelModel.from_pretrained(model_name, map_location="cpu")
    model.eval()

    # Configure for streaming
    print("Configuring for streaming...")
    model.sortformer_modules.chunk_len = 6
    model.sortformer_modules.chunk_right_context = 1
    model.sortformer_modules.chunk_left_context = 1
    model.sortformer_modules.fifo_len = 40
    model.sortformer_modules.spkcache_len = 120
    model.sortformer_modules.spkcache_update_period = 32

    modules = model.sortformer_modules
    preprocessor = model.preprocessor
    pre_encoder_mlmodel = None
    head_mlmodel = None

    if hasattr(preprocessor, 'pad_to'):
        preprocessor.pad_to = 0

    # Calculate dimensions
    chunk_len = modules.chunk_len
    input_chunk_time = (chunk_len + modules.chunk_left_context + modules.chunk_right_context) * modules.subsampling_factor
    fc_d_model = modules.fc_d_model  # 512 - Conformer output
    tf_d_model = modules.tf_d_model  # 192 - Transformer input (after projection)
    spkcache_len = modules.spkcache_len
    fifo_len = modules.fifo_len

    # Get feature dim
    feat_dim = 128
    if hasattr(model, 'encoder') and hasattr(model.encoder, '_feat_in'):
        feat_dim = model.encoder._feat_in

    # Pre-encode output size (after subsampling)
    pre_encode_out_len = input_chunk_time // modules.subsampling_factor
    total_concat_len = spkcache_len + fifo_len + pre_encode_out_len

    print(f"Input chunk frames: {input_chunk_time}")
    print(f"Pre-encode output: {pre_encode_out_len}")
    print(f"Total concat len: {total_concat_len}")
    print(f"Feature dim: {feat_dim}, FC d_model: {fc_d_model}, TF d_model: {tf_d_model}")

    # Audio samples for preprocessor
    stride = 160
    window = 400
    audio_samples = (input_chunk_time - 1) * stride + window
    print(audio_samples)

    # =========================================================
    # 1. Export Preprocessor
    # =========================================================

    if not skip_modules:
        print("\n[1/4] Exporting Preprocessor...")

        preproc_wrapper = PreprocessorWrapper(preprocessor)
        preproc_wrapper.eval()

        dummy_wav = torch.randn(1, audio_samples)
        dummy_len = torch.tensor([audio_samples], dtype=torch.long)

        traced_preproc = torch.jit.trace(preproc_wrapper, (dummy_wav, dummy_len))

        preproc_mlmodel = ct.convert(
            traced_preproc,
            inputs=[
                ct.TensorType(name="audio_signal", shape=dummy_wav.shape),
                ct.TensorType(name="length", shape=dummy_len.shape, dtype=np.int32)
            ],
            outputs=[
                ct.TensorType(name="features", dtype=np.float32),
                ct.TensorType(name="feature_lengths", dtype=np.int32)
            ],
            minimum_deployment_target=ct.target.iOS16,
            compute_precision=get_precision(preproc_precision),
            compute_units=ct.ComputeUnit.ALL
        )
        preproc_mlmodel.save(os.path.join(output_dir, "Pipeline_Preprocessor.mlpackage"))
        print("  Saved Pipeline_Preprocessor.mlpackage")

    # =========================================================
    # 2. Export Pre-Encoder
    # =========================================================

    input_chunk = torch.randn(1, input_chunk_time, feat_dim)
    input_chunk_len = torch.tensor([input_chunk_time], dtype=torch.long)
    input_spkcache = torch.randn(1, spkcache_len, fc_d_model)
    input_spkcache_len = torch.tensor([spkcache_len], dtype=torch.long)
    input_fifo = torch.randn(1, fifo_len, fc_d_model)
    input_fifo_len = torch.tensor([fifo_len], dtype=torch.long)

    if not skip_modules:
        print("\n[2/4] Exporting Pre-Encoder...")
        pre_encoder_mlmodel, _ = convert_pre_encoder(
            model,
            get_precision(pre_encoder_precision),
            os.path.join(output_dir, "Pipeline_PreEncoder.mlpackage"),
            input_chunk, input_chunk_len,
            input_spkcache, input_spkcache_len,
            input_fifo, input_fifo_len
        )
        print("  Saved Pipeline_PreEncoder.mlpackage")

    # =========================================================
    # 3. Export Conformer Encoder
    # =========================================================

    pre_encoder_embs = torch.randn(1, total_concat_len, fc_d_model)
    pre_encoder_lengths = torch.tensor([total_concat_len], dtype=torch.long)
    chunk_pre_encoder_embs = torch.randn(1, pre_encode_out_len, fc_d_model)
    chunk_pre_encoder_lengths = torch.tensor([pre_encode_out_len], dtype=torch.long)

    if not skip_modules:
        print("\n[3/4] Exporting Head Module...")
        head_mlmodel, _ = convert_head(
            model,
            get_precision(head_precision),
            os.path.join(output_dir, "Pipeline_Head.mlpackage"),
            pre_encoder_embs, pre_encoder_lengths,
            chunk_pre_encoder_embs, chunk_pre_encoder_lengths
        )
        print("  Saved Pipeline_Head.mlpackage")

    # =========================================================
    # 5. Create Combined Pipelines
    # =========================================================
    print("\n[4/4] Creating Combined ML Pipelines...")

    from coremltools.models.pipeline import Pipeline
    from coremltools.models import datatypes

    # Load the exported models
    if skip_modules and not verify:
        print('Loading Pipeline CoreML Modules...')
        pre_encoder_mlmodel = ct.models.MLModel(
            os.path.join(output_dir, "Pipeline_PreEncoder.mlpackage")
        )
        head_mlmodel = ct.models.MLModel(
            os.path.join(output_dir, "Pipeline_Head.mlpackage")
        )

        assert pre_encoder_mlmodel is not None and head_mlmodel is not None

    # Create Full Pipeline: PreEncoder → Conformer → Transformer
    # Inputs: chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths
    # Output: preds

    if verify:
        pipeline_model = ct.models.MLModel('coreml_models/SortformerPipeline.mlpackage')
        spec = pipeline_model.get_spec()
        print(pipeline_model.input_description)
        print(pipeline_model.output_description)
        print(spec)
    else:
        try:
            pipeline_model = ct.utils.make_pipeline(pre_encoder_mlmodel, head_mlmodel)

            # Save the pipeline
            pipeline_model.save(os.path.join(output_dir, "SortformerPipeline.mlpackage"))
            print("  Saved SortformerPipeline.mlpackage (PreEncoder + Conformer + Transformer)")
        except Exception as e:
            print(f"  Warning: Could not create full pipeline: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("Pipeline Export Complete!")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print("\nExported models:")
    print(f"  1. Pipeline_Preprocessor.mlpackage        ({preproc_precision})")
    print(f"  2. Pipeline_PreEncoder.mlpackage          ({pre_encoder_precision})")
    print(f"  3. Pipeline_Head.mlpackage                ({head_precision})")
    print(f"  5. SortformerPipeline.mlpackage           (combined: PreEncoder+Head)")
    print("\nUsage in inference:")
    print("  audio -> Preprocessor -> features")
    print("  features + spkcache + fifo -> SortformerPipeline -> predictions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="nvidia/diar_streaming_sortformer_4spk-v2.1",
                        help="NeMo model name or path")
    parser.add_argument("--output_dir", default="coreml_models", help="Output directory")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for single model export")

    # Pipeline options
    parser.add_argument("--pipeline", action="store_true", help="Export as pipeline with separate models")
    parser.add_argument("--preproc_precision", default="fp32", choices=["fp16", "fp32"], help="Preprocessor precision")
    parser.add_argument("--pre_encoder_precision", default="fp32", choices=["fp16", "fp32"],
                        help="Pre-encoder precision")
    parser.add_argument("--head_precision", default="fp16", choices=["fp16", "fp32"],
                        help="Conformer encoder precision")
    parser.add_argument("--skip_modules", action="store_true", help="Skip modules in pipeline export")
    parser.add_argument("--verify", action="store_true", help="Skip pipeline in pipeline export")

    args = parser.parse_args()

    print(f"CoreMLTools Version: {ct.__version__}")

    if args.pipeline:
        export_pipeline(
            args.model_name,
            args.output_dir,
            preproc_precision=args.preproc_precision,
            pre_encoder_precision=args.pre_encoder_precision,
            head_precision=args.head_precision,
            skip_modules=args.skip_modules,
            verify=args.verify,
        )
    else:
        export_model(args.model_name, args.output_dir, use_fp16=args.fp16)

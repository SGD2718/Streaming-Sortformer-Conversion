import torch
import numpy as np
import coremltools as ct
import os
import argparse
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeaturesTA

def verify_models(model_name, coreml_model_dir):
    print(f"Loading NeMo model: {model_name}")
    if os.path.exists(model_name):
        nemo_model = SortformerEncLabelModel.restore_from(model_name, map_location="cpu")
    else:
        nemo_model = SortformerEncLabelModel.from_pretrained(model_name, map_location="cpu")
    nemo_model.eval()

    nemo_model.sortformer_modules.chunk_len = 4
    nemo_model.sortformer_modules.chunk_right_context = 1
    nemo_model.sortformer_modules.fifo_len = 125
    nemo_model.sortformer_modules.spkcache_len = 125
    nemo_model.sortformer_modules.spkcache_update_period = 63
    
    # Load CoreML models
    preproc_path = os.path.join(coreml_model_dir, "SortformerPreprocessor.mlpackage")
    main_path = os.path.join(coreml_model_dir, "Sortformer.mlpackage")
    
    print(f"Loading CoreML models from {coreml_model_dir}...")
    coreml_preproc = ct.models.MLModel(preproc_path)
    coreml_main = ct.models.MLModel(main_path)
    
    # Setup Inputs
    modules = nemo_model.sortformer_modules
    chunk_len = modules.chunk_len
    subsampling_factor = modules.subsampling_factor # 8
    input_chunk_time = chunk_len * subsampling_factor # 1504
    
    # Encoder feat_in
    if hasattr(nemo_model, 'encoder') and hasattr(nemo_model.encoder, '_feat_in'):
        feat_dim = nemo_model.encoder._feat_in
    else:
        feat_dim = 80  # Fallback
        
    fc_d_model = modules.fc_d_model
    spkcache_len = modules.spkcache_len
    fifo_len = modules.fifo_len
    
    print(f"Config: Chunk Time={input_chunk_time}, Feat Dim={feat_dim}, SpkCache={spkcache_len}, Fifo={fifo_len}")
    
    # Calculate expected audio samples for preprocessor
    stride = 160
    window = 400
    expected_audio_samples = (input_chunk_time - 1) * stride + window
    print(f"Expected Audio Samples for CoreML: {expected_audio_samples}")

    # 1. Verify Preprocessor with 'audio.wav'
    print("\n--- Verifying Preprocessor (audio.wav) ---")
    
    import librosa
    if os.path.exists("audio.wav"):
        audio_path = "audio.wav"
        print(f"Loading {audio_path}...")
        wav, sr_native = librosa.load(audio_path, sr=16000, mono=True)
    else:
        print("audio.wav not found, generating dummy audio.")
        wav = np.random.randn(expected_audio_samples).astype(np.float32)
        sr_native = 16000

    # Handle length
    current_samples = wav.shape[0]
    if current_samples < expected_audio_samples:
        pad_amt = expected_audio_samples - current_samples
        print(f"Padding audio with {pad_amt} samples of silence.")
        wav = np.pad(wav, (0, pad_amt))
    elif current_samples > expected_audio_samples:
        print(f"Truncating audio to {expected_audio_samples} samples.")
        wav = wav[:expected_audio_samples]
        
    wav_tensor = torch.from_numpy(wav).unsqueeze(0).float() # [1, L]
    len_tensor = torch.tensor([wav.shape[0]], dtype=torch.int32)
    
    # NeMo Preprocessor
    with torch.no_grad():
        # Ensure pad_to=0
        if hasattr(nemo_model.preprocessor, 'pad_to'):
            nemo_model.preprocessor.pad_to = 0
            
        nemo_feats, nemo_feat_lens = nemo_model.preprocessor(input_signal=wav_tensor, length=len_tensor)
    
    # CoreML Preprocessor
    preproc_inputs = {
        "audio_signal": wav_tensor.numpy(),
        "length": len_tensor.numpy().astype(np.int32)
    }
    coreml_out = coreml_preproc.predict(preproc_inputs)
    coreml_feats = torch.from_numpy(coreml_out["features"])
    
    print(f"NeMo Feats: {nemo_feats.shape}, CoreML Feats: {coreml_feats.shape}")
    
    diff = (nemo_feats - coreml_feats).abs()
    print(f"Preprocessor Max Diff: {diff.max().item()}")
    print(f"Preprocessor Mean Diff: {diff.mean().item()}")
    print(f"Nemo stddev: {torch.std(nemo_feats)}")
    
    # 2. Verify Main Model
    print("\n--- Verifying Main Model (First Chunk) ---")
    
    # Slice first chunk from FEATURES
    # We need exactly input_chunk_time frames.
    if nemo_feats.shape[2] < input_chunk_time:
        print(f"Warning: Audio too short for one chunk! Need {input_chunk_time} frames, have {nemo_feats.shape[2]}. code will crash or pad.")
        # Pad if needed for verification
        pad_amt = input_chunk_time - nemo_feats.shape[2]
        chunk = torch.nn.functional.pad(nemo_feats, (0, pad_amt))
    else:
        chunk = nemo_feats[:, :, :input_chunk_time]
        
    # Input is (B, D, T) -> CoreML model expects (B, T, D) if we transposed it? 
    # Logic in convert_to_coreml: 
    # input_chunk = torch.randn(1, input_chunk_time, feat_dim) # [B, T, D]
    # And we determined forward_for_export expects [B, T, D] because it skips Conformer transpose.
    # nemo_feats output is usually [B, D, T] (from FilterbankFeaturesTA).
    # So we MUST TRANSPOSE here.
    
    chunk = chunk.transpose(1, 2) # [B, T, D]
    input_chunk = chunk.float()
    
    chunk_lengths = torch.tensor([input_chunk_time], dtype=torch.int32)
    
    spkcache = torch.randn(1, spkcache_len, fc_d_model) # Random state is fine
    spkcache_lengths = torch.tensor([spkcache_len], dtype=torch.int32)
    
    fifo = torch.randn(1, fifo_len, fc_d_model)
    fifo_lengths = torch.tensor([fifo_len], dtype=torch.int32)
    
    # Output expected names
    # preds, chunk_embs, chunk_emb_lengths
    
    # Sync patch for NeMo model?
    # verify_coreml.py runs in a fresh process. 
    # If the standard NeMo model uses `concat_and_pad` which has `len()` issue, it might fail in JIT trace but RUNS fine in Eager PyTorch.
    # So we don't need to patch it for inference comparison, usually.
    # Unless `forward_for_export` calls `self.concat_and_pad_script` which doesn't exist?
    # We found `concat_and_pad_script` was not defined.
    # So we MUST patch it or use the one I patched in `convert_to_coreml`?
    # Wait, if `concat_and_pad_script` is missing, NeMo inference will fail.
    # If NeMo's original code fails, then `convert_to_coreml` patch MUST be intended to fix a broken model or provide the missing piece.
    # The reference `sortformer_diar_models.py` calls it.
    # Let's try running NeMo inference. If it fails, I'll add the patch here too.
    
    # We need to define `patched_concat_and_pad` identically or import it?
    # To keep this script standalone, I'll redefine it IF needed.
    
    try:
        # Patching if needed
        from nemo.collections.asr.models.sortformer_modules import SortformerModules
        # If we can't import, we skip this check and see if it runs.
        # But earlier I couldn't import it.
        pass 
    except ImportError:
        pass

    # Basic Patch (Function)
    def simple_concat_and_pad(embs, lengths):
        # Emulates concat_and_pad for B=1
        parts = [emb[:, :l[0], :] for emb, l in zip(embs, lengths)]
        out = torch.cat(parts, dim=1)
        total = torch.sum(torch.stack(lengths), dim=0)
        return out, total
        
    # Check if model has the method
    if not hasattr(nemo_model, 'concat_and_pad_script'):
        print("Model missing concat_and_pad_script, applying patch...")
        nemo_model.concat_and_pad_script = simple_concat_and_pad
        
    # Run NeMo
    with torch.no_grad():
        # Casting to correct types for PyTorch model (it might expect LongTensor for lengths)
        # CoreML inputs are Int32. PyTorch usually handles Int32/Int64 fine, or prefers Int64 for embedding/indexing.
        n_c_len = chunk_lengths.long()
        n_s_len = spkcache_lengths.long()
        n_f_len = fifo_lengths.long()
        
        n_preds, n_embs, n_emb_lens = nemo_model.forward_for_export(
            chunk=chunk,
            chunk_lengths=n_c_len,
            spkcache=spkcache,
            spkcache_lengths=n_s_len,
            fifo=fifo,
            fifo_lengths=n_f_len
        )

    # Run CoreML
    coreml_inputs = {
        "chunk": chunk.numpy(),
        "chunk_lengths": chunk_lengths.numpy().astype(np.int32),
        "spkcache": spkcache.numpy(),
        "spkcache_lengths": spkcache_lengths.numpy().astype(np.int32),
        "fifo": fifo.numpy(),
        "fifo_lengths": fifo_lengths.numpy().astype(np.int32)
    }
    
    c_out = coreml_main.predict(coreml_inputs)
    c_preds = torch.from_numpy(c_out["preds"])
    c_embs = torch.from_numpy(c_out["chunk_embs"])
    # c_emb_lens = torch.from_numpy(c_out["chunk_emb_lengths"]) # Check if exists
    
    print("\n--- Comparison ---")
    print(f"Preds Shape: NeMo {n_preds.shape} vs CoreML {c_preds.shape}")
    diff_preds = (n_preds - c_preds).abs()
    print(f"Preds Diff: Max {diff_preds.max().item():.6f}, Mean {diff_preds.mean().item():.6f}")
    
    print(f"Embs Shape: NeMo {n_embs.shape} vs CoreML {c_embs.shape}")
    diff_embs = (n_embs - c_embs).abs()
    print(f"Embs Diff: Max {diff_embs.max().item():.6f}, Mean {diff_embs.mean().item():.6f}")
    
    print(f"NeMo Embs Range: {n_embs.min():.4f} to {n_embs.max():.4f}")
    print(f"CoreML Embs Range: {c_embs.min():.4f} to {c_embs.max():.4f}")

    if diff_preds.max().item() > 1e-3:
        print("WARNING: High Preds Difference!")
    if diff_embs.max().item() > 1.0: # 0.5 is suspicious but maybe scale dependent
        print("WARNING: High Embs Difference!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    parser.add_argument("--coreml_dir", default="coreml_models")
    args = parser.parse_args()
    
    verify_models(args.model_name, args.coreml_dir)

class Config:
    chunk_len = 6  # Core diarization frames per chunk
    chunk_right_context = 7
    chunk_left_context = 1
    fifo_len = 40
    spkcache_len = 188
    spkcache_update_period = 31

    # do not touch these
    subsampling_factor = 8
    sample_rate = 16000
    mel_window = 400
    mel_stride = 160
    frame_duration = 0.08

    # Full chunk size for model input (includes context)
    chunk_frames = (chunk_len + chunk_right_context + chunk_left_context) * subsampling_factor  # 112

    # CoreML model input size (what the preprocessor was exported with)
    coreml_audio_samples = (chunk_frames - 1) * mel_stride + mel_window  # 18160 samples for chunk_frames=112

    # Preprocessing stride - how much NEW audio per preproc call for ~2Hz updates
    preproc_feature_frames = chunk_len * subsampling_factor  # 48 mel frames = 480ms worth
    preproc_audio_hop = preproc_feature_frames * mel_stride  # 7680 samples = 480ms

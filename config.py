class Config:
    chunk_len = 6
    chunk_right_context = 7
    chunk_left_context = 1
    fifo_len = 40
    spkcache_len = 188
    spkcache_update_period = 40

    # do not touch these
    subsampling_factor = 8
    sample_rate = 16000
    mel_window = 400
    mel_stride = 160
    frame_duration = 0.08

    # computed
    chunk_frames = (chunk_len + chunk_right_context + chunk_left_context) * subsampling_factor
    preproc_audio_samples = (chunk_frames - 1) * mel_stride + mel_window

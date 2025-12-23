import torch
import random
import math
import tqdm


def forward_streaming(
        self,
        processed_signal,
        processed_signal_length,
):
    """
    The main forward pass for diarization inference in streaming mode.

    Args:
        processed_signal (torch.Tensor): Tensor containing audio waveform
            Shape: (batch_size, num_samples)
        processed_signal_length (torch.Tensor): Tensor containing lengths of audio waveforms
            Shape: (batch_size,)

    Returns:
        total_preds (torch.Tensor): Tensor containing predicted speaker labels for the current chunk
            and all previous chunks
            Shape: (batch_size, pred_len, num_speakers)
    """
    streaming_state = self.sortformer_modules.init_streaming_state(
        batch_size=processed_signal.shape[0], async_streaming=self.async_streaming, device=self.device
    )

    batch_size, ch, sig_length = processed_signal.shape
    processed_signal_offset = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
    max_n_frames = sig_length

    if sig_length < max_n_frames:  # need padding to have the same feature length for all GPUs
        pad_tensor = torch.full(
            (batch_size, ch, max_n_frames - sig_length),
            self.negative_init_val,
            dtype=processed_signal.dtype,
            device=processed_signal.device,
        )
        processed_signal = torch.cat([processed_signal, pad_tensor], dim=2)

    att_mod = False

    total_preds = torch.zeros((batch_size, 0, self.sortformer_modules.n_spk), device=self.device)

    feat_len = processed_signal.shape[2]
    num_chunks = math.ceil(
        feat_len / (self.sortformer_modules.chunk_len * self.sortformer_modules.subsampling_factor)
    )
    streaming_loader = self.sortformer_modules.streaming_feat_loader(
        feat_seq=processed_signal,
        feat_seq_length=processed_signal_length,
        feat_seq_offset=processed_signal_offset,
    )
    for _, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in tqdm(
            streaming_loader,
            total=num_chunks,
            desc="Streaming Steps",
            disable=self.training,
    ):
        streaming_state, total_preds = self.forward_streaming_step(
            processed_signal=chunk_feat_seq_t,
            processed_signal_length=feat_lengths,
            streaming_state=streaming_state,
            total_preds=total_preds,
            left_offset=left_offset,
            right_offset=right_offset,
        )

    del processed_signal, processed_signal_length

    if sig_length < max_n_frames:  # Discard preds corresponding to padding
        n_frames = math.ceil(sig_length / self.encoder.subsampling_factor)
        total_preds = total_preds[:, :n_frames, :]
    return total_preds


def forward_streaming_step(
        self,
        processed_signal,
        processed_signal_length,
        streaming_state,
        total_preds,
        drop_extra_pre_encoded=0,
        left_offset=0,
        right_offset=0,
):
    """
    One-step forward pass for diarization inference in streaming mode.

    Args:
        processed_signal (torch.Tensor): Tensor containing audio waveform
            Shape: (batch_size, num_samples)
        processed_signal_length (torch.Tensor): Tensor containing lengths of audio waveforms
            Shape: (batch_size,)
        streaming_state (SortformerStreamingState):
                Tensor variables that contain the streaming state of the model.
                Find more details in the `SortformerStreamingState` class in `sortformer_modules.py`.

            Attributes:
                spkcache (torch.Tensor): Speaker cache to store embeddings from start
                spkcache_lengths (torch.Tensor): Lengths of the speaker cache
                spkcache_preds (torch.Tensor): The speaker predictions for the speaker cache parts
                fifo (torch.Tensor): FIFO queue to save the embedding from the latest chunks
                fifo_lengths (torch.Tensor): Lengths of the FIFO queue
                fifo_preds (torch.Tensor): The speaker predictions for the FIFO queue parts
                spk_perm (torch.Tensor): Speaker permutation information for the speaker cache

        total_preds (torch.Tensor): Tensor containing total predicted speaker activity probabilities
            Shape: (batch_size, cumulative pred length, num_speakers)
        left_offset (int): left offset for the current chunk
        right_offset (int): right offset for the current chunk

    Returns:
        streaming_state (SortformerStreamingState):
                Tensor variables that contain the updated streaming state of the model from
                this function call.
        total_preds (torch.Tensor):
            Tensor containing the updated total predicted speaker activity probabilities.
            Shape: (batch_size, cumulative pred length, num_speakers)
    """
    chunk_pre_encode_embs, chunk_pre_encode_lengths = self.encoder.pre_encode(
        x=processed_signal, lengths=processed_signal_length
    )
    # To match the output of the ASR model, we need to drop the extra pre-encoded embeddings
    if drop_extra_pre_encoded > 0:
        chunk_pre_encode_embs = chunk_pre_encode_embs[:, drop_extra_pre_encoded:, :]
        chunk_pre_encode_lengths = chunk_pre_encode_lengths - drop_extra_pre_encoded

    spkcache_fifo_chunk_pre_encode_embs = self.sortformer_modules.concat_embs(
        [streaming_state.spkcache, streaming_state.fifo, chunk_pre_encode_embs], dim=1, device=self.device
    )
    spkcache_fifo_chunk_pre_encode_lengths = (
            streaming_state.spkcache.shape[1] + streaming_state.fifo.shape[1] + chunk_pre_encode_lengths
    )
    spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.frontend_encoder(
        processed_signal=spkcache_fifo_chunk_pre_encode_embs,
        processed_signal_length=spkcache_fifo_chunk_pre_encode_lengths,
        bypass_pre_encode=True,
    )
    spkcache_fifo_chunk_preds = self.forward_infer(
        emb_seq=spkcache_fifo_chunk_fc_encoder_embs, emb_seq_length=spkcache_fifo_chunk_fc_encoder_lengths
    )

    spkcache_fifo_chunk_preds = self.sortformer_modules.apply_mask_to_preds(
        spkcache_fifo_chunk_preds, spkcache_fifo_chunk_fc_encoder_lengths
    )

    streaming_state, chunk_preds = self.sortformer_modules.streaming_update(
        streaming_state=streaming_state,
        chunk=chunk_pre_encode_embs,
        preds=spkcache_fifo_chunk_preds,
        lc=round(left_offset / self.encoder.subsampling_factor),
        rc=math.ceil(right_offset / self.encoder.subsampling_factor),
    )
    total_preds = torch.cat([total_preds, chunk_preds], dim=1)

    return streaming_state, total_preds


def streaming_update(self, streaming_state, chunk, preds, lc: int = 0, rc: int = 0):
    """
    Update the speaker cache and FIFO queue with the chunk of embeddings and speaker predictions.
    Synchronous version, which means speaker cahce, FIFO queue and chunk have same lengths within a batch.
    Should be used for training and evaluation, not for real streaming applications.

    Args:
        streaming_state (SortformerStreamingState): previous streaming state including speaker cache and FIFO
        chunk (torch.Tensor): chunk of embeddings to be predicted
            Shape: (batch_size, lc+chunk_len+rc, emb_dim)
        preds (torch.Tensor): speaker predictions of the [spkcache + fifo + chunk] embeddings
            Shape: (batch_size, spkcache_len + fifo_len + lc+chunk_len+rc, num_spks)
        lc and rc (int): left & right offset of the chunk,
            only the chunk[:, lc:chunk_len+lc] is used for update of speaker cache and FIFO queue

    Returns:
        streaming_state (SortformerStreamingState): current streaming state including speaker cache and FIFO
        chunk_preds (torch.Tensor): speaker predictions of the chunk embeddings
            Shape: (batch_size, chunk_len, num_spks)
    """

    batch_size, _, emb_dim = chunk.shape

    spkcache_len, fifo_len, chunk_len = (
        streaming_state.spkcache.shape[1],
        streaming_state.fifo.shape[1],
        chunk.shape[1] - lc - rc,
    )
    if streaming_state.spk_perm is not None:
        inv_spk_perm = torch.stack(
            [torch.argsort(streaming_state.spk_perm[batch_index]) for batch_index in range(batch_size)]
        )
        preds = torch.stack(
            [preds[batch_index, :, inv_spk_perm[batch_index]] for batch_index in range(batch_size)]
        )

    streaming_state.fifo_preds = preds[:, spkcache_len : spkcache_len + fifo_len]
    chunk = chunk[:, lc : chunk_len + lc]
    chunk_preds = preds[:, spkcache_len + fifo_len + lc : spkcache_len + fifo_len + chunk_len + lc]

    # append chunk to fifo
    streaming_state.fifo = torch.cat([streaming_state.fifo, chunk], dim=1)
    streaming_state.fifo_preds = torch.cat([streaming_state.fifo_preds, chunk_preds], dim=1)

    if fifo_len + chunk_len > self.fifo_len:
        # extract pop_out_len first frames from FIFO queue
        pop_out_len = self.spkcache_update_period
        pop_out_len = max(pop_out_len, chunk_len - self.fifo_len + fifo_len)
        pop_out_len = min(pop_out_len, fifo_len + chunk_len)

        pop_out_embs = streaming_state.fifo[:, :pop_out_len]
        pop_out_preds = streaming_state.fifo_preds[:, :pop_out_len]
        streaming_state.mean_sil_emb, streaming_state.n_sil_frames = self._get_silence_profile(
            streaming_state.mean_sil_emb,
            streaming_state.n_sil_frames,
            pop_out_embs,
            pop_out_preds,
        )
        streaming_state.fifo = streaming_state.fifo[:, pop_out_len:]
        streaming_state.fifo_preds = streaming_state.fifo_preds[:, pop_out_len:]

        # append pop_out_embs to spkcache
        streaming_state.spkcache = torch.cat([streaming_state.spkcache, pop_out_embs], dim=1)
        if streaming_state.spkcache_preds is not None:  # if speaker cache has been already updated at least once
            streaming_state.spkcache_preds = torch.cat([streaming_state.spkcache_preds, pop_out_preds], dim=1)
        if streaming_state.spkcache.shape[1] > self.spkcache_len:
            if streaming_state.spkcache_preds is None:  # if this is a first update of speaker cache
                streaming_state.spkcache_preds = torch.cat([preds[:, :spkcache_len], pop_out_preds], dim=1)
            streaming_state.spkcache, streaming_state.spkcache_preds, streaming_state.spk_perm = (
                self._compress_spkcache(
                    emb_seq=streaming_state.spkcache,
                    preds=streaming_state.spkcache_preds,
                    mean_sil_emb=streaming_state.mean_sil_emb,
                    permute_spk=self.training,
                )
            )

    return streaming_state, chunk_preds

def forward_infer(self, emb_seq, emb_seq_length):
    """
    The main forward pass for diarization for offline diarization inference.

    Args:
        emb_seq (torch.Tensor): Tensor containing FastConformer encoder states (embedding vectors).
            Shape: (batch_size, diar_frame_count, emb_dim)
        emb_seq_length (torch.Tensor): Tensor containing lengths of FastConformer encoder states.
            Shape: (batch_size,)

    Returns:
        preds (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
            Shape: (batch_size, diar_frame_count, num_speakers)
    """
    encoder_mask = self.sortformer_modules.length_to_mask(emb_seq_length, emb_seq.shape[1])
    trans_emb_seq = self.transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
    _preds = self.sortformer_modules.forward_speaker_sigmoids(trans_emb_seq)
    preds = _preds * encoder_mask.unsqueeze(-1)
    return preds
import torch
from torch import nn
from safe_concat import *
from nemo.collections.asr.models import SortformerEncLabelModel


# @torch.jit.script
def fixed_concat_and_pad(embs, lengths, max_total_len=188+188+6):
    # 1. Collect Valid Slices (Same as before)
    sliced_parts = []
    for emb, length in zip(embs, lengths):
        limit = length[0]
        if limit > 0:
            sliced_parts.append(emb[:, :limit, :])

    # 2. Concatenate valid data -> Shape: (1, current_len, D)
    actual_content = torch.cat(sliced_parts, dim=1)

    # 3. Calculate how much to pad
    B, current_len, D = actual_content.shape
    pad_len = max_total_len - current_len

    # 4. Create the padding block manually
    # We create a block of zeros with shape (1, pad_len, D)
    # Ensure it matches the device/dtype of your content
    padding_block = torch.zeros(
        (B, pad_len, D),
        dtype=actual_content.dtype,
        device=actual_content.device
    )

    # 5. Concatenate them: [Content | Zeros] -> Shape: (1, max_total_len, D)
    fixed_output = torch.cat([actual_content, padding_block], dim=1)

    total_length = sum(lengths)

    return fixed_output, total_length


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


class SortformerHeadWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pre_encoder_embs, pre_encoder_lengths, chunk_pre_encoder_embs, chunk_pre_encoder_lengths):
        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.model.frontend_encoder(
            processed_signal=pre_encoder_embs,
            processed_signal_length=pre_encoder_lengths,
            bypass_pre_encode=True,
        )

        # forward pass for inference
        spkcache_fifo_chunk_preds = self.model.forward_infer(
            spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths
        )
        return spkcache_fifo_chunk_preds, chunk_pre_encoder_embs, chunk_pre_encoder_lengths


class SortformerCoreMLWrapper(nn.Module):
    """
    Wraps the entire Sortformer pipeline (Encoder + Streaming Logic for Export)
    The 'forward_for_export' method in the model is the target.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pre_encoder = PreEncoderWrapper(model)

    def forward(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
        (spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths,
         chunk_pre_encode_embs, chunk_pre_encode_lengths) = self.pre_encoder(
            chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths
        )

        # encode the concatenated embeddings
        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.model.frontend_encoder(
            processed_signal=spkcache_fifo_chunk_pre_encode_embs,
            processed_signal_length=spkcache_fifo_chunk_pre_encode_lengths,
            bypass_pre_encode=True,
        )

        # forward pass for inference
        spkcache_fifo_chunk_preds = self.model.forward_infer(
            spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths
        )
        return spkcache_fifo_chunk_preds, chunk_pre_encode_embs, chunk_pre_encode_lengths


class PreEncoderWrapper(nn.Module):
    """
    Wraps the entire Sortformer pipeline (Encoder + Streaming Logic for Export)
    The 'forward_for_export' method in the model is the target.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        modules = model.sortformer_modules
        chunk_length = modules.chunk_left_context + modules.chunk_len + modules.chunk_right_context
        self.pre_encoder_length = modules.spkcache_len + modules.fifo_len + chunk_length

    def forward(self, *args):
        if len(args) == 6:
            return self.forward_concat(*args)
        else:
            return self.forward_pre_encode(*args)

    def forward_concat(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
        chunk_pre_encode_embs, chunk_pre_encode_lengths = self.model.encoder.pre_encode(x=chunk, lengths=chunk_lengths)
        chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)
        spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths = fixed_concat_and_pad(
            [spkcache, fifo, chunk_pre_encode_embs],
            [spkcache_lengths, fifo_lengths, chunk_pre_encode_lengths],
            self.pre_encoder_length
        )
        return (spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths,
                chunk_pre_encode_embs, chunk_pre_encode_lengths)

    def forward_pre_encode(self, chunk, chunk_lengths):
        chunk_pre_encode_embs, chunk_pre_encode_lengths = self.model.encoder.pre_encode(x=chunk, lengths=chunk_lengths)
        chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)

        return chunk_pre_encode_embs, chunk_pre_encode_lengths


class ConformerEncoderWrapper(nn.Module):
    """
    Wraps the entire Sortformer pipeline (Encoder + Streaming Logic for Export)
    The 'forward_for_export' method in the model is the target.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pre_encode_embs, pre_encode_lengths):
        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.model.frontend_encoder(
            processed_signal=pre_encode_embs,
            processed_signal_length=pre_encode_lengths,
            bypass_pre_encode=True,
        )
        return spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths


class SortformerEncoderWrapper(nn.Module):
    """
    Wraps the entire Sortformer pipeline (Encoder + Streaming Logic for Export)
    The 'forward_for_export' method in the model is the target.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, encoder_embs, encoder_lengths):
        spkcache_fifo_chunk_preds = self.model.forward_infer(
            encoder_embs, encoder_lengths
        )
        return spkcache_fifo_chunk_preds

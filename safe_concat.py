import torch
from torch import Tensor
from torch.library import Library
import coremltools as ct
import numpy as np
from typing import List
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import register_torch_op, _get_inputs

# ==========================================
# 1. Define Library and Schema (Returns Tuple now)
# ==========================================
lib = Library("nemo_custom", "DEF")
lib.define('safe_concat(Tensor[] embs, Tensor[] lengths) -> (Tensor, Tensor)')


# ==========================================
# 2. PyTorch Implementation Logic
# ==========================================
def safe_concat_impl(embs: List[Tensor], lengths: List[Tensor]):
    # 1. Collect Valid Slices (Same as before)
    sliced_parts = []
    max_total_length = 0
    for emb, length in zip(embs, lengths):
        limit = length[0]
        max_total_length += emb.shape[1]
        sliced_parts.append(emb[:, :limit, :])

    # 2. Concatenate valid data -> Shape: (1, current_len, D)
    actual_content = torch.cat(sliced_parts, dim=1)

    # 3. Calculate how much to pad
    B, current_len, D = actual_content.shape
    pad_len = max_total_length - current_len

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


# --- Register Implementations ---
@torch.library.impl("nemo_custom::safe_concat", "CPU")
def safe_concat_cpu(embs, lengths):
    return safe_concat_impl(embs, lengths)


@torch.library.impl("nemo_custom::safe_concat", "MPS")
def safe_concat_mps(embs, lengths):
    return safe_concat_impl(embs, lengths)


@torch.library.impl("nemo_custom::safe_concat", "CUDA")
def safe_concat_cuda(embs, lengths):
    return safe_concat_impl(embs, lengths)


# ==========================================
# 3. Fake Implementation (For Tracing)
# ==========================================
@torch.library.register_fake("nemo_custom::safe_concat")
def safe_concat_fake(embs, lengths):
    # We must return (Tensor, Tensor) to match the schema
    try:
        total_len_val = lengths.sum().item()
    except:
        total_len_val = embs[0].shape[1] + embs[1].shape[1] + embs[2].shape[1]

    out_tensor = embs[0].new_empty(embs[0].shape[0], total_len_val, embs[0].shape[2])
    out_length = lengths.new_empty(1)  # Shape (1,)
    return out_tensor, out_length


# ==========================================
# 4. Helper Wrapper
# ==========================================
def safe_concat_and_pad(embs, lengths):
    """Drop-in replacement for patched_concat_and_pad"""
    return torch.ops.nemo_custom.safe_concat(embs, lengths)


# ==========================================
# 5. CoreML MIL Converter (The Magic Part)
# ==========================================
@register_torch_op(torch_alias=["nemo_custom::safe_concat"])
def convert_safe_concat(context, node):
    x = context[node.inputs[0]]

    x = mb.custom_layer(
        inputs={"input": x},
        className="SafeConcatLayer",  # <--- THIS IS THE LINK
    )

    # 4. Add the result back to the graph so subsequent layers can use it
    context.add(x, torch_name=node.name)

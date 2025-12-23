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
# 2. PyTorch Implementation Logic (ANE-safe)
# ==========================================
def safe_concat_impl(embs: List[Tensor], lengths: List[Tensor]):
    """
    ANE-safe concat and pad using gather with arithmetic-computed indices.
    
    Args:
        embs: List of 3 tensors [spkcache, fifo, chunk], each (B, seq_len, D)
        lengths: List of 3 length tensors, each (1,) or scalar
                 First two may be 0, third is always > 0
    
    Returns:
        output: (B, max_total_len, D) with valid frames packed at the start
        total_length: sum of lengths
    """
    B, _, D = embs[0].shape
    device = embs[0].device
    
    # Fixed sizes (known at trace time)
    size0, size1, size2 = embs[0].shape[1], embs[1].shape[1], embs[2].shape[1]
    total_input_size = size0 + size1 + size2
    max_total_len = total_input_size
    
    # Concatenate all embeddings at full size
    full_concat = torch.cat(embs, dim=1)
    
    # Get lengths (reshape to scalar for efficient broadcast)
    len0 = lengths[0].reshape(())
    len1 = lengths[1].reshape(())
    len2 = lengths[2].reshape(())
    total_length = len0 + len1 + len2
    
    # Output positions
    out_pos = torch.arange(max_total_len, device=device, dtype=torch.long)
    
    # Compute gather indices using arithmetic
    cumsum0 = len0
    cumsum1 = len0 + len1
    
    # Segment indicators (bool -> long for arithmetic)
    in_seg1_or_2 = (out_pos >= cumsum0).long()
    in_seg2 = (out_pos >= cumsum1).long()
    
    # Compute offset and gather index
    offset = in_seg1_or_2 * (size0 - len0) + in_seg2 * (size1 - len1)
    gather_idx = (out_pos + offset).clamp(0, total_input_size - 1)
    
    # Expand for gather
    gather_idx = gather_idx.unsqueeze(0).unsqueeze(-1).expand(B, max_total_len, D)
    
    # Gather and mask padding
    output = torch.gather(full_concat, dim=1, index=gather_idx)
    output = output * (out_pos < total_length).float().unsqueeze(0).unsqueeze(-1)
    
    return output, total_length


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

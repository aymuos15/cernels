from ._ops import ops  # type: ignore  # _ops is generated at build time


def sam_decomposed_rel_pos(query, Rh, Rw, attn=None):
    # attn=None writes the bias directly (the SDPA attn_mask path); a tensor gets it added in.
    return ops.sam_decomposed_rel_pos(query, Rh, Rw, attn)


__all__ = ["sam_decomposed_rel_pos"]

"""Per-model kernel swaps: explicit instance-level replacements, one function per model.

SWAPS maps a ModelProfile.name to (apply, variants). apply(model, variant) mutates the model
in place; 'stock' deletes the instance attributes, restoring the class methods. variants is
the benchmark sweep order, stock first. No auto-discovery: adding a model = writing its swap
function and listing it here.
"""

import torch

from kops.registry._local import load

DECODE_MAX_TOKENS = 4  # at or below this many tokens the decode entry point handles the op
# The sam_decomposed_rel_pos kernel computes each rel dot once per query row (GEMM-equivalent
# traffic), so it is shape-safe at both the windowed (k = 14x14) and global (k = 64x64) layers;
# the gate is kept as an escape hatch, set so the global shape's 4096 key positions pass.
SAM_REL_POS_MAX_KEYS = 4096  # above this many key positions the stock einsum path stays


def _experts_fwd(mod, kernel, decode_kernel):
    def fwd(h, idx, w, _m=mod):
        op = decode_kernel if h.size(0) <= DECODE_MAX_TOKENS else kernel
        if op is None:  # this shape's entry point not enabled in this variant
            return type(_m).forward(_m, h, idx, w)
        return op(h, idx, w, _m)

    return fwd


def _swap_experts(model, cls_name, kernel, decode_kernel):
    """Set (or, with both kernels None, restore) the forward on every <cls_name> instance."""
    n = 0
    for mod in model.modules():
        if type(mod).__name__ != cls_name:
            continue
        n += 1
        if "forward" in vars(mod):
            del mod.forward
        if kernel is not None or decode_kernel is not None:
            mod.forward = _experts_fwd(mod, kernel, decode_kernel)
    if not n:
        raise RuntimeError(f"no {cls_name} modules found — wrong model for this swap")


def north_mini_code(model, variant):
    from kops.registry.cohere2_moe_experts import decode_kernel, kernel

    _swap_experts(
        model,
        "Cohere2MoeExperts",
        None if variant == "stock" else kernel,
        decode_kernel if variant == "custom" else None,
    )


def deepseek_ocr_2(model, variant):
    from kops.registry.deepseek_ocr2_moe_experts import decode_kernel, kernel

    moe = variant != "stock"
    _swap_experts(
        model,
        "DeepseekOcr2TextExperts",
        kernel if moe else None,
        decode_kernel if moe else None,
    )
    # SAM rel-pos: replace the decomposed-bias builder on every SAM attention instance. Both
    # the eager and sdpa forwards call self.get_decomposed_rel_pos and reshape the result, so
    # a contiguous (B*heads, q_h*q_w, k_h*k_w) bias is shape-compatible with either. With
    # attn=None the kernel writes the bias directly (the sdpa path feeds it to
    # F.scaled_dot_product_attention as attn_mask) — no zeros materialization. get_rel_pos
    # (interpolate + gather) stays the model's own method, as in the op benchmark.
    for mod in model.modules():
        if "SamVision" not in type(mod).__name__ or not hasattr(mod, "rel_pos_h"):
            continue
        if "get_decomposed_rel_pos" in vars(mod):
            del mod.get_decomposed_rel_pos
        if variant != "custom_full":
            continue

        def bias(query, rel_pos_h, rel_pos_w, q_size, k_size, _m=mod):
            (qh, qw), (kh, kw) = q_size, k_size
            if kh * kw > SAM_REL_POS_MAX_KEYS:  # global-attention layer: stock einsum path
                return type(_m).get_decomposed_rel_pos(_m, query, rel_pos_h, rel_pos_w, q_size, k_size)
            rh = _m.get_rel_pos(qh, kh, rel_pos_h)
            rw = _m.get_rel_pos(qw, kw, rel_pos_w)
            return load("sam_decomposed_rel_pos").sam_decomposed_rel_pos(query, rh.contiguous(), rw.contiguous(), None)

        mod.get_decomposed_rel_pos = bias

    # custom_compile_vision: MoE kernels + torch.compile of the whole vision tower (SAM +
    # CLIP encoders) — prefill is 67% vision, and most of it is fusable copy_/elementwise
    # overhead (window partitioning, rel-pos broadcast add), which Inductor can attack.
    vt = model.model.vision_tower
    if "forward" in vars(vt):
        del vt.forward
    if variant == "custom_compile_vision":
        vt.forward = torch.compile(type(vt).forward.__get__(vt))


SWAPS = {
    "north_mini_code": (north_mini_code, ("stock", "custom_prefill", "custom")),
    # custom = MoE kernels; custom_full adds the SAM rel-pos swap (windowed + global layers);
    # custom_compile_vision = MoE kernels + compiled vision tower.
    "deepseek_ocr_2": (deepseek_ocr_2, ("stock", "custom", "custom_full", "custom_compile_vision")),
}

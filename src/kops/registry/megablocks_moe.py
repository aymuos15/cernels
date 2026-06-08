"""megablocks_moe loader."""

import torch

from kops.registry._local import load

_ACT_IDS = {"gelu": 0, "gelu_new": 0, "gelu_fast": 0, "silu": 1, "swish": 1, "relu": 2}


def kernel(x, moe):
    if moe is None:
        raise RuntimeError("moe module not available (meta device)")
    topk = moe.experts.args.moe_top_k
    with torch.no_grad():
        _scores, expert_weights, expert_indices = moe.router(x.unsqueeze(1))
    indices = expert_indices.to(torch.int32).contiguous()
    wts = expert_weights.float().contiguous()
    w1 = moe.experts.mlp.w1.data.contiguous()
    w2 = moe.experts.mlp.w2.data.contiguous()
    act_fn = getattr(moe.experts.args, "activation_fn", "gelu")
    act_fn_name = getattr(act_fn, "__name__", "gelu") if callable(act_fn) else str(act_fn)
    act_id = _ACT_IDS.get(act_fn_name.lower(), 0)
    return load("megablocks_moe").megablocks_moe(x.contiguous(), w1, w2, indices, wts, topk, act_id)

"""MegaBlocks MoE grouped-GEMM benchmark.

The Hub kernel IS the reference (reference_is_hub): no library exposes a plain MoE op
matching megablocks' layout. Timed path: router linear + softmax + top-k, token
gather/permute, grouped GEMM (gate/up/down), scatter-combine — all inside the module call.
Precomputed in inputs(): the raw hidden-state tensor and the MoE module (weight storage).
"""

import torch

from configs.base import Config
from kops.registry.megablocks_moe import kernel as moe_kernel


class MegablocksMoe(Config):
    name = "megablocks_moe"
    repo = "kernels-community/megablocks"
    version = 1
    dtype = torch.bfloat16
    op = "kernels-community/megablocks MoE"
    reference_is_hub = True
    use_compile = False

    _tokens: int = 4096
    _hidden: int = 2048
    _ffn: int = 5632
    _num_experts: int = 8
    _top_k: int = 2
    _kernel = None

    def _moe(self, device, dtype):
        from kernels import get_kernel

        if self._kernel is None:
            self._kernel = get_kernel(self.repo, version=self.version)
        m = self._kernel
        args = m.Arguments(
            hidden_size=self._hidden,
            ffn_hidden_size=self._ffn,
            num_layers=1,
            bias=False,
            return_bias=False,
            moe_num_experts=self._num_experts,
            moe_top_k=self._top_k,
            moe_jitter_eps=None,  # deterministic routing
            moe_capacity_factor=0,  # dynamic capacity, no token dropping
            fp16=False,
            bf16=True,
            mlp_type="mlp",
            mlp_impl="grouped",  # sparse impl unsupported on triton>=3.2
            device=torch.device(device),
        )
        moe = m.MoE(args)
        moe.eval()
        return moe

    def inputs(self, device, dtype):
        if str(device) == "meta":
            return torch.empty(self._tokens, self._hidden, device=device, dtype=dtype), None
        moe = self._moe(device, dtype)
        torch.manual_seed(0)
        x = torch.randn(self._tokens, self._hidden, device=device, dtype=dtype)
        return x, moe

    def baseline(self, x, moe):
        if moe is None:
            raise RuntimeError("moe module not available (meta device)")
        x3d = x.unsqueeze(1)  # MoE.forward_once expects [sl, bs, hs]
        with torch.no_grad():
            result = moe(x3d)
        result = result if isinstance(result, torch.Tensor) else result[0]
        return result.squeeze(1)

    custom = staticmethod(moe_kernel)

    def verify(self, out, ref):
        return bool(torch.allclose(out, ref, atol=2e-2))

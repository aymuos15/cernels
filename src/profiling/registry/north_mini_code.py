"""CohereLabs/North-Mini-Code-1.0 — 30B-A3B coding MoE (cohere2_moe, released 2026-06-09).

49 decoder layers (layer 0 dense MLP, 1..48 MoE: 128 experts top-8, sigmoid router,
no shared experts, per-expert intermediate 768), parallel attn+MLP blocks, GQA 32q/4kv
head_dim 128, sliding window 4096. We profile to learn where prefill/decode time lives:
the expert grouped-matmuls, the router, attention, or launch overhead at b=1.
"""

import torch

from profiling.registry.base import ModelProfile

_SNIPPET = '''\
def quantile_bucket(values: list[float], q: int) -> list[int]:
    """Assign each value to one of q quantile buckets."""
    order = sorted(range(len(values)), key=values.__getitem__)
    buckets = [0] * len(values)
    for rank, idx in enumerate(order):
        buckets[idx] = min(q - 1, rank * q // len(values))
    return buckets

'''


class NorthMiniCode(ModelProfile):
    name = "north_mini_code"
    model_id = "CohereLabs/North-Mini-Code-1.0"
    dtype = torch.bfloat16
    decode_tokens = 32
    compile_submodule = "model"

    _prompt_copies = 24  # ~2k prefill tokens of representative code context

    def load(self, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # 57 GB of weights on a 121 GB UMA box: stream shards straight to the device —
        # a CPU load followed by .to(device) holds two full copies and OOMs.
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=self.dtype, device_map=device)
        model.eval()
        return model, tokenizer

    def inputs(self, processor, device):
        code = "\n".join(
            _SNIPPET.replace("quantile_bucket", f"quantile_bucket_{i}") for i in range(self._prompt_copies)
        )
        messages = [
            {"role": "user", "content": f"Review this module and suggest improvements:\n```python\n{code}\n```"}
        ]
        ins = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in ins.items()}

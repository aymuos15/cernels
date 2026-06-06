# config list

| config | repo | baseline | op | custom |
|---|---|---|---|---|
| [relu](registry/relu.py) | [kernels-community/relu](https://huggingface.co/kernels-community/relu) | [torch.relu](https://pytorch.org/docs/stable/generated/torch.relu.html) | relu | — |
| [gelu_fast](registry/gelu_fast.py) | [kernels-community/activation](https://huggingface.co/kernels-community/activation) | [torch.nn.functional.gelu](https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html) | gelu_fast | — |
| [rotary](registry/rotary.py) | [kernels-community/rotary](https://huggingface.co/kernels-community/rotary) | [apply_rotary_pos_emb](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | apply_rotary_transformers | [rope](../kops/rope.cu) |
| [silu_and_mul](registry/silu_and_mul.py) | [kernels-community/activation](https://huggingface.co/kernels-community/activation) | [F.silu](https://pytorch.org/docs/stable/generated/torch.nn.functional.silu.html) · gated | silu_and_mul | [silu_and_mul](../kops/silu_and_mul.cu) |

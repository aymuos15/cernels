# config list

| config | eager (baseline) | lib | op | custom |
|---|---|---|---|---|
| [relu](../src/configs/registry/relu.py) | [torch.relu](https://pytorch.org/docs/stable/generated/torch.relu.html) | [kernels-community/relu](https://huggingface.co/kernels-community/relu) | relu | — |
| [gelu_fast](../src/configs/registry/gelu_fast.py) | [torch.nn.functional.gelu](https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html) | [kernels-community/activation](https://huggingface.co/kernels-community/activation) | gelu_fast | — |
| [rotary](../src/configs/registry/rotary.py) | [apply_rotary_pos_emb](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | [kernels-community/rotary](https://huggingface.co/kernels-community/rotary) | apply_rotary_transformers | [rope](../src/kops/registry/rope.cu) |
| [nms](../src/configs/registry/nms.py) | [torchvision.ops.nms](https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html) | — | torchvision.ops.nms | [nms](../src/kops/registry/nms.cu) |

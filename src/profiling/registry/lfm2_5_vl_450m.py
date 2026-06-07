"""LiquidAI/LFM2.5-VL-450M — small hybrid VLM.

Text backbone (lfm2): 16 decoder layers = 10 Lfm2ShortConv (gated conv mixer) + 6
Lfm2Attention, each with Lfm2MLP + 2x Lfm2RMSNorm. Vision: 12-layer SigLIP2 tower +
a multimodal projector. The conv/attention split is exactly why we profile: we don't
yet know whether decode time lives in the conv mixers, attention, MLP, norms, or (at
prefill) the vision encode.
"""

import numpy as np
import torch
from PIL import Image

from profiling.registry.base import ModelProfile


class Lfm2_5Vl450M(ModelProfile):
    name = "lfm2_5_vl_450m"
    model_id = "LiquidAI/LFM2.5-VL-450M"
    dtype = torch.bfloat16
    decode_tokens = 32
    compile_submodule = "model.language_model"  # text backbone for the inductor lens

    _img_size = 256
    _prompt = "Describe this image in detail."

    def load(self, device):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(self.model_id, dtype=self.dtype, trust_remote_code=True).to(
            device
        )
        model.eval()
        return model, processor

    def inputs(self, processor, device):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, size=(self._img_size, self._img_size, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img}, {"type": "text", "text": self._prompt}],
            }
        ]
        ins = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in ins.items()}

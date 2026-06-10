"""deepseek-community/DeepSeek-OCR-2 — 3.4B OCR VLM (deepseek_ocr2, in-tree HF format).

SAM ViT-B encoder (12 layers, decomposed rel-pos attention bias — the op our shipped
sam_decomposed_rel_pos kernel covers) + a 12-layer DeepSeek-V2-lite MoE decoder
(64 routed experts top-6 softmax, 2 shared experts, moe_intermediate 896, hidden 1280).
We profile to learn the prefill split between vision encode and the MoE decoder, and
whether decode repeats the North Mini Code expert-loop pathology. Note the official
deepseek-ai repo is trust_remote_code; this is the converted in-tree checkpoint.
"""

import torch
from PIL import Image, ImageDraw

from profiling.registry.base import ModelProfile


class DeepseekOcr2(ModelProfile):
    name = "deepseek_ocr_2"
    model_id = "deepseek-community/DeepSeek-OCR-2"
    dtype = torch.bfloat16
    decode_tokens = 32
    compile_submodule = "model.language_model"

    _img_size = 1024
    _prompt = "Convert the document to markdown."

    def load(self, device):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForImageTextToText.from_pretrained(self.model_id, dtype=self.dtype).to(device)
        model.eval()
        return model, processor

    def inputs(self, processor, device):
        # A deterministic synthetic "document": ruled lines of text on white, so the OCR
        # path sees realistic content (a noise image routes/decodes degenerately).
        img = Image.new("RGB", (self._img_size, self._img_size), "white")
        draw = ImageDraw.Draw(img)
        for i in range(40):
            draw.text(
                (40, 16 + 24 * i),
                f"Invoice line {i:02d}: part SKU-{i:04d}, quantity {i % 9 + 1}, unit price {i}.50",
                fill="black",
            )
        # No chat template on this processor: it takes raw text with an <image> placeholder
        # (the documented form, e.g. "<image>\nFree OCR."). Pixel tensors arrive fp32 and the
        # model is bf16, so floating inputs are cast to the profile dtype.
        ins = processor(images=img, text=f"<image>\n{self._prompt}", return_tensors="pt")

        def move(v):
            if not hasattr(v, "to"):
                return v
            return v.to(device, self.dtype) if v.is_floating_point() else v.to(device)

        return {k: move(v) for k, v in ins.items()}

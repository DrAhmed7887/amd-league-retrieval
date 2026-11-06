"""The vision-language model, loaded once on the AMD GPU.

Matches the notebook run that scored 200/200 on the Match 3 sample (notes/
notebook_2026-11-06_qwen_reader.py): Qwen3-VL-8B-Instruct in bf16, greedy decoding.
torch and transformers are imported only when the model is built, so the rest of
the engine stays testable on a laptop.
"""

from __future__ import annotations

import os
from typing import Optional

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/Qwen3-VL-8B-Instruct")
MAX_IMAGE_SIDE = 1600  # bounds vision tokens, so one huge photo cannot eat the time budget


def load_image(path: str):
    from PIL import Image

    with Image.open(path) as img:
        img.seek(0)  # multi-page TIFF: first page
        img = img.convert("RGB")
    longest = max(img.size)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    return img


class VisionLanguageModel:
    def __init__(self, model_dir: str = MODEL_DIR):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("no GPU visible to torch; the harness rejects CPU-only runs")
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map="cuda"
        ).eval()

    def ask(self, prompt: str, image_path: Optional[str] = None, max_new_tokens: int = 64) -> str:
        content = [{"type": "image", "image": load_image(image_path)}] if image_path else []
        content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to("cuda")
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

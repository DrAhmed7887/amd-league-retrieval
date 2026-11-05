"""Notebook experiment, 6 Nov 2026, AMD hackathon notebook (Radeon Pro W7900D, gfx1100, 48 GiB).

This is the exact code that scored 200/200 on the Match 3 sample. It is a record to port into
engine/ (resident server + reader), not production code: it loads the model in-process.

Measured: model load 23.2 s, 16.3 GiB after load, 16.6 GiB peak; index + 2 image transcriptions
5.6 s; 0.5-1.7 s per question.
"""

import json
import re
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

M = "/workspace/models/Qwen3-VL-8B-Instruct"  # snapshot_download("Qwen/Qwen3-VL-8B-Instruct")
proc = AutoProcessor.from_pretrained(M)
model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()


def ask(prompt, image=None, max_new=48):
    content = ([{"type": "image", "image": Image.open(image).convert("RGB")}] if image else []) + [
        {"type": "text", "text": prompt}
    ]
    inputs = proc.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to("cuda")
    t = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    return text, round(time.time() - t, 1)


# Index time: each image is transcribed once and becomes a normal text chunk.
TRANSCRIBE = "Transcribe all text in this image, line by line."
# for img in index["images"]:
#     text, _ = ask(TRANSCRIBE, corpus / img, 300)
#     index["chunks"].append({"file": img, "location": "image text", "text": text})

PROMPT = """You answer a question using ONLY the numbered excerpts below. The products in them are fictional, so never use outside knowledge.
Reply with one line of JSON and nothing else: {{"answer": "...", "files": ["..."]}}
- answer: the exact value only, copied as written in the excerpt (a number, code, version, part number, quarter). Keep qualifiers that are part of the value, such as "Q3 FY27" or "REV-C2". No sentence, no units phrase, no explanation.
- files: the file paths you needed. If one excerpt gave an identifier (a ticket or error code) and another gave the value for it, list both. Never list a file you did not need.
- If a document says it is withdrawn or superseded, use the current one.
- If the excerpts do not contain the answer, reply {{"answer": "", "files": []}}.

Question: {q}

Excerpts:
{ctx}"""


class QwenReader:
    def read(self, q, context):
        ctx = "\n\n".join(
            f"[{i + 1}] file: {c['file']} ({c['location']})\n{c['text'][:1500]}" for i, c in enumerate(context)
        )
        raw, secs = ask(PROMPT.format(q=q, ctx=ctx), None, 80)
        self.last = (raw, secs)
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {"answer": "", "files": []}

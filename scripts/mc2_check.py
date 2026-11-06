#!/usr/bin/env python3
"""Match 2 check on the brief's 10 sample images plus harder variants of each.

Runs on a GPU box (the AMD notebook) against the real model and the same prompt and
cleaner the image uses. Scores exactly as the grader does: uppercase, no whitespace,
no - . · _ ; pass/fail per image.

    python3 scripts/mc2_check.py --brief brief.pdf --out /tmp/mc2
"""

from __future__ import annotations

import argparse
import io
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The brief's samples, in page order, with the format the brief says each one ships in.
SAMPLES = [
    ("01_ca_plate", "png", "7ABC123"),
    ("02_beijing_plate", "png", "京A·12345"),
    ("03_ny_plate_angled", "jpg", "JHT 2951"),
    ("04_plate_blur", "png", "5XYZ891"),
    ("05_shanghai_plate_dark", "jpg", "沪B·88888"),
    ("06_stop", "png", "STOP"),
    ("07_stop_noise", "tiff", "STOP"),
    ("08_speed_limit", "jpg", "SPEED LIMIT 65"),
    ("09_road_work", "png", "ROAD WORK AHEAD"),
    ("10_advisory_35", "tiff", "35"),
]


def normalize(text: str) -> str:
    return re.sub(r"[\s\-\.·_]", "", (text or "").upper())


def extract_samples(brief: Path):
    """The sample images, from the brief PDF or from a folder of them in page order."""
    from PIL import Image

    if brief.is_dir():
        images = [Image.open(p).convert("RGB") for p in sorted(brief.iterdir()) if p.is_file()]
    else:
        import pypdf
        images = [Image.open(io.BytesIO(im.data)).convert("RGB")
                  for page in pypdf.PdfReader(str(brief)).pages for im in page.images]
    if len(images) != len(SAMPLES):
        raise SystemExit(f"expected {len(SAMPLES)} sample images in the brief, found {len(images)}")
    return images


def variants(img):
    """Harder versions, in the spirit of the brief's 'adverse conditions'."""
    from PIL import Image, ImageEnhance, ImageFilter

    rng = random.Random(7)
    w, h = img.size
    big = img.resize((w * 5, h * 5), Image.LANCZOS)  # larger than the model's native input
    tilt = img.rotate(9, resample=Image.BICUBIC, expand=True, fillcolor=(90, 90, 90))
    blur = img.filter(ImageFilter.GaussianBlur(2.2))
    dark = ImageEnhance.Brightness(img).enhance(0.35)
    noisy = dark.copy()
    px = noisy.load()
    for _ in range(w * h // 6):
        x, y = rng.randrange(w), rng.randrange(h)
        v = rng.randrange(256)
        px[x, y] = (v, v, v)
    small = img.resize((max(w // 3, 40), max(h // 3, 20)), Image.BILINEAR)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=25)
    lowq = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return {"big": ("png", big), "tilt": ("jpg", tilt), "blur": ("png", blur),
            "dark_noise": ("tiff", noisy), "small_lowq": ("jpg", lowq)}


def save(img, path: Path, fmt: str) -> None:
    kind = {"png": "PNG", "jpg": "JPEG", "tiff": "TIFF"}[fmt]
    img.save(path, kind, **({"quality": 90} if kind == "JPEG" else {}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/mc2"))
    ap.add_argument("--samples-only", action="store_true")
    args = ap.parse_args()

    from engine.model import VisionLanguageModel
    from engine.ocr import OCR_PROMPT, clean_ocr_text

    OCR_TOKENS = 128  # same cap as app.py
    args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    for (stem, fmt, want), img in zip(SAMPLES, extract_samples(args.brief)):
        path = args.out / f"{stem}.{fmt}"
        save(img, path, fmt)
        cases.append((path, want, "sample"))
        if not args.samples_only:
            for tag, (vfmt, vimg) in variants(img).items():
                vpath = args.out / f"{stem}__{tag}.{vfmt}"
                save(vimg, vpath, vfmt)
                cases.append((vpath, want, tag))

    t0 = time.monotonic()
    model = VisionLanguageModel()
    print(f"model loaded in {time.monotonic() - t0:.1f}s", flush=True)

    by_kind, worst = {}, 0.0
    for path, want, kind in cases:
        t = time.monotonic()
        got = clean_ocr_text(model.ask(OCR_PROMPT, str(path), OCR_TOKENS))
        secs = time.monotonic() - t
        worst = max(worst, secs)
        ok = normalize(got) == normalize(want)
        by_kind.setdefault(kind, []).append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {secs:4.1f}s  {path.name:<40} want {want!r:<20} got {got!r}", flush=True)

    print()
    for kind, oks in by_kind.items():
        print(f"{kind:<12} {sum(oks)}/{len(oks)}")
    samples = by_kind.get("sample", [])
    print(f"SAMPLE SCORE {20 * sum(samples)}/{20 * len(samples)}   slowest image {worst:.1f}s (limit 30 s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

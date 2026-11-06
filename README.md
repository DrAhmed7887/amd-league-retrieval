# amd-league-retrieval

An answer-with-source engine for the AMD AI League: it reads a folder of mixed documents (or web pages), answers a question with the exact value only, and cites only the files the answer needed. It refuses with an empty answer when the source isn't there.

- **Match 3 (RAG):** a folder of PDF, Word, Excel, CSV, logs, code and images.
- **Match 4 (web retrieval):** the same engine over web pages. Planned.

## Status

| Part | State |
|---|---|
| Corpus walk that survives unreadable, encrypted, unknown-type files and empty folders | Done, tested |
| Parsers: PDF, DOCX (with tables), XLSX (all sheets), CSV, text, logs, code | Done, tested |
| Persisted index + BM25 search | Done, tested |
| Answer step: drops withdrawn revisions, follows ticket/error IDs to a second file, accepts a value only if it is written in a cited file, prunes citations to the chain, turns prose and crashes into refusals | Done, tested |
| `app.py` to the AMD contract (always writes both graded keys) | Done, tested |
| Resident model service on a unix socket (`server.py`, `engine/server.py`, `engine/client.py`); fails fast if the model can't load | Done, tested with a stand-in model |
| Qwen reader (the 200/200 prompt) and image transcription during `--index` | Done, tested with a stand-in model |
| **Match 2 mode**: `--input-image` → `<stem>_output.json` `{"text": ...}` | Done; 10/10 brief samples and 50/50 harder variants on the AMD GPU (`scripts/mc2_check.py`) |
| Dockerfile (mandated base, torch pinned as a constraint, weights at a fixed revision in one layer per shard, offline) | Built and pushed by `.github/workflows/build-image.yml`: 45.5 GiB, base layers intact, AMD self-check passes (no GPU in CI) |
| `scripts/build_and_check.sh`: build, size, real-GPU sample run, OCR smoke, AMD selfcheck | For a Docker + AMD GPU machine |

## Sample score

`scripts/score_sample.py` scores the kit's 10 questions exactly as the grader does (6 Nov 2026):

| Reader | Score | Meaning |
|---|---|---|
| `oracle` (knows the right answer) | 160/200 | Ceiling of retrieval + checks. Misses only the 2 image questions |
| `none` (refuses everything) | 20/200 | Only the unanswerable question |
| **Qwen3-VL-8B-Instruct** on the AMD notebook (Radeon Pro W7900D, gfx1100) | **200/200** | Notebook experiment, 6 Nov 2026: images transcribed once at index time, then the same checks. 0.5–1.7 s per question, 23 s model load, 16.6 GiB peak VRAM |

| **Full submission path** on the AMD notebook (`server.py` + one `app.py` process per question, commit a3f3a5f) | **200/200** | 6 Nov 2026: requirements install keeps torch `2.13.0+rocm10.0.0`; model load 25.6 s; `--index` with 2 image reads 13.9 s; questions 0.6–1.8 s each; Match 2 mode 1.6–2.6 s per image; 17.5 GiB VRAM in use |

The sample corpus is small; the graded one is larger and harder, so 200/200 here is a smoke test, not a forecast.

**Match 2** (6 Nov 2026, same notebook GPU, `scripts/mc2_check.py`): the brief's 10 sample images read 10/10 (200/200), and 50 harder variants (5x size, tilt, blur, dark with noise, tiny low-quality JPEG) read 50/50. Model load 24.6 s; slowest image 8.3 s (the first, warm-up), most 0.1–1.5 s.

Not yet tested: the pushed image itself on an AMD GPU (CI has no GPU; the same code ran on the notebook GPU).

## Model

`Qwen/Qwen3-VL-8B-Instruct` (Apache-2.0, 16.3 GiB bf16). Image budget: 29.2 GiB base + 16.3 GiB weights ≈ 46 of 60 GiB.

Notebook setup that keeps the AMD torch build (pin it, then let pip resolve the rest):

```bash
pip list --format=freeze | grep -iE '^(torch|torchvision|torchaudio|triton)==' > torch-pins.txt
pip install -c torch-pins.txt transformers accelerate huggingface_hub pypdf python-docx openpyxl
```

`--no-deps` alone breaks huggingface_hub 2.x (it needs httpx2 and httpcore2).

## Run the tests

The AMD starter kit is kept out of git. Put it at `kit/` and run its setup step once:

```bash
unzip mc3-starter-kit.zip && mv mc3-starter-kit kit && sh kit/setup.sh
python3 -m venv .venv && .venv/bin/pip install pypdf python-docx openpyxl pytest
.venv/bin/python -m pytest -q
```

## Build

`.github/workflows/build-image.yml` builds the image on a free GitHub runner and runs every check that needs no GPU. With `push=true` it pushes to Docker Hub (ghcr.io caps layers at 10 GB; the mandated base has an 18.6 GB layer). On a Docker + AMD GPU machine, `bash scripts/build_and_check.sh <tag>` adds the real-GPU run.

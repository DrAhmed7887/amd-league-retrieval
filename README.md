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
| Images (read by a vision model on the GPU) | Not started |
| The real reader: an open model kept resident on the GPU behind a unix socket | Not started |
| Dockerfile on the mandated ROCm base, `--no-deps` installs, self-check | Not started |

## Sample score

`scripts/score_sample.py` scores the kit's 10 questions exactly as the grader does (6 Nov 2026):

| Reader | Score | Meaning |
|---|---|---|
| `oracle` (knows the right answer) | 160/200 | Ceiling of retrieval + checks. Misses only the 2 image questions |
| `none` (refuses everything) | 20/200 | Only the unanswerable question |
| **Qwen3-VL-8B-Instruct** on the AMD notebook (Radeon Pro W7900D, gfx1100) | **200/200** | Notebook experiment, 6 Nov 2026: images transcribed once at index time, then the same checks. 0.5–1.7 s per question, 23 s model load, 16.6 GiB peak VRAM |

The sample corpus is small; the graded one is larger and harder, so 200/200 here is a smoke test, not a forecast.

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

## Next

1. Port the notebook reader into `engine/`: a resident model server on a unix socket (each question is a new process, and a 23 s load per question would break the 30 s limit), image transcription during `--index`, and the JSON prompt.
2. Dockerfile (weights in `/models`, torch pins as constraints) and AMD's `selfcheck.py`.
3. Build and push from a cloud machine with Docker (the notebook has none).

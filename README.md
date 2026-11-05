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

## Run the tests

The AMD starter kit is kept out of git. Put it at `kit/` and run its setup step once:

```bash
unzip mc3-starter-kit.zip && mv mc3-starter-kit kit && sh kit/setup.sh
python3 -m venv .venv && .venv/bin/pip install pypdf python-docx openpyxl pytest
.venv/bin/python -m pytest -q
```

## Next

1. The real reader: pick and load an open vision-language model on the AMD notebook (it must read the two image questions too), kept loaded between questions.
2. Images: read them once during `--index`, so questions stay under 30 s.
3. Dockerfile and AMD's `selfcheck.py`, then build and push from a cloud machine.

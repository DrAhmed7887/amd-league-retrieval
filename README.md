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
| Images (read by a vision model on the GPU) | Not started |
| Answer step: value only, superseded documents, multi-file chains, refusals | Not started |
| Resident model process (the harness starts a new process per question) | Not started |
| Dockerfile on the mandated ROCm base, `--no-deps` installs, self-check | Not started |

## Run the tests

The AMD starter kit is kept out of git. Put it at `kit/` and run its setup step once:

```bash
unzip mc3-starter-kit.zip && mv mc3-starter-kit kit && sh kit/setup.sh
python3 -m venv .venv && .venv/bin/pip install pypdf python-docx openpyxl pytest
.venv/bin/python -m pytest -q
```

## Known next problem

On the sample corpus the WITHDRAWN datasheet (r1) outranks the current one (r2) for the junction-temperature question. The answer step must prefer the current revision, or the citation is wrong.

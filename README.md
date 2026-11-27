# amd-league-retrieval

An answer-with-source engine for the AMD AI League. It reads a folder of mixed
documents or web pages, returns only the exact value requested, cites only the
sources that support the answer, and refuses with an empty answer when the
source is missing.

- **Match 2 (OCR):** `--input-image` text extraction via Qwen3-VL-8B-Instruct.
- **Match 3 (RAG):** `--index` and `--corpus` retrieval over PDF, Word, Excel,
  CSV, logs, code, and images.
- **Match 4 (Web retrieval):** live crawl, search, and saved-page retrieval on
  the assumed Match 3 contract with expanded CLI/JSON compatibility.

## League standing (November 27, 2026)

The AMD and lablab AI Academy League ("One League. Four matches.") runs through
**December 1, 2026, 20:00 CET**.

| Entry | Image | Verified status |
|---|---|---|
| Match 2 + Match 3 (`FallRx-Sentinel`) | `docker.io/drzayed87/amd-league:v1` | Officially graded on lablab: **200/200 (10/10)** on Match 2 and **180/200 (9/10)** on Match 3 (`380` pts, Rank **#10 / 86** as of November 27, 2026). |
| Match 4 (`Intelligent Web Information Retrieval`, Final Match) | `docker.io/drzayed87/amd-league:v2` | Rebuilt from `main` via `.github/workflows/build-image.yml` (`tag=v2`, superseding the November 7 `048eda8` build `sha256:9877f957...`) with defensive CLI/JSON handling; staged for `mini_challenge_4_image`. |

## Match 4: web retrieval

No official Match 4 requirements document has been published by the organizers.
`app.py` preserves the Match 3 contract while also accepting wide Match 4
invocation variants (`parse_known_args`, silent `stdout`, atomic 3-key output):

```bash
python3 /app/app.py --query-id find_01 --query "Who is the CEO of AMD?" [--url https://start.page]
# -> /app/output/find_01_output.json  {"answer": "Lisa Su", "citations": ["https://..."], "confidence": 0.9}
```

Accepted CLI flags include `--query-id`/`--id`/`--qid`/`--find-id`/`--find`/`--task-id`,
`--query`/`--question`/`--task`/`--prompt`/`-q`, `--url`/`--website`/`--start-url`,
`--pages`, `--question-file`/`--input`/`--input-file`, `--input-dir`,
`--output-dir`, `--output`/`-o`, and bare `--index`. Calls with `--corpus` and
no `--url` (unless `MATCH_MODE=web`) still run Match 3.

With a start URL, the engine crawls same-site links up to 10 pages. Without one,
it queries package registries, DuckDuckGo/Bing, and Wikipedia, fetches top hits
in parallel, and ranks passages by keywords plus lead passages. Fetching caps at
2 MB per page and 17 s; each question is capped at 26 s total including model
inference.

## Known limits and checks

- **Match 4 spec unknown:** No official Match 4 brief or sample corpus exists;
  every public `mini-challenge-4` entry on lablab currently shows `0/10` or
  `error`/`failed` (November 27, 2026).
- **Self-check on AMD GPU:** Our November 7 check scored **8/8** on
  `scripts/m4_check.py` (pages fetched on Mac, replayed to Qwen3-VL-8B-Instruct
  on Radeon PRO W7900D, 1.8–3.3 s model time), not an official grader score.
- **Grader network access:** Outbound web access inside the Match 4 grading
  container is untested; if `--network none` or blocked search engines apply,
  live search returns empty refusals.
- **Local Docker check:** `kit/selfcheck.py` requires Docker (absent on the
  build Mac) and runs in GitHub Actions without a GPU; the pure-Python `pytest`
  suite passes locally (kit-dependent tests require `kit/` unpacked).
- **Match 3 sample check:** `scripts/score_sample.py` scored `200/200` on the
  AMD notebook (`160/200` with `oracle` text-only reader, November 6, 2026).

## Model and container

- **Model:** `Qwen/Qwen3-VL-8B-Instruct` (Apache-2.0, 16.3 GiB bf16, loaded once
  by `server.py` and served over a unix socket).
- **Base image:** `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`
  (~45.5 GiB total; base layers unchanged; `torch` pinned via constraints).
- **Build:** `.github/workflows/build-image.yml` builds on Ubuntu runners and
  pushes to Docker Hub (GHCR caps layers at 10 GB; the base has an 18.6 GB
  layer). On a Docker + AMD GPU machine, run `bash scripts/build_and_check.sh <tag>`.

## Run the tests

The AMD starter kit stays out of git. Place it at `kit/` to run kit-dependent
checks, or run the pure-Python suite directly:

```bash
/usr/bin/python3 -m pytest -q
```

## License

Code: [MIT](LICENSE). Model weights (`Qwen/Qwen3-VL-8B-Instruct`) are published
under Apache-2.0.

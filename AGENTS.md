# AGENTS.md

Guidance for agents (and humans) working in this repo.

## What this is
`quant-bench` benchmarks 1–5 GGUF quantizations of one model — MMLU + aider polyglot
coding + PPL + a perf probe — serving each model one-at-a-time via a system
`llama-server`. Python, managed with **uv**.

## Naming (do not "fix")
- Distribution / CLI / console script: `quant-bench` (hyphen).
- Python import package: `quant_bench` (underscore).
Both are intentional (PEP 8 distribution-vs-import naming).

## Commands
- Sync env: `uv sync`
- One-time setup (required before the coding stage): `uv run quant-bench setup`
  — clones `aider` (pinned to the installed `aider-chat` version) + `polyglot-benchmark`
  into `tmp.benchmarks/`; needs `git` on PATH.
- Run a benchmark: `uv run quant-bench run --config models.yaml --results-dir <dir> --port 8126`
  (or `./run_example.sh`).
- Tests: `uv run pytest` (config `testpaths = ["tests"]`). Pure unit tests — external
  llama-server / lm-eval / aider calls are mocked; no GPU, network, or model files needed.
  - Single test: `uv run pytest tests/test_config.py::test_server_flags_typed_fields`
- Lint: `uvx ruff check .` (fix: `uvx ruff check --fix .`).
  **`ruff` is NOT a project dependency** — run it via `uvx` (or `uv run --with ruff`),
  not `uv run ruff`.
- No typecheck, no CI, no pre-commit, no task-runner/Makefile are configured.

## Tooling (pyproject.toml)
- Python `>=3.10,<3.13` (venv is 3.11). Build: hatchling; package at `src/quant_bench`.
- ruff: `target-version = "py310"`, `line-length = 120`, rules `E,F,W,I,B`;
  `per-file-ignores` `cli.py = ["E402"]`; bugbear
  `extend-immutable-calls = ["click.option","click.argument","click.Path"]`
  (so cli.py reuses shared option objects).
- `pydantic` is a direct dependency (backs `LlamaServerFlags` in `config.py`).

## Layout / entrypoints
- `cli.py` — click CLI. Group `app` (bare `quant-bench` prints help); subcommands `setup`, `run`.
  Console entry `quant-bench = quant_bench.cli:main`; also `python -m quant_bench`.
- `config.py` — `models.yaml` parsing (`load_models` → `ModelSpec`), `ServerProfile`,
  `LlamaServerFlags` (pydantic, `.argv()`), `server_flags_for`, device→ngl resolution.
- `runner.py` — per-model orchestration (`run_model`): MMLU/perf server → separate coding
  server → PPL. `RunConfig` dataclass.
- `llamaserver.py` — `LlamaServer` process manager (spawn/health/stop), `find_llama_server`.
- `llama_lm.py` — custom lm-eval model (OpenAI client) for MMLU loglikelihood.
- `coding.py` — aider polyglot driver (`find_benchmarks`, `run_coding`); **needs the cloned
  repos from `setup`**.
- `mmlu.py`, `perf.py`, `ppl.py` — stage runners. `report.py` — `ModelScore`, composite
  scoring, `report.md` / `report.json`.

## Gotchas
- **Run order**: `uv sync` → `quant-bench setup` (once) → `quant-bench run`. Skipping `setup`
  leaves the coding stage with unknown attempt counts.
- Requires a **system `llama-server` on PATH** (or `--llama-server /path`); PPL additionally
  needs `llama-perplexity` alongside it.
- `--results-dir` is **required** — use a distinct dir per config so reports don't overwrite.
- Avoid ports `8080`/`8000` (other local servers); the example uses `--port 8126`.
- `models.yaml`: a model's `flags` **replace** (do not merge) the group's `flags`; quote
  numeric flag values (`"0"` not `0`).
- `--benchmark-root` defaults to `tmp.benchmarks/` for both `setup` and `run` — keep them in
  sync if you override one.

## Git conventions (observed from history)
- Conventional commits, scoped, imperative: `refactor(cli): …`, `docs(quant_bench): …`, `fix: …`.
- Branches like `refactor/<short-desc>`; PRs target `main`
  (remote `origin` → `github.com/CATech-org/quant_bench`).

## Gitignored scratch (don't commit)
`results*/`, `*-results/`, `tmp.benchmarks/`, `local/`, `.env`, `*.log`, `.venv/`.

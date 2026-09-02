# quant-bench

Benchmark 1–5 GGUF quantizations of the same model, one at a time, on:

- **MMLU** (lm-evaluation-harness, `mmlu` multiple-choice loglikelihood by default)
- **aider's official polyglot coding harness** (pass@1 / pass@2)
- **PPL** (perplexity on a fixed reference, via `llama-perplexity` — the primary quant-*fidelity* signal; lower = closer to the base model = better)

plus a short throughput/latency probe (TTFT, tok/s) for each model. Models are
served by your system `llama-server`; each model is loaded, benchmarked, and
unloaded before the next one starts (never two models in VRAM at once).

**Why PPL matters:** MMLU and aider pass@N saturate at the noise floor for small models (both quants score near chance / near zero), so they can't separate close quantizations. PPL is continuous and high-resolution, so the composite is PPL-weighted by default and the ranking is reproducible.

## Install

```bash
uv sync
uv run quant-bench setup   # clones aider (matching your aider-chat version) + polyglot-benchmark
```

## Configure

Edit `models.yaml` (default; any file works via `--config`) — a list of model families. All quants of a family share one tokenizer, so the tokenizer lives in the config, per family (no separate `--tokenizer` flag):

```yaml example
# 1-5 models total, benchmarked sequentially
- tokenizer: ~/Desktop/llama/models/gemma4-e2b   # required: HuggingFace tokenizer dir
  models-dir: ~/Desktop/llama/models/gemma4-e2b  # optional: where the .gguf files live (default: tokenizer's dir)
  flags: ["--split-mode", "none", "--main-gpu", "0", "--jinja"]   # optional: shared by the group (one argv token per item; quote numbers). Any llama-server flag works.
  models:
    - gemma-4-E2B-it-Q3_K_M.gguf                 # relative to models-dir; absolute paths OK
    - path: gemma-4-E2B-it-Q4_K_M.gguf           # mapping form = per-model flags
      flags: [--jinja]                           # (replace the group's for this model)
```

Multiple config files are fine — one per benchmark set. A ready example ships as
the default `models.yaml`.

## Run

```bash
uv run quant-bench run \
  --config models.yaml \
  --results-dir results/gemma \
  --port 8126
```

(`--port 8126` because 8080/8000 are taken by other local servers; default
`--device vram` = `-ngl all`.) An equivalent non-interactive script is
`run_example.sh` (it adds `--weights 0.5 --yes`; point `--llama-server` at your
binary if it is not on PATH).

`--config` defaults to `models.yaml` and fails fast with a clear error if the
file is missing or invalid (bad YAML, missing gguf, tokenizer dir without a
tokenizer file, >5 models, duplicates). `--results-dir` is required — use a
different one per config so reports don't overwrite each other.

Before anything starts you get a conservative **estimated runtime** (with the scope of each stage) and a `Continue? [Y/n]` prompt, then — unless `--weights` is given — the **MMLU weight** of the capability half of the composite score. The composite folds in PPL: with the defaults (`--ppl-weight 0.5`, `--weights 0.5`)

```
score = 0.50 × PPL(0-100) + 0.25 × MMLU% + 0.25 × aider pass@2%
```

where `PPL(0-100) = 100 × min_ppl / model_ppl` (best quant in the run = 100).
Pass `--yes` to skip the confirmation (scripted runs). Pass `--ppl-weight 0` to
revert to the old MMLU+coding-only composite, or `--skip-ppl` to not run PPL.

Results:

- `results/report.md` + `results/report.json` — ranked table with MMLU,
  pass@1/pass@2, TTFT, tok/s, composite score, and the exact
  `llama-server` command + version used per model
- `results/mmlu_<task>_<model>.json` — full lm-eval output
- `results/server_<model>.log`, `results/coding_<model>.log`
- `tmp.benchmarks/YYYY-MM-DD-HH-MM-SS--<model>/` — raw polyglot run dirs

## Useful options

| option | default | meaning |
|---|---|---|
| `--config models.yaml` | `models.yaml` | YAML model config (see Configure) |
| `--results-dir DIR` | **required** | where report/logs go (created if missing); one dir per config |
| `--host 0.0.0.0` | `0.0.0.0` | interface llama-server binds to (client still connects via 127.0.0.1) |
| `--device vram\|cpu\|hybrid` | `vram` | `vram` = `-ngl all`, `cpu` = `-ngl 0`, `hybrid` requires `--ngl N` |
| `--ngl N` | – | explicit GPU layer count (overrides `--device`) |
| `--threads N` | – | llama-server CPU threads (`-t`) |
| `--parallel N` | `--mmlu-concurrency` | llama-server slots |
| `--ctx N` | `8192` | per-request context; server gets `-c N x slots` (total KV grows with `--parallel` — lower `--ctx` or `--mmlu-concurrency` if you OOM) |
| `--extra-flags -fa` | – | extra single-token llama-server flags (repeatable); multi-value flags go in `models.yaml` |
| `--log-level N` | `2` | llama-server `-lv` (1=error, 2=warn, 3=info, 4=trace, 5=debug). Default 2 keeps the per-request info flood out of the server logs; raise to 3+ to debug (MMLU logs grow to multi-GB) |
| `--mmlu-task mmlu\|mmlu_generative` | `mmlu` | loglikelihood (default) or generation-based MMLU |
| `--mmlu-limit N` | – | N docs per subtask (smoke tests; applied to each of the 57 MMLU subjects) |
| `--mmlu-concurrency N` | `8` | parallel MMLU scoring requests |
| `--languages python` | `python` | polyglot languages (needs the matching toolchain: pytest, cargo, go, node, gradle) |
| `--edit-format whole` | `whole` | aider edit format |
| `--tries N` | `2` | polyglot tries per test (pass@N) |
| `--coding-limit N` | – | only N polyglot tests — the harness shuffles unseeded, so results are not comparable across models |
| `--weights W` | prompt | MMLU weight **within the capability half** (rest goes to aider coding), skips the prompt |
| `--ppl-weight W` | `0.5` | PPL (fidelity) weight in the composite, 0.0-1.0 (0 = old MMLU+coding-only composite) |
| `--ppl-reference PATH` | `scripts/ppl_ref.txt` | fixed reference text for the PPL fidelity metric |
| `--ppl-ctx N` | `1024` | context window for the `llama-perplexity` probe |
| `--ppl-runs N` | `2` | PPL runs per model (mean + run-to-run reproducibility) |
| `--coding-kv-fix {f16,no-cache,off}` | `f16` | make the coding stage reproducible: f16 KV cache, `--no-cache-prompt`, or none |
| `--skip-mmlu` / `--skip-coding` / `--skip-perf` / `--skip-ppl` | – | run a subset of the benchmarks |

## How it works

- A custom lm-eval model (`llama-server`, `src/quant_bench/llama_lm.py`)
  scores MMLU by asking llama-server for `top_logprobs` on one generated
  token at a time — llama.cpp's OAI-compatible endpoint returns logprobs for
  generated tokens only (no prompt echo), which breaks the stock
  `local-completions` loglikelihood path.
- The polyglot harness runs from the cloned aider repo in-process
  (`AIDER_DOCKER=1`, `OPENAI_API_BASE` pointed at llama-server).
  **Note:** without Docker, LLM-generated code executes on your host.
- Python polyglot tests use `pytest` from this project's venv (PATH is set
  automatically).
- **PPL** runs `llama-perplexity` (found next to `llama-server`) as its own
  process on the model's device flags against a fixed reference, a few times per
  model; it runs *after* the serving servers stop, so never two models in VRAM.
- **Coding runs on its own server** with the `--coding-kv-fix` flags (default
  f16 KV cache) so the greedy pass@N is reproducible run-to-run — the MMLU and
  perf stages keep the default server/cache and are untouched.
- Progress: MMLU shows a live per-doc progress bar with ETA; the coding stage
  prints one line per finished exercise (full `benchmark.py` output stays in
  `results/coding_<model>.log`).

## Notes

- Keep `--ctx` ≥ 4096 for MMLU; larger only matters for long coding contexts.
- A model with a broken/incomplete GGUF will fail at server startup; the run
  logs the last server lines and continues with the next model.
- A model gets a composite score from whichever components completed: PPL +
  MMLU + pass@2 when all are present, otherwise PPL alone or MMLU+pass@2 alone;
  partial runs are still reported in the table with `n/a` columns.
- All stages run greedy (MMLU: temperature 0 + top_k 1; coding: aider sends
  temperature 0 for these models). The coding stage runs on its own server with
  a determinism fix (`--coding-kv-fix`, default f16 KV cache) so pass@N is
  reproducible run-to-run; the MMLU/perf server keeps the default prompt cache
  and is untouched.
- `--tries` is a repair loop, not repeated sampling: test failures are fed
  back into the same chat for the next try, so pass@2 ≥ pass@1 by
  construction.
- Uncertainty: MMLU shows ±1.96×SE, coding shows Wilson 95% CIs (wide at
  n=34 exercises), PPL shows its bootstrap 95% CI (per-token SE; tighter the
  longer the reference — the shipped one is ~274K tokens). Each model's score is
  shown as its component breakdown (e.g. `0.50×PPL(…) + 0.25×MMLU(…) + 0.25×pass@2(…)`).
  A `*` next to a score marks an adjacent-rank difference that is not significant
  at the 95% CI level — do not read fine-grained order into unmarked gaps smaller
  than the CIs.

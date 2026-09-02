#!/usr/bin/env bash
# Example benchmark run: benchmarks the models listed in models.yaml.
# If your llama-server is not on PATH, add:  --llama-server /path/to/llama-server

uv run quant-bench run \
  --config models.yaml \
  --results-dir results/gemma \
  --port 8126 \
  --weights 0.5 \
  --yes

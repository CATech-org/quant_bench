"""Latency / throughput probe for a running llama-server (streaming completions)."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass

import requests

PROBE_PROMPT = (
    "Write a short Python function that returns the nth Fibonacci number "
    "using iteration, with a docstring. Then briefly explain its time complexity. "
)


@dataclass
class PerfResult:
    n_requests: int
    ttft_ms_mean: float
    ttft_ms_p50: float
    ttft_ms_p95: float
    tok_s_mean: float
    tok_s_median: float
    total_tokens: int
    duration_s: float


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def probe(
    *,
    base_url: str,
    model_name: str,
    n_requests: int = 20,
    max_tokens: int = 128,
    timeout: float = 600.0,
) -> PerfResult:
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model_name,
        "prompt": PROBE_PROMPT,
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "stream": True,
    }
    ttfts: list[float] = []
    tok_s: list[float] = []
    total_tokens = 0
    t0 = time.perf_counter()
    with requests.Session() as sess:
        for _ in range(n_requests):
            t_start = time.perf_counter()
            first_tok: float | None = None
            n = 0
            with sess.post(url, json=payload, stream=True, timeout=(10.0, timeout)) as r:
                r.raise_for_status()
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    try:
                        delta = chunk["choices"][0]["text"]
                    except (KeyError, IndexError, TypeError):
                        continue
                    if not delta:
                        continue
                    now = time.perf_counter()
                    if first_tok is None:
                        first_tok = now
                    n += 1
            t_end = time.perf_counter()
            if first_tok is None or n == 0:
                continue
            ttfts.append((first_tok - t_start) * 1000.0)
            tok_s.append(n / max(t_end - first_tok, 1e-6))
            total_tokens += n
    return PerfResult(
        n_requests=len(ttfts),
        ttft_ms_mean=statistics.fmean(ttfts),
        ttft_ms_p50=_pctl(ttfts, 0.50),
        ttft_ms_p95=_pctl(ttfts, 0.95),
        tok_s_mean=statistics.fmean(tok_s),
        tok_s_median=statistics.median(tok_s),
        total_tokens=total_tokens,
        duration_s=time.perf_counter() - t0,
    )

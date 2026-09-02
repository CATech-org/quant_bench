"""Run MMLU (lm-evaluation-harness) against a running llama-server."""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

from quant_bench.llama_lm import LlamaServerLM  # noqa: F401  # registers "llama-server" in the lm-eval registry

PRIMARY_METRIC = {"mmlu": "acc", "mmlu_generative": "exact_match"}
TOP_LOGPROBS = 50
MMLU_SUBJECTS = 57
DOCS_PER_SUBJECT = 140
console = Console()


@dataclass
class MMLUResult:
    """Aggregated results of one MMLU run.

    Attributes:
        task: The lm-eval task name (e.g. ``mmlu``).
        score: The primary metric as a fraction (0-1).
        score_metric: The metric name that ``score`` refers to (e.g. ``acc``).
        score_stderr: Standard error of the score (bootstrap), if reported.
        n_samples: Number of documents scored.
        duration_s: Wall-clock run time in seconds.
        raw: Full lm-eval output (excluded from ``repr``).
        result_path: Path to the saved raw JSON, if any.
    """

    task: str
    score: float
    score_metric: str
    score_stderr: Optional[float]
    n_samples: int
    duration_s: float
    raw: dict = field(repr=False)
    result_path: Optional[Path] = None


@contextlib.contextmanager
def _silence_stderr():
    """Temporarily redirect file-descriptor 2 (stderr) to /dev/null.

    Used to hide fd-level noise (e.g. transformers' env-info probe shelling
    out to git) that bypasses logging configuration. Yields unconditionally.
    """
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        yield
        return
    old_fd = os.dup(2)
    os.dup2(null_fd, 2)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        os.close(null_fd)


def _metric(metrics: dict, name: str) -> Optional[float]:
    """Look up a metric by name in an lm-eval metrics dict.

    Tries the ``"<name>,none"`` key first (lm-eval's grouping convention),
    then the bare name.

    Args:
        metrics: An lm-eval metrics dictionary.
        name: The metric name to look up.

    Returns:
        Optional[float]: The metric as a float, or ``None`` if absent.
    """
    for key in (f"{name},none", name):
        v = metrics.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def run_mmlu(
    *,
    base_url: str,
    model_name: str,
    tokenizer: str,
    task: str = "mmlu",
    limit: Optional[int] = None,
    num_concurrent: int = 8,
    max_length: int = 8192,
    results_dir: Path = Path("results"),
) -> MMLUResult:
    """Run an MMLU evaluation against a running llama-server via lm-eval.

    Args:
        base_url: Base URL of the running llama-server.
        model_name: The served model id to send in requests.
        tokenizer: HuggingFace tokenizer id or directory matching the GGUF.
        task: The MMLU task: ``mmlu`` (loglikelihood) or ``mmlu_generative``.
        limit: Docs per subject (for smoke tests); ``None`` uses the default.
        num_concurrent: Number of parallel scoring requests.
        max_length: Maximum context length for the evaluation.
        results_dir: Where to save the raw per-run JSON.

    Returns:
        MMLUResult: The aggregated score, standard error, and sample count.

    Raises:
        ValueError: If ``task`` is not a supported MMLU task.
        RuntimeError: If lm-eval returns no results or the metric is missing.
    """
    if task not in PRIMARY_METRIC:
        raise ValueError(f"unsupported mmlu task {task!r} (use: {', '.join(PRIMARY_METRIC)})")
    from lm_eval import simple_evaluate

    model_args = {
        "base_url": f"{base_url}/v1/completions",
        "model": model_name,
        "tokenizer": tokenizer,
        "num_concurrent": int(num_concurrent),
        "max_length": int(max_length),
        "top_logprobs": TOP_LOGPROBS,
        "timeout": 600,
        # generous: transient server-side 500s do happen under load; backoff has jitter
        "max_retries": 8,
    }
    n_docs = MMLU_SUBJECTS * (int(limit) if limit else DOCS_PER_SUBJECT)
    console.print(f"MMLU: {n_docs:,} docs ({MMLU_SUBJECTS} subjects), {num_concurrent} concurrent")
    t0 = time.time()
    # transformers' env-info probe shells out to git (fd-level stderr noise)
    with _silence_stderr():
        res = simple_evaluate(
            model="llama-server",
            model_args=model_args,
            tasks=[task],
            num_fewshot=0,
            batch_size=1,
            limit=limit,
            log_samples=False,
            bootstrap_iters=1000,
        )
    duration_s = time.time() - t0
    if not res:
        raise RuntimeError("lm-eval returned no results")

    groups = res.get("groups") or {}
    metrics = groups.get(task) or (res.get("results") or {}).get(task)
    if not metrics:
        raise RuntimeError(f"no results for task {task!r}")
    metric_name = PRIMARY_METRIC[task]
    score = _metric(metrics, metric_name)
    if score is None:
        raise RuntimeError(f"metric {metric_name!r} not found in {sorted(metrics)}")
    stderr = _metric(metrics, f"{metric_name}_stderr")
    n_samples = int(metrics.get("sample_len", 0))

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(model_name)).strip("-.")
    result_path = results_dir / f"mmlu_{task}_{slug}.json"
    raw = {
        "task": task,
        "model": model_name,
        "base_url": base_url,
        "tokenizer": tokenizer,
        "limit": limit,
        "score": score,
        "score_metric": metric_name,
        "score_stderr": stderr,
        "n_samples": n_samples,
        "duration_s": duration_s,
        "results": res.get("results"),
        "groups": res.get("groups"),
        "n-samples": res.get("n-samples"),
    }
    result_path.write_text(json.dumps(raw, indent=2, default=str))
    return MMLUResult(
        task=task,
        score=score,
        score_metric=metric_name,
        score_stderr=stderr,
        n_samples=n_samples,
        duration_s=duration_s,
        raw=raw,
        result_path=result_path,
    )

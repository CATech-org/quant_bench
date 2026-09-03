"""Per-model benchmark execution: serve for MMLU/perf, re-serve for coding, then run PPL.

This is the execution layer behind the ``quant-bench run`` command. It takes a
resolved :class:`RunConfig` and a single model and drives the serving/coding/PPL
stages, capturing per-stage failures on the returned :class:`ModelScore`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

from quant_bench.coding import run_coding
from quant_bench.config import ModelSpec, ServerProfile, server_args_for
from quant_bench.llamaserver import LlamaServer, ServerError
from quant_bench.mmlu import run_mmlu
from quant_bench.perf import probe
from quant_bench.ppl import run_ppl
from quant_bench.report import ModelScore

console = Console()


@dataclass
class RunConfig:
    """Resolved, immutable settings shared across the per-model benchmark stages.

    Attributes:
        server: Global llama-server settings (port, ctx, device, ...).
        binary: Path to the llama-server binary.
        version: The llama-server version banner.
        config: Path to the models.yaml config.
        results_dir: Directory for reports and logs.
        benchmark_root: Directory holding the cloned benchmark repos.
        startup_timeout: Seconds to wait for server + model load.
        skip_mmlu: Whether the MMLU stage is skipped.
        skip_perf: Whether the perf probe is skipped.
        skip_coding: Whether the coding stage is skipped.
        mmlu_task: The MMLU task name.
        mmlu_limit: MMLU docs-per-subject override, if any.
        mmlu_concurrency: Number of parallel MMLU scoring requests.
        edit_format: aider edit format for the coding stage.
        languages: Comma-separated polyglot languages.
        tries: Number of repair-loop tries per test.
        coding_limit: Max polyglot tests, if any.
        coding_kv_fix: Coding-server determinism fix (f16/no-cache/off).
        perf_requests: Number of perf probe requests.
        perf_max_tokens: Max tokens generated per perf request.
        ppl_reference: Path to the PPL reference text.
        ppl_ctx: Context window for the PPL probe.
        ppl_runs: PPL runs per model.
        ppl_weight: PPL weight in the composite score.
        ppl_available: Whether the PPL stage will run.
        perplexity_bin: Path to llama-perplexity, if available.
    """

    server: ServerProfile
    binary: str
    version: str
    config: Path
    results_dir: Path
    benchmark_root: Path
    startup_timeout: int
    skip_mmlu: bool
    skip_perf: bool
    skip_coding: bool
    mmlu_task: str
    mmlu_limit: Optional[int]
    mmlu_concurrency: int
    edit_format: str
    languages: str
    tries: int
    coding_limit: Optional[int]
    coding_kv_fix: str
    perf_requests: int
    perf_max_tokens: int
    ppl_reference: Path
    ppl_ctx: int
    ppl_runs: int
    ppl_weight: float
    ppl_available: bool
    perplexity_bin: Optional[str]


def _coding_kv_fix_args(fix: str) -> list[str]:
    """Extra llama-server flags for the coding stage to make it reproducible.

    Args:
        fix: The determinism fix: ``f16`` (f16 KV cache), ``no-cache``
            (``--no-cache-prompt``), or ``off`` (none).

    Returns:
        list[str]: The extra llama-server flags, or an empty list for ``off``.
    """
    if fix == "f16":
        return ["--cache-type-k", "f16", "--cache-type-v", "f16"]
    if fix == "no-cache":
        return ["--no-cache-prompt"]
    return []


def run_model(cfg: RunConfig, model: ModelSpec) -> ModelScore:
    """Benchmark a single model across the enabled stages.

    Serves the model for MMLU/perf (server #1), re-serves it with the coding
    determinism fix for the polyglot benchmark (server #2), then runs PPL as a
    separate process. Per-stage failures are captured on the returned score; a
    server that fails to start is recorded as a coding-stage server error.

    Args:
        cfg: The resolved run settings.
        model: The model to benchmark.

    Returns:
        ModelScore: The populated per-model results.
    """
    args = server_args_for(model, cfg.server)
    score = ModelScore(
        label=model.label,
        slug=model.slug,
        path=str(model.path),
        flags=list(model.flags),
        server_cmd=f"{cfg.binary} {' '.join(args)}",
        server_version=cfg.version,
    )
    try:
        _run_mmlu_and_perf(cfg, args, model, score)
        _run_coding(cfg, args, model, score)
        _run_ppl(cfg, model, score)
    except ServerError as e:
        console.print(f"[red]server error:[/red] {e}")
        score.coding_error = score.coding_error or f"server error: {e}"
    return score


def _run_mmlu_and_perf(cfg: RunConfig, args: list[str], model: ModelSpec, score: ModelScore) -> None:
    """Server #1: run the MMLU and perf stages on a fresh llama-server.

    The server is started outside the per-stage guards, so a startup failure
    propagates as a ``ServerError`` (handled by :func:`run_model`). Each stage
    that runs is individually guarded so one failure does not stop the other.

    Args:
        cfg: The resolved run settings.
        args: Base llama-server args for this model.
        model: The model to benchmark.
        score: The per-model results to populate.
    """
    srv = LlamaServer(
        binary=cfg.binary,
        args=args,
        log_path=cfg.results_dir / f"server_{model.slug}.log",
        startup_timeout=cfg.startup_timeout,
    )
    srv.start()
    model_name = srv.model_name() or model.slug
    try:
        if not cfg.skip_mmlu:
            try:
                score.mmlu = run_mmlu(
                    base_url=srv.url,
                    model_name=model_name,
                    tokenizer=str(model.tokenizer),
                    task=cfg.mmlu_task,
                    limit=cfg.mmlu_limit,
                    num_concurrent=cfg.mmlu_concurrency,
                    max_length=cfg.server.ctx,
                    results_dir=cfg.results_dir,
                )
                console.print(
                    f"MMLU ({cfg.mmlu_task}) {score.mmlu.score_metric}: "
                    f"[bold]{score.mmlu.score * 100:.2f}%[/bold] "
                    f"(n={score.mmlu.n_samples}, {score.mmlu.duration_s:.0f}s)"
                )
            except Exception as e:  # noqa: BLE001
                score.mmlu_error = f"{type(e).__name__}: {e}"
                console.print(f"[red]MMLU failed:[/red] {score.mmlu_error}")
        if not cfg.skip_perf:
            try:
                score.perf = probe(
                    base_url=srv.url,
                    model_name=model_name,
                    n_requests=cfg.perf_requests,
                    max_tokens=cfg.perf_max_tokens,
                )
                console.print(
                    f"perf: TTFT p50 {score.perf.ttft_ms_p50:.0f} ms, "
                    f"{score.perf.tok_s_median:.1f} tok/s ({score.perf.n_requests} requests)"
                )
            except Exception as e:  # noqa: BLE001
                score.perf_error = f"{type(e).__name__}: {e}"
                console.print(f"[red]perf probe failed:[/red] {score.perf_error}")
    finally:
        srv.stop()


def _run_coding(cfg: RunConfig, args: list[str], model: ModelSpec, score: ModelScore) -> None:
    """Server #2: run the polyglot coding benchmark on its own llama-server.

    Uses a fresh server with the coding determinism fix, isolated from the
    MMLU/perf server. A server-start failure is captured as a coding error.

    Args:
        cfg: The resolved run settings.
        args: Base llama-server args for this model.
        model: The model to benchmark.
        score: The per-model results to populate.
    """
    if cfg.skip_coding:
        return
    csrv = LlamaServer(
        binary=cfg.binary,
        args=args + _coding_kv_fix_args(cfg.coding_kv_fix),
        log_path=cfg.results_dir / f"server_{model.slug}_coding.log",
        startup_timeout=cfg.startup_timeout,
    )
    try:
        csrv.start()
        score.coding = run_coding(
            benchmark_root=cfg.benchmark_root,
            server_url=csrv.url,
            model_slug=model.slug,
            server_ctx=cfg.server.ctx,
            edit_format=cfg.edit_format,
            languages=cfg.languages,
            tries=cfg.tries,
            num_tests=cfg.coding_limit or -1,
            log_path=cfg.results_dir / f"coding_{model.slug}.log",
        )
        c = score.coding
        console.print(
            f"polyglot: pass@1 [bold]{c.pass_rate_1:.1f}%[/bold], "
            f"pass@2 [bold]{c.pass_rate_2:.1f}%[/bold] "
            f"({c.completed_tests}/{c.total_tests} tests, {c.duration_s:.0f}s)"
        )
    except Exception as e:  # noqa: BLE001
        score.coding_error = f"{type(e).__name__}: {e}"
        console.print(f"[red]coding benchmark failed:[/red] {score.coding_error}")
    finally:
        csrv.stop()


def _run_ppl(cfg: RunConfig, model: ModelSpec, score: ModelScore) -> None:
    """Run the PPL fidelity metric as a separate process.

    Args:
        cfg: The resolved run settings.
        model: The model to benchmark.
        score: The per-model results to populate.
    """
    if not cfg.ppl_available:
        return
    try:
        score.ppl = run_ppl(
            binary=cfg.perplexity_bin,
            model_path=model.path,
            reference=cfg.ppl_reference,
            server=cfg.server,
            ctx=cfg.ppl_ctx,
            runs=cfg.ppl_runs,
            tokenizer=str(model.tokenizer),
        )
        se = f" ±{score.ppl.ppl_se:.2f}" if score.ppl.ppl_se is not None else ""
        console.print(
            f"PPL: [bold]{score.ppl.ppl:.2f}{se}[/bold] (lower is better; "
            f"~{score.ppl.n_tokens / 1000:.0f}K tokens, "
            f"{score.ppl.num_runs} runs, {score.ppl.duration_s:.0f}s)"
        )
    except Exception as e:  # noqa: BLE001
        score.ppl_error = f"{type(e).__name__}: {e}"
        console.print(f"[red]PPL failed:[/red] {score.ppl_error}")


def build_meta(cfg: RunConfig, duration_s: float) -> dict:
    """Build the run metadata dict embedded in the reports.

    Args:
        cfg: The resolved run settings.
        duration_s: Total wall-clock benchmark time in seconds.

    Returns:
        dict: Metadata keyed by stage; skipped stages and (when PPL is
        unavailable) the PPL fields are recorded as ``None``.
    """
    return {
        "llama_server": cfg.binary,
        "llama_server_version": cfg.version,
        "config": str(cfg.config),
        "total_duration_s": round(duration_s, 1),
        "mmlu_task": None if cfg.skip_mmlu else cfg.mmlu_task,
        "mmlu_limit": cfg.mmlu_limit,
        "languages": None if cfg.skip_coding else cfg.languages,
        "coding_limit": None if cfg.skip_coding else cfg.coding_limit,
        "coding_kv_fix": None if cfg.skip_coding else cfg.coding_kv_fix,
        "ppl_weight": None if not cfg.ppl_available else cfg.ppl_weight,
        "ppl_reference": None if not cfg.ppl_available else str(cfg.ppl_reference),
    }

"""quant-bench CLI: `setup` clones the benchmark repos, `run` benchmarks each model in models.yaml."""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

import click
from rich.console import Console
from rich.prompt import FloatPrompt

from quant_bench.coding import CodingError, _total_exercises, find_benchmarks, run_coding
from quant_bench.config import ConfigError, ServerProfile, load_models, server_args_for
from quant_bench.llamaserver import LlamaServer, ServerError, find_llama_server, llama_server_version
from quant_bench.mmlu import DOCS_PER_SUBJECT, MMLU_SUBJECTS, run_mmlu
from quant_bench.perf import probe
from quant_bench.ppl import PPLError, find_llama_perplexity, run_ppl
from quant_bench.report import ModelScore, compute_scores, write_report

console = Console()

AIDER_REPO = "https://github.com/Aider-AI/aider.git"
POLYGLOT_REPO = "https://github.com/Aider-AI/polyglot-benchmark.git"


@click.group(invoke_without_command=True)
@click.pass_context
def app(ctx: click.Context) -> None:
    """Benchmark 1-5 GGUF quantizations: MMLU + aider polyglot coding, via llama-server."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


def _clone(repo: str, dest: Path, tag: Optional[str] = None) -> None:
    """Shallow-clone a git repo, skipping if the destination already exists.

    Args:
        repo: The git repository URL to clone.
        dest: The directory to clone into (created as needed).
        tag: Optional tag to check out; omitted clones the default branch.

    Raises:
        subprocess.CalledProcessError: If the clone fails (a partial destination
            is removed first).
    """
    if (dest / ".git").is_dir():
        console.print(f"[yellow]Already cloned, skipping:[/yellow] {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"]
    if tag:
        cmd += ["--branch", tag]
    cmd += [repo, str(dest)]
    console.print(f"[bold]Cloning:[/bold] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if dest.exists():
            subprocess.run(["rm", "-rf", str(dest)], check=False)
        raise


def _aider_version() -> str:
    """Return the installed aider-chat version.

    Returns:
        str: The aider-chat version string.

    Raises:
        ConfigError: If aider-chat is not installed (run ``uv sync`` first).
    """
    try:
        return importlib.metadata.version("aider-chat")
    except importlib.metadata.PackageNotFoundError as e:
        raise ConfigError("aider-chat is not installed; run `uv sync` first") from e


def _fail(msg: str) -> None:
    """Print an error message and exit the CLI with a non-zero status.

    Args:
        msg: The error message to display.
    """
    console.print(f"[red]error:[/red] {msg}")
    sys.exit(1)


def _fmt_dur(seconds: float) -> str:
    """Format a duration in seconds as a short human-friendly estimate.

    Args:
        seconds: The duration in seconds.

    Returns:
        str: e.g. ``"~1 h 05 min"`` or ``"~7 min"``.
    """
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m = rem // 60
    return f"~{h} h {m:02d} min" if h else f"~{m} min"


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


def _estimate_runtime(
    n_models: int,
    *,
    skip_mmlu: bool,
    skip_perf: bool,
    skip_coding: bool,
    skip_ppl: bool,
    mmlu_limit: Optional[int],
    coding_attempts: Optional[int],
) -> tuple[float, float, list[str]]:
    """Rough (low, high) seconds plus scope lines, assuming a 27B-class model on a decent GPU.

    Args:
        n_models: Number of models in the config.
        skip_mmlu: Whether the MMLU stage is skipped.
        skip_perf: Whether the perf probe is skipped.
        skip_coding: Whether the coding stage is skipped.
        skip_ppl: Whether the PPL stage is skipped.
        mmlu_limit: MMLU docs-per-subject override, if any.
        coding_attempts: Estimated number of coding attempts, or ``None`` if unknown.

    Returns:
        tuple[float, float, list[str]]: The low and high runtime estimates in
        seconds, plus one scope line per stage that will run.
    """
    lo = hi = 0.0
    scope: list[str] = []
    if not skip_mmlu:
        n_docs = MMLU_SUBJECTS * (int(mmlu_limit) if mmlu_limit else DOCS_PER_SUBJECT)
        frac = n_docs / (MMLU_SUBJECTS * DOCS_PER_SUBJECT)
        lo += n_models * 45 * 60 * frac
        hi += n_models * 60 * 60 * frac
        scope.append(f"MMLU: {n_docs:,} docs/model")
    if not skip_perf:
        lo += n_models * 60
        hi += n_models * 180
        scope.append("perf probe: ~1-2 min/model")
    if not skip_coding:
        if coding_attempts:
            lo += n_models * coding_attempts * 30
            hi += n_models * coding_attempts * 60
            scope.append(f"coding: {coding_attempts} attempts/model")
        else:
            scope.append("coding: attempt count unknown (run `quant-bench setup` first)")
    if not skip_ppl:
        lo += n_models * 120
        hi += n_models * 600
        scope.append("PPL: ~2-10 min/model (a few runs on a fixed reference)")
    return lo, hi, scope


@app.command()
@click.option(
    "--benchmark-root",
    type=click.Path(path_type=Path),
    default="tmp.benchmarks",
    show_default=True,
    help="Where to clone the benchmark repos",
)
def setup(benchmark_root: Path) -> None:
    """Clone aider (tag matching installed aider-chat) and polyglot-benchmark."""
    if shutil.which("git") is None:
        _fail("git not found on PATH; it is required for setup")
    v = _aider_version()
    try:
        _clone(AIDER_REPO, Path(benchmark_root) / "aider", f"v{v}")
    except subprocess.CalledProcessError:
        console.print(f"[yellow]Tag v{v} not found, retrying with tag {v} ...[/yellow]")
        _clone(AIDER_REPO, Path(benchmark_root) / "aider", v)
    _clone(POLYGLOT_REPO, Path(benchmark_root) / "polyglot-benchmark")
    console.print(f"[green]setup complete:[/green] aider v{v} + polyglot-benchmark under {benchmark_root}")


@app.command()
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    default="models.yaml",
    show_default=True,
    help="YAML model config: list of family groups (tokenizer, models-dir, flags, models); 1-5 models total",
)
@click.option("--llama-server", default="llama-server", show_default=True, help="llama-server binary (name or path)")
@click.option("--port", type=int, default=8080, show_default=True)
@click.option(
    "--host",
    default="0.0.0.0",
    show_default=True,
    help="Interface llama-server binds to (client still connects via 127.0.0.1)",
)
@click.option(
    "--device", default="vram", show_default=True, help="vram (all layers on GPU), cpu or hybrid (needs --ngl)"
)
@click.option("--ngl", type=int, default=None, help="GPU layers; overrides --device")
@click.option("--threads", type=int, default=None, help="CPU threads for llama-server")
@click.option("--parallel", type=int, default=None, help="llama-server slots (default: --mmlu-concurrency)")
@click.option(
    "--ctx",
    type=int,
    default=8192,
    show_default=True,
    help=(
        "Per-request context size; the server gets -c ctx x slots "
        "(total KV grows with --mmlu-concurrency, lower either if you OOM)"
    ),
)
@click.option(
    "--extra-flags",
    multiple=True,
    default=(),
    help="Extra llama-server flags, single tokens only (repeatable), e.g. --extra-flags -fa",
)
@click.option(
    "--log-level",
    type=int,
    default=2,
    show_default=True,
    help=(
        "llama-server -lv (1=error, 2=warn, 3=info, 4=trace, 5=debug). "
        "Default 2 (warn) keeps the per-request info flood out of the server logs; "
        "raise to 3+ to debug (logs grow to multi-GB on MMLU)."
    ),
)
@click.option("--mmlu-task", default="mmlu", show_default=True, help="mmlu (loglikelihood) or mmlu_generative")
@click.option(
    "--mmlu-limit",
    type=int,
    default=None,
    help="MMLU docs per subject (smoke tests; applied to each of the 57 MMLU subjects)",
)
@click.option(
    "--mmlu-concurrency", type=int, default=8, show_default=True, help="parallel MMLU scoring requests"
)
@click.option("--edit-format", default="whole", show_default=True, help="aider edit format (whole, udiff, ...)")
@click.option("--languages", default="python", show_default=True, help="polyglot languages, comma separated")
@click.option("--tries", type=int, default=2, show_default=True, help="polyglot tries per test")
@click.option(
    "--coding-limit",
    type=int,
    default=None,
    help="Run only N polyglot tests (unseeded shuffle: results not comparable across models)",
)
@click.option("--perf-requests", type=int, default=20, show_default=True)
@click.option("--perf-max-tokens", type=int, default=128, show_default=True)
@click.option(
    "--weights", type=float, default=None, help="MMLU weight for the composite score, 0.0-1.0 (skips the prompt)"
)
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Where report/logs/server logs go (created if missing); use a different dir per config",
)
@click.option(
    "--benchmark-root",
    type=click.Path(path_type=Path),
    default="tmp.benchmarks",
    show_default=True,
)
@click.option(
    "--startup-timeout", type=int, default=900, show_default=True, help="Max seconds to wait for server+model load"
)
@click.option("--skip-mmlu", is_flag=True, default=False)
@click.option("--skip-perf", is_flag=True, default=False)
@click.option("--skip-coding", is_flag=True, default=False)
@click.option(
    "--ppl-reference",
    type=click.Path(path_type=Path),
    default="scripts/ppl_ref.txt",
    show_default=True,
    help="Reference text for the PPL fidelity metric (lower PPL = better quant)",
)
@click.option("--skip-ppl", is_flag=True, default=False)
@click.option(
    "--ppl-weight",
    type=float,
    default=0.5,
    show_default=True,
    help="PPL (fidelity) weight in the composite score, 0.0-1.0",
)
@click.option("--ppl-ctx", type=int, default=1024, show_default=True, help="Context window for the PPL probe")
@click.option("--ppl-runs", type=int, default=2, show_default=True, help="PPL runs per model (mean + reproducibility)")
@click.option(
    "--coding-kv-fix",
    default="f16",
    show_default=True,
    help="Coding-server determinism fix: f16 (f16 KV cache), no-cache (--no-cache-prompt), or off",
)
@click.option("-y", "--yes", is_flag=True, default=False, help="Skip the preflight confirmation prompt")
def run(
    config: Path,
    llama_server: str,
    port: int,
    host: str,
    device: str,
    ngl: Optional[int],
    threads: Optional[int],
    parallel: Optional[int],
    ctx: int,
    extra_flags: list[str],
    log_level: int,
    mmlu_task: str,
    mmlu_limit: Optional[int],
    mmlu_concurrency: int,
    edit_format: str,
    languages: str,
    tries: int,
    coding_limit: Optional[int],
    perf_requests: int,
    perf_max_tokens: int,
    weights: Optional[float],
    results_dir: Path,
    benchmark_root: Path,
    startup_timeout: int,
    skip_mmlu: bool,
    skip_perf: bool,
    skip_coding: bool,
    ppl_reference: Path,
    skip_ppl: bool,
    ppl_weight: float,
    ppl_ctx: int,
    ppl_runs: int,
    coding_kv_fix: str,
    yes: bool,
) -> None:
    """Benchmark every model in --config, one at a time, and write a ranked report.

    Runs the enabled stages (MMLU, perf, coding, PPL) for each model in --config,
    computes a composite score, and writes report.md / report.json plus per-model
    logs into --results-dir.
    """
    if not skip_mmlu and mmlu_task not in ("mmlu", "mmlu_generative"):
        _fail(f"unknown --mmlu-task {mmlu_task!r} (use mmlu or mmlu_generative)")
    if not (0.0 <= (weights or 0.0) <= 1.0) and weights is not None:
        _fail("--weights must be between 0.0 and 1.0")
    if not (0.0 <= ppl_weight <= 1.0):
        _fail("--ppl-weight must be between 0.0 and 1.0")
    if coding_kv_fix not in ("f16", "no-cache", "off"):
        _fail("--coding-kv-fix must be one of: f16, no-cache, off")
    if not skip_ppl and not Path(ppl_reference).expanduser().is_file():
        _fail(f"--ppl-reference not found: {ppl_reference} (or pass --skip-ppl)")

    try:
        models = load_models(config)
    except ConfigError as e:
        _fail(str(e))

    server = ServerProfile(
        port=port,
        host=host,
        ctx=ctx,
        device=device,
        ngl=ngl,
        threads=threads,
        parallel=parallel if parallel is not None else mmlu_concurrency,
        extra_flags=extra_flags or [],
        log_level=log_level,
    )
    try:
        server.base_args()
    except ConfigError as e:
        _fail(str(e))
    try:
        binary = find_llama_server(llama_server)
    except ServerError as e:
        _fail(str(e))
    version = llama_server_version(binary)
    console.print(f"llama-server: {binary}\n{version}\n")

    ppl_available = not skip_ppl
    perplexity_bin: Optional[str] = None
    if ppl_available:
        try:
            perplexity_bin = str(find_llama_perplexity(binary))
        except PPLError as e:
            console.print(f"[yellow]PPL disabled:[/yellow] {e}")
            ppl_available = False

    # ------------------------------------------------- preflight: scope + confirm
    coding_attempts: Optional[int] = None
    if not skip_coding:
        try:
            _, _, polyglot = find_benchmarks(benchmark_root)
            coding_attempts = _total_exercises(polyglot, languages, coding_limit or -1) * tries
        except CodingError:
            coding_attempts = None
    lo, hi, scope = _estimate_runtime(
        len(models),
        skip_mmlu=skip_mmlu,
        skip_perf=skip_perf,
        skip_coding=skip_coding,
        skip_ppl=not ppl_available,
        mmlu_limit=mmlu_limit,
        coding_attempts=coding_attempts,
    )
    console.print(f"[bold]Estimated runtime:[/bold] {_fmt_dur(lo)} - {_fmt_dur(hi)} (rough, depends on hardware)")
    for line in scope:
        console.print(f"  [dim]{line}[/dim]")
    if not yes:
        if not click.confirm("Continue?"):
            console.print("Aborted.")
            sys.exit(0)
    if weights is None:
        weight = FloatPrompt.ask(
            "MMLU weight for the composite score (0.0-1.0, rest goes to aider coding)", default=0.5
        )
        while not (0.0 <= weight <= 1.0):
            weight = FloatPrompt.ask("Enter a weight between 0.0 and 1.0", default=0.5)
    else:
        weight = float(weights)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    scores: list[ModelScore] = []
    t_start = time.time()

    for i, model in enumerate(models, start=1):
        console.rule(f"[bold]{i}/{len(models)}: {model.label}[/bold]")
        args = server_args_for(model, server)
        score = ModelScore(
            label=model.label,
            slug=model.slug,
            path=str(model.path),
            flags=list(model.flags),
            server_cmd=f"{binary} {' '.join(args)}",
            server_version=version,
        )
        try:
            # --- server #1: MMLU + perf (default prompt cache; MMLU left untouched) ---
            srv = LlamaServer(
                binary=binary,
                args=args,
                log_path=results_dir / f"server_{model.slug}.log",
                startup_timeout=startup_timeout,
            )
            srv.start()
            model_name = srv.model_name() or model.slug
            try:
                if not skip_mmlu:
                    try:
                        score.mmlu = run_mmlu(
                            base_url=srv.url,
                            model_name=model_name,
                            tokenizer=str(model.tokenizer),
                            task=mmlu_task,
                            limit=mmlu_limit,
                            num_concurrent=mmlu_concurrency,
                            max_length=ctx,
                            results_dir=results_dir,
                        )
                        console.print(
                            f"MMLU ({mmlu_task}) {score.mmlu.score_metric}: "
                            f"[bold]{score.mmlu.score * 100:.2f}%[/bold] "
                            f"(n={score.mmlu.n_samples}, {score.mmlu.duration_s:.0f}s)"
                        )
                    except Exception as e:  # noqa: BLE001
                        score.mmlu_error = f"{type(e).__name__}: {e}"
                        console.print(f"[red]MMLU failed:[/red] {score.mmlu_error}")
                if not skip_perf:
                    try:
                        score.perf = probe(
                            base_url=srv.url,
                            model_name=model_name,
                            n_requests=perf_requests,
                            max_tokens=perf_max_tokens,
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
            # --- server #2: coding, on its own server with the determinism fix (isolated from MMLU/perf) ---
            if not skip_coding:
                csrv = LlamaServer(
                    binary=binary,
                    args=args + _coding_kv_fix_args(coding_kv_fix),
                    log_path=results_dir / f"server_{model.slug}_coding.log",
                    startup_timeout=startup_timeout,
                )
                try:
                    csrv.start()
                    score.coding = run_coding(
                        benchmark_root=Path(benchmark_root),
                        server_url=csrv.url,
                        model_slug=model.slug,
                        server_ctx=ctx,
                        edit_format=edit_format,
                        languages=languages,
                        tries=tries,
                        num_tests=coding_limit or -1,
                        log_path=results_dir / f"coding_{model.slug}.log",
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
            # --- PPL: a separate process (loads the model itself; one model in VRAM at a time) ---
            if ppl_available:
                try:
                    score.ppl = run_ppl(
                        binary=perplexity_bin,
                        model_path=model.path,
                        reference=Path(ppl_reference),
                        server=server,
                        ctx=ppl_ctx,
                        runs=ppl_runs,
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
        except ServerError as e:
            console.print(f"[red]server error:[/red] {e}")
            score.coding_error = score.coding_error or f"server error: {e}"
        scores.append(score)

    console.rule("[bold]Scoring[/bold]")
    compute_scores(scores, weight, ppl_weight if ppl_available else 0.0)

    meta = {
        "llama_server": binary,
        "llama_server_version": version,
        "config": str(config),
        "total_duration_s": round(time.time() - t_start, 1),
        "mmlu_task": None if skip_mmlu else mmlu_task,
        "mmlu_limit": mmlu_limit,
        "languages": None if skip_coding else languages,
        "coding_limit": None if skip_coding else coding_limit,
        "coding_kv_fix": None if skip_coding else coding_kv_fix,
        "ppl_weight": None if not ppl_available else ppl_weight,
        "ppl_reference": None if not ppl_available else str(ppl_reference),
    }
    report_md, report_json = write_report(
        scores, weight=weight, results_dir=results_dir, meta=meta, ppl_weight=ppl_weight if ppl_available else 0.0
    )
    console.print(f"\n[bold]report:[/bold] {report_md}\n[bold]json:  [/bold] {report_json}")


def main() -> None:
    """Entry point that launches the quant-bench CLI."""
    app()


if __name__ == "__main__":
    main()

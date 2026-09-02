"""quant-bench CLI: `setup` clones the benchmark repos, `run` benchmarks each model in models.yaml."""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

import typer
from rich.console import Console
from rich.prompt import FloatPrompt

from quant_bench.coding import CodingError, _total_exercises, find_benchmarks, run_coding
from quant_bench.config import ConfigError, ServerProfile, load_models, server_args_for
from quant_bench.llamaserver import LlamaServer, ServerError, find_llama_server, llama_server_version
from quant_bench.mmlu import DOCS_PER_SUBJECT, MMLU_SUBJECTS, run_mmlu
from quant_bench.perf import probe
from quant_bench.ppl import PPLError, find_llama_perplexity, run_ppl
from quant_bench.report import ModelScore, compute_scores, write_report

app = typer.Typer(
    help="Benchmark 1-5 GGUF quantizations: MMLU + aider polyglot coding, via llama-server.",
    no_args_is_help=True,
)
console = Console()

AIDER_REPO = "https://github.com/Aider-AI/aider.git"
POLYGLOT_REPO = "https://github.com/Aider-AI/polyglot-benchmark.git"


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
    raise typer.Exit(1)


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
def setup(
    benchmark_root: Path = typer.Option(
        Path("tmp.benchmarks"), "--benchmark-root", help="Where to clone the benchmark repos"
    ),
) -> None:
    """Clone aider (tag matching installed aider-chat) and polyglot-benchmark.

    Args:
        benchmark_root: Directory to clone the benchmark repos into.

    Raises:
        typer.Exit: If git is not available.
    """
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
def run(
    config_file: Path = typer.Option(
        Path("models.yaml"),
        "--config",
        help="YAML model config: list of family groups (tokenizer, models-dir, flags, models); 1-5 models total",
    ),
    llama_server: str = typer.Option("llama-server", "--llama-server", help="llama-server binary (name or path)"),
    port: int = typer.Option(8080, "--port"),
    host: str = typer.Option(
        "0.0.0.0", "--host", help="Interface llama-server binds to (client still connects via 127.0.0.1)"
    ),
    device: str = typer.Option("vram", "--device", help="vram (all layers on GPU), cpu or hybrid (needs --ngl)"),
    ngl: Optional[int] = typer.Option(None, "--ngl", help="GPU layers; overrides --device"),
    threads: Optional[int] = typer.Option(None, "--threads", help="CPU threads for llama-server"),
    parallel: Optional[int] = typer.Option(
        None, "--parallel", help="llama-server slots (default: --mmlu-concurrency)"
    ),
    ctx: int = typer.Option(
        8192,
        "--ctx",
        help=(
            "Per-request context size; the server gets -c ctx x slots "
            "(total KV grows with --mmlu-concurrency, lower either if you OOM)"
        ),
    ),
    extra_flags: Optional[list[str]] = typer.Option(
        None, "--extra-flags", help="Extra llama-server flags, single tokens only (repeatable), e.g. --extra-flags -fa"
    ),
    log_level: int = typer.Option(
        2,
        "--log-level",
        help=(
            "llama-server -lv (1=error, 2=warn, 3=info, 4=trace, 5=debug). "
            "Default 2 (warn) keeps the per-request info flood out of the server logs; "
            "raise to 3+ to debug (logs grow to multi-GB on MMLU)."
        ),
    ),
    mmlu_task: str = typer.Option("mmlu", "--mmlu-task", help="mmlu (loglikelihood) or mmlu_generative"),
    mmlu_limit: Optional[int] = typer.Option(
        None, "--mmlu-limit", help="MMLU docs per subject (smoke tests; applied to each of the 57 MMLU subjects)"
    ),
    mmlu_concurrency: int = typer.Option(8, "--mmlu-concurrency", help="parallel MMLU scoring requests"),
    edit_format: str = typer.Option("whole", "--edit-format", help="aider edit format (whole, udiff, ...)"),
    languages: str = typer.Option("python", "--languages", help="polyglot languages, comma separated"),
    tries: int = typer.Option(2, "--tries", help="polyglot tries per test"),
    coding_limit: Optional[int] = typer.Option(
        None,
        "--coding-limit",
        help="Run only N polyglot tests (unseeded shuffle: results not comparable across models)",
    ),
    perf_requests: int = typer.Option(20, "--perf-requests"),
    perf_max_tokens: int = typer.Option(128, "--perf-max-tokens"),
    weights: Optional[float] = typer.Option(
        None, "--weights", help="MMLU weight for the composite score, 0.0-1.0 (skips the prompt)"
    ),
    results_dir: Path = typer.Option(
        ...,
        "--results-dir",
        help="Where report/logs/server logs go (created if missing); use a different dir per config",
    ),
    benchmark_root: Path = typer.Option(Path("tmp.benchmarks"), "--benchmark-root"),
    startup_timeout: int = typer.Option(900, "--startup-timeout", help="Max seconds to wait for server+model load"),
    skip_mmlu: bool = typer.Option(False, "--skip-mmlu"),
    skip_perf: bool = typer.Option(False, "--skip-perf"),
    skip_coding: bool = typer.Option(False, "--skip-coding"),
    ppl_reference: Path = typer.Option(
        Path("scripts/ppl_ref.txt"),
        "--ppl-reference",
        help="Reference text for the PPL fidelity metric (lower PPL = better quant)",
    ),
    skip_ppl: bool = typer.Option(False, "--skip-ppl"),
    ppl_weight: float = typer.Option(
        0.5, "--ppl-weight", help="PPL (fidelity) weight in the composite score, 0.0-1.0"
    ),
    ppl_ctx: int = typer.Option(1024, "--ppl-ctx", help="Context window for the PPL probe"),
    ppl_runs: int = typer.Option(2, "--ppl-runs", help="PPL runs per model (mean + reproducibility)"),
    coding_kv_fix: str = typer.Option(
        "f16",
        "--coding-kv-fix",
        help="Coding-server determinism fix: f16 (f16 KV cache), no-cache (--no-cache-prompt), or off",
    ),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip the preflight confirmation prompt"),
) -> None:
    """Benchmark every model in --config, one at a time, and write a ranked report.

    Loads each model from --config in turn, runs the enabled stages (MMLU, perf,
    coding, PPL), computes a composite score, and writes report.md / report.json
    plus per-model logs into --results-dir. A runtime estimate and a MMLU-weight
    prompt are shown before anything runs (suppressible with --yes / --weights).

    The many options mirror the CLI flags (device, ctx, MMLU/coding/PPL tuning,
    skip flags, ...); see ``--help`` for each.

    Raises:
        typer.Exit: On invalid options or a failed preflight/config check.
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
        models = load_models(config_file)
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
        if not typer.confirm("Continue?"):
            console.print("Aborted.")
            raise typer.Exit()
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
        "config": str(config_file),
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
    """Entry point that launches the quant-bench Typer CLI."""
    app()


if __name__ == "__main__":
    main()

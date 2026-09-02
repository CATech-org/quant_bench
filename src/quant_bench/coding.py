"""Run aider's official polyglot coding benchmark against a running llama-server."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


class CodingError(Exception):
    """Raised when the polyglot benchmark cannot be located, run, or parsed."""


@dataclass
class CodingResult:
    """Aggregated results of one polyglot coding benchmark run.

    Attributes:
        model: The model name used (``openai/<slug>``).
        edit_format: aider edit format (e.g. ``whole``).
        languages: Polyglot languages run (comma-separated).
        tries: Number of repair-loop tries per test.
        pass_rate_1: pass@1 as a percentage (``None`` if no completed tests).
        pass_rate_2: pass@2 as a percentage (``None`` if no completed tests).
        pass_num_1: Number of exercises passed on try 1.
        pass_num_2: Number of exercises passed within the tries.
        completed_tests: Number of exercises with recorded outcomes.
        total_tests: Total number of exercises targeted.
        duration_s: Wall-clock run time in seconds.
        prompt_tokens: Total prompt tokens across exercises.
        completion_tokens: Total completion tokens across exercises.
        run_dir: Directory holding the raw per-exercise results.
        raw: Raw per-exercise result records (excluded from ``repr``).
    """

    model: str
    edit_format: str
    languages: str
    tries: int
    pass_rate_1: Optional[float]
    pass_rate_2: Optional[float]
    pass_num_1: int
    pass_num_2: int
    completed_tests: int
    total_tests: int
    duration_s: float
    prompt_tokens: int
    completion_tokens: int
    run_dir: Path
    raw: list = field(repr=False, default_factory=list)


def find_benchmarks(benchmark_root: Path) -> tuple[Path, Path, Path]:
    """Locate the aider and polyglot-benchmark repos under a benchmark root.

    Args:
        benchmark_root: Directory containing the cloned benchmark repos.

    Returns:
        tuple[Path, Path, Path]: The resolved root, the aider repo, and the
        polyglot-benchmark directory.

    Raises:
        CodingError: If the benchmark files are missing (setup not run).
    """
    root = Path(benchmark_root).resolve()
    aider_repo = root / "aider"
    benchmark_py = aider_repo / "benchmark" / "benchmark.py"
    polyglot = root / "polyglot-benchmark"
    if not benchmark_py.is_file():
        raise CodingError(f"benchmark.py not found at {benchmark_py}; run `quant-bench setup` first")
    if not polyglot.is_dir():
        raise CodingError(f"polyglot-benchmark not found at {polyglot}; run `quant-bench setup` first")
    return root, aider_repo, polyglot


def _find_run_dir(root: Path, name: str) -> Path:
    """Find the most recent polyglot run directory for a model.

    Args:
        root: The benchmark root holding run directories.
        name: The model slug to search for in the run directory name.

    Returns:
        Path: The run directory with the most recent modification time.

    Raises:
        CodingError: If no run directory can be found.
    """
    dirs = [d for d in root.glob(f"*--{name}") if d.is_dir()]
    if not dirs:
        dirs = [
            d
            for d in root.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and d.name not in ("polyglot-benchmark", "aider")
        ]
    if not dirs:
        raise CodingError(f"could not find benchmark run dir under {root} for {name!r}")
    return max(dirs, key=lambda d: d.stat().st_mtime)


def _load_results(run_dir: Path, languages: str) -> list[dict]:
    """Load per-exercise ``.aider.results.json`` records from a run directory.

    Args:
        run_dir: The polyglot run directory.
        languages: Comma-separated languages to load (empty = all languages).

    Returns:
        list[dict]: The parsed result records, each tagged with its exercise
        name and language.
    """
    if languages:
        pats = [f"{lang.strip()}/exercises/practice/*/.aider.results.json" for lang in languages.split(",")]
    else:
        pats = ["*/exercises/practice/*/.aider.results.json"]
    out: list[dict] = []
    for pat in pats:
        for f in sorted(run_dir.glob(pat)):
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if data:
                data["exercise"] = f.parent.name
                data["language"] = f.relative_to(run_dir).parts[0]
                out.append(data)
    return out


def _count_done(run_dir: Path, languages: str) -> int:
    """Count completed exercises (result files present) in a run directory.

    Args:
        run_dir: The polyglot run directory.
        languages: Comma-separated languages to count (empty = all languages).

    Returns:
        int: The number of result files present.
    """
    if languages:
        pats = [f"{lang.strip()}/exercises/practice/*/.aider.results.json" for lang in languages.split(",")]
    else:
        pats = ["*/exercises/practice/*/.aider.results.json"]
    n = 0
    for pat in pats:
        n += sum(1 for _ in run_dir.glob(pat))
    return n


def _total_exercises(polyglot: Path, languages: str, num_tests: int) -> int:
    """Count the total practice exercises available for the given languages.

    Args:
        polyglot: The polyglot-benchmark directory.
        languages: Comma-separated languages (empty = all present languages).
        num_tests: If positive, cap the total at this many tests.

    Returns:
        int: The number of practice exercises (capped by ``num_tests``).
    """
    langs = [part.strip() for part in languages.split(",") if part.strip()] or ["*"]
    total = 0
    for lang in langs:
        d = polyglot / lang / "exercises" / "practice"
        if d.is_dir():
            total += sum(1 for p in d.iterdir() if p.is_dir())
    if num_tests and num_tests > 0:
        total = min(total, num_tests)
    return total


def _progress_poller(
    root: Path,
    model_slug: str,
    languages: str,
    total: int,
    t0: float,
    stop: threading.Event,
    pre_existing: set,
    interval: float = 30.0,
) -> None:
    """Poll a run in a background thread and print per-exercise progress.

    Prints one line per newly finished exercise (and a periodic heartbeat if an
    exercise is slow). Swallows all errors so it can never crash the run.

    Args:
        root: The benchmark root holding run directories.
        model_slug: The model slug to locate the run directory for.
        languages: Comma-separated languages being run.
        total: Total exercises expected (for the progress denominator).
        t0: Start timestamp (``time.time()``) for elapsed-time display.
        stop: Event whose set stops the polling loop.
        pre_existing: Run directories that existed before this run (excluded).
        interval: Seconds between polls.
    """
    last_count = -1
    last_print = 0.0
    while not stop.wait(interval):
        try:
            dirs = [d for d in root.glob(f"*--{model_slug}") if d.is_dir() and d not in pre_existing]
            if not dirs:
                continue
            run_dir = max(dirs, key=lambda d: d.stat().st_mtime)
            count = _count_done(run_dir, languages)
            now = time.time()
            if count != last_count or now - last_print >= 3 * interval:
                console.print(f"coding: {count}/{total} exercises ({(now - t0) / 60:.1f} min)")
                last_count = count
                last_print = now
        except Exception:  # noqa: BLE001 - polling must never crash the run
            pass


def _summarize(
    run_dir: Path,
    model: str,
    edit_format: str,
    languages: str,
    tries: int,
    duration_s: float,
) -> CodingResult:
    """Aggregate a finished run directory into a CodingResult.

    Args:
        run_dir: The polyglot run directory to summarize.
        model: The model name to record.
        edit_format: The aider edit format that was used.
        languages: The languages that were run.
        tries: The number of tries per test.
        duration_s: Wall-clock run time in seconds.

    Returns:
        CodingResult: The aggregated result (pass@1, pass@2, tokens, ...).
    """
    results = _load_results(run_dir, languages)
    if languages:
        total_tests = sum(
            len(list(run_dir.glob(f"{lang.strip()}/exercises/practice/*")))
            for lang in languages.split(",")
        )
    else:
        total_tests = len(list(run_dir.glob("*/exercises/practice/*")))
    completed = len(results)
    pass1 = pass2 = 0
    for r in results:
        oc = r.get("tests_outcomes") or []
        if oc and oc[0]:
            pass1 += 1
        if any(oc):
            pass2 += 1
    pr1 = 100.0 * pass1 / completed if completed else None
    pr2 = 100.0 * pass2 / completed if completed else None
    return CodingResult(
        model=model,
        edit_format=edit_format,
        languages=languages,
        tries=tries,
        pass_rate_1=pr1,
        pass_rate_2=pr2,
        pass_num_1=pass1,
        pass_num_2=pass2,
        completed_tests=completed,
        total_tests=total_tests,
        duration_s=duration_s,
        prompt_tokens=sum(int(r.get("prompt_tokens") or 0) for r in results),
        completion_tokens=sum(int(r.get("completion_tokens") or 0) for r in results),
        run_dir=run_dir,
        raw=results,
    )


def run_coding(
    *,
    benchmark_root: Path,
    server_url: str,
    model_slug: str,
    server_ctx: int,
    edit_format: str = "whole",
    languages: str = "python",
    tries: int = 2,
    num_tests: int = -1,
    threads: int = 1,
    log_path: Path,
) -> CodingResult:
    """Run the polyglot coding benchmark against a running llama-server.

    Invokes the cloned aider ``benchmark.py`` in-process with ``AIDER_DOCKER=1``
    and the OpenAI API base pointed at the server, in a background progress
    poller, then summarizes the resulting run directory.

    Args:
        benchmark_root: Directory containing the aider and polyglot repos.
        server_url: Base URL of the running llama-server.
        model_slug: The model slug/alias to benchmark.
        server_ctx: The server's per-request context, passed to ``--num-ctx``.
        edit_format: aider edit format to use.
        languages: Comma-separated polyglot languages.
        tries: Number of repair-loop tries per test.
        num_tests: If positive, cap the number of tests to run.
        threads: Number of parallel benchmark threads (``--threads``).
        log_path: Where the benchmark process's stdout/stderr are written.

    Returns:
        CodingResult: The aggregated coding result.

    Raises:
        CodingError: If the benchmark exits non-zero (with the log tail).
    """
    root, aider_repo, polyglot = find_benchmarks(benchmark_root)
    root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["AIDER_DOCKER"] = "1"
    env["AIDER_BENCHMARK_DIR"] = str(root)
    env["OPENAI_API_BASE"] = f"{server_url}/v1"
    env["OPENAI_API_KEY"] = "llama"
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"

    cmd = [
        sys.executable,
        "benchmark/benchmark.py",
        model_slug,
        "--new",
        "--model",
        f"openai/{model_slug}",
        "--edit-format",
        edit_format,
        "--tries",
        str(tries),
        "--num-ctx",
        str(server_ctx),
    ]
    if languages:
        cmd += ["--languages", languages]
    if num_tests and num_tests > 0:
        cmd += ["--num-tests", str(num_tests)]
    if threads and threads > 1:
        cmd += ["--threads", str(threads)]

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Running polyglot benchmark:[/bold] {' '.join(cmd)}")
    console.print(f"(exercises: {polyglot})")
    t0 = time.time()
    total = _total_exercises(polyglot, languages, num_tests)
    pre_existing = {d for d in root.glob(f"*--{model_slug}") if d.is_dir()}
    stop = threading.Event()
    poller = None
    if total > 0:
        poller = threading.Thread(
            target=_progress_poller,
            args=(root, model_slug, languages, total, t0, stop, pre_existing),
            daemon=True,
        )
        poller.start()
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, cwd=aider_repo, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    finally:
        stop.set()
        if poller is not None:
            poller.join(timeout=5)
    duration = time.time() - t0
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-25:]
        raise CodingError(
            f"polyglot benchmark exited with code {proc.returncode}; log: {log_path}\n" + "\n".join(tail)
        )

    run_dir = _find_run_dir(root, model_slug)
    return _summarize(run_dir, f"openai/{model_slug}", edit_format, languages, tries, duration)

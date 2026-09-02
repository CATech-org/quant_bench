"""Perplexity (PPL) fidelity metric for a GGUF, via llama.cpp's ``llama-perplexity``.

Perplexity is a continuous, high-resolution measure of how faithfully a quantization
reproduces its base model, and it is far more sensitive to quant quality than MMLU or
aider pass@N when the model is small (where those saturate near their floor). For a
set of quants of the *same* model, lower PPL == closer to the original == better.

``llama-perplexity`` is a separate binary that loads the model itself, so it runs as
its own process (not against the running llama-server). It prints the result on
stderr as ``Final estimate: PPL = <ppl> +/- <se>``.
"""

from __future__ import annotations

import re
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from quant_bench.config import ServerProfile

DEFAULT_PPL_CTX = 1024
DEFAULT_PPL_RUNS = 2
DEFAULT_PPL_TIMEOUT = 3600.0

# `Final estimate: PPL = 67.9739 +/- 8.71330`  (newer builds)
_FINAL_ESTIMATE_RE = re.compile(
    r"Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\+/-\s*([0-9]+(?:\.[0-9]+)?))?",
    re.IGNORECASE,
)
# `overall_ppl = 12.345`  (older/alternate builds)
_OVERALL_PPL_RE = re.compile(r"overall_ppl\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


class PPLError(Exception):
    """Raised when the perplexity binary is missing, fails, or cannot be parsed."""


@dataclass
class PPLResult:
    """Aggregated perplexity measurements for one model.

    Attributes:
        ppl: Mean perplexity across the runs (lower is better).
        ppl_se: Mean standard error reported by llama-perplexity (bootstrap), if any.
        runs: Per-run perplexity values (excluded from ``repr``).
        runs_se: Per-run standard errors, if reported (excluded from ``repr``).
        n_tokens: Number of tokens in the reference text.
        duration_s: Total wall-clock time across all runs.
        num_runs: Number of runs that were executed.
        reference: Path to the reference text file.
        cmd: The command line used for each run.
    """

    ppl: float
    ppl_se: Optional[float]
    runs: list[float] = field(repr=False, default_factory=list)
    runs_se: list[Optional[float]] = field(repr=False, default_factory=list)
    n_tokens: int = 0
    duration_s: float = 0.0
    num_runs: int = 0
    reference: str = ""
    cmd: str = ""


def find_llama_perplexity(llama_server_path: str) -> Path:
    """Locate the ``llama-perplexity`` binary (expected next to llama-server).

    Args:
        llama_server_path: Path to llama-server, used to find the sibling binary.

    Returns:
        Path: Path to the ``llama-perplexity`` binary.

    Raises:
        PPLError: If the binary is not found next to llama-server or on PATH.
    """
    cand = Path(llama_server_path).resolve().with_name("llama-perplexity")
    if cand.is_file():
        return cand
    import shutil

    on_path = shutil.which("llama-perplexity")
    if on_path:
        return Path(on_path)
    raise PPLError(
        f"llama-perplexity not found (looked next to {llama_server_path!r} and on PATH). "
        "Build llama.cpp with the perplexity example or pass a matching binary."
    )


def parse_ppl(text: str) -> Optional[tuple[float, Optional[float]]]:
    """Extract ``(ppl, se)`` from llama-perplexity output (stdout+stderr).

    Args:
        text: The combined stdout+stderr from the perplexity process.

    Returns:
        Optional[tuple[float, Optional[float]]]: The perplexity and its standard
        error (``None`` if not reported), or ``None`` if neither could be parsed.
    """
    text = text.replace("\r", "\n")
    m = _FINAL_ESTIMATE_RE.search(text)
    if m:
        ppl = float(m.group(1))
        se = float(m.group(2)) if m.group(2) else None
        return ppl, se
    m2 = _OVERALL_PPL_RE.search(text)
    if m2:
        return float(m2.group(1)), None
    return None


def _count_tokens(text: str, tokenizer: Optional[str]) -> int:
    """Token count of the reference (exact via the model tokenizer, else a ~chars/4 estimate).

    More scored tokens => a tighter PPL estimate (llama-perplexity's SE scales as 1/sqrt(n)),
    so this is surfaced in the report so the CI is interpretable.

    Args:
        text: The reference text to count tokens for.
        tokenizer: HuggingFace tokenizer id/directory, or ``None`` to estimate.

    Returns:
        int: The token count (exact if a tokenizer was available, else an estimate).
    """
    if tokenizer:
        try:
            from transformers import AutoTokenizer

            return len(AutoTokenizer.from_pretrained(tokenizer).encode(text))
        except Exception:  # noqa: BLE001 - fall back to a character estimate
            pass
    return max(0, len(text)) // 4


def build_ppl_args(model_path: Path, reference: Path, server: ServerProfile, ctx: int) -> list[str]:
    """Command-line args for llama-perplexity, honoring the profile's device settings.

    Args:
        model_path: Path to the ``.gguf`` model to load.
        reference: Path to the reference text file.
        server: The server profile (provides shared device/thread flags).
        ctx: Context window (``-c``) for the perplexity run.

    Returns:
        list[str]: argv tokens for llama-perplexity (excluding the binary itself).
    """
    return [
        "-m",
        str(model_path),
        "-f",
        str(reference),
        "-c",
        str(ctx),
        *server.ppl_base_args(),
        "-sm",
        "none",
        "--no-warmup",
    ]


def run_ppl(
    *,
    binary: str | Path,
    model_path: Path,
    reference: Path,
    server: ServerProfile,
    ctx: int = DEFAULT_PPL_CTX,
    runs: int = DEFAULT_PPL_RUNS,
    timeout: float = DEFAULT_PPL_TIMEOUT,
    tokenizer: Optional[str] = None,
) -> PPLResult:
    """Run llama-perplexity ``runs`` times and aggregate the results.

    Args:
        binary: Path to the ``llama-perplexity`` binary.
        model_path: Path to the ``.gguf`` model to load.
        reference: Path to the reference text file.
        server: The server profile (provides shared device/thread flags).
        ctx: Context window for the perplexity run.
        runs: Number of runs to perform.
        timeout: Per-run timeout in seconds.
        tokenizer: HuggingFace tokenizer for an exact token count, or ``None``.

    Returns:
        PPLResult: Mean perplexity, standard error, and per-run details.

    Raises:
        PPLError: If the reference is missing, a run times out, or output
            cannot be parsed.
    """
    reference = Path(reference)
    if not reference.is_file():
        raise PPLError(f"PPL reference file not found: {reference}")
    tokens = _count_tokens(reference.read_text(encoding="utf-8", errors="ignore"), tokenizer)

    args = build_ppl_args(model_path, reference, server, ctx)
    cmd = [str(binary), *args]
    ppls: list[float] = []
    ses: list[Optional[float]] = []
    t0 = time.time()
    for _ in range(max(1, int(runs))):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise PPLError(f"llama-perplexity timed out after {timeout:.0f}s") from e
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        parsed = parse_ppl(text)
        if parsed is None:
            tail = text.strip().splitlines()[-12:]
            raise PPLError(
                f"could not parse PPL from llama-perplexity (exit {proc.returncode}); "
                f"last lines:\n" + "\n".join(tail)
            )
        ppl, se = parsed
        ppls.append(ppl)
        ses.append(se)
    duration_s = time.time() - t0
    return PPLResult(
        ppl=statistics.fmean(ppls),
        ppl_se=statistics.fmean([s for s in ses if s is not None]) if any(s is not None for s in ses) else None,
        runs=ppls,
        runs_se=ses,
        n_tokens=tokens,
        duration_s=duration_s,
        num_runs=len(ppls),
        reference=str(reference),
        cmd=" ".join(cmd),
    )

"""Composite scoring and report generation."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from quant_bench.coding import CodingResult
from quant_bench.mmlu import MMLUResult
from quant_bench.perf import PerfResult
from quant_bench.ppl import PPLResult

console = Console()

# Human-readable names for the composite's 0-100 components (used in the score breakdown).
_COMPONENT_LABELS = {"ppl": "PPL", "mmlu": "MMLU", "coding": "aider pass@2"}


@dataclass
class ModelScore:
    label: str
    slug: str
    path: str
    flags: list[str]
    server_cmd: str = ""
    server_version: str = ""
    mmlu: Optional[MMLUResult] = None
    perf: Optional[PerfResult] = None
    coding: Optional[CodingResult] = None
    ppl: Optional[PPLResult] = None
    ppl_score: Optional[float] = None
    ppl_score_runs: list[float] = field(default_factory=list)
    ppl_score_se: Optional[float] = None
    mmlu_error: Optional[str] = None
    perf_error: Optional[str] = None
    coding_error: Optional[str] = None
    ppl_error: Optional[str] = None
    score: Optional[float] = None
    rank: Optional[int] = None
    nonsignificant: bool = False
    # The composite's present components as (name, normalized_weight, value_0_100),
    # filled by compute_scores so the report can show exactly how the score was built.
    score_components: list[tuple[str, float, float]] = field(default_factory=list)


def _component_weights(mmlu_weight: float, ppl_weight: float) -> tuple[float, float, float]:
    """Base (pre-renormalization) weights for the three 0-100 components.

    The final benchmark score is a flat weighted sum:
        score = w_ppl * PPL_fidelity + w_mmlu * MMLU% + w_coding * aider-pass@2%
    where w_ppl = ppl_weight and the remaining (1 - ppl_weight) is split between MMLU and
    aider pass@2 by mmlu_weight.
    """
    w_ppl = ppl_weight
    w_mmlu = (1.0 - ppl_weight) * mmlu_weight
    w_coding = (1.0 - ppl_weight) * (1.0 - mmlu_weight)
    return w_ppl, w_mmlu, w_coding


def _component_value(s: ModelScore, name: str) -> Optional[float]:
    """0-100 value of a single component (None if that metric is unavailable)."""
    if name == "ppl":
        return s.ppl_score
    if name == "mmlu":
        return s.mmlu.score * 100.0 if s.mmlu is not None else None
    if name == "coding":
        return s.coding.pass_rate_2 if (s.coding is not None and s.coding.pass_rate_2 is not None) else None
    return None


def _composite_components(s: ModelScore, mmlu_weight: float, ppl_weight: float) -> list[tuple[str, float, float]]:
    """The present components as [(name, normalized_weight, value_0_100)], weights renormalized to sum to 1.

    A missing metric simply drops out and the remaining weights are rescaled, so a partial run
    still yields a score on the same 0-100 scale (no silent dilution of the score).
    """
    w_ppl, w_mmlu, w_coding = _component_weights(mmlu_weight, ppl_weight)
    raw: list[tuple[str, float, float]] = []
    for name, w in (("ppl", w_ppl), ("mmlu", w_mmlu), ("coding", w_coding)):
        if w <= 0.0:
            continue
        v = _component_value(s, name)
        if v is not None:
            raw.append((name, w, v))
    total = sum(w for _, w, _ in raw)
    if total <= 0:
        return []
    return [(n, w / total, v) for n, w, v in raw]


def compute_scores(scores: list[ModelScore], mmlu_weight: float, ppl_weight: float = 0.0) -> None:
    # PPL is lower-is-better and unbounded, so normalize across the scored set:
    # the best (lowest) PPL -> 100, others scale by the ratio. This is what makes a
    # PPL-dominant composite reproducible and comparable.
    with_ppl = [s for s in scores if s.ppl is not None and s.ppl.ppl > 0]
    if with_ppl:
        min_ppl = min(s.ppl.ppl for s in with_ppl)
        for s in with_ppl:
            s.ppl_score = 100.0 * min_ppl / s.ppl.ppl
            s.ppl_score_runs = [100.0 * min_ppl / r for r in s.ppl.runs]
            # Propagate the PPL estimate SE (per-token bootstrap) into score points via the
            # first-order derivative of score = 100*min/ppl: d(score)/d(ppl) = -100*min/ppl^2.
            if s.ppl.ppl_se is not None and s.ppl.ppl > 0:
                s.ppl_score_se = (100.0 * min_ppl / (s.ppl.ppl * s.ppl.ppl)) * s.ppl.ppl_se
    for s in scores:
        comps = _composite_components(s, mmlu_weight, ppl_weight)
        s.score_components = comps
        s.score = sum(w * v for _, w, v in comps) if comps else None
    ranked = [s for s in scores if s.score is not None]
    ranked.sort(key=lambda s: s.score, reverse=True)
    for i, s in enumerate(ranked, start=1):
        s.rank = i


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.1f}%"


def _num(v: Optional[float], fmt: str = "{:.0f}") -> str:
    return "n/a" if v is None else fmt.format(v)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion, in percent."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return (max(0.0, (center - half) * 100.0), min(100.0, (center + half) * 100.0))


def _mmlu_ci_half(s: ModelScore) -> Optional[float]:
    """Half-width (score points) of the 95% CI on the MMLU component."""
    if s.mmlu is None or s.mmlu.score_stderr is None:
        return None
    return 1.96 * s.mmlu.score_stderr * 100.0


def _coding_ci_half(s: ModelScore) -> Optional[float]:
    """Half-width (percentage points) of the Wilson 95% CI on pass@2."""
    if s.coding is None or s.coding.completed_tests <= 0 or s.coding.pass_rate_2 is None:
        return None
    lo, hi = _wilson_ci(s.coding.pass_num_2, s.coding.completed_tests)
    return (hi - lo) / 2.0


def _ppl_ci_half(s: ModelScore) -> Optional[float]:
    """Half-width (score points) of the 95% CI on the PPL component.

    Primarily the PPL estimate's per-token bootstrap SE (``ppl_score_se``), widened to 95%
    (x1.96). RMS-combined with any run-to-run spread so between-run jitter is captured too.
    Using the bootstrap SE (not just the run spread) is what keeps the composite CI honest:
    with a long reference the PPL SE is small, so a real PPL gap is flagged significant.
    """
    parts: list[float] = []
    if s.ppl_score_se is not None:
        parts.append(1.96 * s.ppl_score_se)
    if s.ppl is not None and len(s.ppl_score_runs) >= 2:
        parts.append((max(s.ppl_score_runs) - min(s.ppl_score_runs)) / 2.0)
    if not parts:
        return None
    return math.sqrt(sum(x * x for x in parts))


def _score_ci_half(s: ModelScore, mmlu_weight: float, ppl_weight: float = 0.0) -> Optional[float]:
    """Half-width (score points) of the 95% CI on the composite score (PPL + MMLU + coding).

    Weighted RMS of the present components' CIs, using the same renormalized weights as the
    score itself. Returns None if any present component lacks a CI (we won't claim significance).
    """
    comps = _composite_components(s, mmlu_weight, ppl_weight)
    if not comps:
        return None
    comp_ci = {"ppl": _ppl_ci_half(s), "mmlu": _mmlu_ci_half(s), "coding": _coding_ci_half(s)}
    for name, _, _ in comps:
        if comp_ci[name] is None:
            return None
    return math.sqrt(sum((w * comp_ci[name]) ** 2 for name, w, _ in comps))


def _mark_nonsignificant(scores: list[ModelScore], mmlu_weight: float, ppl_weight: float = 0.0) -> None:
    """Flag models whose adjacent-rank score difference is within the combined 95% CIs."""
    for s in scores:
        s.nonsignificant = False
    ranked = sorted((s for s in scores if s.rank is not None), key=lambda s: s.rank)
    for a, b in zip(ranked, ranked[1:], strict=False):
        ha, hb = _score_ci_half(a, mmlu_weight, ppl_weight), _score_ci_half(b, mmlu_weight, ppl_weight)
        if ha is None or hb is None or a.score is None or b.score is None:
            continue
        if a.score - b.score <= math.sqrt(ha * ha + hb * hb):
            a.nonsignificant = True
            b.nonsignificant = True


def _mmlu_cell(s: ModelScore) -> str:
    if s.mmlu is None:
        return "n/a"
    h = _mmlu_ci_half(s)
    return f"{s.mmlu.score * 100:.1f}% ±{h:.1f}" if h is not None else _pct(s.mmlu.score * 100.0)


def _pass_cell(s: ModelScore, which: int) -> str:
    if s.coding is None or s.coding.completed_tests <= 0:
        return "n/a"
    rate = s.coding.pass_rate_1 if which == 1 else s.coding.pass_rate_2
    k = s.coding.pass_num_1 if which == 1 else s.coding.pass_num_2
    if rate is None:
        return "n/a"
    lo, hi = _wilson_ci(k, s.coding.completed_tests)
    return f"{rate:.1f}% ±{(hi - lo) / 2:.1f}"


def _ppl_cell(s: ModelScore) -> str:
    if s.ppl is None:
        return "n/a"
    if s.ppl.ppl_se is not None:
        return f"{s.ppl.ppl:.2f}\u00b1{1.96 * s.ppl.ppl_se:.2f}"
    return f"{s.ppl.ppl:.2f}"


def _composite_label(weight: float, ppl_weight: float) -> str:
    """Human-readable composite formula. `weight` = MMLU weight within the capability half."""
    if ppl_weight <= 0.0:
        return f"{weight:.2f} x MMLU + {1 - weight:.2f} x aider pass@2"
    w_ppl = ppl_weight
    w_mmlu = (1.0 - ppl_weight) * weight
    w_coding = (1.0 - ppl_weight) * (1.0 - weight)
    return f"{w_ppl:.2f} x PPL + {w_mmlu:.2f} x MMLU + {w_coding:.2f} x aider pass@2"


def _table(scores: list[ModelScore], weight: float, ppl_weight: float = 0.0) -> Table:
    t = Table(title=f"quant-bench results (score = {_composite_label(weight, ppl_weight)})")
    t.add_column("rank", justify="right")
    t.add_column("model")
    t.add_column("PPL", justify="right")
    t.add_column("MMLU", justify="right")
    t.add_column("aider pass@1", justify="right")
    t.add_column("aider pass@2", justify="right")
    t.add_column("TTFT p50 (ms)", justify="right")
    t.add_column("tok/s", justify="right")
    t.add_column("score", justify="right")
    ordered = sorted(scores, key=lambda s: (s.rank is None, s.rank or 0))
    for s in ordered:
        t.add_row(
            str(s.rank) if s.rank else "-",
            s.label,
            _ppl_cell(s),
            _mmlu_cell(s),
            _pass_cell(s, 1),
            _pass_cell(s, 2),
            _num(s.perf.ttft_ms_p50) if s.perf else "n/a",
            _num(s.perf.tok_s_median) if s.perf else "n/a",
            f"{s.score:.2f}*" if s.score is not None and s.nonsignificant else (
                f"{s.score:.2f}" if s.score is not None else "n/a"
            ),
        )
    return t


def _model_entry(s: ModelScore, weight: float, ppl_weight: float = 0.0) -> dict[str, Any]:
    h_m = _mmlu_ci_half(s)
    return {
        "label": s.label,
        "slug": s.slug,
        "path": s.path,
        "per_model_flags": s.flags,
        "server_cmd": s.server_cmd,
        "server_version": s.server_version,
        "rank": s.rank,
        "score": s.score,
        "nonsignificant": s.nonsignificant,
        "score_ci_95_half": _score_ci_half(s, weight, ppl_weight),
        "score_components": [
            {"component": n, "weight": round(w, 4), "value_0_100": round(v, 2)}
            for n, w, v in s.score_components
        ],
        "ppl": (
            {
                "ppl": s.ppl.ppl,
                "ppl_se": s.ppl.ppl_se,
                "runs": list(s.ppl.runs),
                "num_runs": s.ppl.num_runs,
                "n_tokens": s.ppl.n_tokens,
                "reference": s.ppl.reference,
                "cmd": s.ppl.cmd,
                "duration_s": s.ppl.duration_s,
                "score_0_100": s.ppl_score,
                "score_runs": list(s.ppl_score_runs),
                "ci_95_half_pts": _ppl_ci_half(s),
            }
            if s.ppl
            else ({"error": s.ppl_error} if s.ppl_error else None)
        ),
        "mmlu": {
            "task": s.mmlu.task,
            "score": s.mmlu.score,
            "score_metric": s.mmlu.score_metric,
            "score_stderr": s.mmlu.score_stderr,
            "ci_95_half_pts": h_m,
            "n_samples": s.mmlu.n_samples,
            "duration_s": s.mmlu.duration_s,
        }
        if s.mmlu
        else {"error": s.mmlu_error},
        "perf": {
            "n_requests": s.perf.n_requests,
            "ttft_ms_mean": s.perf.ttft_ms_mean,
            "ttft_ms_p50": s.perf.ttft_ms_p50,
            "ttft_ms_p95": s.perf.ttft_ms_p95,
            "tok_s_mean": s.perf.tok_s_mean,
            "tok_s_median": s.perf.tok_s_median,
            "total_tokens": s.perf.total_tokens,
            "duration_s": s.perf.duration_s,
        }
        if s.perf
        else ({"error": s.perf_error} if s.perf_error else None),
        "coding": {
            "model": s.coding.model,
            "edit_format": s.coding.edit_format,
            "languages": s.coding.languages,
            "tries": s.coding.tries,
            "pass_rate_1": s.coding.pass_rate_1,
            "pass_rate_2": s.coding.pass_rate_2,
            "pass_num_1": s.coding.pass_num_1,
            "pass_num_2": s.coding.pass_num_2,
            "pass_rate_1_ci_95": list(_wilson_ci(s.coding.pass_num_1, s.coding.completed_tests)),
            "pass_rate_2_ci_95": list(_wilson_ci(s.coding.pass_num_2, s.coding.completed_tests)),
            "completed_tests": s.coding.completed_tests,
            "total_tests": s.coding.total_tests,
            "duration_s": s.coding.duration_s,
            "prompt_tokens": s.coding.prompt_tokens,
            "completion_tokens": s.coding.completion_tokens,
            "run_dir": str(s.coding.run_dir),
        }
        if s.coding
        else ({"error": s.coding_error} if s.coding_error else None),
    }


def _mmlu_subjects_table(scores: list[ModelScore]) -> list[str]:
    ranked = [s for s in scores if s.mmlu and (s.mmlu.raw or {}).get("results")]
    if len(ranked) < 2:
        return []
    subj: dict[str, dict[str, float]] = {}
    for s in ranked:
        for k, v in s.mmlu.raw["results"].items():
            if k == "mmlu" or not isinstance(v, dict):
                continue
            a = v.get("acc,none")
            if a is None:
                continue
            subj.setdefault(k.replace("mmlu_", ""), {})[s.slug] = float(a)
    rows = [
        (max(per.values()) - min(per.values()), name, per)
        for name, per in subj.items()
        if len(per) >= 2
    ]
    rows.sort(key=lambda r: (-r[0], r[1]))
    slugs = [s.slug for s in ranked]
    lines = [
        "",
        "## MMLU by subject (sorted by max-min spread)",
        "",
        "| spread (pts) | subject | " + " | ".join(slugs) + " |",
        "|---:|---|" + "---:|" * len(slugs),
    ]
    for spread, name, per in rows:
        mark = " †" if spread >= 0.04 else ""
        cells = " | ".join(
            f"{per[slu] * 100:.1f}" if slu in per else "–" for slu in slugs
        )
        lines.append(f"| {spread * 100:.1f}{mark} | {name} | {cells} |")
    lines += ["", "† spread ≥ 4 pts between models"]
    return lines


def _coding_exercises_table(scores: list[ModelScore]) -> list[str]:
    ranked = [s for s in scores if s.coding and s.coding.raw]
    if not ranked:
        return []
    ex: dict[str, dict[str, tuple[bool, bool]]] = {}
    for s in ranked:
        for r in s.coding.raw:
            name = r.get("exercise") or "?"
            oc = r.get("tests_outcomes") or []
            p1 = bool(oc and oc[0])
            p2 = bool(oc and (len(oc) > 1 and (oc[0] or oc[1])))
            ex.setdefault(name, {})[s.slug] = (p1, p2)
    slugs = [s.slug for s in ranked]
    lines = [
        "",
        "## Coding by exercise",
        "",
        "| exercise | " + " | ".join(slugs) + " |",
        "|---|" + "---:|" * len(slugs),
    ]
    for name in sorted(ex):
        cells = []
        for slu in slugs:
            v = ex[name].get(slu)
            if v is None:
                cells.append("–")
            elif v[0]:
                cells.append("1")
            elif v[1]:
                cells.append("2")
            else:
                cells.append("·")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "1 = passed on try 1, 2 = passed on try 2 only, · = failed, – = not run"]
    return lines


def _markdown(scores: list[ModelScore], weight: float, meta: dict, ppl_weight: float = 0.0) -> str:
    lines: list[str] = []
    lines.append("# quant-bench results")
    lines.append("")
    lines.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for k, v in meta.items():
        lines.append(f"- {k}: {v}")
    lines.append(f"- composite: `score = {_composite_label(weight, ppl_weight)}`")
    lines.append("")
    lines.append(
        "| rank | model | PPL | MMLU | aider pass@1 | aider pass@2 | TTFT p50 (ms) | tok/s | score |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    ordered = sorted(scores, key=lambda s: (s.rank is None, s.rank or 0))
    for s in ordered:
        lines.append(
            "| {rank} | {label} | {ppl} | {mmlu} | {p1} | {p2} | {ttft} | {tps} | {score} |".format(
                rank=s.rank or "-",
                label=s.label,
                ppl=_ppl_cell(s),
                mmlu=_mmlu_cell(s),
                p1=_pass_cell(s, 1),
                p2=_pass_cell(s, 2),
                ttft=_num(s.perf.ttft_ms_p50) if s.perf else "n/a",
                tps=_num(s.perf.tok_s_median) if s.perf else "n/a",
                score=(f"{s.score:.2f}*" if s.nonsignificant else f"{s.score:.2f}")
                if s.score is not None
                else "n/a",
            )
        )
    if any(s.nonsignificant for s in scores):
        lines.append("")
        lines.append(
            "* adjacent-rank difference not significant at the 95% CI level "
            "(PPL bootstrap SE, MMLU ±1.96×SE, coding Wilson CI, propagated into the composite)"
        )
    lines.append("")
    lines.append("## per-model details")
    for s in ordered:
        lines.append("")
        lines.append(f"### {('#' + str(s.rank) + ' ') if s.rank else ''}{s.label}")
        lines.append("")
        if s.score is not None and s.score_components:
            parts = " + ".join(
                f"{w:.2f}\u00d7{_COMPONENT_LABELS.get(n, n)}({v:.1f})" for n, w, v in s.score_components
            )
            ns = " *" if s.nonsignificant else ""
            lines.append(f"- **score {s.score:.2f}{ns}** = {parts}")
            lines.append("")
        lines.append(f"- path: `{s.path}`")
        if s.flags:
            lines.append(f"- per-model flags: `{' '.join(s.flags)}`")
        lines.append(f"- server command: `{s.server_cmd}`")
        lines.append(f"- llama-server: {s.server_version}")
        if s.mmlu:
            h = _mmlu_ci_half(s)
            ci = f" ± {h:.2f} pts (95% CI)" if h is not None else ""
            lines.append(
                f"- MMLU ({s.mmlu.task}) {s.mmlu.score_metric}: "
                f"**{s.mmlu.score * 100:.2f}%**{ci} (n={s.mmlu.n_samples}, {s.mmlu.duration_s:.0f}s)"
            )
        if s.mmlu_error:
            lines.append(f"- MMLU error: {s.mmlu_error}")
        if s.perf:
            lines.append(
                f"- perf: TTFT p50 {s.perf.ttft_ms_p50:.0f} ms (p95 {s.perf.ttft_ms_p95:.0f} ms), "
                f"{s.perf.tok_s_median:.1f} tok/s over {s.perf.n_requests} requests"
            )
        if s.perf_error:
            lines.append(f"- perf error: {s.perf_error}")
        if s.coding:
            if s.coding.pass_rate_1 is not None and s.coding.pass_rate_2 is not None:
                lo1, hi1 = _wilson_ci(s.coding.pass_num_1, s.coding.completed_tests)
                lo2, hi2 = _wilson_ci(s.coding.pass_num_2, s.coding.completed_tests)
                lines.append(
                    f"- polyglot ({s.coding.languages}, {s.coding.edit_format}): "
                    f"pass@1 {s.coding.pass_rate_1:.1f}% [{lo1:.1f}-{hi1:.1f}], "
                    f"pass@2 {s.coding.pass_rate_2:.1f}% [{lo2:.1f}-{hi2:.1f}] (Wilson 95% CI) "
                    f"({s.coding.completed_tests}/{s.coding.total_tests} tests, {s.coding.duration_s:.0f}s), "
                    f"run dir `{s.coding.run_dir}`"
                )
            else:
                lines.append(
                    f"- polyglot ({s.coding.languages}, {s.coding.edit_format}): "
                    f"no completed tests ({s.coding.completed_tests}/{s.coding.total_tests}), "
                    f"run dir `{s.coding.run_dir}`"
                )
        if s.coding_error:
            lines.append(f"- coding error: {s.coding_error}")
        if s.ppl:
            runs_str = ", ".join(f"{r:.2f}" for r in s.ppl.runs)
            ci = 1.96 * s.ppl.ppl_se if s.ppl.ppl_se is not None else None
            ci_str = f" \u00b1{ci:.2f} (95% CI)" if ci is not None else ""
            tok = f", ~{s.ppl.n_tokens / 1000:.0f}K tokens" if s.ppl.n_tokens else ""
            norm = f", fidelity {s.ppl_score:.1f}/100" if s.ppl_score is not None else ""
            lines.append(
                f"- PPL (lower is better): **{s.ppl.ppl:.2f}**{ci_str}{tok} "
                f"({s.ppl.num_runs} runs: {runs_str}){norm}, {s.ppl.duration_s:.0f}s"
            )
        if s.ppl_error:
            lines.append(f"- PPL error: {s.ppl_error}")
    for extra in (_mmlu_subjects_table(ordered), _coding_exercises_table(ordered)):
        if extra:
            lines.append("")
            lines.extend(extra)
    lines.append("")
    return "\n".join(lines)


def write_report(
    scores: list[ModelScore],
    *,
    weight: float,
    results_dir: Path,
    meta: Optional[dict] = None,
    ppl_weight: float = 0.0,
) -> tuple[Path, Path]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    meta = meta or {}

    _mark_nonsignificant(scores, weight, ppl_weight)
    console.print(_table(scores, weight, ppl_weight))

    report_md = results_dir / "report.md"
    report_md.write_text(_markdown(scores, weight, meta, ppl_weight))

    report_json = results_dir / "report.json"
    report_json.write_text(
        json.dumps(
            {
                "weight_mmlu": weight,
                "weight_ppl": ppl_weight,
                "composite": _composite_label(weight, ppl_weight),
                "meta": meta,
                "models": [_model_entry(s, weight, ppl_weight) for s in scores],
            },
            indent=2,
            default=str,
        )
    )
    return report_md, report_json

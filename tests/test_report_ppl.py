"""Unit tests for the PPL-folded composite score, its normalization, and significance logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_bench.coding import CodingResult
from quant_bench.mmlu import MMLUResult
from quant_bench.ppl import PPLResult
from quant_bench.report import ModelScore, _mark_nonsignificant, compute_scores, write_report


def _ppl(ppl: float, runs: list[float]) -> PPLResult:
    return PPLResult(
        ppl=ppl,
        ppl_se=None,
        runs=list(runs),
        runs_se=[None] * len(runs),
        num_runs=len(runs),
        reference="ref",
        cmd="cmd",
    )


def _mmlu(acc: float, se: float) -> MMLUResult:
    return MMLUResult(
        task="mmlu",
        score=acc,
        score_metric="acc",
        score_stderr=se,
        n_samples=6100,
        duration_s=100.0,
        raw={"results": {"mmlu": {"acc,none": acc}}},
        result_path=None,
    )


def _coding(pass_n: int, n: int) -> CodingResult:
    raw = [
        {
            "exercise": f"ex{i}",
            "language": "python",
            "tests_outcomes": ([True] if i < pass_n else [False, False]),
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        for i in range(n)
    ]
    p1 = sum(1 for r in raw if r["tests_outcomes"][0])
    p2 = sum(1 for r in raw if (r["tests_outcomes"][0] or (len(r["tests_outcomes"]) > 1 and r["tests_outcomes"][1])))
    return CodingResult(
        model="m",
        edit_format="whole",
        languages="python",
        tries=2,
        pass_rate_1=100.0 * p1 / n,
        pass_rate_2=100.0 * p2 / n,
        pass_num_1=p1,
        pass_num_2=p2,
        completed_tests=n,
        total_tests=n,
        duration_s=1.0,
        prompt_tokens=0,
        completion_tokens=0,
        run_dir=Path("/tmp/run"),
        raw=raw,
    )


def _model(slug: str, ppl: float, acc: float, se: float, pass_n: int, n: int = 34) -> ModelScore:
    s = ModelScore(label=slug, slug=slug, path=f"/p/{slug}", flags=[])
    s.ppl = _ppl(ppl, [ppl, ppl])  # two identical runs -> deterministic PPL component
    s.mmlu = _mmlu(acc, se)
    s.coding = _coding(pass_n, n)
    return s


def test_ppl_normalization_best_is_100():
    a = _model("a", 50.0, 0.25, 0.01, 0)
    b = _model("b", 60.0, 0.25, 0.01, 0)
    compute_scores([a, b], 0.5, ppl_weight=0.5)
    assert a.ppl_score == pytest.approx(100.0)
    assert b.ppl_score == pytest.approx(100.0 * 50.0 / 60.0)
    assert a.rank == 1 and b.rank == 2


def test_composite_folds_ppl():
    a = _model("a", 40.0, 0.30, 0.01, 0)
    b = _model("b", 50.0, 0.30, 0.01, 0)
    compute_scores([a, b], 0.5, ppl_weight=0.5)
    # cap = 0.5 * (0.30*100) + 0.5 * 0 = 15 for both
    assert a.score == pytest.approx(0.5 * 100.0 + 0.5 * 15.0)
    assert b.score == pytest.approx(0.5 * (100.0 * 40.0 / 50.0) + 0.5 * 15.0)
    assert a.rank == 1


def test_ppl_dominates_for_weak_models():
    # MMLU and coding are at the noise floor for both (weak small model); only PPL differs.
    a = _model("a", 53.0, 0.258, 0.0037, 1)  # e.g. Q3
    b = _model("b", 68.0, 0.261, 0.0037, 0)  # e.g. Q4
    compute_scores([a, b], 0.5, ppl_weight=0.5)
    assert a.rank == 1 and b.rank == 2  # lower PPL wins even though MMLU/coding are a wash


def test_ppl_weight_zero_keeps_capability_only():
    a = _model("a", 53.0, 0.258, 0.0037, 1)
    b = _model("b", 68.0, 0.261, 0.0037, 0)
    compute_scores([a, b], 0.5, ppl_weight=0.0)
    assert a.score == pytest.approx(0.5 * 25.8 + 0.5 * (100.0 * 1 / 34))
    assert b.score == pytest.approx(0.5 * 26.1)
    assert a.rank == 1


def test_ppl_dominant_ranking_is_significant():
    a = _model("a", 53.0, 0.258, 0.0037, 1)
    b = _model("b", 68.0, 0.261, 0.0037, 0)
    compute_scores([a, b], 0.5, ppl_weight=0.5)
    _mark_nonsignificant([a, b], 0.5, ppl_weight=0.5)
    # PPL component CI is ~0 (identical runs) and the gap is large -> significant (no '*').
    assert not a.nonsignificant and not b.nonsignificant


def test_write_report_includes_ppl(tmp_path: Path):
    a = _model("a", 53.0, 0.258, 0.0037, 1)
    b = _model("b", 68.0, 0.261, 0.0037, 0)
    compute_scores([a, b], 0.5, ppl_weight=0.5)
    md, js = write_report([a, b], weight=0.5, results_dir=tmp_path, meta={}, ppl_weight=0.5)
    text = md.read_text()
    assert "PPL" in text
    assert "x PPL" in text  # composite label mentions the PPL term
    data = json.loads(js.read_text())
    assert data["weight_ppl"] == 0.5
    for m in data["models"]:
        assert m["ppl"] is not None
        assert m["ppl"]["score_0_100"] is not None
        assert len(m["ppl"]["runs"]) == 2


def _model_with_se(slug: str, ppl: float, ppl_se: float, acc: float, se: float, pass_n: int, n: int = 34) -> ModelScore:
    """Like _model but with a PPL bootstrap SE and a token count (mimics a real run)."""
    s = ModelScore(label=slug, slug=slug, path=f"/p/{slug}", flags=[])
    s.ppl = PPLResult(
        ppl=ppl,
        ppl_se=ppl_se,
        runs=[ppl, ppl],
        runs_se=[ppl_se, ppl_se],
        n_tokens=300_000,
        num_runs=2,
        reference="ref",
        cmd="cmd",
    )
    s.mmlu = _mmlu(acc, se)
    s.coding = _coding(pass_n, n)
    return s


def test_renormalize_ppl_only():
    a = _model("a", 50.0, 0.25, 0.01, 0)
    b = _model("b", 60.0, 0.25, 0.01, 0)
    b.mmlu = None
    b.coding = None  # PPL-only model
    compute_scores([a, b], 0.5, ppl_weight=0.5)
    assert b.score == pytest.approx(b.ppl_score)  # weights renormalize to PPL = 1.0
    assert [n for n, _, _ in b.score_components] == ["ppl"]
    assert b.score_components[0][1] == pytest.approx(1.0)


def test_renormalize_mmlu_only():
    s = _model("a", 50.0, 0.30, 0.01, 0)
    s.ppl = None
    s.coding = None  # only MMLU remains
    compute_scores([s], 0.5, ppl_weight=0.5)
    assert s.score == pytest.approx(30.0)
    assert [n for n, _, _ in s.score_components] == ["mmlu"]


def test_ppl_bootstrap_se_keeps_gap_significant():
    # Tight PPL (small SE) -> the ~11-point score gap is well outside the CI -> significant.
    a = _model_with_se("a", 53.0, 0.7, 0.258, 0.0037, 1)
    b = _model_with_se("b", 68.0, 0.7, 0.261, 0.0037, 0)
    compute_scores([a, b], 0.5, ppl_weight=0.5)
    _mark_nonsignificant([a, b], 0.5, ppl_weight=0.5)
    assert not a.nonsignificant and not b.nonsignificant


def test_ppl_large_se_makes_gap_nonsignificant():
    # Loose PPL (huge SE) -> the same gap is within the CI -> not significant.
    a = _model_with_se("a", 53.0, 30.0, 0.258, 0.0037, 1)
    b = _model_with_se("b", 68.0, 30.0, 0.261, 0.0037, 0)
    compute_scores([a, b], 0.5, ppl_weight=0.5)
    _mark_nonsignificant([a, b], 0.5, ppl_weight=0.5)
    assert a.nonsignificant and b.nonsignificant


def test_ppl_cell_shows_95_ci():
    from quant_bench.report import _ppl_cell

    s = _model_with_se("a", 53.0, 0.7, 0.258, 0.0037, 1)
    cell = _ppl_cell(s)
    assert cell == "53.00\u00b11.37"


def test_write_report_includes_breakdown(tmp_path: Path):
    a = _model_with_se("a", 53.0, 0.7, 0.258, 0.0037, 1)
    b = _model_with_se("b", 68.0, 0.7, 0.261, 0.0037, 0)
    compute_scores([a, b], 0.5, ppl_weight=0.5)
    md, js = write_report([a, b], weight=0.5, results_dir=tmp_path, meta={}, ppl_weight=0.5)
    data = json.loads(js.read_text())
    for m in data["models"]:
        comps = m["score_components"]
        assert {c["component"] for c in comps} == {"ppl", "mmlu", "coding"}
        assert sum(c["weight"] for c in comps) == pytest.approx(1.0)
    text = md.read_text()
    assert "= 0.50\u00d7PPL" in text  # per-model score breakdown
    assert "K tokens" in text  # PPL token count surfaced

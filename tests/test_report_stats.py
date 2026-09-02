"""Unit tests for report statistics: Wilson CIs, n.s. marking, appendix rendering."""

from __future__ import annotations

import json
from pathlib import Path

from quant_bench.coding import CodingResult
from quant_bench.mmlu import MMLUResult
from quant_bench.report import ModelScore, _mark_nonsignificant, _wilson_ci, write_report


def _mmlu(acc: float, se: float) -> MMLUResult:
    return MMLUResult(
        task="mmlu",
        score=acc,
        score_metric="acc",
        score_stderr=se,
        n_samples=6100,
        duration_s=100.0,
        raw={
            "results": {
                "mmlu": {"acc,none": acc},
                "mmlu_math": {"acc,none": acc, "sample_len": 100},
                "mmlu_physics": {"acc,none": acc - 0.05, "sample_len": 100},
            }
        },
        result_path=None,
    )


def _coding(outcomes: dict[str, list[bool]]) -> CodingResult:
    n = len(outcomes)
    p1 = sum(1 for oc in outcomes.values() if oc and oc[0])
    p2 = sum(1 for oc in outcomes.values() if oc and (len(oc) > 1 and (oc[0] or oc[1])))
    raw = [
        {"exercise": name, "language": "python", "tests_outcomes": oc, "prompt_tokens": 0, "completion_tokens": 0}
        for name, oc in outcomes.items()
    ]
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
        duration_s=100.0,
        prompt_tokens=0,
        completion_tokens=0,
        run_dir=Path("/tmp/run"),
        raw=raw,
    )


def _score(slug: str, acc: float, se: float, outcomes: dict[str, list[bool]]) -> ModelScore:
    s = ModelScore(label=slug, slug=slug, path=f"/p/{slug}", flags=[])
    s.mmlu = _mmlu(acc, se)
    s.coding = _coding(outcomes)
    s.score = 0.5 * acc * 100.0 + 0.5 * s.coding.pass_rate_2
    return s


def _outcomes(n: int, n_pass: int, on_try1: bool = True) -> dict[str, list[bool]]:
    out = {}
    for i in range(n):
        if i < n_pass:
            out[f"ex{i}"] = [True] if on_try1 else [False, True]
        else:
            out[f"ex{i}"] = [False, False]
    return out


def test_wilson_ci_basic() -> None:
    lo, hi = _wilson_ci(8, 34)
    assert 10.0 < lo < 15.0
    assert 38.0 < hi < 42.0
    assert lo < 23.5 < hi


def test_wilson_ci_edges() -> None:
    assert _wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = _wilson_ci(5, 5)
    assert hi == 100.0
    assert 0.0 <= lo < 100.0


def test_mark_nonsignificant_close_scores() -> None:
    a = _score("a", 0.30, 0.01, _outcomes(34, 2))
    b = _score("b", 0.29, 0.01, _outcomes(34, 1))
    a.rank, b.rank = 1, 2
    _mark_nonsignificant([a, b], 0.5)
    assert a.nonsignificant and b.nonsignificant


def test_mark_nonsignificant_clear_gap() -> None:
    a = _score("a", 0.50, 0.005, _outcomes(34, 20))
    b = _score("b", 0.30, 0.005, _outcomes(34, 5))
    a.rank, b.rank = 1, 2
    _mark_nonsignificant([a, b], 0.5)
    assert not a.nonsignificant and not b.nonsignificant


def test_mark_nonsignificant_skips_unscored() -> None:
    a = _score("a", 0.50, 0.005, _outcomes(34, 20))
    c = ModelScore(label="c", slug="c", path="/p/c", flags=[])
    a.rank = 1
    _mark_nonsignificant([a, c], 0.5)
    assert not a.nonsignificant and not c.nonsignificant


def test_write_report_includes_stats_and_appendices(tmp_path: Path) -> None:
    a = _score("model-a", 0.30, 0.01, _outcomes(34, 2))
    b = _score("model-b", 0.29, 0.01, _outcomes(34, 1))
    a.rank, b.rank = 1, 2
    md, js = write_report([a, b], weight=0.5, results_dir=tmp_path, meta={})
    text = md.read_text()
    assert "not significant" in text
    assert "±" in text
    assert "Wilson 95% CI" in text
    assert "## MMLU by subject" in text
    assert "## Coding by exercise" in text
    data = json.loads(js.read_text())
    assert data["models"][0]["nonsignificant"] is True
    assert data["models"][0]["score_ci_95_half"] is not None
    assert len(data["models"][0]["coding"]["pass_rate_2_ci_95"]) == 2

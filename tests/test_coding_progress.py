"""Unit tests for the coding progress helpers (no server needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_bench.coding import _count_done, _total_exercises

POLYGLOT = Path(__file__).resolve().parent.parent / "tmp.benchmarks" / "polyglot-benchmark"


@pytest.fixture()
def fake_run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "2026-01-01-00-00-00--model"
    for lang in ("python", "typescript"):
        base = run / lang / "exercises" / "practice"
        for name in ("ex1", "ex2", "ex3"):
            d = base / name
            d.mkdir(parents=True)
            if name != "ex3":
                (d / ".aider.results.json").write_text("{}")
    return run


def test_count_done_specific_language(fake_run_dir: Path) -> None:
    assert _count_done(fake_run_dir, "python") == 2
    assert _count_done(fake_run_dir, "python,typescript") == 4
    assert _count_done(fake_run_dir, "") == 4


def test_total_exercises_needs_polyglot() -> None:
    if not POLYGLOT.is_dir():
        pytest.skip("polyglot-benchmark not set up; run `quant-bench setup`")
    total = _total_exercises(POLYGLOT, "python", -1)
    assert total > 0
    assert _total_exercises(POLYGLOT, "python", 5) == 5
    assert _total_exercises(POLYGLOT, "no-such-language", -1) == 0

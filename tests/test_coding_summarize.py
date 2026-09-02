"""Unit tests for the pass@1 / pass@2 math in coding._summarize.

aider's benchmark.py feeds a failing try's test output back to the model on the
next try, so `tests_outcomes[0]` is the cold single-shot attempt and a test that
passes on the first try records a single-element outcome `[True]`. pass@2 must be
a superset of pass@1 ("passed on at least one try"), so a first-try pass counts
in both.
"""

import json
from pathlib import Path

from quant_bench.coding import _summarize


def _write_run_dir(run_dir: Path, outcomes: dict[str, list]) -> None:
    for exercise, oc in outcomes.items():
        ex_dir = run_dir / "python" / "exercises" / "practice" / exercise
        ex_dir.mkdir(parents=True)
        (ex_dir / ".aider.results.json").write_text(
            json.dumps(
                {
                    "model": "m",
                    "edit_format": "whole",
                    "tests_outcomes": oc,
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                }
            )
        )


def test_pass2_is_superset_of_pass1(tmp_path):
    # [True]         -> passed try 1  (counts in pass@1 AND pass@2)
    # [False, True]  -> failed then fixed (counts in pass@2 only)
    # [False, False] -> failed both   (counts in neither)
    # []             -> no outcomes   (counts in neither, still completed)
    run_dir = tmp_path / "run"
    _write_run_dir(
        run_dir,
        {
            "ex_first": [True],
            "ex_fix": [False, True],
            "ex_fail": [False, False],
            "ex_none": [],
        },
    )
    res = _summarize(run_dir, "m", "whole", "python", tries=2, duration_s=0.0)
    assert res.completed_tests == 4
    assert res.pass_num_1 == 1
    assert res.pass_num_2 == 2  # includes the first-try pass, not just fail-then-fix
    assert res.pass_rate_1 == 100.0 * 1 / 4
    assert res.pass_rate_2 == 100.0 * 2 / 4
    # pass@2 must always be >= pass@1
    assert res.pass_rate_2 >= res.pass_rate_1


def test_no_passes(tmp_path):
    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, {"ex_fail": [False, False], "ex_fail2": [False]})
    res = _summarize(run_dir, "m", "whole", "python", tries=2, duration_s=0.0)
    assert res.completed_tests == 2
    assert res.pass_num_1 == 0
    assert res.pass_num_2 == 0
    assert res.pass_rate_1 == 0.0
    assert res.pass_rate_2 == 0.0

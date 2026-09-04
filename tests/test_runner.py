"""Unit tests for the per-model benchmark execution layer (quant_bench.runner).

The serving/coding/PPL stages are exercised with the external calls (llama-server
process and the mmlu/perf/coding/ppl runners) mocked out, so no GPU or network is
needed. Verifies stage ordering, skip flags, per-stage error capture, the model-level
server-error fallback, and the report metadata builder.
"""

from __future__ import annotations

from pathlib import Path

import quant_bench.runner as runner
from quant_bench.coding import CodingResult
from quant_bench.config import LlamaServerFlags, ModelSpec, ServerProfile
from quant_bench.llamaserver import ServerError
from quant_bench.mmlu import MMLUResult
from quant_bench.perf import PerfResult
from quant_bench.ppl import PPLResult
from quant_bench.report import ModelScore


def _cfg(**overrides) -> runner.RunConfig:
    base = dict(
        server=ServerProfile(device="vram"),
        binary="/bin/llama-server",
        version="b1234",
        config=Path("models.yaml"),
        results_dir=Path("results"),
        benchmark_root=Path("tmp.benchmarks"),
        startup_timeout=900,
        skip_mmlu=False,
        skip_perf=False,
        skip_coding=False,
        mmlu_task="mmlu",
        mmlu_limit=None,
        mmlu_concurrency=8,
        edit_format="whole",
        languages="python",
        tries=2,
        coding_limit=None,
        coding_kv_fix="f16",
        perf_requests=20,
        perf_max_tokens=128,
        ppl_reference=Path("scripts/ppl_ref.txt"),
        ppl_ctx=1024,
        ppl_runs=2,
        ppl_weight=0.5,
        ppl_available=True,
        perplexity_bin="/bin/llama-perplexity",
    )
    base.update(overrides)
    return runner.RunConfig(**base)


def _model() -> ModelSpec:
    return ModelSpec(path=Path("/models/q3.gguf"), tokenizer=Path("/tok"), flags=[])


def _flags() -> LlamaServerFlags:
    return LlamaServerFlags(model="/m.gguf", port=8080, host="0.0.0.0", alias="m", ctx=8192, ngl="all")


def _blank_score() -> ModelScore:
    return ModelScore(label="m", slug="m", path="/p/m", flags=[])


def _mmlu() -> MMLUResult:
    return MMLUResult(
        task="mmlu",
        score=0.30,
        score_metric="acc",
        score_stderr=0.01,
        n_samples=6100,
        duration_s=10.0,
        raw={},
        result_path=None,
    )


def _perf() -> PerfResult:
    return PerfResult(
        n_requests=20,
        ttft_ms_mean=50.0,
        ttft_ms_p50=48.0,
        ttft_ms_p95=90.0,
        tok_s_mean=40.0,
        tok_s_median=42.0,
        total_tokens=2000,
        duration_s=5.0,
    )


def _coding() -> CodingResult:
    return CodingResult(
        model="m",
        edit_format="whole",
        languages="python",
        tries=2,
        pass_rate_1=50.0,
        pass_rate_2=60.0,
        pass_num_1=1,
        pass_num_2=2,
        completed_tests=2,
        total_tests=2,
        duration_s=1.0,
        prompt_tokens=0,
        completion_tokens=0,
        run_dir=Path("/tmp/run"),
        raw=[],
    )


def _ppl() -> PPLResult:
    return PPLResult(
        ppl=50.0,
        ppl_se=None,
        runs=[50.0, 50.0],
        runs_se=[None, None],
        n_tokens=273536,
        duration_s=120.0,
        num_runs=2,
        reference="ref",
        cmd="cmd",
    )


def _fake_server(monkeypatch) -> dict:
    state = {"started": 0, "stopped": 0}

    class _S:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            state["started"] += 1

        def model_name(self):
            return "mockmodel"

        def stop(self):
            state["stopped"] += 1

        @property
        def url(self):
            return "http://127.0.0.1:8080"

    monkeypatch.setattr(runner, "LlamaServer", _S)
    return state


def test_build_meta_all_stages():
    cfg = _cfg()
    meta = runner.build_meta(cfg, 123.456)
    assert meta["llama_server"] == cfg.binary
    assert meta["llama_server_version"] == cfg.version
    assert meta["config"] == str(cfg.config)
    assert meta["total_duration_s"] == 123.5
    assert meta["mmlu_task"] == "mmlu"
    assert meta["languages"] == "python"
    assert meta["ppl_weight"] == 0.5
    assert meta["ppl_reference"] == str(cfg.ppl_reference)


def test_build_meta_skips_record_none():
    cfg = _cfg(skip_mmlu=True, skip_coding=True, ppl_available=False)
    meta = runner.build_meta(cfg, 10.0)
    assert meta["mmlu_task"] is None
    assert meta["languages"] is None
    assert meta["coding_limit"] is None
    assert meta["coding_kv_fix"] is None
    assert meta["ppl_weight"] is None
    assert meta["ppl_reference"] is None


def test_run_model_calls_stages_in_order(monkeypatch):
    cfg = _cfg()
    order = []
    monkeypatch.setattr(runner, "_run_mmlu_and_perf", lambda *a: order.append("mmlu_perf"))
    monkeypatch.setattr(runner, "_run_coding", lambda *a: order.append("coding"))
    monkeypatch.setattr(runner, "_run_ppl", lambda *a: order.append("ppl"))
    score = runner.run_model(cfg, _model())
    assert order == ["mmlu_perf", "coding", "ppl"]
    assert score.label == "q3.gguf"
    assert score.slug == "q3"
    assert score.server_version == cfg.version
    assert score.server_cmd.startswith(cfg.binary)


def test_run_model_catches_server_error(monkeypatch):
    cfg = _cfg()

    def boom(*args, **kwargs):
        raise ServerError("boom")

    monkeypatch.setattr(runner, "_run_mmlu_and_perf", boom)
    coding_called = []
    monkeypatch.setattr(runner, "_run_coding", lambda *a: coding_called.append(1))
    monkeypatch.setattr(runner, "_run_ppl", lambda *a: None)
    score = runner.run_model(cfg, _model())
    assert coding_called == []  # subsequent stages are skipped after a server error
    assert score.coding_error == "server error: boom"


def test_run_mmlu_and_perf_runs_both(monkeypatch):
    state = _fake_server(monkeypatch)
    monkeypatch.setattr(runner, "run_mmlu", lambda **kw: _mmlu())
    monkeypatch.setattr(runner, "probe", lambda **kw: _perf())
    score = _blank_score()
    runner._run_mmlu_and_perf(_cfg(), _flags(), _model(), score)
    assert score.mmlu is not None
    assert score.perf is not None
    assert state["started"] == 1 and state["stopped"] == 1


def test_run_mmlu_and_perf_respects_skip(monkeypatch):
    state = _fake_server(monkeypatch)
    called = []
    monkeypatch.setattr(runner, "run_mmlu", lambda **kw: called.append("mmlu"))
    monkeypatch.setattr(runner, "probe", lambda **kw: called.append("perf"))
    score = _blank_score()
    runner._run_mmlu_and_perf(_cfg(skip_mmlu=True, skip_perf=True), _flags(), _model(), score)
    assert called == []
    assert score.mmlu is None and score.perf is None
    assert state["started"] == 1 and state["stopped"] == 1  # server still cycles for the enabled set


def test_run_mmlu_and_perf_isolates_stage_failure(monkeypatch):
    _fake_server(monkeypatch)

    def bad_mmlu(**kw):
        raise RuntimeError("mmlu broke")

    monkeypatch.setattr(runner, "run_mmlu", bad_mmlu)
    monkeypatch.setattr(runner, "probe", lambda **kw: _perf())
    score = _blank_score()
    runner._run_mmlu_and_perf(_cfg(), _flags(), _model(), score)
    assert "mmlu broke" in score.mmlu_error
    assert score.perf is not None  # the perf stage still runs after MMLU fails


def test_run_coding_skipped(monkeypatch):
    state = _fake_server(monkeypatch)
    monkeypatch.setattr(runner, "run_coding", lambda **kw: _coding())
    score = _blank_score()
    runner._run_coding(_cfg(skip_coding=True), _flags(), _model(), score)
    assert state["started"] == 0  # no coding server is spawned
    assert score.coding is None


def test_run_coding_runs(monkeypatch):
    state = _fake_server(monkeypatch)
    monkeypatch.setattr(runner, "run_coding", lambda **kw: _coding())
    score = _blank_score()
    runner._run_coding(_cfg(coding_kv_fix="no-cache"), _flags(), _model(), score)
    assert score.coding is not None
    assert state["started"] == 1 and state["stopped"] == 1


def test_run_coding_captures_error(monkeypatch):
    _fake_server(monkeypatch)

    def bad(**kw):
        raise RuntimeError("coding broke")

    monkeypatch.setattr(runner, "run_coding", bad)
    score = _blank_score()
    runner._run_coding(_cfg(), _flags(), _model(), score)
    assert "coding broke" in score.coding_error


def test_run_ppl_unavailable(monkeypatch):
    called = []
    monkeypatch.setattr(runner, "run_ppl", lambda **kw: called.append("ppl"))
    score = _blank_score()
    runner._run_ppl(_cfg(ppl_available=False), _model(), score)
    assert called == []
    assert score.ppl is None


def test_run_ppl_runs(monkeypatch):
    monkeypatch.setattr(runner, "run_ppl", lambda **kw: _ppl())
    score = _blank_score()
    runner._run_ppl(_cfg(), _model(), score)
    assert score.ppl is not None
    assert score.ppl.ppl == 50.0


def test_run_ppl_captures_error(monkeypatch):
    def bad(**kw):
        raise RuntimeError("ppl broke")

    monkeypatch.setattr(runner, "run_ppl", bad)
    score = _blank_score()
    runner._run_ppl(_cfg(), _model(), score)
    assert "ppl broke" in score.ppl_error


def test_coding_kv_fix_args():
    assert runner._coding_kv_fix_args("f16") == ["--cache-type-k", "f16", "--cache-type-v", "f16"]
    assert runner._coding_kv_fix_args("no-cache") == ["--no-cache-prompt"]
    assert runner._coding_kv_fix_args("off") == []

"""Unit tests for the PPL fidelity metric: output parsing, binary discovery, arg building."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_bench.config import ServerProfile
from quant_bench.ppl import PPLError, build_ppl_args, find_llama_perplexity, parse_ppl, run_ppl


def test_parse_ppl_final_estimate_with_se():
    text = "  I Final estimate: PPL = 67.9739 +/- 8.71330\n"
    ppl, se = parse_ppl(text)
    assert ppl == pytest.approx(67.9739)
    assert se == pytest.approx(8.71330)


def test_parse_ppl_final_estimate_without_se():
    ppl, se = parse_ppl("Final estimate: PPL = 12.5\n")
    assert ppl == pytest.approx(12.5)
    assert se is None


def test_parse_ppl_overall_ppl_fallback():
    ppl, se = parse_ppl("llama_perplexity_impl: overall_ppl = 53.6409 (tokens = 1234)\n")
    assert ppl == pytest.approx(53.6409)
    assert se is None


def test_parse_ppl_handles_carriage_returns():
    text = "progress\rprogress\rFinal estimate: PPL = 9.99 +/- 1.0\n"
    ppl, se = parse_ppl(text)
    assert ppl == pytest.approx(9.99)
    assert se == pytest.approx(1.0)


def test_parse_ppl_no_match():
    assert parse_ppl("some unrelated log\nno numbers here\n") is None


def test_find_llama_perplexity_sibling(tmp_path: Path):
    server = tmp_path / "bin" / "llama-server"
    server.parent.mkdir(parents=True)
    server.write_text("")
    perplexity = tmp_path / "bin" / "llama-perplexity"
    perplexity.write_text("")
    assert find_llama_perplexity(str(server)) == perplexity


def test_find_llama_perplexity_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    server = tmp_path / "bin" / "llama-server"
    server.parent.mkdir(parents=True)
    server.write_text("")
    monkeypatch.setattr("shutil.which", lambda _name: None)  # hide any system binary
    with pytest.raises(PPLError):
        find_llama_perplexity(str(server))


def test_build_ppl_args_vram():
    prof = ServerProfile(device="vram", threads=None)
    args = build_ppl_args(Path("/m/q.gguf"), Path("/ref.txt"), prof, ctx=1024)
    assert args[:2] == ["-m", "/m/q.gguf"]
    assert args[args.index("-f") + 1] == "/ref.txt"
    assert args[args.index("-c") + 1] == "1024"
    assert args[args.index("-ngl") + 1] == "all"
    assert "-t" not in args
    assert args[-3:] == ["-sm", "none", "--no-warmup"]


def test_build_ppl_args_cpu_with_threads():
    prof = ServerProfile(device="cpu", threads=16)
    args = build_ppl_args(Path("/m/q.gguf"), Path("/ref.txt"), prof, ctx=2048)
    assert args[args.index("-ngl") + 1] == "0"
    assert args[args.index("-t") + 1] == "16"
    assert args[args.index("-c") + 1] == "2048"


def test_run_ppl_missing_reference(tmp_path: Path):
    with pytest.raises(PPLError, match="reference file not found"):
        run_ppl(
            binary="/bin/true",
            model_path=tmp_path / "q.gguf",
            reference=tmp_path / "missing.txt",
            server=ServerProfile(device="cpu"),
        )

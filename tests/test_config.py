"""Unit tests for models.yaml parsing and llama-server argument construction."""

import textwrap

import pytest

from quant_bench.config import (
    ConfigError,
    LlamaServerFlags,
    ModelSpec,
    ServerProfile,
    load_models,
    server_flags_for,
)


def _arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def _spec(tmp_path) -> ModelSpec:
    gguf = tmp_path / "m.gguf"
    gguf.write_text("")
    return ModelSpec(path=gguf, tokenizer=tmp_path / "tok", flags=[])


def test_server_flags_ctx_is_multiplied_by_slots(tmp_path):
    spec = _spec(tmp_path)
    prof = ServerProfile(port=1234, ctx=4096, device="cpu", parallel=8)
    args = server_flags_for(spec, prof).argv()
    assert _arg(args, "-c") == str(4096 * 8)
    assert _arg(args, "--parallel") == "8"
    assert _arg(args, "-ngl") == "0"
    assert _arg(args, "--port") == "1234"
    assert _arg(args, "--alias") == "m"


def test_server_flags_host_default_and_override(tmp_path):
    spec = _spec(tmp_path)
    prof = ServerProfile(port=1234, ctx=4096, device="cpu")
    assert _arg(server_flags_for(spec, prof).argv(), "--host") == "0.0.0.0"
    prof_local = ServerProfile(port=1234, ctx=4096, device="cpu", host="127.0.0.1")
    assert _arg(server_flags_for(spec, prof_local).argv(), "--host") == "127.0.0.1"


def test_server_flags_without_parallel_defaults_to_one_slot(tmp_path):
    spec = _spec(tmp_path)
    prof = ServerProfile(port=1, ctx=8192, device="vram")
    args = server_flags_for(spec, prof).argv()
    assert _arg(args, "-c") == "8192"
    assert _arg(args, "-ngl") == "all"
    assert "--parallel" not in args


def test_server_args_hybrid_requires_ngl():
    prof = ServerProfile(device="hybrid")
    with pytest.raises(ConfigError):
        prof.base_args()


def test_server_flags_typed_fields(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_text("")
    spec = ModelSpec(path=gguf, tokenizer=tmp_path / "tok", flags=["--mlock"])
    prof = ServerProfile(port=1234, ctx=4096, device="cpu", parallel=2, threads=4, extra_flags=["--no-mmap"])
    flags = server_flags_for(spec, prof)
    assert isinstance(flags, LlamaServerFlags)
    assert flags.port == 1234 and isinstance(flags.port, int)
    assert flags.ctx == 4096 * 2
    assert flags.ngl == "0"
    assert flags.extra == ["--no-mmap", "--mlock"]
    assert flags.argv()[:2] == ["-m", str(gguf)]


# ---------------------------------------------------------------- models.yaml


def _write_config(tmp_path, text: str):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(textwrap.dedent(text))
    return cfg


def _make_family(tmp_path, ggufs=("model.gguf",), tok_dirname="tok"):
    tok = tmp_path / tok_dirname
    tok.mkdir(exist_ok=True)
    (tok / "tokenizer.json").write_text("")
    paths = []
    for name in ggufs:
        p = tok / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
        paths.append(p)
    return tok, paths


def test_load_models_group_and_per_model_flags(tmp_path):
    tok, (a, b) = _make_family(tmp_path, ggufs=("a.gguf", "b.gguf"))
    cfg = _write_config(
        tmp_path,
        f"""
    - tokenizer: {tok}
      flags: ["--split-mode", "none", "--jinja"]
      models:
        - a.gguf
        - path: b.gguf
          flags: ["--mlock"]
    """,
    )
    models = load_models(cfg)
    assert len(models) == 2
    assert models[0].path == a
    assert models[0].flags == ["--split-mode", "none", "--jinja"]
    assert models[0].tokenizer == tok
    assert models[0].slug == "a"
    assert models[1].path == b
    assert models[1].flags == ["--mlock"]
    assert models[1].tokenizer == tok


def test_load_models_models_dir_defaults_to_tokenizer_dir(tmp_path):
    tok, (a,) = _make_family(tmp_path, ggufs=("q4.gguf",))
    cfg = _write_config(tmp_path, f"- tokenizer: {tok}\n  models: [q4.gguf]\n")
    models = load_models(cfg)
    assert len(models) == 1
    assert models[0].path == a


def test_load_models_explicit_models_dir(tmp_path):
    tok = tmp_path / "tok"
    tok.mkdir()
    (tok / "tokenizer.json").write_text("")
    mdir = tmp_path / "ggufs"
    mdir.mkdir()
    gguf = mdir / "m.gguf"
    gguf.write_text("")
    cfg = _write_config(tmp_path, f"- tokenizer: {tok}\n  models-dir: {mdir}\n  models: [m.gguf]\n")
    models = load_models(cfg)
    assert models[0].path == gguf
    assert models[0].tokenizer == tok


def test_load_models_relative_paths_resolve_against_config_file(tmp_path):
    sub = tmp_path / "configs"
    sub.mkdir()
    models_root = tmp_path / "models"
    tok = models_root / "tok"
    tok.mkdir(parents=True)
    (tok / "tokenizer.json").write_text("")
    (models_root / "m.gguf").write_text("")
    cfg = sub / "models.yaml"
    cfg.write_text("- tokenizer: ../models/tok\n  models-dir: ../models\n  models: [m.gguf]\n")
    models = load_models(cfg)
    assert models[0].path == models_root / "m.gguf"
    assert models[0].tokenizer == tok


def test_load_models_rejects_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_models(tmp_path / "nope.yaml")


def test_load_models_rejects_missing_gguf(tmp_path):
    tok, _ = _make_family(tmp_path, ggufs=())
    cfg = _write_config(tmp_path, f"- tokenizer: {tok}\n  models: [ghost.gguf]\n")
    with pytest.raises(ConfigError, match="file not found"):
        load_models(cfg)


def test_load_models_rejects_bad_tokenizer_dir(tmp_path):
    empty = tmp_path / "emptytok"
    empty.mkdir()
    cfg = _write_config(tmp_path, f"- tokenizer: {empty}\n  models: []\n")
    with pytest.raises(ConfigError, match="tokenizer"):
        load_models(cfg)
    cfg2 = _write_config(tmp_path, f"- tokenizer: {tmp_path / 'ghost'}/tok\n  models: []\n")
    with pytest.raises(ConfigError, match="tokenizer"):
        load_models(cfg2)


def test_load_models_rejects_too_many_models(tmp_path):
    names = [f"m{i}.gguf" for i in range(6)]
    tok, _ = _make_family(tmp_path, ggufs=names)
    cfg = _write_config(tmp_path, f"- tokenizer: {tok}\n  models: [{', '.join(names)}]\n")
    with pytest.raises(ConfigError, match="between 1 and 5"):
        load_models(cfg)


def test_load_models_rejects_duplicates(tmp_path):
    tok, (a,) = _make_family(tmp_path, ggufs=("dup.gguf",))
    cfg = _write_config(tmp_path, f"- tokenizer: {tok}\n  models: [dup.gguf, {a}]\n")
    with pytest.raises(ConfigError, match="duplicate"):
        load_models(cfg)


def test_load_models_rejects_invalid_yaml(tmp_path):
    cfg = _write_config(tmp_path, "tokenizer: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_models(cfg)


def test_load_models_rejects_non_list_top_level(tmp_path):
    cfg = _write_config(tmp_path, "tokenizer: x\n")
    with pytest.raises(ConfigError, match="non-empty list"):
        load_models(cfg)


def test_load_models_rejects_unknown_keys(tmp_path):
    tok, (a,) = _make_family(tmp_path, ggufs=("m.gguf",))
    cfg = _write_config(tmp_path, f"- tokenizer: {tok}\n  model: [m.gguf]\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_models(cfg)

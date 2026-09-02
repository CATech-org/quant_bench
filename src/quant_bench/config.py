"""Configuration types and models.yaml parsing for quant-bench."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

MAX_MODELS = 5
MIN_MODELS = 1

TOKENIZER_BACKEND_FILES = ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer.model")
ENTRY_KEYS = {"tokenizer", "models-dir", "flags", "models"}
MODEL_KEYS = {"path", "flags"}


class ConfigError(Exception):
    pass


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return slug or "model"


@dataclass
class ModelSpec:
    path: Path
    tokenizer: Path
    flags: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return _slugify(self.path.stem)

    @property
    def label(self) -> str:
        return self.path.name


@dataclass
class ServerProfile:
    """Global llama-server settings (per-model flags in models.yaml override these)."""

    port: int = 8080
    host: str = "0.0.0.0"
    ctx: int = 8192
    device: str = "vram"
    ngl: Optional[int] = None
    threads: Optional[int] = None
    parallel: Optional[int] = None
    extra_flags: list[str] = field(default_factory=list)
    ppl_ctx: int = 1024
    log_level: int = 2

    def _device_args(self) -> list[str]:
        if self.ngl is not None:
            return ["-ngl", str(self.ngl)]
        if self.device == "vram":
            return ["-ngl", "all"]
        if self.device == "cpu":
            return ["-ngl", "0"]
        if self.device == "hybrid":
            raise ConfigError("--device hybrid requires --ngl N (number of GPU layers)")
        raise ConfigError(f"unknown --device {self.device!r} (use vram, cpu or hybrid)")

    def base_args(self) -> list[str]:
        args: list[str] = self._device_args()
        if self.threads is not None:
            args += ["-t", str(self.threads)]
        if self.parallel is not None:
            args += ["--parallel", str(self.parallel)]
        args += ["-lv", str(self.log_level)]
        args += list(self.extra_flags)
        return args

    def ppl_base_args(self) -> list[str]:
        """Device/offload flags for llama-perplexity (same offload as the server, no slots/extra flags)."""
        args: list[str] = self._device_args()
        if self.threads is not None:
            args += ["-t", str(self.threads)]
        return args


def _resolve(base: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _check_flags(flags: Any, what: str) -> list[str]:
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        raise ConfigError(f"{what}: flags must be a list of strings")
    return list(flags)


def _check_tokenizer_dir(tok: Path, what: str) -> None:
    if not tok.is_dir():
        raise ConfigError(f"{what}: tokenizer directory not found: {tok}")
    if not any((tok / f).is_file() for f in TOKENIZER_BACKEND_FILES):
        raise ConfigError(
            f"{what}: tokenizer directory {tok} contains no tokenizer files "
            f"({' or '.join(TOKENIZER_BACKEND_FILES)}); download the HuggingFace "
            "tokenizer for this model and point 'tokenizer' at that folder"
        )


def load_models(path: Path) -> list[ModelSpec]:
    """Parse a models.yaml: a list of family groups, each with a tokenizer dir,
    a models-dir (defaults to the tokenizer dir), shared flags, and 1+ .gguf models."""
    if not path.is_file():
        raise ConfigError(
            f"model config file not found: {path}\n"
            "Create models.yaml (one entry per model family) or pass --config <file>."
        )
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{path}: expected a non-empty list of model-family entries")
    models: list[ModelSpec] = []
    seen: set[Path] = set()
    for i, entry in enumerate(raw, start=1):
        what = f"entry {i}"
        if not isinstance(entry, dict):
            raise ConfigError(f"{what}: expected a mapping with 'tokenizer' and 'models' keys")
        unknown = set(entry) - ENTRY_KEYS
        if unknown:
            raise ConfigError(f"{what}: unknown key(s) {sorted(unknown)} (expected {sorted(ENTRY_KEYS)})")
        if not isinstance(entry.get("tokenizer"), str) or not entry.get("tokenizer"):
            raise ConfigError(f"{what}: 'tokenizer' is required (HuggingFace tokenizer dir for this family)")
        tok = _resolve(path.parent, entry["tokenizer"])
        _check_tokenizer_dir(tok, what)
        if "models-dir" in entry:
            if not isinstance(entry["models-dir"], str) or not entry["models-dir"]:
                raise ConfigError(f"{what}: 'models-dir' must be a path string")
            base = _resolve(path.parent, entry["models-dir"])
        else:
            base = tok
        if not base.is_dir():
            raise ConfigError(f"{what}: models-dir not found: {base}")
        group_flags = _check_flags(entry.get("flags") or [], what)
        models_raw = entry.get("models")
        if not isinstance(models_raw, list) or not models_raw:
            raise ConfigError(f"{what}: 'models' must be a non-empty list of .gguf paths")
        for j, m in enumerate(models_raw, start=1):
            mwhat = f"{what} models[{j}]"
            if isinstance(m, str):
                mp_raw, mflags = m, group_flags
            elif isinstance(m, dict):
                munknown = set(m) - MODEL_KEYS
                if munknown:
                    raise ConfigError(f"{mwhat}: unknown key(s) {sorted(munknown)} (expected {sorted(MODEL_KEYS)})")
                if not isinstance(m.get("path"), str) or not m.get("path"):
                    raise ConfigError(f"{mwhat}: 'path' is required")
                mp_raw = m["path"]
                mflags = _check_flags(m.get("flags") or group_flags, mwhat)
            else:
                raise ConfigError(f"{mwhat}: expected a path string or a mapping with 'path'")
            mp = _resolve(base, mp_raw)
            if not mp.name.lower().endswith(".gguf"):
                raise ConfigError(f"{mwhat}: {mp} does not end in .gguf")
            if not mp.is_file():
                raise ConfigError(f"{mwhat}: file not found: {mp}")
            if mp in seen:
                raise ConfigError(f"{mwhat}: duplicate model {mp}")
            seen.add(mp)
            models.append(ModelSpec(path=mp, tokenizer=tok, flags=list(mflags)))
    if not (MIN_MODELS <= len(models) <= MAX_MODELS):
        raise ConfigError(
            f"expected between {MIN_MODELS} and {MAX_MODELS} models in {path}, found {len(models)}"
        )
    return models


def server_args_for(model: ModelSpec, server: ServerProfile) -> list[str]:
    # This llama-server build treats -c as the TOTAL KV budget, split evenly
    # across --parallel slots. We want --ctx to be the per-request context,
    # so pass ctx * slots.
    slots = server.parallel or 1
    args = [
        "-m",
        str(model.path),
        "--port",
        str(server.port),
        "--host",
        server.host,
        "--alias",
        model.slug,
        "-c",
        str(server.ctx * slots),
    ]
    args += server.base_args()
    args += model.flags
    return args

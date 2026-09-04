"""Configuration types and models.yaml parsing for quant-bench."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

MAX_MODELS = 5
MIN_MODELS = 1

TOKENIZER_BACKEND_FILES = ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer.model")
ENTRY_KEYS = {"tokenizer", "models-dir", "flags", "models"}
MODEL_KEYS = {"path", "flags"}


class ConfigError(Exception):
    """Raised when models.yaml is missing, malformed, or points at invalid files."""


def _slugify(name: str) -> str:
    """Turn a model filename stem into a path/URL-safe slug.

    Args:
        name: The name to slugify (typically a GGUF filename stem).

    Returns:
        str: The slugified name, or ``"model"`` if nothing usable remains.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return slug or "model"


@dataclass
class ModelSpec:
    """One GGUF model to benchmark, sharing its model family's tokenizer.

    Attributes:
        path: Path to the ``.gguf`` model file.
        tokenizer: Path to the HuggingFace tokenizer directory for the family.
        flags: Extra llama-server flags specific to this model (override group flags).
    """

    path: Path
    tokenizer: Path
    flags: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        """Path/URL-safe identifier derived from the model filename stem.

        Returns:
            str: The slug, used as the server ``--alias`` and in result filenames.
        """
        return _slugify(self.path.stem)

    @property
    def label(self) -> str:
        """Display label for the model (its filename).

        Returns:
            str: The model filename, e.g. ``gemma-4-E2B-it-Q3_K_M.gguf``.
        """
        return self.path.name


@dataclass
class ServerProfile:
    """Global llama-server settings (per-model flags in models.yaml override these).

    Attributes:
        port: TCP port for llama-server to bind.
        host: Interface for llama-server to bind to (clients still use 127.0.0.1).
        ctx: Per-request context size (the server is given ``ctx * parallel``).
        device: Device mode: ``vram`` (all layers on GPU), ``cpu`` or ``hybrid``.
        ngl: Explicit GPU layer count; overrides ``device`` when set.
        threads: llama-server CPU threads (``-t``).
        parallel: Number of llama-server slots (``--parallel``).
        extra_flags: Extra single-token llama-server flags.
        ppl_ctx: Context window for the ``llama-perplexity`` probe.
        log_level: llama-server ``-lv`` verbosity (1=error .. 5=debug).
    """

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

    def resolved_ngl(self) -> str:
        """Resolve the GPU layer count (``-ngl`` value) from the device/offload settings.

        Returns:
            str: The ``-ngl`` token value: an explicit count, ``"all"`` (vram),
                or ``"0"`` (cpu).

        Raises:
            ConfigError: If ``device`` is ``hybrid`` without an ``ngl`` value, or
                is an unrecognized value.
        """
        if self.ngl is not None:
            return str(self.ngl)
        if self.device == "vram":
            return "all"
        if self.device == "cpu":
            return "0"
        if self.device == "hybrid":
            raise ConfigError("--device hybrid requires --ngl N (number of GPU layers)")
        raise ConfigError(f"unknown --device {self.device!r} (use vram, cpu or hybrid)")

    def _device_args(self) -> list[str]:
        """Translate the device/offload settings into llama-server flags.

        Returns:
            list[str]: Device/offload argv tokens (``-ngl ...``).

        Raises:
            ConfigError: If ``device`` is ``hybrid`` without an ``ngl`` value, or
                is an unrecognized value.
        """
        return ["-ngl", self.resolved_ngl()]

    def base_args(self) -> list[str]:
        """Device, threading, slot and logging flags for the serving llama-server.

        Returns:
            list[str]: argv tokens to append to the llama-server command line.
        """
        args: list[str] = self._device_args()
        if self.threads is not None:
            args += ["-t", str(self.threads)]
        if self.parallel is not None:
            args += ["--parallel", str(self.parallel)]
        args += ["-lv", str(self.log_level)]
        args += list(self.extra_flags)
        return args

    def ppl_base_args(self) -> list[str]:
        """Device/offload flags for llama-perplexity (same offload as the server).

        The perplexity binary loads the model itself, so it takes no slot or
        extra-flags arguments; only the device and thread settings are shared.

        Returns:
            list[str]: Device/thread argv tokens for the perplexity command.
        """
        args: list[str] = self._device_args()
        if self.threads is not None:
            args += ["-t", str(self.threads)]
        return args


class LlamaServerFlags(BaseModel):
    """Typed llama-server flags for one model, expandable to an argv list.

    The well-known flags are first-class fields so ``port`` is a real ``int``
    rather than a token parsed out of a list; free-form options (per-model and
    ``--extra`` flags) are kept in ``extra``.

    Attributes:
        model: Path to the ``.gguf`` model file (``-m``).
        port: TCP port to bind (``--port``).
        host: Interface to bind (``--host``).
        alias: Served model alias (``--alias``).
        ctx: Total KV budget (``-c``); per-request context times slot count.
        ngl: GPU layer count (``-ngl``): ``"all"``, ``"0"``, or a number.
        threads: CPU threads (``-t``), if set.
        parallel: Number of slots (``--parallel``), if set.
        log_level: ``-lv`` verbosity (1=error .. 5=debug).
        extra: Free-form flags appended last (per-model + ``--extra`` options).
    """

    model: str
    port: int
    host: str
    alias: str
    ctx: int
    ngl: str
    threads: Optional[int] = None
    parallel: Optional[int] = None
    log_level: int = 2
    extra: list[str] = Field(default_factory=list)

    def argv(self) -> list[str]:
        """Expand the flags into llama-server argv tokens (excluding the binary).

        Returns:
            list[str]: The ordered command-line tokens for llama-server.
        """
        out: list[str] = [
            "-m",
            self.model,
            "--port",
            str(self.port),
            "--host",
            self.host,
            "--alias",
            self.alias,
            "-c",
            str(self.ctx),
            "-ngl",
            self.ngl,
        ]
        if self.threads is not None:
            out += ["-t", str(self.threads)]
        if self.parallel is not None:
            out += ["--parallel", str(self.parallel)]
        out += ["-lv", str(self.log_level)]
        out += list(self.extra)
        return out


def _resolve(base: Path, raw: str) -> Path:
    """Resolve a possibly-relative path against a base directory.

    Args:
        base: Directory to resolve relative paths against (the config file's parent).
        raw: The path as written in the config; may be relative or ``~/``-anchored.

    Returns:
        Path: The fully-resolved absolute path.
    """
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _check_flags(flags: Any, what: str) -> list[str]:
    """Validate that a ``flags`` value is a list of strings.

    Args:
        flags: The value to validate.
        what: Human-readable label for the entry, used in error messages.

    Returns:
        list[str]: The flags as a fresh list.

    Raises:
        ConfigError: If ``flags`` is not a list of strings.
    """
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        raise ConfigError(f"{what}: flags must be a list of strings")
    return list(flags)


def _check_tokenizer_dir(tok: Path, what: str) -> None:
    """Verify a tokenizer directory exists and contains a usable tokenizer.

    Args:
        tok: The tokenizer directory to check.
        what: Human-readable label for the entry, used in error messages.

    Raises:
        ConfigError: If the directory is missing or has no tokenizer files.
    """
    if not tok.is_dir():
        raise ConfigError(f"{what}: tokenizer directory not found: {tok}")
    if not any((tok / f).is_file() for f in TOKENIZER_BACKEND_FILES):
        raise ConfigError(
            f"{what}: tokenizer directory {tok} contains no tokenizer files "
            f"({' or '.join(TOKENIZER_BACKEND_FILES)}); download the HuggingFace "
            "tokenizer for this model and point 'tokenizer' at that folder"
        )


def load_models(path: Path) -> list[ModelSpec]:
    """Parse a models.yaml into the list of models to benchmark.

    The file is a list of model-family groups; each group has a tokenizer dir,
    an optional models-dir (defaulting to the tokenizer dir), shared flags, and
    one or more ``.gguf`` models.

    Args:
        path: Path to the models.yaml config file.

    Returns:
        list[ModelSpec]: The flattened list of models to benchmark (1 to
            ``MAX_MODELS`` total).

    Raises:
        ConfigError: If the file is missing, malformed, or references
            non-existent or invalid models or tokenizer directories.
    """
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


def server_flags_for(model: ModelSpec, server: ServerProfile) -> LlamaServerFlags:
    """Build the typed llama-server flags for one model.

    Args:
        model: The model to serve.
        server: The global server profile (port, ctx, device, etc.).

    Returns:
        LlamaServerFlags: The typed flags; call ``.argv()`` for the argv tokens.

    Note:
        This llama-server build treats ``-c`` as the total KV budget, split
        evenly across ``--parallel`` slots, so ``ctx * slots`` is passed to make
        ``--ctx`` the per-request context.
    """
    slots = server.parallel or 1
    return LlamaServerFlags(
        model=str(model.path),
        port=server.port,
        host=server.host,
        alias=model.slug,
        ctx=server.ctx * slots,
        ngl=server.resolved_ngl(),
        threads=server.threads,
        parallel=server.parallel,
        log_level=server.log_level,
        extra=[*server.extra_flags, *model.flags],
    )

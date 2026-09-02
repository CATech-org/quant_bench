"""Spawn, health-check and stop a local llama-server process."""

from __future__ import annotations

import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


class ServerError(Exception):
    pass


def find_llama_server(explicit: str = "llama-server") -> str:
    path = shutil.which(explicit)
    if path is None:
        raise ServerError(
            f"llama-server binary {explicit!r} not found on PATH. "
            "Install llama.cpp system-wide or pass --llama-server /path/to/llama-server."
        )
    return path


def llama_server_version(binary: str) -> str:
    try:
        out = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30
        )
        text = (out.stdout or "") + (out.stderr or "")
        first = text.strip().splitlines()[0] if text.strip() else "unknown"
        return first.removeprefix("version:").strip()
    except Exception:
        return "unknown"


def _get(url: str, timeout: float = 3.0) -> Optional[dict]:
    import json

    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def wait_port_free(port: int, timeout: float = 120.0) -> None:
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(2)
    raise ServerError(f"port {port} still in use after {timeout:.0f}s; refusing to start")


@dataclass
class LlamaServer:
    binary: str
    args: list[str]
    log_path: Path
    startup_timeout: int = 900

    proc: Optional[subprocess.Popen] = None
    port: int = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        i = self.args.index("--port")
        self.port = int(self.args[i + 1])
        wait_port_free(self.port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [self.binary, *self.args]
        console.print(f"[bold]Starting server:[/bold] {' '.join(cmd)}")
        with open(self.log_path, "w") as log:
            self.proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.time() + self.startup_timeout
        last_log_hint = 0.0
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                tail = self.log_path.read_text().splitlines()[-15:]
                raise ServerError(
                    "llama-server exited during startup "
                    f"(code {self.proc.returncode}). Last log lines:\n" + "\n".join(tail)
                )
            health = _get(f"{self.url}/health")
            if health and health.get("status") == "ok":
                console.print("[green]Server ready.[/green]")
                return
            if time.time() - last_log_hint > 30:
                console.print("  waiting for model to load ...")
                last_log_hint = time.time()
            time.sleep(2)
        self.stop()
        raise ServerError(f"llama-server did not become healthy within {self.startup_timeout}s")

    def model_name(self) -> Optional[str]:
        data = _get(f"{self.url}/v1/models")
        try:
            return data["data"][0]["id"]
        except (TypeError, KeyError, IndexError):
            return None

    def _signal_group(self, signum: int) -> None:
        import os

        try:
            os.killpg(os.getpgid(self.proc.pid), signum)
        except (ProcessLookupError, PermissionError, OSError):
            if signum == signal.SIGTERM:
                self.proc.terminate()
            else:
                self.proc.kill()

    def stop(self, timeout: int = 60) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            console.print("Stopping server ...")
            self._signal_group(signal.SIGTERM)
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                console.print("[yellow]Server did not exit, killing ...[/yellow]")
                self._signal_group(signal.SIGKILL)
                self.proc.wait(timeout=30)
        self.proc = None
        import socket

        deadline = time.time() + 60
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", self.port)) != 0:
                    return
            time.sleep(1)
        console.print("[yellow]Warning: port still bound after stop.[/yellow]")

#!/usr/bin/env python3
"""PCA48 server client — wraps Unix socket JSON communication.

设计: 1 server 进程 + N concurrent clients
- server 启动一次, 加载 Qwen (~30-40s)
- 后续 client 调用 server 即可(无 model load)
- 多 client 可并发 (server 内部 threading)

用法:
    from syntax_subspace_pca48_server_client import ModelClient

    client = ModelClient()  # auto-starts server if not running
    client.load_ckpt("/path/to/ckpt.pt")
    candidates = client.generate(z_tensor, attrs_list, ...)
    client.quit()
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
SERVER_SCRIPT = REPO_ROOT / "gen_query" / "syntax_subspace_pca48_server.py"
PYTHON_BIN = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"
SERVER_LOG = SCRATCH / "logs" / "pca48_server.log"
SERVER_PID_FILE = SCRATCH / "tmp" / "pca48_server.pid"
SERVER_SOCKET = SCRATCH / "tmp" / "pca48_server.sock"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [client] {msg}", file=sys.stderr, flush=True)


def is_server_running() -> bool:
    """Check if server socket exists AND PID alive."""
    if not SERVER_SOCKET.exists():
        return False
    if not SERVER_PID_FILE.exists():
        return False
    try:
        pid = int(SERVER_PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, FileNotFoundError, OSError):
        return False


def start_server() -> int:
    """Start server in background. Returns PID."""
    if is_server_running():
        pid = int(SERVER_PID_FILE.read_text().strip())
        log(f"  server already running (PID {pid})")
        return pid

    SCRATCH.joinpath("tmp").mkdir(parents=True, exist_ok=True)
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)

    log(f"  starting server: {SERVER_SCRIPT}")
    proc = subprocess.Popen(
        [PYTHON_BIN, "-u", str(SERVER_SCRIPT)],
        stdin=subprocess.DEVNULL,
        stdout=open(SERVER_LOG, "a"),
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        start_new_session=True,
    )
    SERVER_PID_FILE.write_text(str(proc.pid))
    log(f"  server PID {proc.pid}, waiting for socket...")

    # Wait for socket file
    start = time.time()
    while time.time() - start < 240:
        time.sleep(2)
        if SERVER_SOCKET.exists():
            log(f"  socket ready ({time.time() - start:.0f}s)")
            return proc.pid
        if proc.poll() is not None:
            raise RuntimeError(f"server died, log:\n{SERVER_LOG.read_text()[-2000:]}")

    raise RuntimeError(f"server socket timeout, log:\n{SERVER_LOG.read_text()[-2000:]}")


class ModelClient:
    """Talk to the long-running model server via Unix socket."""

    def __init__(self, auto_start: bool = True):
        if auto_start:
            start_server()

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(str(SERVER_SOCKET))
        log(f"  connected to server socket")

    def _send_recv(self, req: dict) -> dict:
        line = json.dumps(req) + "\n"
        self.sock.sendall(line.encode("utf-8"))
        # Read until \n
        buf = b""
        while b"\n" not in buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("server closed socket")
            buf += chunk
        resp_line, _, _ = buf.partition(b"\n")
        return json.loads(resp_line.decode("utf-8"))

    def load_ckpt(self, ckpt_path: str) -> None:
        resp = self._send_recv({"op": "load_ckpt", "ckpt_path": ckpt_path})
        if resp.get("status") != "ok":
            raise RuntimeError(f"load_ckpt failed: {resp}")
        log(f"  loaded ckpt: {ckpt_path}")

    def generate(
        self,
        z: torch.Tensor,
        attrs_list: list[list[str]],
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
        seed: Optional[int] = None,
    ) -> list[str]:
        z_list = z.cpu().numpy().tolist() if isinstance(z, torch.Tensor) else z
        req = {
            "op": "generate",
            "z": z_list,
            "attrs_list": attrs_list,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        }
        if seed is not None:
            req["seed"] = seed
        resp = self._send_recv(req)
        if resp.get("status") != "ok":
            raise RuntimeError(f"generate failed: {resp}")
        return resp["candidates"]

    def quit(self) -> None:
        """Send quit request (does NOT kill server — server stays up for other clients)."""
        try:
            self._send_recv({"op": "quit"})
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    log("=== ModelClient self-test ===")
    client = ModelClient(auto_start=False)
    if not is_server_running():
        start_server()
        client = ModelClient(auto_start=False)
    ping = client._send_recv({"op": "ping"})
    log(f"ping: {ping}")
    client.quit()
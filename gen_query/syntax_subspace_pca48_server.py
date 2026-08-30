#!/usr/bin/env python3
"""Phase 6 — long-running PCA48 + Qwen model server (Unix socket + JSON protocol).

设计目的 (用户 2026-08-30 加速): 避免每次 eval 重启都加载 Qwen2-7B (~30-40s)。

启动:
  nohup python syntax_subspace_pca48_server.py < /dev/zero > server.log 2>&1 &
  启动后保持 model 加载, 监听 Unix socket, 接受多 client 并发请求

协议 (JSON per request, socket.sendall):
  Request:  {"op": "load_ckpt", "ckpt_path": "/path/to/ckpt.pt"}
  Response: {"status": "ok", "ckpt": "/path/to/ckpt.pt"}

  Request:  {"op": "generate", "z": [[...48 floats...], ...],
             "attrs_list": [[attr1, attr2, ...], ...],
             "max_new_tokens": 50, "temperature": 0.8, "top_p": 0.95,
             "do_sample": true, "seed": 42}
  Response: {"candidates": ["query1", "query2", ...]}

  Request:  {"op": "quit"}
  Response: {"status": "bye"}

每个 client 连接由独立 thread 处理(避免阻塞主循环)。

参数全部硬编码 (Rule 3): 路径等通过 stdin JSON 传入。
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "gen_query"))

SOCKET_PATH = "/home/wlia0047/hj82_scratch2/wenyu/tmp/pca48_server.sock"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [server] {msg}", file=sys.stderr, flush=True)


def recv_json(sock: socket.socket) -> dict:
    """Read one JSON line (terminated by \n)."""
    buf = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            return None
        buf += chunk
        if b"\n" in buf:
            line, _, rest = buf.partition(b"\n")
            return json.loads(line.decode("utf-8"))


def send_json(sock: socket.socket, obj: dict) -> None:
    line = json.dumps(obj) + "\n"
    sock.sendall(line.encode("utf-8"))


class ServerState:
    """Shared model state across all client connections."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.current_ckpt = "<init>"

    def init_model(self):
        log("  loading Qwen2-7B + LoRA + projector (default init) ...")
        from syntax_subspace_pca48_projector import Qwen2WithPrefix
        t0 = time.time()
        self.model = Qwen2WithPrefix()
        self.model.eval()
        log(f"  model ready in {time.time()-t0:.0f}s, trainable params: "
            f"{sum(p.numel() for p in self.model.trainable_parameters()):,}")

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        t0 = time.time()

        if op == "ping":
            return {"status": "ok", "ckpt": self.current_ckpt, "elapsed": time.time() - t0}

        if op == "load_ckpt":
            ckpt_path = req["ckpt_path"]
            if not Path(ckpt_path).exists():
                return {"status": "error", "error": f"ckpt not found: {ckpt_path}"}
            with self.lock:
                try:
                    self.model.load_projector(ckpt_path)
                    self.current_ckpt = ckpt_path
                    log(f"  loaded ckpt: {ckpt_path}")
                    return {"status": "ok", "ckpt": ckpt_path, "elapsed": time.time() - t0}
                except Exception as e:
                    return {"status": "error", "error": f"load_projector failed: {e}"}

        if op == "generate":
            try:
                z_np = np.asarray(req["z"], dtype=np.float32)
                z_tensor = torch.from_numpy(z_np).to("cuda")
                attrs_list = req["attrs_list"]
                max_new_tokens = int(req.get("max_new_tokens", 50))
                temperature = float(req.get("temperature", 0.8))
                top_p = float(req.get("top_p", 0.95))
                do_sample = bool(req.get("do_sample", True))
                seed = req.get("seed", None)

                if seed is not None:
                    torch.manual_seed(int(seed))

                with self.lock, torch.no_grad():
                    candidates = self.model.generate(
                        z_tensor, attrs_list,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                return {
                    "status": "ok",
                    "candidates": candidates,
                    "elapsed": time.time() - t0,
                    "n_batch": int(z_tensor.size(0)),
                }
            except Exception as e:
                return {"status": "error", "error": f"generate failed: {e}"}

        if op == "quit":
            return {"status": "bye"}

        return {"status": "error", "error": f"unknown op: {op}"}


def handle_client(client_sock: socket.socket, addr, state: ServerState) -> None:
    """Handle one client connection in its own thread."""
    log(f"  client connected: {addr}")
    try:
        while True:
            req = recv_json(client_sock)
            if req is None:
                break
            op = req.get("op")
            resp = state.handle(req)
            send_json(client_sock, resp)
            if op == "quit":
                break
    except Exception as e:
        log(f"  client error: {e}")
    finally:
        client_sock.close()
        log(f"  client disconnected: {addr}")


def main():
    log("=== PCA48 model server starting (Unix socket) ===")

    # Init model
    state = ServerState()
    state.init_model()

    # Setup socket
    Path(SOCKET_PATH).parent.mkdir(parents=True, exist_ok=True)
    if Path(SOCKET_PATH).exists():
        Path(SOCKET_PATH).unlink()

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(SOCKET_PATH)
    server_sock.listen(8)
    log(f"  socket listening at {SOCKET_PATH}")

    try:
        while True:
            client_sock, addr = server_sock.accept()
            t = threading.Thread(target=handle_client, args=(client_sock, addr, state), daemon=True)
            t.start()
    except KeyboardInterrupt:
        log("  shutting down")
    finally:
        server_sock.close()
        Path(SOCKET_PATH).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
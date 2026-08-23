#!/usr/bin/env python3
"""Long-running Qwen2-7B server with hidden-state injection.

Each client connection gets its own thread → recv blocking stays in the thread,
not the main event loop.

Request (pickle):
    {"id": int, "method": str, "kwargs": dict}

Response (pickle):
    {"id": int, "result": Any, "error": str or None}
"""
from __future__ import annotations
import argparse
import io
import os
import pickle
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import torch

# === Hardcoded paths ===
MODEL_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
DEFAULT_SOCKET = "/home/wlia0047/hj82_scratch2/wenyu/tmp/qwen_hidden_server.sock"
PID_FILE = "/home/wlia0047/hj82_scratch2/wenyu/tmp/qwen_hidden_server.pid"
LOG_FILE = "/home/wlia0047/hj82_scratch2/wenyu/tmp/qwen_hidden_server.log"


class QwenHiddenServer:
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._running = True
        self._model_loaded = False
        self._client = None  # QwenLocalClient

    def _load_model(self) -> None:
        t0 = time.time()
        sys.path.insert(0, str(Path(__file__).parent))
        from llm_client import create_qwen_local_client
        self._client = create_qwen_local_client(with_vllm=False)
        self._model_loaded = True
        self._log(f"model loaded in {time.time()-t0:.1f}s")

    def _log(self, msg: str) -> None:
        print(msg, flush=True)

    def _handle_client(self, client_sock: socket.socket) -> None:
        """Run in a dedicated thread per client."""
        try:
            # Receive until we have complete data (pickle frame)
            data = b""
            while True:
                chunk = client_sock.recv(65536)
                if not chunk:
                    break
                data += chunk
                # Try to unpickle — if it succeeds, we have a complete frame
                try:
                    request = pickle.loads(data)
                    break
                except Exception:
                    # Incomplete frame, keep receiving
                    continue
            if not data:
                return

            req_id = request.get("id")
            method = request.get("method")
            kwargs = request.get("kwargs", {})

            if method == "_ping":
                response = {"id": req_id, "result": "pong", "error": None}
            elif method == "_shutdown":
                response = {"id": req_id, "result": "bye", "error": None}
                client_sock.sendall(pickle.dumps(response))
                self._running = False
                return
            else:
                try:
                    gen_fn = getattr(self._client, method, None)
                    if gen_fn is None:
                        raise AttributeError(f"unknown method: {method}")

                    # Deserialize torch.Tensor bytes in injection_per_row
                    kw = {}
                    for k, v in kwargs.items():
                        if isinstance(v, list) and k == "injection_per_row":
                            deserialized = []
                            for item in v:
                                if item is None:
                                    deserialized.append(None)
                                elif isinstance(item, bytes):
                                    deserialized.append(
                                        torch.load(io.BytesIO(item), weights_only=False)
                                    )
                                else:
                                    deserialized.append(item)
                            kw[k] = deserialized
                        else:
                            kw[k] = v

                    result = gen_fn(**kw)
                    response = {"id": req_id, "result": result, "error": None}
                except Exception as exc:
                    response = {
                        "id": req_id,
                        "result": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

            client_sock.sendall(pickle.dumps(response))
        except Exception as exc:
            try:
                client_sock.sendall(
                    pickle.dumps({"id": -1, "result": None, "error": str(exc)})
                )
            except Exception:
                pass
        finally:
            client_sock.close()

    def run(self) -> None:
        # Remove stale socket
        sock_path = Path(self.socket_path)
        if sock_path.exists():
            sock_path.unlink()
        sock_path.parent.mkdir(parents=True, exist_ok=True)

        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(self.socket_path)
        server_sock.listen(32)
        server_sock.settimeout(1.0)  # accept timeout so we can check _running

        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

        self._log(f"listening on {self.socket_path}")
        self._log(f"pid={os.getpid()}")

        self._load_model()

        while self._running:
            try:
                client_sock, _ = server_sock.accept()
            except socket.timeout:
                continue
            t = threading.Thread(target=self._handle_client, args=(client_sock,), daemon=True)
            t.start()

        # Shutdown
        try:
            server_sock.close()
        except Exception:
            pass
        if sock_path.exists():
            sock_path.unlink()
        if Path(PID_FILE).exists():
            Path(PID_FILE).unlink()
        self._log("shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", default=DEFAULT_SOCKET)
    args = parser.parse_args()

    server = QwenHiddenServer(socket_path=args.socket_path)

    def signal_handler(sig, frame):
        server._running = False

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    server.run()

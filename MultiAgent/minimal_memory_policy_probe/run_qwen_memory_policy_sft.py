#!/usr/bin/env python3
"""Launch LLaMA-Factory SFT for the memory-policy trajectory student."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


LLAMA_FACTORY_DIR = Path("/home/wlia0047/ar57/wenyu/MultiAgent/Agent_Foundation_Models/LLaMA-Factory")
CONFIG_FILE = LLAMA_FACTORY_DIR / "examples/train_lora/qwen2_5_0_5b_memory_policy_sft.yaml"


def main() -> None:
    if not LLAMA_FACTORY_DIR.is_dir():
        raise FileNotFoundError(f"LLaMA-Factory directory does not exist: {LLAMA_FACTORY_DIR}")
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(f"Training config does not exist: {CONFIG_FILE}")

    env = os.environ.copy()
    src_path = str(LLAMA_FACTORY_DIR / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if existing_pythonpath is None else f"{src_path}:{existing_pythonpath}"

    command = [
        sys.executable,
        "-m",
        "llamafactory.cli",
        "train",
        str(CONFIG_FILE.relative_to(LLAMA_FACTORY_DIR)),
    ]
    print(f"Launching memory-policy SFT from {LLAMA_FACTORY_DIR}", flush=True)
    print("Command: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=LLAMA_FACTORY_DIR, env=env, check=True)


if __name__ == "__main__":
    main()

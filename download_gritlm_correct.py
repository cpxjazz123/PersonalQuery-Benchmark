#!/usr/bin/env python3
import os
import warnings
warnings.filterwarnings("ignore")

# 设置正确的缓存目录
os.environ["HF_HOME"] = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from huggingface_hub import snapshot_download

print("开始下载 GritLM/GritLM-7B 模型...")
print(f"HF_HOME: {os.environ.get('HF_HOME')}")

snapshot_download(
    "GritLM/GritLM-7B",
    cache_dir="/home/wlia0047/ar57_scratch/wenyu/hf_models",
    local_files_only=False
)

print("下载完成!")
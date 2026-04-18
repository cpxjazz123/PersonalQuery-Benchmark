#!/usr/bin/env python3
import os
import warnings
warnings.filterwarnings("ignore")

# 取消离线模式
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from huggingface_hub import snapshot_download

print("开始下载 GritLM/GritLM-7B 模型...")
print(f"缓存目录: /home/wlia0047/ar57_scratch/wenyu/huggingface_cache")

snapshot_download(
    "GritLM/GritLM-7B",
    cache_dir="/home/wlia0047/ar57_scratch/wenyu/huggingface_cache",
    local_files_only=False
)

print("下载完成!")
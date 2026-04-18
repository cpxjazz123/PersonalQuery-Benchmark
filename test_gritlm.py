#!/usr/bin/env python3
"""测试 GritLM 编码功能"""

import os
import torch
os.environ['HF_HOME'] = '/home/wlia0047/ar57_scratch/wenyu/hf_models'

print("=" * 60)
print("开始测试 GritLM")
print("=" * 60)

from gritlm import GritLM
print("Import GritLM 成功")

print("正在加载模型...")
model = GritLM('GritLM/GritLM-7B', torch_dtype=torch.bfloat16, is_inference=True)
print("模型加载成功")

print("正在编码...")
result = model.encode(['hello world'], instruction='<|user|>\n<|embed|>\n', max_length=512)
print(f"编码结果 shape: {result.shape}")
print("测试完成!")

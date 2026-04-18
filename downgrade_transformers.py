#!/usr/bin/env python3
import subprocess
import sys

print("开始降级 transformers 到 4.44.0...")

result = subprocess.run([
    sys.executable, "-m", "pip", "install",
    "transformers==4.44.0",
    "--upgrade"
], capture_output=True, text=True)

print("stdout:", result.stdout)
print("stderr:", result.stderr)

if result.returncode == 0:
    print("transformers 降级成功!")
else:
    print("降级失败!")

# 验证版本
import transformers
print(f"当前 transformers 版本: {transformers.__version__}")
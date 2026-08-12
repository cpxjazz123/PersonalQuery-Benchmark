#!/usr/bin/env python3
"""使用用户真实错误训练的 LambdaMART 模型生成噪声查询 - Grocery_and_Gourmet_Food"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR / "common"))

from apply_lambdamart_userbased_noisy import main

if __name__ == "__main__":
    main("Grocery_and_Gourmet_Food")
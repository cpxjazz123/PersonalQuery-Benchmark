# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

PersonalQuery 是一个个性化搜索查询生成研究项目，用户评论转化为高度差异化的用户画像及个性化搜索查询。

## 核心执行规则

### 所有脚本必须使用 /root/stark 环境

Python 虚拟环境位于 `/root/stark`（Python 3.11.10，torch 2.11.0+cu128）：

```bash
# 激活环境后执行
source /root/stark/bin/activate
python script.py
```
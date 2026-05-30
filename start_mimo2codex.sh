#!/bin/bash

# 清理之前占用 8788 端口的进程
if lsof -i :8788 >/dev/null 2>&1; then
    echo "清理占用 8788 端口的旧进程..."
    lsof -i :8788 -t | xargs -r kill -9 2>/dev/null
    sleep 1
fi

cd /fs04/ar57/wenyu/mimo2codex
npm start -- --model minimax
echo ""
echo "=========================================="
echo "mimo2codex 已启动"
echo "Admin UI: http://127.0.0.1:8788/admin/"
echo "健康检查: http://127.0.0.1:8788/healthz"
echo "=========================================="

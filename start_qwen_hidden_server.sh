#!/bin/bash
# ===========================================================================
# Qwen Hidden Server 启动脚本
# ===========================================================================
#
# 启动一个 long-running Qwen2-7B 服务进程，监听 Unix socket。
# 后续所有 generate_with_hidden_injection 调用走此 socket，无需重复加载模型。
#
# 用法:
#   ./start_qwen_hidden_server.sh start   # 后台启动（nohup）
#   ./start_qwen_hidden_server.sh stop   # 停止服务
#   ./start_qwen_hidden_server.sh status # 检查是否在运行
#
# 产物:
#   /home/wlia0047/hj82_scratch2/wenyu/tmp/qwen_hidden_server.sock  (socket)
#   /home/wlia0047/hj82_scratch2/wenyu/tmp/qwen_hidden_server.pid  (PID file)
#   /home/wlia0047/hj82_scratch2/wenyu/tmp/qwen_hidden_server.log  (stdout/stderr)
# ===========================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"
cd "${REPO_DIR}"

SOCKET_PATH="/home/wlia0047/hj82_scratch2/wenyu/tmp/qwen_hidden_server.sock"
PID_FILE="/home/wlia0047/hj82_scratch2/wenyu/tmp/qwen_hidden_server.pid"
LOG_FILE="/home/wlia0047/hj82_scratch2/wenyu/tmp/qwen_hidden_server.log"
TMPDIR="/home/wlia0047/hj82_scratch2/wenyu/tmp"

PYTHON="/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"

check_alive() {
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat "${PID_FILE}")
        if [ -n "${PID}" ] && kill -0 "${PID}" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

do_start() {
    if check_alive; then
        PID=$(cat "${PID_FILE}")
        echo "Server already running with PID ${PID}"
        return 0
    fi

    mkdir -p "${TMPDIR}"

    echo "[$(date)] Starting Qwen hidden server..."
    nohup env TMPDIR="${TMPDIR}" \
        "${PYTHON}" \
            qwen_hidden_server.py \
            --socket-path "${SOCKET_PATH}" \
            > "${LOG_FILE}" 2>&1 &

    SERVER_PID=$!
    echo "${SERVER_PID}" > "${PID_FILE}"
    echo "[$(date)] Server started with PID ${SERVER_PID}"
    echo "Log: ${LOG_FILE}"

    # Wait for model to load (up to 120s)
    echo "Waiting for model to load..."
    for i in $(seq 1 60); do
        sleep 2
        if grep -q "model loaded in" "${LOG_FILE}" 2>/dev/null; then
            echo "Model loaded successfully!"
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: Server process died. Check ${LOG_FILE}"
            tail -20 "${LOG_FILE}"
            return 1
        fi
    done

    echo "WARNING: Model load timeout (120s), but server may still be initializing..."
    echo "Check ${LOG_FILE} for status"
}

do_stop() {
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat "${PID_FILE}")
        echo "Stopping server PID ${PID}..."
        kill "${PID}" 2>/dev/null || true
        sleep 1
        kill -9 "${PID}" 2>/dev/null || true
        rm -f "${PID_FILE}"
        rm -f "${SOCKET_PATH}"
        echo "Server stopped."
    else
        echo "No PID file found, nothing to stop."
    fi
}

do_status() {
    if check_alive; then
        PID=$(cat "${PID_FILE}")
        echo "Server is RUNNING (PID ${PID})"
        if [ -f "${LOG_FILE}" ]; then
            echo "--- last 5 lines of log ---"
            tail -5 "${LOG_FILE}"
        fi
    else
        echo "Server is NOT running"
        if [ -f "${LOG_FILE}" ]; then
            echo "--- last 10 lines of log ---"
            tail -10 "${LOG_FILE}"
        fi
    fi
}

case "${1}" in
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_stop
        sleep 2
        do_start
        ;;
    status)
        do_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac

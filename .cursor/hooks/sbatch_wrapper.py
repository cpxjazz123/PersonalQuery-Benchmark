#!/usr/bin/env python3
# Wrapper script to execute Python commands via sbatch and monitor logs

import sys
import os
import re
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def parse_command(args):
    """Parse command arguments, handling quoted strings like bash script."""
    if not args:
        return None
    
    if len(args) == 1:
        cmd = args[0]
        # Remove outer quotes if present
        if (cmd.startswith("'") and cmd.endswith("'")) or \
           (cmd.startswith('"') and cmd.endswith('"')):
            return cmd[1:-1]
        return cmd
    else:
        # Combine all arguments
        return ' '.join(args)


def is_job_running(job_id):
    """Check if SLURM job is still running."""
    try:
        result = subprocess.run(
            ['squeue', '-j', str(job_id), '-h'],
            capture_output=True,
            timeout=5,
            text=True
        )
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return True
    except Exception:
        return True


def find_running_jobs_by_name(script_name, current_user=None):
    """
    查找当前用户正在运行的同名脚本任务。
    """
    if current_user is None:
        current_user = os.environ.get('USER', os.environ.get('LOGNAME', ''))
    
    if not script_name or script_name == 'job':
        return []
    
    try:
        result = subprocess.run(
            ['squeue', '-u', current_user, '-o', '%.18i %.100j %.8T', '--noheader'],
            capture_output=True,
            timeout=10,
            text=True
        )
        
        if result.returncode != 0:
            return []
        
        running_jobs = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
            
            job_id = parts[0]
            status = parts[-1]
            job_name = ' '.join(parts[1:-1])
            
            if status in ['RUNNING', 'PENDING']:
                if script_name in job_name or job_name.startswith(f"submit_{script_name}"):
                    running_jobs.append({
                        'job_id': job_id,
                        'name': job_name,
                        'status': status
                    })
        
        return running_jobs
    
    except subprocess.TimeoutExpired:
        print("[sbatch_wrapper] ⚠️  查询运行中任务超时", file=sys.stderr)
        return []
    except Exception as e:
        print(f"[sbatch_wrapper] ⚠️  查询运行中任务时出错: {e}", file=sys.stderr)
        return []


def cancel_jobs(job_ids):
    """
    取消指定的SLURM任务。
    
    Args:
        job_ids: 任务ID列表
    
    Returns:
        tuple: (成功取消的数量, 失败的数量)
    """
    if not job_ids:
        return (0, 0)
    
    success_count = 0
    fail_count = 0
    
    for job_id in job_ids:
        try:
            result = subprocess.run(
                ['scancel', str(job_id)],
                capture_output=True,
                timeout=5,
                text=True
            )
            
            if result.returncode == 0:
                print(f"[sbatch_wrapper] ✅ 已取消任务 {job_id}", file=sys.stderr)
                success_count += 1
            else:
                print(f"[sbatch_wrapper] ⚠️  取消任务 {job_id} 失败: {result.stderr}", file=sys.stderr)
                fail_count += 1
        
        except subprocess.TimeoutExpired:
            print(f"[sbatch_wrapper] ⚠️  取消任务 {job_id} 超时", file=sys.stderr)
            fail_count += 1
        except Exception as e:
            print(f"[sbatch_wrapper] ⚠️  取消任务 {job_id} 时出错: {e}", file=sys.stderr)
            fail_count += 1
    
    return (success_count, fail_count)


def monitor_logs(log_file, err_file, job_id, max_wait=30):
    """Monitor job status until it completes, outputting log content to terminal."""
    log_path = Path(log_file)
    err_path = Path(err_file)

    # Wait for log files to be created (with timeout)
    wait_count = 0
    while wait_count < max_wait:
        if log_path.exists() or err_path.exists():
            break
        # Check job status while waiting
        if not is_job_running(job_id):
            print("[sbatch_wrapper] ⚠️  作业已完成但日志文件尚未创建")
            return
        time.sleep(1)
        wait_count += 1

    # Monitor job status and output log content in real-time
    last_log_size = 0
    last_err_size = 0
    last_status_time = time.time()
    status_interval = 5

    try:
        while True:
            current_time = time.time()
            is_running = is_job_running(job_id)
            
            output_log_content = False
            if log_path.exists() and log_path.stat().st_size > last_log_size:
                with open(log_path, 'r') as f:
                    f.seek(last_log_size)
                    new_content = f.read()
                    if new_content:
                        print(new_content, end='')
                        output_log_content = True
                last_log_size = log_path.stat().st_size

            output_err_content = False
            if err_path.exists() and err_path.stat().st_size > last_err_size:
                with open(err_path, 'r') as f:
                    f.seek(last_err_size)
                    new_content = f.read()
                    if new_content:
                        print(new_content, end='')
                        output_err_content = True
                last_err_size = err_path.stat().st_size
            
            if not is_running:
                if output_log_content or output_err_content:
                    print(f"[sbatch_wrapper] 作业已完成 (Job ID: {job_id})")
                time.sleep(0.5)
                break
            
            if current_time - last_status_time >= status_interval:
                print(f"[sbatch_wrapper] 作业正在运行中 (Job ID: {job_id})...")
                last_status_time = current_time

            time.sleep(1)

    except KeyboardInterrupt:
        # User interrupted, but job continues running
        raise
    except Exception as e:
        print(f"[sbatch_wrapper] ⚠️  监控作业状态时出错: {e}")
        # Continue monitoring even if there's an error
        while True:
            if not is_job_running(job_id):
                break
            time.sleep(1)


def is_python_script_command(command: str) -> bool:
    """Check if command is executing a Python script."""
    if not command:
        return False
    
    command = command.strip()
    
    # Patterns that indicate Python script execution
    patterns = [
        r'^python\s+.*\.py',           # python script.py
        r'^python3\s+.*\.py',           # python3 script.py
        r'\spython\s+.*\.py',           # ... python script.py
        r'\spython3\s+.*\.py',          # ... python3 script.py
        r'\.py\s',                      # .py file with arguments
        r'\.py$',                       # .py file at end
    ]
    
    for pattern in patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    
    return False


def has_sbatch_in_command(command: str) -> bool:
    """Check if command already contains sbatch or sbatch_wrapper."""
    if not command:
        return False
    
    command_lower = command.lower()
    script_name = "/home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py"
    
    # Check for various sbatch patterns
    sbatch_patterns = [
        'sbatch',
        'sbatch_wrapper',
        script_name.lower(),
        'sbatch_wrapper.py',
    ]
    
    for pattern in sbatch_patterns:
        if pattern in command_lower or pattern in command:
            return True
    
    return False


def read_input() -> Dict[str, Any]:
    """Read and parse JSON input from stdin (for hook mode)."""
    try:
        if not sys.stdin.isatty():
            raw_input = sys.stdin.read()
            if raw_input.strip():
                return json.loads(raw_input)
    except (json.JSONDecodeError, Exception):
        pass
    return {}


def get_gpu_node_status():
    """
    查询GPU分区的节点状态，返回每个节点的GPU空闲情况。
    """
    try:
        result = subprocess.run(
            ['sinfo', '-p', 'gpu', '-N', '-o', '%N %G %C', '--noheader'],
            capture_output=True,
            text=True,
            timeout=20
        )
        
        if result.returncode != 0:
            print(f"[sbatch_wrapper] ⚠️  无法获取GPU节点状态: {result.stderr}", file=sys.stderr)
            return []
        
        nodes = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
            
            node_name = parts[0]
            gpu_info = parts[1]
            cpu_info = parts[2]
            
            gpu_type = 'Unknown'
            gpu_total = 0
            if gpu_info.startswith('gpu:'):
                gpu_parts = gpu_info[4:].split(':')
                if len(gpu_parts) >= 2:
                    gpu_type = gpu_parts[0]
                    gpu_count_str = gpu_parts[1].split('(')[0]
                    try:
                        gpu_total = int(gpu_count_str)
                    except ValueError:
                        gpu_total = 1
            
            cpu_parts = cpu_info.split('/')
            if len(cpu_parts) >= 4:
                cpu_allocated = int(cpu_parts[0])
                cpu_idle = int(cpu_parts[1])
                cpu_total = int(cpu_parts[3])
            else:
                cpu_allocated = 0
                cpu_idle = 0
                cpu_total = 0
            
            idle_ratio = cpu_idle / cpu_total if cpu_total > 0 else 0
            
            nodes.append({
                'node': node_name,
                'gpu_type': gpu_type,
                'gpu_total': gpu_total,
                'gpu_allocated': 0,
                'gpu_idle': gpu_total,
                'cpu_allocated': cpu_allocated,
                'cpu_idle': cpu_idle,
                'cpu_total': cpu_total,
                'idle_ratio': idle_ratio
            })
        
        return _fill_gpu_allocation(nodes)
    
    except subprocess.TimeoutExpired:
        print("[sbatch_wrapper] ⚠️  查询GPU节点超时", file=sys.stderr)
        return []
    except Exception as e:
        print(f"[sbatch_wrapper] ⚠️  查询GPU节点时出错: {e}", file=sys.stderr)
        return []


def _fill_gpu_allocation(nodes):
    """
    查询每个节点的实际GPU分配情况。
    """
    if not nodes:
        return nodes
    
    node_dict = {n['node']: n for n in nodes}
    
    try:
        result = subprocess.run(
            ['squeue', '-t', 'RUNNING', '-o', '%N %b', '--noheader'],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode != 0:
            return nodes
        
        gpu_usage = {}
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            node_name = parts[0]
            gres_info = parts[1] if len(parts) > 1 else ''
            
            if 'gres/gpu' in gres_info:
                match = re.search(r'gres/gpu:(\w*):?(\d+)', gres_info)
                if match:
                    gpu_count = int(match.group(2))
                else:
                    gpu_count = 1
                
                gpu_usage[node_name] = gpu_usage.get(node_name, 0) + gpu_count
        
        for node_name, allocated in gpu_usage.items():
            if node_name in node_dict:
                node_dict[node_name]['gpu_allocated'] = allocated
                node_dict[node_name]['gpu_idle'] = max(0, node_dict[node_name]['gpu_total'] - allocated)
        
        return nodes
    
    except Exception:
        return nodes


def get_cpu_node_status():
    """
    查询CPU分区的节点状态（非GPU分区）。
    """
    try:
        result = subprocess.run(
            ['sinfo', '-p', 'comp', '-N', '-o', '%N %C', '--noheader'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return []
        
        nodes = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            node_name = parts[0]
            cpu_info = parts[1]
            
            cpu_parts = cpu_info.split('/')
            if len(cpu_parts) >= 4:
                cpu_allocated = int(cpu_parts[0])
                cpu_idle = int(cpu_parts[1])
                cpu_total = int(cpu_parts[3])
            else:
                cpu_allocated = 0
                cpu_idle = 0
                cpu_total = 0
            
            idle_ratio = cpu_idle / cpu_total if cpu_total > 0 else 0
            
            nodes.append({
                'node': node_name,
                'cpu_allocated': cpu_allocated,
                'cpu_idle': cpu_idle,
                'cpu_total': cpu_total,
                'idle_ratio': idle_ratio
            })
        
        return nodes
    
    except Exception:
        return []


def select_best_gpu_node(nodes, prefer_gpu_type=None):
    """
    从可用节点中选择最佳的GPU节点（优先选择有空闲GPU的节点）。
    """
    if not nodes:
        return None
    
    if prefer_gpu_type:
        prefer_type = prefer_gpu_type.upper()
        filtered_nodes = [
            n for n in nodes 
            if n['gpu_type'].upper().startswith(prefer_type) and n['gpu_idle'] > 0
        ]
        if filtered_nodes:
            filtered_nodes.sort(key=lambda n: (-n['gpu_idle'], -n['idle_ratio']))
            return filtered_nodes[0]
    
    gpu_priority = {
        'A100': 100,
        'L40S': 80,
        'A40': 60,
        'A10': 40,
        'T4': 20,
    }
    
    available_nodes = [n for n in nodes if n['gpu_idle'] > 0]
    
    if not available_nodes:
        available_nodes = [n for n in nodes if n['cpu_idle'] > 0]
    
    if not available_nodes:
        available_nodes = nodes
    
    def sort_key(node):
        gpu_type = node['gpu_type'].upper()
        base_type = gpu_type.split('-')[0]
        priority = gpu_priority.get(base_type, 0)
        return (-priority, -node.get('gpu_idle', 0), -node['idle_ratio'])
    
    available_nodes.sort(key=sort_key)
    return available_nodes[0] if available_nodes else None


def select_best_cpu_node(nodes):
    """
    从可用节点中选择最佳的CPU节点。
    """
    if not nodes:
        return None

    available_nodes = [n for n in nodes if n['cpu_idle'] > 0]

    if not available_nodes:
        available_nodes = nodes

    available_nodes.sort(key=lambda n: (-n['cpu_idle'], -n['idle_ratio'], n['node']))
    return available_nodes[0] if available_nodes else None


def find_completely_idle_gpu_nodes(prefer_gpu_type=None):
    """
    查找完全空闲的GPU节点（所有CPU和GPU都未被占用）。
    使用此方法可以绕过Fairshare调度，直接指定节点保证100%立即分配。

    Returns:
        list: 完全空闲的节点列表，每个节点包含 node, gpu_type, gpu_total, cpu_total
    """
    try:
        # 获取预留节点列表（sinfo -p gpu -N 不显示 resv 状态，需要用 sinfo -a 补充）
        reserved_nodes = set()
        result_resv = subprocess.run(
            ['sinfo', '-a', '-o', '%N %a', '--noheader'],
            capture_output=True,
            text=True,
            timeout=15
        )
        if result_resv.returncode == 0:
            for line in result_resv.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[1] == 'resv':
                    # 解析节点列表，如 m3a[105-107] 或 m3a105
                    node_expr = parts[0]
                    if '[' in node_expr:
                        # 格式: m3a[105-107]
                        prefix = node_expr.split('[')[0]
                        range_part = node_expr.split('[')[1].rstrip(']')
                        for r in range_part.split(','):
                            if '-' in r:
                                start, end = r.split('-')
                                for i in range(int(start), int(end) + 1):
                                    reserved_nodes.add(f"{prefix}{i}")
                            else:
                                reserved_nodes.add(f"{prefix}{r}")
                    else:
                        reserved_nodes.add(node_expr)

        # 查询GPU节点的CPU分配状态 (A/I/O/T 格式)
        # %N = 节点名, %a = 状态, %C = CPU分配 (allocated/idle/other/total), %e = 内存, %G = GPU信息
        result = subprocess.run(
            ['sinfo', '-p', 'gpu', '-N', '-o', '%N %a %C %e %G', '--noheader'],
            capture_output=True,
            text=True,
            timeout=15
        )

        if result.returncode != 0:
            print(f"[sbatch_wrapper] ⚠️  查询空闲节点失败: {result.stderr}", file=sys.stderr)
            return []

        idle_nodes = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            node_name = parts[0]
            node_state = parts[1]  # up, down, drain, etc.
            cpu_info = parts[2]    # A/I/O/T 格式

            # 跳过非UP状态的节点
            if node_state != 'up':
                continue

            # 跳过预留状态的节点
            if node_name in reserved_nodes:
                continue

            # 解析CPU分配状态
            cpu_parts = cpu_info.split('/')
            if len(cpu_parts) >= 4:
                cpu_allocated = int(cpu_parts[0])
                cpu_idle = int(cpu_parts[1])
                cpu_total = int(cpu_parts[3])
            else:
                continue

            # 只选择完全空闲的节点（所有CPU都idle）
            if cpu_allocated != 0 or cpu_idle != cpu_total:
                continue

            # 解析GPU信息
            gpu_type = "unknown"
            gpu_total = 0
            if len(parts) >= 5 and parts[4]:
                gres_info = parts[4]
                # 格式: gpu:xxx:Y 或 gres/gpu:xxx:Y
                match = re.search(r'gpu:(\w+):?(\d+)', gres_info)
                if match:
                    gpu_type = match.group(1)
                    gpu_total = int(match.group(2))
                else:
                    gpu_total = 1

            # 过滤GPU类型
            if prefer_gpu_type:
                prefer_type = prefer_gpu_type.upper()
                if not gpu_type.upper().startswith(prefer_type):
                    continue

            idle_nodes.append({
                'node': node_name,
                'gpu_type': gpu_type,
                'gpu_total': gpu_total,
                'cpu_total': cpu_total
            })

        # 按GPU类型优先级排序
        gpu_priority = {'H100': 150, 'A100': 100, 'L40S': 80, 'A40': 60, 'A10': 40, 'T4': 20}
        idle_nodes.sort(key=lambda n: (-gpu_priority.get(n['gpu_type'].upper(), 0), -n['gpu_total']))

        return idle_nodes

    except subprocess.TimeoutExpired:
        print("[sbatch_wrapper] ⚠️  查询空闲节点超时", file=sys.stderr)
        return []
    except Exception as e:
        print(f"[sbatch_wrapper] ⚠️  查询空闲节点时出错: {e}", file=sys.stderr)
        return []


def print_gpu_status(nodes, selected_node=None):
    """打印GPU节点状态信息。"""
    if not nodes:
        print("[sbatch_wrapper] ⚠️  未获取到GPU节点信息", file=sys.stderr)
        return
    
    print("\n[sbatch_wrapper] 📊 GPU节点状态:", file=sys.stderr)
    print("-" * 90, file=sys.stderr)
    print(f"{'节点':<12} {'GPU类型':<10} {'GPU空闲/总数':<14} {'CPU空闲/总数':<15} {'空闲率':<10}", file=sys.stderr)
    print("-" * 90, file=sys.stderr)
    
    sorted_nodes = sorted(nodes, key=lambda n: (n['gpu_type'], -n.get('gpu_idle', 0), -n['idle_ratio']))
    
    for node in sorted_nodes:
        selected_marker = "✅ " if selected_node and node['node'] == selected_node['node'] else "   "
        gpu_ratio = f"{node.get('gpu_idle', 0)}/{node['gpu_total']}"
        cpu_ratio = f"{node['cpu_idle']}/{node['cpu_total']}"
        idle_pct = f"{node['idle_ratio']*100:.1f}%"
        print(f"{selected_marker}{node['node']:<10} {node['gpu_type']:<10} {gpu_ratio:<14} {cpu_ratio:<15} {idle_pct:<10}", file=sys.stderr)
    
    print("-" * 90, file=sys.stderr)
    
    if selected_node:
        print(f"[sbatch_wrapper] 🎯 已选择节点: {selected_node['node']} (GPU空闲: {selected_node.get('gpu_idle', 0)}, 类型: {selected_node['gpu_type']})", file=sys.stderr)


def main():
    # Check if running as a hook (has JSON input from stdin)
    input_data = read_input()
    is_hook_mode = bool(input_data)

    # Always log when hook is called (even if no input)
    print(f"[sbatch_wrapper] Hook被调用 - is_hook_mode: {is_hook_mode}, stdin_isatty: {sys.stdin.isatty()}", file=sys.stderr)

    if is_hook_mode:
        # Hook mode: intercept command and wrap with sbatch
        command = input_data.get("command", "").strip()
        working_dir = (
            input_data.get("working_directory") or
            input_data.get("cwd") or
            input_data.get("workingDirectory") or
            os.getcwd()
        )

        # Debug output
        print(f"[sbatch_wrapper] 命令: {command[:200]}", file=sys.stderr)
        print(f"[sbatch_wrapper] 工作目录: {working_dir}", file=sys.stderr)

        # Check if this is a Python script command
        is_python_script = is_python_script_command(command)
        has_sbatch = has_sbatch_in_command(command)

        print(f"[sbatch_wrapper] 是 Python 脚本命令: {is_python_script}", file=sys.stderr)
        print(f"[sbatch_wrapper] 包含 sbatch: {has_sbatch}", file=sys.stderr)

        # Always allow in hook mode - let commands proceed
        if has_sbatch:
            # Already wrapped, allow as-is
            output = {
                "continue": True,
                "permission": "allow"
            }
        elif command:
            # Not a Python script, allow as-is
            output = {
                "continue": True,
                "permission": "allow"
            }
        else:
            # No command, allow as-is
            output = {
                "continue": True,
                "permission": "allow"
            }

        # Output JSON response for hook mode
        print(json.dumps(output, ensure_ascii=False))
        return

    # Direct mode: execute command via sbatch
    # Parse command from command line arguments
    args = sys.argv[1:]
    use_gpu = False
    use_fit = False         # 使用 fit 分区
    num_gpus = 1            # GPU数量，默认为1
    time_limit = None
    prefer_gpu_type = None  # 首选GPU类型
    list_gpu_only = False   # 仅列出GPU状态
    nodelist = None         # 指定节点列表
    partition = None        # 指定分区

    # Simple argument parser for wrapper options
    remaining_args = []
    i = 0
    while i < len(args):
        if args[i] == "--gpu":
            use_gpu = True
            i += 1
        elif args[i] == "--fit":
            use_fit = True
            use_gpu = True  # fit 分区自动启用 GPU
            i += 1
        elif args[i] == "--gpu-type" and i + 1 < len(args):
            prefer_gpu_type = args[i+1]
            use_gpu = True  # 指定GPU类型自动启用GPU
            i += 2
        elif args[i] == "--num-gpus" and i + 1 < len(args):
            num_gpus = int(args[i+1])
            use_gpu = True  # 指定GPU数量自动启用GPU
            i += 2
        elif args[i] == "--list-gpu":
            list_gpu_only = True
            i += 1
        elif args[i] == "--time" and i + 1 < len(args):
            time_limit = args[i+1]
            i += 2
        elif args[i] == "--nodelist" and i + 1 < len(args):
            nodelist = args[i+1]
            print(f"[sbatch_wrapper] 🎯 指定节点列表: {nodelist}", file=sys.stderr)
            i += 2
        elif args[i] == "--partition" and i + 1 < len(args):
            partition = args[i+1]
            print(f"[sbatch_wrapper] 🎯 指定分区: {partition}", file=sys.stderr)
            i += 2
        else:
            remaining_args.append(args[i])
            i += 1
    
    # 如果只是列出GPU状态
    if list_gpu_only:
        nodes = get_gpu_node_status()
        print_gpu_status(nodes)
        return
    
    original_command = parse_command(remaining_args)
    
    if not original_command:
        print("[sbatch_wrapper] ❌ 错误: 未提供要执行的命令", file=sys.stderr)
        sys.exit(1)
    
    # Setup paths
    log_dir = Path("/home/wlia0047/ar57/wenyu/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Clean up ALL .sh files in log directory before starting as requested
    print(f"[sbatch_wrapper] 🗑️  清理日志目录中的所有残留 .sh 脚本...", file=sys.stderr)
    sh_cleanup_count = 0
    for sh_file in log_dir.glob("*.sh"):
        try:
            sh_file.unlink()
            sh_cleanup_count += 1
        except Exception:
            pass
    
    if sh_cleanup_count > 0:
        print(f"[sbatch_wrapper] ✅ 已全局清理 {sh_cleanup_count} 个 .sh 脚本", file=sys.stderr)

    # Extract script name for logging
    script_match = re.search(r'([\w-]+)\.py', original_command)
    if script_match:
        full_script_name = script_match.group(1)
        base_name = re.sub(r'_\d{8}_\d{6}$', '', full_script_name)
        base_name = re.sub(r'_v\d+$', '', base_name)
        script_name = base_name if base_name else full_script_name
    else:
        script_name = "job"
    
    # 检测并取消同名脚本的运行中任务
    print(f"[sbatch_wrapper] 🔍 检查是否有同名脚本 '{script_name}' 的运行中任务...", file=sys.stderr)
    running_jobs = find_running_jobs_by_name(script_name)
    
    if running_jobs:
        print(f"[sbatch_wrapper] ⚠️  发现 {len(running_jobs)} 个同名脚本的运行中任务:", file=sys.stderr)
        for job in running_jobs:
            print(f"[sbatch_wrapper]    - Job ID: {job['job_id']}, 名称: {job['name']}, 状态: {job['status']}", file=sys.stderr)
        
        job_ids_to_cancel = [job['job_id'] for job in running_jobs]
        print(f"[sbatch_wrapper] 🔄 正在取消这些任务...", file=sys.stderr)
        success, fail = cancel_jobs(job_ids_to_cancel)
        
        if success > 0:
            print(f"[sbatch_wrapper] ✅ 已取消 {success} 个任务", file=sys.stderr)
        if fail > 0:
            print(f"[sbatch_wrapper] ⚠️  {fail} 个任务取消失败", file=sys.stderr)
        
        # 等待任务完全取消
        time.sleep(2)
    else:
        print(f"[sbatch_wrapper] ✅ 未发现同名脚本的运行中任务", file=sys.stderr)
    
    # Clean up old log files for THIS script name
    print(f"[sbatch_wrapper] 🗑️  清理与 '{script_name}' 相关的旧日志文件...", file=sys.stderr)
    cleanup_count = 0

    # Pattern 1: Clean up files matching current script name (e.g., 09_generate_noisy_queries_v31_*)
    for old_file in log_dir.glob(f"{script_name}_*"):
        if old_file.suffix in ['.log', '.err']:
            try:
                old_file.unlink()
                cleanup_count += 1
            except Exception as e:
                print(f"[sbatch_wrapper] ⚠️  删除文件失败 {old_file.name}: {e}", file=sys.stderr)

    # Pattern 2: For stage 9 scripts, also clean up old pattern files (without version suffix)
    # This handles the legacy pattern from SLURM scripts: 09_generate_noisy_queries_<job_id>.log
    if script_name.startswith('09_generate_noisy_queries'):
        base_pattern = script_name.split('_v')[0] if '_v' in script_name else '09_generate_noisy_queries'
        for old_file in log_dir.glob(f"{base_pattern}_*"):
            # Only clean if it doesn't match current script name (avoid duplicate work)
            if not old_file.name.startswith(script_name):
                if old_file.suffix in ['.log', '.err']:
                    try:
                        old_file.unlink()
                        cleanup_count += 1
                    except Exception as e:
                        print(f"[sbatch_wrapper] ⚠️  删除文件失败 {old_file.name}: {e}", file=sys.stderr)

    if cleanup_count > 0:
        print(f"[sbatch_wrapper] ✅ 已清理 {cleanup_count} 个旧日志文件", file=sys.stderr)

    timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Using %j for SLURM job ID in the output/error filenames
    log_pattern = log_dir / f"{script_name}_%j.log"
    err_pattern = log_dir / f"{script_name}_%j.err"
    # The temporary submission script still uses a timestamp to be unique
    script_file = log_dir / f"submit_{script_name}_{timestamp_suffix}.sh"

    # Get current directory
    current_dir = os.getcwd() or "/home/wlia0047/ar57/wenyu"

    # 节点分配：GPU任务使用 gpu 分区，CPU任务使用 comp 分区，由SLURM自动分配
    # fit 分区用于长时间运行的 GPU 任务，需要 fitq QOS
    # 自定义分区优先于默认逻辑
    qos_option = ""
    if partition:
        partition_option = f"#SBATCH -p {partition}"
        print(f"[sbatch_wrapper] 🎯 使用自定义分区: {partition}", file=sys.stderr)
    elif use_fit:
        partition_option = "#SBATCH -p fit"
        qos_option = "#SBATCH --qos=fitq"
        print(f"[sbatch_wrapper] 🎯 使用 fit 分区 + fitq QOS", file=sys.stderr)
    elif use_gpu:
        partition_option = "#SBATCH -p gpu"
    else:
        partition_option = "#SBATCH -p comp"

    # GPU任务：指定GPU类型偏好（如果提供）
    constraint_option = ""
    if use_gpu and prefer_gpu_type:
        constraint_option = f"#SBATCH --constraint={prefer_gpu_type}"
        print(f"[sbatch_wrapper] 🎯 GPU类型约束: --constraint={prefer_gpu_type}", file=sys.stderr)

    # 指定节点列表
    nodelist_option = ""
    if nodelist:
        nodelist_option = f"#SBATCH --nodelist={nodelist}"
        print(f"[sbatch_wrapper] 🎯 指定节点: {nodelist}", file=sys.stderr)

    # Create sbatch script
    memory_allocation = "#SBATCH --mem=64G" if is_python_script_command(original_command) else ""
    gpu_allocation = f"#SBATCH --gres=gpu:{num_gpus}" if use_gpu else ""
    ntasks_option = f"#SBATCH --ntasks={num_gpus}" if use_gpu and num_gpus > 1 else ""
    time_allocation = f"#SBATCH --time={time_limit}" if time_limit else ""

    script_content = f"""#!/bin/bash
#SBATCH --output={log_pattern}
#SBATCH --error={err_pattern}
{partition_option}
{qos_option}
{memory_allocation}
{gpu_allocation}
{ntasks_option}
{time_allocation}
{constraint_option}
{nodelist_option}
set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/stark
cd "{current_dir}"
export PYTHONUNBUFFERED=1
{original_command}
"""
    
    script_file.write_text(script_content, encoding='utf-8')
    script_file.chmod(0o755)

    # Submit the job
    print("[sbatch_wrapper] 提交作业到 SLURM...")

    try:
        result = subprocess.run(
            ['sbatch', str(script_file)],
            capture_output=True,
            text=True,
            check=True
        )

        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if not match:
            print("[sbatch_wrapper] ❌ 提交 sbatch 作业失败: 无法解析作业 ID")
            sys.exit(1)

        job_id = match.group(1)
        log_file = log_dir / f"{script_name}_{job_id}.log"
        err_file = log_dir / f"{script_name}_{job_id}.err"

    except subprocess.CalledProcessError as e:
        print(f"[sbatch_wrapper] ❌ 提交 sbatch 作业失败: {e}")
        if e.stderr:
            print(e.stderr)
        sys.exit(1)

    print(f"[sbatch_wrapper] ✅ 作业已提交，Job ID: {job_id}")
    print(f"[sbatch_wrapper] 日志文件: {log_file}")
    print(f"[sbatch_wrapper] 错误文件: {err_file}")
    print("[sbatch_wrapper] 开始监控作业状态 (Ctrl+C 停止监控，作业将继续运行)...")

    # Monitor logs
    job_completed = False
    try:
        monitor_logs(str(log_file), str(err_file), job_id)
        job_completed = True
    except KeyboardInterrupt:
        print("\n[sbatch_wrapper] 监控已停止，作业将继续在后台运行...")

    print(f"[sbatch_wrapper] ✅ 作业已完成 (Job ID: {job_id})")
    if log_file.exists() or err_file.exists():
        print(f"[sbatch_wrapper] 查看完整日志: tail -f {log_file} {err_file}")
    
    # Clean up temporary script file after job completion (keep log files for inspection)
    if job_completed:
        cleaned_files = []
        failed_files = []

        # Only clean up script file, keep log and error files
        if script_file.exists():
            try:
                script_file.unlink()
                cleaned_files.append(f"脚本文件: {script_file.name}")
            except Exception as e:
                failed_files.append(f"脚本文件: {e}")

        # Print cleanup results
        if cleaned_files:
            print(f"[sbatch_wrapper] 🗑️  已清理临时文件: {', '.join(cleaned_files)}", file=sys.stderr)
        if failed_files:
            print(f"[sbatch_wrapper] ⚠️  清理文件失败: {', '.join(failed_files)}", file=sys.stderr)


if __name__ == "__main__":
    main()

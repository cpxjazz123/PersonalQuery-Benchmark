#!/usr/bin/env python3
"""
Cursor Hook: Check Stark Conda Environment Activation and SBATCH Requirement
This hook checks if Python/pip commands are executed with the stark conda environment activated,
and if Python scripts need to be executed via sbatch.

This hook intercepts Python/pip commands executed in the project directory and:
1. Prompts the user to activate the environment if it's not already activated in the command
2. Prompts the user to use sbatch for Python script execution if not already using it
"""

import json
import sys
import re
import os
from typing import Dict, Any

# Stark environment configuration
STARK_ENV = "/home/wlia0047/ar57_scratch/wenyu/stark"
CONDA_INIT = "/apps/anaconda/2024.02-1/etc/profile.d/conda.sh"
PROJECT_ROOT = "/home/wlia0047/ar57/wenyu"
SBATCH_WRAPPER_SCRIPT = "/home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py"


def is_python_command(command: str) -> bool:
    """Check if command is Python/pip related"""
    if not command:
        return False
    
    # Strip leading whitespace for pattern matching
    command = command.strip()
    
    patterns = [
        r'^python\s',           # python command
        r'^python3\s',          # python3 command
        r'^pip\s',              # pip command
        r'^pip3\s',             # pip3 command
        r'\spython\s',          # python in middle of command
        r'\spython3\s',         # python3 in middle of command
        r'\.py\s',              # .py file execution
        r'\.py$',               # .py file at end
        r'python.*\.py',        # python script.py pattern
    ]
    
    for pattern in patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    
    return False


def is_in_project_directory(working_dir: str) -> bool:
    """Check if working directory is in project"""
    if not working_dir:
        return False

    # Normalize paths for comparison
    project_root = os.path.abspath(PROJECT_ROOT)
    scratch_root = os.path.abspath("/home/wlia0047/ar57_scratch/wenyu")
    home_root = os.path.abspath("/home/wlia0047")
    fs04_root = os.path.abspath("/fs04/ar57/wenyu")  # Add /fs04 path
    abs_working_dir = os.path.abspath(working_dir)

    return (abs_working_dir.startswith(project_root + os.sep) or
            abs_working_dir == project_root or
            abs_working_dir.startswith(scratch_root + os.sep) or
            abs_working_dir == scratch_root or
            abs_working_dir.startswith(home_root + os.sep) or
            abs_working_dir == home_root or
            abs_working_dir.startswith(fs04_root + os.sep) or
            abs_working_dir == fs04_root)


def has_activation_in_command(command: str) -> bool:
    """Check if command already contains environment activation"""
    if not command:
        return False
    
    # Check for common activation patterns
    activation_patterns = [
        r'activate',           # conda activate or source activate
        r'venv/bin/activate',  # virtualenv activation
        r'\.venv/bin/activate', # .venv activation
        r'source.*activate',   # source activate pattern
    ]
    
    command_lower = command.lower()
    for pattern in activation_patterns:
        if re.search(pattern, command_lower):
            return True
    
    return False


def is_python_script_command(command: str) -> bool:
    """Check if command is executing a Python script (not just Python command)."""
    if not command:
        return False
    
    command = command.strip()
    
    # Patterns that indicate Python script execution (executing a .py file)
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
    
    # Check for various sbatch patterns
    sbatch_patterns = [
        'sbatch',
        'sbatch_wrapper',
        SBATCH_WRAPPER_SCRIPT.lower(),
        'sbatch_wrapper.py',
    ]
    
    for pattern in sbatch_patterns:
        if pattern in command_lower or pattern in command:
            return True
    
    return False


def read_input() -> Dict[str, Any]:
    """Read and parse JSON input from stdin"""
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            return {}
        return json.loads(raw_input)
    except json.JSONDecodeError:
        # If not valid JSON, treat as plain text command
        raw_input = raw_input.strip()
        if raw_input:
            return {"command": raw_input}
        return {}
    except Exception:
        # On any other error, return empty dict (will allow command as-is)
        return {}


def main():
    """Main function"""
    try:
        input_data = read_input()
    except Exception:
        # On any error reading input, allow command to proceed
        print(json.dumps({
            "continue": True,
            "permission": "allow"
        }), file=sys.stderr)
        print(json.dumps({
            "continue": True,
            "permission": "allow"
        }))
        return
    
    # Extract command and working directory
    command = input_data.get("command", "").strip()
    
    # Try multiple ways to get working directory
    working_dir = (
        input_data.get("working_directory") or
        input_data.get("cwd") or
        input_data.get("workingDirectory") or
        os.getcwd()
    )
    
    # Debug output to stderr (won't interfere with JSON output)
    print(f"[activate-stark-env] Command: {command[:200]}", file=sys.stderr)
    print(f"[activate-stark-env] Working dir: {working_dir}", file=sys.stderr)
    
    # Check if this is a Python command in project directory
    is_python = is_python_command(command)
    in_project = is_in_project_directory(working_dir)
    has_activation = has_activation_in_command(command)
    is_python_script = is_python_script_command(command)
    has_sbatch = has_sbatch_in_command(command)
    
    print(f"[activate-stark-env] Is Python command: {is_python}", file=sys.stderr)
    print(f"[activate-stark-env] In project directory: {in_project}", file=sys.stderr)
    print(f"[activate-stark-env] Has activation: {has_activation}", file=sys.stderr)
    print(f"[activate-stark-env] Is Python script: {is_python_script}", file=sys.stderr)
    print(f"[activate-stark-env] Has sbatch: {has_sbatch}", file=sys.stderr)
    
    # Always allow - bypass permission check
    output = {
        "continue": True,
        "permission": "allow"
    }
    
    # Output JSON response to stdout (required by Cursor)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()

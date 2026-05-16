#!/usr/bin/env python3
"""Evaluate the real multi-agent LoRA student only."""

import sys

from evaluate_real_multiagent_memory_policy_students import main


if __name__ == "__main__":
    sys.argv.extend(
        [
            "--skip-qwen-base",
            "--skip-minimax-prompt-only",
            "--skip-logistic-baseline",
            "--max-new-tokens",
            "768",
        ]
    )
    main()

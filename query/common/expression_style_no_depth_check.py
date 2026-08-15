"""Compatibility shim: iter #169 renamed consumer-side references from syntax_depth to expression_style, but the underlying implementation module kept its filename. This shim re-exports everything from the original module so cat-specific scripts (e.g. 04_generate_by_syntax_depth_no_depth_check_10_<Cat>.py) that import `from common.expression_style_no_depth_check import main` keep working without duplicating the module body.

iter #198 finding: 04_query/common/expression_style_no_depth_check.py was missing — Stage 04 scripts for Grocery_and_Gourmet_Food and Pet_Supplies (and re-runs of Baby_Products) fail with ModuleNotFoundError. This shim restores importability by forwarding all public symbols from syntax_depth_no_depth_check.

Note: per CLAUDE.md Rule 7 (no fallback), this is a rename-compatibility layer, NOT a behavior-fallback — the actual implementation is unchanged.
"""

from __future__ import annotations

# Re-export every public symbol from the original module so that
# `from common.expression_style_no_depth_check import <name>` resolves to the
# same object as `from common.syntax_depth_no_depth_check import <name>`.
from common.syntax_depth_no_depth_check import (  # noqa: F401
    main,
    process_one_user,
    load_user_syntax_depths,
    build_syntax_depth_prompt,
    build_syntax_depth_prewarm_prompt,
    build_user_tasks,
    prewarm_syntax_depth_cache,
    parse_syntax_depth_response,
    call_llm_no_empty_retry,
    count_words,
    log,
    get_category_config,
    load_minimax_client,
    ThreadPoolExecutor,
    as_completed,
    Path,
)

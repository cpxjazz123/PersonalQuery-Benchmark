"""Common helpers and configs shared across 06_query stage-6 scripts.

Layout:
  - attribute_helpers.py: attribute extraction, normalization, 5-attr validation
  - llm_runner.py: MiniMax LLM client lifecycle
  - syntax_depth_no_depth_check.py: shared body of
    06_generate_by_syntax_depth_no_depth_check_10_<Cat>.py
  - config.py: loaders for 06_query_config.json / query_config.json
  - 06_query_config.json: per-category paths (attr_density / attr_values)
  - query_config.json: global query settings (num_users_to_test / max_workers)
"""

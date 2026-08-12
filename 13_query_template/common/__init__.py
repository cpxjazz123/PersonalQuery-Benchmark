"""Common helpers and configs shared across 13_query_template stage-14 scripts.

Layout:
  - config.py: loaders for 13_query_template_config.json (template / paths)
  - template_query.py: shared body of 14_generate_query_template_<Cat>.py
    (template rendering + 5-attr validation + incremental write)
  - user_data_loader.py: load query_template.json -> 08 evaluate_* expected format
  - cache_path_overrides.py: 8 cache path/loader overrides used by 14 driver
  - generate_template_cache.py: per-retriever cache generator (clean + noisy)
  - eval_template_driver.py: importlib-load 08 eval + monkey-patch + run 4 evaluators
  - template_noisy_query.py: noisy query generator (07 lambdamart_userbased rule-based)
"""

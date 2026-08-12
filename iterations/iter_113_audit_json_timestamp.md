# Iteration #113 — Audit JSON `generated_at` timestamp + dashboard display

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` (add timestamp) + `PersoanlQuery/_generate_audit_dashboard.py` (render timestamp)
**prior**: dashboard already showed status / summary / provenance; this iter adds audit run time

## §A 审稿意见

iter #108–#112 dashboard 显示 17 claim 状态 + provenance,但 reviewer 翻
dashboard 不知道 **这个 audit 是什么时候跑的**。 audit JSON 文件 mtime
可以给 stale/dirty 提示,但 reviewer 大概率直接看 HTML 不去查文件 mtime。

需求: audit JSON 含 `generated_at` ISO 8601 UTC timestamp,dashboard header
显示这个时间。 Reviewer 看到 "2026-07-20T21:26:10+00:00" 立刻知道:
1. dashboard 是 fresh regenerate,还是 stale (上 git clone 后没重新跑过)
2. 如果 audit 时间与最近 commit 时间差距大 → 可能 commit 后还没重新 audit

## §B 本轮 (iter #113) 改动

### §B.1 paper_claims_audit.py

```python
import json
import os
from datetime import datetime, timezone  # iter #113
from pathlib import Path
...

out_doc = {
    "audit_target": "Paper §2.1-§3 RQ1-4 + Table 1-3 (file existence + value validation, iter #93; CLI iter #104)",
    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # iter #113
    "n_claims": len(CLAIMS),
    "n_audited": len(selected),
    ...
}
```

`datetime.now(timezone.utc)` 给 timezone-aware UTC time,`isoformat(timespec="seconds")`
输出 `2026-07-20T21:26:10+00:00` 格式 (ISO 8601)。

### §B.2 _generate_audit_dashboard.py

`_render_html` 读 `doc.get("generated_at", "")`,若有则 render:

```python
generated_at_html = f"<div class='generated-at'><strong>generated_at:</strong> <code>{generated_at}</code></div>" if generated_at else ""
```

+ CSS `.generated-at` (小字灰色)。

## §C 关键改动点

1. **UTC timezone-aware**: 不依赖本地 timezone,任何 reviewer 看到同一个时间。
2. **ISO 8601**: 机器可读 + 人类可读,diff/git blame 都能识别。
3. **Backward compatible**: dashboard 如果 JSON 没有 `generated_at` (e.g. 旧版
   audit script 生成的 JSON) 不会 crash — `if generated_at else ""` 跳过。
4. **header 位置**: 放在 audit_target 下方,filter_note 上方 — reviewer 第一眼
   看到 audit 是关于什么的,然后立刻看到 "什么时候跑的",然后是 filter。

## §D 测试

```bash
$ python3 -c "
import json
doc = json.load(open('result/personal_query/iterations/paper_claims_audit.json'))
print('generated_at:', doc.get('generated_at'))
"
generated_at: 2026-07-20T21:26:10+00:00

$ grep -oE "generated_at:</strong> <code>[^<]+</code>" \
    result/personal_query/iterations/paper_claims_audit_dashboard.html
generated_at:</strong> <code>2026-07-20T21:26:10+00:00</code>
```

Full audit CI 测试 PASS (regenerate + 4 regression + dashboard)。

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py` (+1 import + 1 field)
- 模块: `PersoanlQuery/_generate_audit_dashboard.py` (+1 line render + 1 CSS)
- 验证: JSON 含 generated_at + dashboard 显示 ✓
- 命令: `bash PersoanlQuery/_run_audit_ci.sh` (full ~2s) → AUDIT CI PASSED

## §F 与 audit infra timeline 的关系

dashboard 演进:
- **#108** — self-contained HTML + sortable table
- **#109** — rel_delta percent fix
- **#110** — vc status color coding
- **#112** — provenance panels (expected_outputs + code_evidence)
- **#113** — generated_at timestamp ✓ 本轮

每一轮都加 reviewer-visible 信息 layer (severity colors, provenance, now
temporal context)。 现在 reviewer 看 dashboard 能回答:
- "audit 跑了吗?" → 是,generated_at 显示
- "哪些 claim verified?" → 6 个 green badge
- "哪些 mismatch?" → 4 个 orange badge + 11.56% / 30.34% / 78.38% / 13.40%
- "mismatch 在哪些文件?" → expected_outputs globs
- "实现 scripts?" → code_evidence script:function refs
- "paper caveat?" → paper_text_caveat yellow box

## §G backlog (remaining)

- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU) — flip 5 unverified → verified
- **iter #88** — Stage 6/9 re-run (1.5-3 h GPU) — flip 1 degenerate → verified
- **iter #101** — re-run audit + update Table 1/3 footnotes (post-#87+#88)
- **iter #102** — Re-collect Table 2 human-eval with larger sample
- **iter #114** — `--diff` mode in audit CLI: compare current claim state vs frozen baseline JSON (reviewer can spot silent flips without running regression test)
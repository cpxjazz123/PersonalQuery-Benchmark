# Iteration #03 — Rule 7 整改(预期外失败修复)

**日期**: 2026-07-20
**scope**: 修复违反 Rule 7 的静默 fallback / bare except
**关联 stage**: 05(inject_noisy) / 01(preference_extraction)

---

## §A 冗余代码发现

**无本轮新增冗余。**

---

## §B 不合理逻辑发现

### [P0] Stage 05 `common.py` JSON fallback 静默吞数据(违反 Rule 7) ✅ 已修复

- **文件**:`05_inject_noisy/common/common.py:25`
- **问题**:`except json.JSONDecodeError: pass` 静默跳过,调用方无感知
- **分析**:逐行解析是合理的 fallback 恢复策略,但 `pass` 让人误以为成功加载了完整数据
- **处理**:改为 `log(f"JSON 整体解析失败({e}), 尝试逐行解析")` + `log(f"跳过无法解析的 JSON 片段: ...")`

### [P0] debug_stage1.py 4 处 bare `except:`(违反 Rule 7) ✅ 已修复

| 行号 | 原写法 | 改为 |
|------|--------|------|
| 20 | `except:` | `except (ValueError, TypeError):` |
| 200 | `except:` | `except (json.JSONDecodeError, ValueError, TypeError):` |
| 224 | `except:` | `except (json.JSONDecodeError, ValueError, TypeError):` |
| 271 | `except:` | `except (json.JSONDecodeError, ValueError, TypeError):` |

- **未改**:Stage 01 `01_extract_preferences_*.py` 中的 `except Exception: details = {}` 等——这些是**预期内 fallback**(字段类型不确定时返回空值),符合 Rule 7,不改。

---

## §C 可合并文件发现

**无本轮变更。**

---

## §D 本轮已实施改动

- **修改**:
  - `05_inject_noisy/common/common.py`:JSON fallback 加 log
  - `01_preference_extraction/debug_stage1.py`:4 处 bare `except:` 改为具体异常类型

---

## §E 验证

- `python3 -m py_compile 05_inject_noisy/common/common.py` PASS
- `python3 -m py_compile 01_preference_extraction/debug_stage1.py` PASS
- Stage 01 其余 `except Exception: details = {}` 经判定为预期内 fallback,保持不变

---

## §F 下轮建议

1. **[iter #04] Stage 07/11/12/13 syntax eval common 统一**:以 Stage 07 为规范,Stage 11/12 改为参数化调用
2. **[iter #05] Stage 02 两文件合并**:`extract_errors_common.py` + `classify_writing_errors_common.py`
3. **[iter #06] Stage 05 `edit_distance` 双实现合并**

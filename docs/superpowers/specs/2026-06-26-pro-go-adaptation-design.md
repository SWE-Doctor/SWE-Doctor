# SWE-bench Pro — Golang Bug 类型 ours 方法适配设计

**分支**: `feat/pipeline-v2-pro-go`（从 V2 Verified `e2e906c` 切出）
**日期**: 2026-06-26
**目标**: 让 ours 三阶段 pipeline 支持 SWE-bench Pro 的 **golang** bug 实例，效果对标 Python，用 **Delve (dlv)** 作为类 PDB 调试器包装 stage2 RCA。

对标参照：Python 基线（pdb）→ JS 适配（node inspect，`feat/pipeline-v2-pro-js`）→ **Go（dlv）**。

---

## 1. 三条不可破的契约边界（照 JS 做法）

1. **调试会话契约**：`debug_agent/pdb_session.py` 的 `StartResult` / `CmdResult` 两个 dataclass + 帧字典 `{file, lineno, qualname}`。Go 版 `from .pdb_session import StartResult, CmdResult` **复用，绝不另定义**（JS 的 `js_session.py:23` 即如此）。
2. **语言开关**：`ctx["language"]` 单一真理源，下游 gating / conclusion_validator / rca_enrich 全部 keyed off 它，**无自动嗅探**。CLI `--language python|go`。
3. **stage1 语言注册**：`reproduction_test_agent/langpack.py` 的 `_REGISTRY` 填一行 dataclass 解决 ~90% stage1 差异；`_pdb_session_log` 记录流产同形 → 所有下游白嫖。

---

## 2. 现状盘点

**已 Go-ready（白捡）**：
- `swebench_pro_baseline/` 语言无关，能跑 Go 实例。
- `run_pro_test/go_runner.py` 已存在：失败测试抽取（`extract_failed_tests:41`）、coverage 文件级定位、build-error 映射、`detect_language → _RUNNERS["go"]` 已接线。

**缺口（要做）**：
- go 分支基于 Verified，**无 langpack 抽象**（config 无 language 字段，纯 python hardcode）。
- `debug_agent` 纯 pdb，无 Go/dlv 感知。
- stage2 深度 RCA `run_pro_test/run_statement_rca.py:66 _is_python_instance` 直接跳过 Go。
- reproduction/repair/debug 三 agent 对 Go 零感知。

---

## 3. 调试器方案（核心，spike 已验证）

**架构选定：dlv REPL stdin-pipe，对标 `PdbSession`**（复用 `_read_until_prompt` 机制），非 headless（规避 JS 端口泄漏教训）。spike 全绿：break/continue/next/step/locals/stack/print 均正常，输出纯文本规整。

**接线参数（硬知识，全部踩过坑，见 memory `pro-go-dlv-spike`）**：
| 项 | 值 / 坑 |
|---|---|
| dlv 安装 | `dlv@v1.25.1`（go1.24 兼容；latest v1.27 要 go≥1.25，GOTOOLCHAIN=local 禁升级）。跨实例 go1.16~1.24 → **预编译 go1.24 静态 dlv 二进制 docker cp 进容器** |
| PATH | `container.exec_bash` 用 `bash -lc`（login）丢 go → 显式 `export PATH=/usr/local/go/bin:/go/bin:$PATH` |
| 去色 | `NO_COLOR=1 TERM=dumb`（dlv REPL 默认 ANSI 色码） |
| tty | `--allow-non-terminal-interactive=true`（pipe stdin 必须） |
| 编译 | `GOFLAGS=-mod=mod`（go1.24 默认 readonly 不补 go.sum）；调试别带 `-cover` |
| ptrace | launch 加 `--cap-add SYS_PTRACE` |
| 启动 | `dlv test <pkg> --allow-non-terminal-interactive=true -- -test.run '^TestX$'`；编译慢 → start timeout 大（对标 JS 160s） |
| 提示符 | `(dlv) ` |
| 停点帧 | `> <pkg>.<func>() <file>:<line> (PC: ..)` |
| stack 帧 | `N  0x.. in <func>` 换行 `   at <file>:<line>` |

**Go 异常类型**（dlv prompt checklist）：nil deref、goroutine race、interface 类型断言失败、defer/recover、channel 阻塞/死锁、map 并发写、切片越界。

---

## 4. 逐文件实现清单

### 阶段1 — stage1 复现（建立语言基础设施）
| 文件 | 动作 | JS 参照 |
|---|---|---|
| `reproduction_test_agent/langpack.py` | **新建**：`LangPack` dataclass + `_REGISTRY{python, go}` + `get_langpack` + `parse_test_output`。go 条目见 §5 | js langpack.py |
| `reproduction_test_agent/config.py` | 加 `language: str = "python"` 字段 | — |
| `reproduction_test_agent/{generator,executor,localizer,pipeline,repair,critic}.py` | 加 `language=` 透传 + 查 langpack 替换 hardcode `*.py`/`pytest`/`def` | js 各模块改造 |
| `reproduction_test_agent/pipeline.py` | **真写新逻辑** `_go_import_hint`：从 `go.mod` 读 module + 目标目录算 import path；repro 落盘到**被测包同目录** `<pkg>/zzz_repro_test.go`（go 测试必须同包目录） | js `_build_js_gen_hint` |
| `reproduction_test_agent/run_batch_go.py` | **新建**：`language="go"`，注入 jefzda 镜像 | run_batch_js.py |

### 阶段2 — stage2 dlv 调试器（核心）

**架构优化（偏离原 JS 做法，更 surgical）**：`pdb_start`（actions.py:105）已通过 `ctx["_pdb_session_factory"]` 预留多后端注入点。故 Go **不新增 go_start/go_cmd/go_script 调试 action**，而是复用 pdb_* 全套 dispatch + `_pdb_session_log` 横切契约，由 run_debug 在 go 模式注入 `GoDlvSession` factory（GoDlvSession 满足 start/cmd/restart/stop + StartResult/CmdResult 契约即无缝替换）。仅新增 `gotest`（非调试测试运行，供 gating 证据）。go prompt 明确"pdb_* 启动的是 dlv 后端调试会话，pdb_cmd 发 dlv 命令"。

| 文件 | 动作 | JS 参照 |
|---|---|---|
| `debug_agent/go_session.py` | **新建** `GoDlvSession`：`import` 复用 StartResult/CmdResult，REPL stdin-pipe，dlv 帧正则，§3 接线参数；run_args→`dlv test` 命令 | js_session.py |
| `debug_agent/gotest_runner.py` | **新建** `run_gotest`/`GoTestResult`/`parse_go_failures`（复用 go_runner.extract_failed_tests） | jest_runner.py |
| `debug_agent/actions.py` | 仅加 `gotest` action（`_TAG` + `analyzer._ALL_ACTIONS`）；调试会话复用 pdb_*（factory 注入，零改 dispatch） | jest action |
| `debug_agent/gating.py` | `_is_real_frame` 排 `/usr/local/go/src`、`vendor/`、`/go/pkg/mod/`；`_had_pytest_evidence` 认 `gotest` 或 go run_args | gating.py js |
| `debug_agent/path_norm.py` | `is_test_file` 加 `_test.go` | path_norm.py js |
| `debug_agent/probe.py` | go 分支注 `fmt.Printf("PROBE %+v\n", expr)` | probe.py js |
| `debug_agent/run_debug.py` | `--language` choices 加 `go`；分派加 `elif language=="go"`；`_find_accepted_repro` globs 加 `*.go` | run_debug.py js |
| `debug_agent/container.py` | `launch` 加 `--cap-add SYS_PTRACE`；dlv 二进制注入 | — |
| `debug_agent/prompts/{action_tutorial_go.md, analyzer_system_go.j2, dlv_checklist.md}` | **新建** dlv 命令教程 + Go 异常清单；`analyzer._system_prompt` 加 go 分支 | prompts/*_js.* |

### 阶段3 — stage3 修复 + Pro 接线
| 文件 | 动作 | JS 参照 |
|---|---|---|
| `repair_agent/repair_config_pro_go.yaml` | **新建**：cwd /app、go build 友好、max_tokens | repair_config_pro_js.yaml |
| `run_pro_test/stage_go_repro.py` | **新建**：收 stage1 repro 写 `_repro_tests/` + `image.txt` | stage_js_repro.py |
| `pipeline_run/run_pipeline_pro_go.sh` | **新建**：4 阶段编排 `--language go` | run_pipeline_pro_js.sh |
| `go_smoke_ids.txt` | **新建**：flipt Go 冒烟实例清单 | js_smoke_ids.txt |

### 横切 — DeepSeek smoke 基础设施
- 预编译 go1.24 静态 dlv 二进制（`docker cp` 进任意容器）。
- DeepSeek 网关 env（thinking toggle 代码已就绪 `run_debug.py:99`/`llm.py:73`，`MSWEA_COST_TRACKING=ignore_errors`）。

---

## 5. Go LangPack 条目（关键值）

```python
"go": LangPack(
    name="go",
    test_filename="repro_test.go",          # go 测试必须 _test.go 后缀
    code_fence="go",
    system_prompt=_GO_SYSTEM,               # testing 包 / func TestXxx(t *testing.T) / t.Fatalf / 同包测试
    repair_prompt_lang_hint="Go",
    source_globs=["*.go"],
    test_path_substrings=["_test.go"],
    grep_type_args="--type go",
    func_def_grep=r"func (\([^)]*\) )?{name}",   # 函数或方法
    error_types=["panic:", "undefined:", "cannot use", "build failed", "FAIL"],
)
```
`parse_test_output("go", out)` 复用 `run_pro_test/go_runner.extract_failed_tests`。

**Go 特殊性**（vs js/python）：
- 测试文件必须放被测包**同目录**（`_test.go`），不能任意落 /tmp。
- `package foo`（同包）测试可访问未导出符号；`package foo_test`（外部）只能导出符号。
- import 用 module path（`go.mod` 派生），非相对路径 → `_go_import_hint` 是 stage1 唯一必须真写的新逻辑。

---

## 6. 验证策略（用户要求：每阶段 DeepSeek smoke）

每阶段落地后：
1. 单测 `pytest tests/...` 绿（TDD）。
2. DeepSeek（deepseek-v4-pro）在一个 flipt Go 实例（本地已有镜像，如 `e42da21a`）上跑该阶段，确认无回归。
3. 最终四阶段完整 smoke：stage1→2→3→eval，验证至少 1 个 Go 实例 RESOLVED。

冒烟实例首选 `instance_flipt-io__flipt-e42da21a07a5ae35835ec54f74004ebd58713874`（go1.24.3，FAIL_TO_PASS=`TestBatchEvaluate, TestEvaluate_FlagDisabled`，spike 已验证 dlv 可调）。

---

## 7. 风险（JS 已付学费）

1. **import 模型不同**：Go 用 module path 非相对路径 → `_go_import_hint` 必须真写。利好：同包测试可直接访问未导出符号，落盘比 JS 简单。
2. **跨实例 go 版本 1.16~1.24**：预编译静态 dlv 二进制规避装 dlv 兼容地狱。
3. **gate 语言词表**：`_is_real_frame` 必须排 `/usr/local/go/src` + `vendor/`，否则被标准库/运行时帧污染。
4. **go test 编译慢**：start timeout 要大；dlv test 首跑可能 1-2 分钟。

---

## 8. 实施进度（2026-06-26）

| 阶段 | 状态 | 验证 |
|---|---|---|
| Spike | ✅ | dlv 真容器断点/单步/变量 |
| Stage1 复现 | ✅ | DeepSeek smoke **2/2 accepted**（修选包 + go 严格性两坑） |
| Stage2 go_session | ✅ | 真容器 dlv 会话（断点/next/帧解析） |
| Stage2 集成 | ✅ | 239 passed（factory 注入/gotest/go prompts/container/run_debug） |
| Stage2 RCA 机制 | ✅ | DeepSeek smoke：repro 放包目录、gotest 复现、dlv 断点/单步、看到 bug 代码 |
| Stage2 RCA 收敛 | ⚠️ | LLM 不下结论（break 错层、没 step into）→ 加 rule7(step-into)+收敛引导；gpt-5.4-mini 诊断中 |
| Stage3 脚本 | ✅ | 语法+接口对齐 |
| 完整测试 | ✅ | 277 passed（我改的全部目录，5 pre-existing 缺依赖错误无关） |

**Stage2 RCA 收敛坑（已修/调优）**：
- repro 必须放包目录（.relpath sidecar），非 _repro/（go 编译）
- break 禁绝对路径（dlv 拒绝）
- analyzer_system_go.j2 是 .format 模板 → 字面 `{}` 触发 KeyError
- LLM 易 break 外层入口 + `next` 打转不 step-into 深层 bug 函数 → 加 rule7 + 收敛引导

待办：gpt-5.4-mini 诊断（机制 vs 模型）→ 完整 4 阶段 smoke。

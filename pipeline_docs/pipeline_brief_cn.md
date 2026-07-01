# mini-swe-agent Pipeline 说明（入口：`run_full_pipeline.sh`）

## 1. 流程概览
当前 pipeline 是一个 **3 阶段串行闭环**：
- Stage 1：生成并筛选可复现测试（reproduction tests）
- Stage 2：统一回放测试并做语句级根因分析（RCA）
- Stage 3：将 RCA + 本地化信号注入修复任务，生成补丁候选

## 2. 入口与运行方式
- 统一入口：`/home3/yaoqi/test_driven_agent/run_full_pipeline.sh`
- 常见命令：
  - 全量执行：`./run_full_pipeline.sh`
  - 仅跑修复：`./run_full_pipeline.sh --skip-stage1 --skip-stage2`
  - 自定义输出根目录：`./run_full_pipeline.sh --run-root /tmp/myrun`
- 输出根目录（`RUN_ROOT`）结构固定为：
  - `stage1_reproduction/`
  - `stage2_rca/`（含 `rca_output/`）
  - `stage3_repair/`

## 3. 配置与环境
- 配置文件优先级：
  - `PIPELINE_ENV_FILE`（默认 `${ROOT_DIR}/.env.pipeline`）：任务规模、模型、并发、目录等
  - `ENV_FILE`（默认 `${ROOT_DIR}/.env.rca`）：API Key 等敏感配置
- 关键默认值（未在 `.env.pipeline` 覆盖时生效）：
  - `ISSUE_IDS=swebench_pro_issue_ids_python_only_49.txt`
  - `DATASET=ScaleAI/SWE-bench_Pro`，`SPLIT=test`
  - `WORKERS=8`，`TIMEOUT_SECONDS=2400`
  - `PIPELINE_MODEL/REPRO_MODEL/RCA_MODEL/REPAIR_MODEL=openai/gpt-5.4`
  - `ENABLE_PHASE3=1`（启用 RCA 的 LLM refinement）
- 模型网关：
  - 统一设置 `OPENAI_API_BASE/OPENAI_BASE_URL=https://api.tu-zi.com/v1`
  - 通过 `TUZI_API_KEY` 注入 `OPENAI_API_KEY`
  - 若 `TUZI_API_KEY` 缺失，脚本会直接退出

## 4. 三阶段详细说明

### 4.1 Stage 1：复现测试生成（`reproduction_test_agent`）
- 根据 `issue_ids` 从 `ScaleAI/SWE-bench_Pro` 拉样本，并在对应 Docker 镜像里运行 e-Otter++ 流程。
- 每个实例产出：
  - `<instance_id>.json`（轻量结果）
  - `<instance_id>_full.json`（完整轨迹）
  - 若干 `accepted` 测试代码（另存为 `*_test_i.py`）
- 关键字段：
  - `accepted[*].final_test`：后续 Stage 2 的主输入
  - `best_test`：当没有 accepted 列表时可作为 fallback
  - `localization.relevant_files/focal_functions`：Stage 3 的辅助定位信号

### 4.2 Stage 2：统一回放 + 语句级 RCA（`run_pro_test`）
- **2a 回放（`run_repro_trace.py`）**
  - 读取 Stage 1 的 `accepted[*].final_test`（无则 fallback `best_test`）。
  - 将同一实例的测试写入 `workspace/_repro_tests/repro_i.py`。
  - 使用 **单次 pytest 调用**执行一个实例的全部 accepted 测试（保证 coverage/trace 在同一执行上下文聚合）。
  - 产出 `phase2_coverage.json`、`focused_*.log`、`stdout.log/stderr.log` 等 RCA 原料。
- **2b RCA（`run_statement_rca.py`）**
  - 基于覆盖率 + 失败日志 +（可选）源码快照做 statement-level root cause ranking。
  - 数据源优先级：`source_snapshot` > Docker 临时提取源码 > GitHub clone 到 `repos_dir`。
  - 输出：
    - `rca_output/<instance_id>_rca.json`（每实例候选）
    - `rca_output/rca_summary.json`、`rca_output/rca_summary.md`（汇总统计）
  - `--enable-phase3 --phase3-model` 可触发 LLM refinement，强化弱信号样本排序

### 4.3 Stage 3：修复生成（`repair_agent`）
- `run_repair.py` 读取 Stage 2 的 `*_rca.json`，提取 top1/top5 可疑文件注入任务模板，驱动修复。
- 同时读取 Stage 1 的 localizer 信号（`relevant_files/focal_functions`）做交叉提示，减少只盯 coverage 热点的偏差。
- 任务构造时会过滤测试文件候选（如 `_repro_tests/`、`tests/`、`test_*.py`），避免误导模型去改测试而非业务代码。
- 输出核心文件：
  - `stage3_repair/preds.json`（补丁结果索引）
  - `stage3_repair/rendered_tasks/*.txt`（每实例最终提示词）
  - `stage3_repair/repair_agent.log`（执行日志）

## 5. 关键机制
- **断点续跑**：通过 `--skip-stage1/2/3` 复用已有产物，适合多轮迭代调参。
- **空样本兜底**：
  - 若 Stage 1 某实例 `accepted=0`，Stage 2 会跳过并记录 `skipped.json` + `*.SKIPPED` 标记。
  - Stage 3 仍会执行；若缺失 RCA 文件则自动按“无 RCA 提示”模式运行。
- **统一模型入口**：三个阶段都通过同一 OpenAI 兼容网关，便于统一切模型与审计流量。
- **并发执行**：Stage 1/2/3 均支持 worker 并行，默认 `WORKERS=8`。

## 6. 重点产物与排错入口
- Stage 1：
  - `stage1_reproduction/preds.json`：每实例 accepted 数量与耗时
  - `stage1_reproduction/<instance>/<instance>.json`：候选/accepted/localization 详情
- Stage 2：
  - `stage2_rca/repro_trace_summary.json`：回放成功/跳过/失败统计
  - `stage2_rca/<instance>/workspace/`：覆盖率与执行日志原始材料
  - `stage2_rca/rca_output/<instance>_rca.json`：根因候选结果
- Stage 3：
  - `stage3_repair/preds.json`：最终补丁结果索引
  - `stage3_repair/rendered_tasks/`：可直接检查 RCA 注入是否符合预期

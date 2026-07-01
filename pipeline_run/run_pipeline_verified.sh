#!/usr/bin/env bash
# V2 pipeline for SWE-bench Verified. Usage:
#   bash pipeline_run/run_pipeline_verified.sh <RUN_DIR> <ISSUE_IDS_FILE>
# RUN_DIR should be timestamped, e.g. pipeline_run/verified_smoke_$(date +%Y%m%d_%H%M%S)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${1:?usage: run_pipeline_verified.sh <RUN_DIR> <ISSUE_IDS_FILE>}"
IDS="${2:?usage: run_pipeline_verified.sh <RUN_DIR> <ISSUE_IDS_FILE>}"
mkdir -p "$RUN_DIR"

# Anthropic relay creds
set -a; source "$REPO_ROOT/claude.env"; set +a
export ANTHROPIC_API_KEY="${ANTHROPIC_AUTH_TOKEN:-${ANTHROPIC_API_KEY:-}}"
export RCA_BACKEND=debug_agent
export DEBUG_AGENT_ENRICH=1

# Optional DeepSeek backend: set DEEPSEEK_ENV to a creds file (API_KEY/BASE_URL)
# to run the pipeline on DeepSeek via litellm instead of the Anthropic relay,
# and set MODEL=deepseek/<name>. Stage 1 disables reasoning (DEEPSEEK_THINKING
# below); Stages 2b/3 keep it on. Both are no-ops for non-DeepSeek models.
if [ -n "${DEEPSEEK_ENV:-}" ]; then
  set -a; source "$DEEPSEEK_ENV"; set +a
  export DEEPSEEK_API_KEY="${API_KEY:-${DEEPSEEK_API_KEY:-}}"
  export DEEPSEEK_API_BASE="${BASE_URL:-${DEEPSEEK_API_BASE:-}}"
  # Stage 3 (repair) uses the minisweagent textbased model, whose cost calc has
  # no price table for the deepseek-v4-pro proxy name; ignore the error.
  export MSWEA_COST_TRACKING=ignore_errors
fi

MODEL="${MODEL:-anthropic/claude-sonnet-4-6}"
WORKERS="${WORKERS:-1}"
LOG="$RUN_DIR/pipeline.log"

S1="$RUN_DIR/stage1_reproduction"
S2="$RUN_DIR/stage2_rca"
S3="$RUN_DIR/stage3_repair"
RUN_ID="verified_$(basename "$RUN_DIR")"

echo "=== Stage 1: reproduction tests ===" | tee -a "$LOG"
if [ ! -f "$RUN_DIR/.stage1.done" ]; then
  DEEPSEEK_THINKING=disabled python -m reproduction_test_agent.run_batch_verified \
    --issue-ids-file "$IDS" --output "$S1" --model "$MODEL" \
    --workers "$WORKERS" --max-repair-attempts 8 --test-timeout 120 >>"$LOG" 2>&1
  touch "$RUN_DIR/.stage1.done"
fi

echo "=== Stage 2a: repro-trace ===" | tee -a "$LOG"
if [ ! -f "$RUN_DIR/.stage2a.done" ]; then
  python -m run_verified_test.run_repro_trace_verified \
    --repro-dir "$S1" --issue-ids-file "$IDS" --output-dir "$S2" \
    --num-workers "$WORKERS" --timeout-seconds 1800 >>"$LOG" 2>&1
  touch "$RUN_DIR/.stage2a.done"
fi

echo "=== Stage 2b: PDB RCA ===" | tee -a "$LOG"
if [ ! -f "$RUN_DIR/.stage2b.done" ]; then
  DEEPSEEK_THINKING=enabled python -m run_verified_test.run_rca_verified \
    --results-dir "$S2" --output-dir "$S2/rca_output" \
    --model "$MODEL" --num-workers "$WORKERS" >>"$LOG" 2>&1
  touch "$RUN_DIR/.stage2b.done"
fi

echo "=== Stage 3: repair ===" | tee -a "$LOG"
if [ ! -f "$RUN_DIR/.stage3.done" ]; then
  DEEPSEEK_THINKING=enabled python -m repair_agent.run_repair_verified \
    --issue-ids-file "$IDS" --rca-dir "$S2/rca_output" --repro-dir "$S1" \
    --output "$S3" --model "$MODEL" --workers "$WORKERS" >>"$LOG" 2>&1
  touch "$RUN_DIR/.stage3.done"
fi

echo "=== Stage 4: eval (official harness) ===" | tee -a "$LOG"
python -m swebench_verified_eval.run_eval_verified \
  --preds "$S3/preds.json" --run-id "$RUN_ID" \
  --output-dir "$RUN_DIR" --max-workers "$WORKERS" >>"$LOG" 2>&1

echo "Pipeline complete" | tee -a "$LOG"

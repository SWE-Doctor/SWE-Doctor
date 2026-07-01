#!/usr/bin/env bash
# SWE-bench Pro GO 4-stage pipeline (OURS) on DeepSeek (deepseek-v4-pro).
# Mirrors run_pipeline_pro_js_deepseek.sh; go uses `go test` + a dlv RCA session
# (no per-instance run_script.sh, no NodeBB-style overrides).
# Stages:
#   Stage 1  reproduction_test_agent.run_batch --language go  (generate go repro tests)
#   Stage 2a run_pro_test/stage_go_repro.py                   (stage repro + image)
#   Stage 2b debug_agent.run_debug --language go              (dlv RCA, per instance)
#   Stage 3  repair_agent.run_repair  -c repair_config_pro_go.yaml
#   Stage 4  swebench_pro_baseline/evaluate.py                (apply patch, run gold tests)
# Usage: bash run_pipeline_pro_go.sh <ids-file> [run-dir]   (env: WORKERS, EVAL_WORKERS, PIPELINE_ENV, SCRIPTS_DIR)
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
IDS="${1:?usage: run_pipeline_pro_go.sh <ids-file> [run-dir]}"
# SWE-bench Pro dataset checkout; override via SCRIPTS_DIR or SWEBENCH_PRO_OS_ROOT.
SCRIPTS_DIR="${SCRIPTS_DIR:-${SWEBENCH_PRO_OS_ROOT:-$(dirname "$REPO")/SWE-bench_Pro-os}/run_scripts}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${2:-$REPO/pipeline_run/run_pro_go_deepseek_${TS}}"
WORKERS="${WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
mkdir -p "$RUN_DIR"; cp "$IDS" "$RUN_DIR/issue_ids.txt"
# Backend model + credentials. Point PIPELINE_ENV at a shell file that exports
# GO_MODEL and any provider keys (e.g. DEEPSEEK_API_KEY / DEEPSEEK_API_BASE);
# see pipeline_run/backend_env.example.sh for the expected variables.
if [ -n "${PIPELINE_ENV:-}" ]; then set -a; source "$PIPELINE_ENV"; set +a; fi
MODEL="${GO_MODEL:?set GO_MODEL (e.g. via PIPELINE_ENV) to the backend model name}"
S1="$RUN_DIR/stage1_reproduction"; S2="$RUN_DIR/stage2_rca"; S3="$RUN_DIR/stage3_repair"
LOG="$RUN_DIR/pipeline.log"

# ── Stage 1: go reproduction tests ──────────────────────────────────────────
if [ ! -f "$RUN_DIR/.stage1.done" ]; then
  echo "[pipeline] Stage 1: go reproduction tests (workers=$WORKERS)" | tee -a "$LOG"
  DEEPSEEK_THINKING=disabled python -m reproduction_test_agent.run_batch \
    --issue-ids-file "$IDS" --output "$S1" --model "$MODEL" --language go \
    --workers "$WORKERS" --max-repair-attempts "${REPAIR_ATTEMPTS:-10}" --test-timeout 300 2>&1 | tee -a "$LOG"
  touch "$RUN_DIR/.stage1.done"
fi

# ── Stage 2a: stage accepted go repro tests ─────────────────────────────────
if [ ! -f "$RUN_DIR/.stage2a.done" ]; then
  echo "[pipeline] Stage 2a: stage go repro tests" | tee -a "$LOG"
  python run_pro_test/stage_go_repro.py \
    --repro-dir "$S1" --output-dir "$S2" --issue-ids-file "$IDS" 2>&1 | tee -a "$LOG"
  touch "$RUN_DIR/.stage2a.done"
fi

# ── Stage 2b: go RCA (dlv), per instance, parallel x$WORKERS ─────────────────
if [ ! -f "$RUN_DIR/.stage2b.done" ]; then
  echo "[pipeline] Stage 2b: go RCA dlv (parallel x$WORKERS)" | tee -a "$LOG"
  mkdir -p "$S2/rca_output"
  _rca_one() {
    local iid="$1"; [ -z "$iid" ] && return 0
    local inst_dir="$S2/$iid"
    if [ ! -f "$inst_dir/image.txt" ]; then
      echo "[stage2b] skip $iid (not staged / no accepted repro)" | tee -a "$LOG"; return 0
    fi
    if [ -f "$S2/rca_output/${iid}_rca.json" ]; then
      echo "[stage2b] skip $iid (rca already exists)" | tee -a "$LOG"; return 0
    fi
    local image; image="$(cat "$inst_dir/image.txt")"
    DEBUG_AGENT_ENRICH=0 python -m debug_agent.run_debug \
      --instance-dir "$inst_dir" --output-dir "$S2/rca_output" \
      --image "$image" --model "$MODEL" --language go --workdir /app \
      --max-rounds "${MAX_ROUNDS:-20}" --wall-timeout "${WALL_TIMEOUT:-900}" 2>&1 | tee -a "$LOG" || true
  }
  export -f _rca_one
  export S2 MODEL LOG
  grep -v '^[[:space:]]*$' "$IDS" | xargs -P "$WORKERS" -I {} bash -c '_rca_one "$@"' _ {} || true
  touch "$RUN_DIR/.stage2b.done"
fi

# Optional early stop (smoke = stages 1-2b only, skip repair + eval).
if [ "${STOP_AFTER:-}" = "stage2b" ]; then
  echo "[pipeline] STOP_AFTER=stage2b — stopping before repair. RCA: $S2/rca_output" | tee -a "$LOG"
  exit 0
fi

# ── Stage 3: repair ─────────────────────────────────────────────────────────
if [ ! -f "$RUN_DIR/.stage3.done" ]; then
  echo "[pipeline] Stage 3: repair (workers=$WORKERS)" | tee -a "$LOG"
  python -m repair_agent.run_repair \
    --issue-ids-file "$IDS" --rca-dir "$S2/rca_output" --repro-dir "$S1" \
    -o "$S3" --model "$MODEL" -c repair_agent/repair_config_pro_go.yaml \
    -w "$WORKERS" 2>&1 | tee -a "$LOG"
  touch "$RUN_DIR/.stage3.done"
fi

# ── Stage 4: eval (Pro harness) ─────────────────────────────────────────────
echo "[pipeline] Stage 4: eval (workers=$EVAL_WORKERS)" | tee -a "$LOG"
export SWEBENCH_PRO_OS_ROOT="$(dirname "$SCRIPTS_DIR")"
python swebench_pro_baseline/evaluate.py -o "$S3" \
  --dataset ScaleAI/SWE-bench_Pro -w "$EVAL_WORKERS" 2>&1 | tee -a "$LOG" || true

echo "[pipeline] done: $RUN_DIR" | tee -a "$LOG"

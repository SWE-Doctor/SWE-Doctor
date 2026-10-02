#!/usr/bin/env bash
set -euo pipefail

# Repo root is the parent of this script's directory; both are overridable via env.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT_DIR="${ROOT_DIR:-$(dirname "${PROJECT_DIR}")}"
HF_HOME_DEFAULT="${ROOT_DIR}/.hf_cache"
ENV_FILE_DEFAULT="${ROOT_DIR}/.env.rca"
LOCAL_SRC="${PROJECT_DIR}/src"

cd "${PROJECT_DIR}"

export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export PYTHONPATH="${LOCAL_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

ENV_FILE="${ENV_FILE:-$ENV_FILE_DEFAULT}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

PREDS="${1:-${PROJECT_DIR}/reproduction_test_agent/outputs/python_only_49/preds.json}"
OUTPUT="${2:-${PROJECT_DIR}/reproduction_test_agent/outputs/python_only_49/eval_results.json}"

python -m reproduction_test_agent.evaluate_repro_test \
  --preds "${PREDS}" \
  --dataset "ScaleAI/SWE-bench_Pro" \
  --split "test" \
  --dockerhub-username "jefzda" \
  --workers 8 \
  --test-timeout 60 \
  --output "${OUTPUT}" \
  "${@:3}"

#!/usr/bin/env bash
set -euo pipefail

# Repo root is the parent of this script's directory; both are overridable via env.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT_DIR="${ROOT_DIR:-$(dirname "${PROJECT_DIR}")}"
HF_HOME_DEFAULT="${ROOT_DIR}/.hf_cache"
LOCAL_SRC="${PROJECT_DIR}/src"

cd "${PROJECT_DIR}"

export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export PYTHONPATH="${LOCAL_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

OUTPUT_DIR="${1:?Usage: $0 <output_dir> [workers]}"
WORKERS="${2:-8}"

python "swebench_pro_baseline/evaluate.py" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset "ScaleAI/SWE-bench_Pro" \
  --split "test" \
  --dockerhub-username "jefzda" \
  --workers "${WORKERS}" \
  "${@:3}"

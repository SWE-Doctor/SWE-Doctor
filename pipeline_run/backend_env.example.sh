# Backend model + credentials template for the SWE-bench Pro GO pipeline.
# Copy this file, fill in your own values, and point PIPELINE_ENV at the copy:
#
#   cp pipeline_run/backend_env.example.sh pipeline_run/backend_env.local.sh
#   # edit backend_env.local.sh with your keys
#   PIPELINE_ENV=pipeline_run/backend_env.local.sh \
#     bash pipeline_run/run_pipeline_pro_go.sh <ids-file>
#
# Do not commit a filled-in copy: *.env and local files are gitignored.

# Model name passed to litellm for every stage (stage 1 turns reasoning off for
# speed; stages 2b/3 keep it on). Any litellm-supported id works.
export GO_MODEL="deepseek/deepseek-v4-pro"

# Provider credentials. For a DeepSeek-compatible endpoint set both; for other
# providers export the keys that provider expects instead.
export DEEPSEEK_API_KEY="<your-api-key>"
export DEEPSEEK_API_BASE="<https://your-endpoint/v1>"

# deepseek-v4-pro has no litellm price table; ignore cost-tracking errors.
export MSWEA_COST_TRACKING=ignore_errors

# Stage 1 disables reasoning for throughput; stages 2b/3 default to enabled.
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-enabled}"

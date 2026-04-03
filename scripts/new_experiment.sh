#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# new_experiment.sh — Scaffold a new training experiment
#
# Usage:
#   ./scripts/new_experiment.sh <model-name> <version> <data-version> \
#     [--base-model HF_ID] [--chat-template NAME]
#
# Example:
#   ./scripts/new_experiment.sh gemma-3-12b v1 v1 \
#     --base-model google/gemma-3-12b-it --chat-template gemma
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Parse arguments ──────────────────────────────────────────
MODEL_NAME="${1:?Usage: $0 <model-name> <version> <data-version> [--base-model HF_ID] [--chat-template NAME]}"
VERSION="${2:?Missing version (e.g., v1)}"
DATA_VERSION="${3:?Missing data version (e.g., v1)}"
shift 3

BASE_MODEL=""
CHAT_TEMPLATE="mistral"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --chat-template) CHAT_TEMPLATE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

EXPERIMENT="${MODEL_NAME}-${VERSION}"
EXPERIMENT_DIR="experiments/${EXPERIMENT}"

# ─── Validate ─────────────────────────────────────────────────
if [[ -d "$EXPERIMENT_DIR" ]]; then
    echo "Error: ${EXPERIMENT_DIR} already exists"
    exit 1
fi

if [[ ! -d "configs" ]]; then
    echo "Error: Run from project root (configs/ directory not found)"
    exit 1
fi

if [[ ! -d "data/${DATA_VERSION}" ]]; then
    echo "Error: data/${DATA_VERSION}/ does not exist"
    exit 1
fi

echo "Creating experiment: ${EXPERIMENT}"
echo "  Base model:     ${BASE_MODEL:-<from template>}"
echo "  Chat template:  ${CHAT_TEMPLATE}"
echo "  Data version:   ${DATA_VERSION}"

# ─── Create directory structure ───────────────────────────────
mkdir -p "${EXPERIMENT_DIR}"/{configs,results,eval,models/{sft_checkpoint,raft_checkpoint,dpo_checkpoint,production}}

# ─── Copy and rewrite SFT config ─────────────────────────────
sed \
    -e "s|data/v1/sft/|data/${DATA_VERSION}/sft/|g" \
    -e "s|data/v1/raft/|data/${DATA_VERSION}/raft/|g" \
    -e "s|data/v1/dpo/|data/${DATA_VERSION}/dpo/|g" \
    -e "s|output_dir: \"models/sft_checkpoint\"|output_dir: \"${EXPERIMENT_DIR}/models/sft_checkpoint\"|" \
    -e "s|logging_dir: \"models/sft_checkpoint/logs\"|logging_dir: \"${EXPERIMENT_DIR}/models/sft_checkpoint/logs\"|" \
    -e "s|chat_template: \"mistral\"|chat_template: \"${CHAT_TEMPLATE}\"|" \
    -e "s|project: \"alvinai-sft\"|project: \"alvinai-training\"|" \
    -e "s|run_name: null|run_name: \"${EXPERIMENT}-sft\"|" \
    configs/sft_config.yaml > "${EXPERIMENT_DIR}/configs/sft_config.yaml"

# Rewrite base_model if provided
if [[ -n "$BASE_MODEL" ]]; then
    sed -i.bak "s|base_model: \"mistralai/Mistral-7B-Instruct-v0.3\"|base_model: \"${BASE_MODEL}\"|" \
        "${EXPERIMENT_DIR}/configs/sft_config.yaml"
    rm -f "${EXPERIMENT_DIR}/configs/sft_config.yaml.bak"
fi

# Add W&B group and tags
cat >> "${EXPERIMENT_DIR}/configs/sft_config.yaml" <<EOF
  group: "${EXPERIMENT}"
  tags: ["${MODEL_NAME}", "sft", "${DATA_VERSION}"]
EOF

# ─── Copy and rewrite RAFT config ────────────────────────────
sed \
    -e "s|base_model: \"models/sft_checkpoint\"|base_model: \"${EXPERIMENT_DIR}/models/sft_checkpoint\"|" \
    -e "s|data/v1/sft/|data/${DATA_VERSION}/sft/|g" \
    -e "s|data/v1/raft/|data/${DATA_VERSION}/raft/|g" \
    -e "s|data/v1/dpo/|data/${DATA_VERSION}/dpo/|g" \
    -e "s|output_dir: \"models/raft_checkpoint\"|output_dir: \"${EXPERIMENT_DIR}/models/raft_checkpoint\"|" \
    -e "s|logging_dir: \"models/raft_checkpoint/logs\"|logging_dir: \"${EXPERIMENT_DIR}/models/raft_checkpoint/logs\"|" \
    -e "s|chat_template: \"mistral\"|chat_template: \"${CHAT_TEMPLATE}\"|" \
    -e "s|project: \"alvinai-raft\"|project: \"alvinai-training\"|" \
    -e "s|run_name: null|run_name: \"${EXPERIMENT}-raft\"|" \
    configs/raft_config.yaml > "${EXPERIMENT_DIR}/configs/raft_config.yaml"

cat >> "${EXPERIMENT_DIR}/configs/raft_config.yaml" <<EOF
  group: "${EXPERIMENT}"
  tags: ["${MODEL_NAME}", "raft", "${DATA_VERSION}"]
EOF

# ─── Copy and rewrite DPO config ─────────────────────────────
sed \
    -e "s|base_model: \"models/raft_checkpoint\"|base_model: \"${EXPERIMENT_DIR}/models/raft_checkpoint\"|" \
    -e "s|data/v1/sft/|data/${DATA_VERSION}/sft/|g" \
    -e "s|data/v1/raft/|data/${DATA_VERSION}/raft/|g" \
    -e "s|data/v1/dpo/|data/${DATA_VERSION}/dpo/|g" \
    -e "s|output_dir: \"models/dpo_checkpoint\"|output_dir: \"${EXPERIMENT_DIR}/models/dpo_checkpoint\"|" \
    -e "s|logging_dir: \"models/dpo_checkpoint/logs\"|logging_dir: \"${EXPERIMENT_DIR}/models/dpo_checkpoint/logs\"|" \
    -e "s|chat_template: \"mistral\"|chat_template: \"${CHAT_TEMPLATE}\"|" \
    -e "s|project: \"alvinai-dpo\"|project: \"alvinai-training\"|" \
    -e "s|run_name: null|run_name: \"${EXPERIMENT}-dpo\"|" \
    configs/dpo_config.yaml > "${EXPERIMENT_DIR}/configs/dpo_config.yaml"

cat >> "${EXPERIMENT_DIR}/configs/dpo_config.yaml" <<EOF
  group: "${EXPERIMENT}"
  tags: ["${MODEL_NAME}", "dpo", "${DATA_VERSION}"]
EOF

# ─── Generate experiment.yaml manifest ────────────────────────
cat > "${EXPERIMENT_DIR}/experiment.yaml" <<EOF
name: ${EXPERIMENT}
description: ""
created: $(date +%Y-%m-%d)
base_model: ${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
chat_template: ${CHAT_TEMPLATE}
data_version: ${DATA_VERSION}
hardware: null
cost_per_hour_usd: null
total_training_hours: null
total_cost_usd: null
status: pending  # pending | sft_done | raft_done | dpo_done | completed
notes: ""
EOF

# ─── Generate skeleton results YAML files ─────────────────────
for stage in sft raft dpo; do
cat > "${EXPERIMENT_DIR}/results/${stage}_results.yaml" <<EOF
stage: ${stage}
experiment: ${EXPERIMENT}
model: null
hardware: null
date: null
duration_minutes: null
total_steps: null
epochs: null

metrics:
  train_loss_avg: null
  eval_loss_final: null

resources:
  vram_peak_gb: null
  gpu: null
  gpu_memory_gb: null

observations: ""
EOF
done

# RAGAS eval skeleton
cat > "${EXPERIMENT_DIR}/results/ragas_eval.yaml" <<EOF
experiment: ${EXPERIMENT}
model: null
eval_date: null
eval_dataset: data/${DATA_VERSION}/eval/

namespaces:
  customer_support:
    faithfulness: null
    answer_relevance: null
    context_precision: null
    context_recall: null
  engineering:
    faithfulness: null
    answer_relevance: null
    context_precision: null
    context_recall: null
  dealer_sales:
    faithfulness: null
    answer_relevance: null
    context_precision: null
    context_recall: null
  compliance:
    faithfulness: null
    answer_relevance: null
    context_precision: null
    context_recall: null
  employee_hr:
    faithfulness: null
    answer_relevance: null
    context_precision: null
    context_recall: null
  vendor:
    faithfulness: null
    answer_relevance: null
    context_precision: null
    context_recall: null

overall:
  faithfulness_avg: null
  answer_relevance_avg: null
  context_precision_avg: null
  context_recall_avg: null
  all_targets_met: null
EOF

# Latency benchmark skeleton
cat > "${EXPERIMENT_DIR}/results/latency_benchmark.yaml" <<EOF
experiment: ${EXPERIMENT}
model: null
serving: vllm
hardware: null
date: null

benchmarks:
  factual_lookup:
    p50_ms: null
    p95_ms: null
    p99_ms: null
    num_queries: null
  document_search_warm:
    p50_ms: null
    p95_ms: null
    p99_ms: null
    num_queries: null
  document_search_cold:
    p50_ms: null
    p95_ms: null
    p99_ms: null
    num_queries: null

tokens_per_second: null
concurrent_users_tested: null
EOF

echo ""
echo "✓ Experiment scaffolded: ${EXPERIMENT_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Edit experiment.yaml — add description and hardware"
echo "  2. Review configs — adjust batch sizes, learning rates if needed"
echo "  3. Run training:"
echo "     python -m scripts.run_sft --config ${EXPERIMENT_DIR}/configs/sft_config.yaml"
echo "     python -m scripts.run_raft --config ${EXPERIMENT_DIR}/configs/raft_config.yaml"
echo "     python -m scripts.run_dpo --config ${EXPERIMENT_DIR}/configs/dpo_config.yaml"
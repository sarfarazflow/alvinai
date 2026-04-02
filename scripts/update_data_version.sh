#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# update_data_version.sh — Re-point an experiment to a new data version
#
# Usage:
#   ./scripts/update_data_version.sh <experiment-dir> <new-data-version>
#
# Examples:
#   # Update a single experiment
#   ./scripts/update_data_version.sh experiments/mistral-nemo-12b-v1 v2
#
#   # Update all experiments at once
#   for exp in experiments/*/; do
#     ./scripts/update_data_version.sh "$exp" v2
#   done
# ──────────────────────────────────────────────────────────────
set -euo pipefail

EXPERIMENT_DIR="${1:?Usage: $0 <experiment-dir> <new-data-version>}"
NEW_VERSION="${2:?Usage: $0 <experiment-dir> <new-data-version>}"

# Strip trailing slash
EXPERIMENT_DIR="${EXPERIMENT_DIR%/}"

# ─── Validate ─────────────────────────────────────────────────
if [[ ! -d "$EXPERIMENT_DIR/configs" ]]; then
    echo "Error: ${EXPERIMENT_DIR}/configs/ not found"
    exit 1
fi

if [[ ! -f "$EXPERIMENT_DIR/experiment.yaml" ]]; then
    echo "Error: ${EXPERIMENT_DIR}/experiment.yaml not found"
    exit 1
fi

if [[ ! -d "data/${NEW_VERSION}" ]]; then
    echo "Warning: data/${NEW_VERSION}/ does not exist yet (will be created by generation scripts)"
fi

# ─── Detect current data version ─────────────────────────────
CURRENT_VERSION=$(grep 'data_version:' "${EXPERIMENT_DIR}/experiment.yaml" | awk '{print $2}')
if [[ -z "$CURRENT_VERSION" ]]; then
    echo "Error: Could not detect current data_version from experiment.yaml"
    exit 1
fi

if [[ "$CURRENT_VERSION" == "$NEW_VERSION" ]]; then
    echo "Already on ${NEW_VERSION}, nothing to do."
    exit 0
fi

EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
echo "Updating ${EXPERIMENT_NAME}: data/${CURRENT_VERSION} → data/${NEW_VERSION}"

# ─── Update config files ─────────────────────────────────────
for config in "${EXPERIMENT_DIR}"/configs/*.yaml; do
    if [[ ! -f "$config" ]]; then
        continue
    fi

    # Update data paths
    sed -i.bak "s|data/${CURRENT_VERSION}/|data/${NEW_VERSION}/|g" "$config"

    # Update W&B tags
    sed -i.bak "s|\"${CURRENT_VERSION}\"|\"${NEW_VERSION}\"|g" "$config"

    rm -f "${config}.bak"
    echo "  Updated $(basename "$config")"
done

# ─── Update experiment.yaml ──────────────────────────────────
sed -i.bak "s|data_version: ${CURRENT_VERSION}|data_version: ${NEW_VERSION}|" \
    "${EXPERIMENT_DIR}/experiment.yaml"

# Update eval dataset path in ragas_eval.yaml if it exists
if [[ -f "${EXPERIMENT_DIR}/results/ragas_eval.yaml" ]]; then
    sed -i.bak "s|data/${CURRENT_VERSION}/|data/${NEW_VERSION}/|g" \
        "${EXPERIMENT_DIR}/results/ragas_eval.yaml"
    rm -f "${EXPERIMENT_DIR}/results/ragas_eval.yaml.bak"
    echo "  Updated ragas_eval.yaml"
fi

rm -f "${EXPERIMENT_DIR}/experiment.yaml.bak"
echo "  Updated experiment.yaml"

echo ""
echo "Done. Verify with:"
echo "  grep -r 'data/' ${EXPERIMENT_DIR}/configs/ ${EXPERIMENT_DIR}/experiment.yaml | grep -v '${NEW_VERSION}' || echo 'All paths updated'"

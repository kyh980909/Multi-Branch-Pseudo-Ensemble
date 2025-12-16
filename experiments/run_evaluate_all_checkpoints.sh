#!/usr/bin/env bash
# Evaluate every checkpoint (MBEE/EDL) independently using experiments/evaluate_ood.py
#
# Usage:
#   ./experiments/run_evaluate_all_checkpoints.sh \
#       [CHECKPOINT_ROOT] [RESULT_ROOT]
#
# Defaults:
#   CHECKPOINT_ROOT=experiments/checkpoints
#   RESULT_ROOT=experiments/evaluation_results/all_models
#
# Notes:
#   - Each experiment directory under CHECKPOINT_ROOT must contain config.json and best_model.pth.
#   - The config "model" field determines whether we call evaluate_ood.py with --mbee_path or --edl_path.
#   - Results for each checkpoint are stored in RESULT_ROOT/<experiment_name>.

set -euo pipefail

CKPT_ROOT="${1:-experiments/checkpoints}"
RESULT_ROOT="${2:-experiments/evaluation_results/all_models}"
PYTHON_BIN="${PYTHON:-python3}"

if [[ ! -d "${CKPT_ROOT}" ]]; then
    echo "Checkpoint directory not found: ${CKPT_ROOT}" >&2
    exit 1
fi

mkdir -p "${RESULT_ROOT}"

mapfile -t MODEL_INFO < <("${PYTHON_BIN}" - <<'PY' "${CKPT_ROOT}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for exp in sorted(root.iterdir()):
    config = exp / "config.json"
    best_model = exp / "best_model.pth"
    if not (config.exists() and best_model.exists()):
        continue
    try:
        with config.open() as f:
            cfg = json.load(f)
    except json.JSONDecodeError:
        continue
    model = str(cfg.get("model", "unknown")).lower()
    print(f"{model}|{exp.name}|{best_model}")
PY
)

if [[ ${#MODEL_INFO[@]} -eq 0 ]]; then
    echo "No checkpoints with config.json and best_model.pth were found under ${CKPT_ROOT}" >&2
    exit 1
fi

echo "Found ${#MODEL_INFO[@]} checkpoints. Results will be written to ${RESULT_ROOT}."

for entry in "${MODEL_INFO[@]}"; do
    IFS='|' read -r model name path <<<"${entry}"
    save_dir="${RESULT_ROOT}/${name}"
    mkdir -p "${save_dir}"

    case "${model}" in
        mbee|mbpe)
            echo
            echo "Evaluating MBEE checkpoint ${name}..."
            "${PYTHON_BIN}" experiments/evaluate_ood.py \
                --mbee_path "${path}" \
                --save_dir "${save_dir}"
            ;;
        edl)
            echo
            echo "Evaluating EDL checkpoint ${name}..."
            "${PYTHON_BIN}" experiments/evaluate_ood.py \
                --edl_path "${path}" \
                --save_dir "${save_dir}"
            ;;
        *)
            echo "Skipping ${name} (unknown model type: ${model})"
            ;;
    esac
done

echo
echo "All evaluations finished. Summary directories are under ${RESULT_ROOT}"

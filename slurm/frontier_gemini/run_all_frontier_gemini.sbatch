#!/bin/bash
#SBATCH --job-name=gemini_all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --output=/cluster/scratch/%u/llmgap/logs/frontier_gemini/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/llmgap/logs/frontier_gemini/%x-%j.err

# Full before-only frontier inference: every (setting, task) combo, every
# model in MODELS, full test split (no --max-rows). Run smoke_frontier_gemini.sbatch
# first to sanity-check before submitting this.

set -euo pipefail
cd /cluster/home/$USER/llmgap

LOG_DIR=/cluster/scratch/$USER/llmgap/logs/frontier_gemini
OUT_DIR=/cluster/scratch/$USER/llmgap/out/frontier_gemini
mkdir -p "$LOG_DIR" "$OUT_DIR"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "ERROR: GEMINI_API_KEY is not set in the environment." >&2
    exit 1
fi
echo "GEMINI_API_KEY is set (value not printed)."

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap   # must contain google-genai

SCRIPT=src/run_inference_frontier_gemini.py

# Add more Gemini model ids here once verified to work with this script.
MODELS=(
    "gemini-3.6-flash"
    # "gemini-3.6-pro"
    # "gemini-2.5-flash"
)

# task_flag | csv_path | suffix
DATASETS=(
    "correct_answer|reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs.csv|_reasoning_efficiency_ca"
    "next_subquestion|reasoning-efficiency/experiments/proof_search/out/next_subquestion_pairs.csv|_reasoning_efficiency_ns"
    "correct_answer|reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs_gsm8k.csv|_gsm8k_ca"
    "next_subquestion|reasoning-efficiency/experiments/proof_search/out/next_subquestion_pairs_gsm8k.csv|_gsm8k_ns"
    "correct_answer|colm-paper-code-cleaned/experiments/csm_mwps/out/correct_answer_distractor_pairs.csv|_synthetic_distractor_ca"
    "distractor|colm-paper-code-cleaned/experiments/csm_mwps/out/distractor_pairs.csv|_synthetic_distractor_dist"
    "correct_answer|eedi/data/processed/correct_answer_pairs_eedi_split_control.csv|_eedi_split_control_ca"
    "distractor|eedi/data/processed/distractor_pairs_eedi_split_control.csv|_eedi_split_control_dist"
)

for MODEL in "${MODELS[@]}"; do
    for ds_row in "${DATASETS[@]}"; do
        IFS='|' read -r TASK CSV SUFFIX <<< "$ds_row"

        echo ""
        echo "=== model=${MODEL} task=${TASK} suffix=${SUFFIX} ==="
        python "$SCRIPT" \
            --api-model "$MODEL" \
            --model-name "$MODEL" \
            --task "$TASK" \
            --data-csv "$CSV" \
            --out-dir "$OUT_DIR" \
            --judge-provider gemini \
            --judge-model "$MODEL" \
            --suffix "$SUFFIX"
    done
done

echo ""
echo "All frontier Gemini runs finished. Summaries in: $OUT_DIR"

#!/bin/bash
#SBATCH --job-name=gemini_gsm8k_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20000m
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=/cluster/scratch/%u/llmgap/logs/frontier_gemini/%x-%j.out
#SBATCH --error=/cluster/scratch/%u/llmgap/logs/frontier_gemini/%x-%j.err

# gsm8k setting, full test split: correct_answer + next_subquestion.
# Needs a GPU now: run_inference_frontier_gemini.py loads a local
# Qwen2.5-7B-Instruct judge (openai_judge.py/prompt.py) for scoring,
# on top of calling the Gemini API for generation.

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
conda activate llmgap   # must contain google-genai, torch, transformers

SCRIPT=src/run_inference_frontier_gemini.py
MODEL="gemini-3.6-flash"

DATASETS=(
    "correct_answer|reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs_gsm8k.csv|_gsm8k_ca"
    "next_subquestion|reasoning-efficiency/experiments/proof_search/out/next_subquestion_pairs_gsm8k.csv|_gsm8k_ns"
)

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
        --thinking-level minimal \
        --suffix "$SUFFIX"
done

echo ""
echo "gsm8k frontier Gemini runs finished. Summaries in: $OUT_DIR"

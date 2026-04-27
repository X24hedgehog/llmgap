#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40000m
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Required env vars (passed via --export from submit_all.sh):
#   MODEL_NAME    e.g. Qwen/Qwen2.5-0.5B-Instruct
#   TASK          correct_answer | next_subquestion
#   MODE          before | after
#
# Optional:
#   CHECKPOINT    path to fine-tuned adapter (only needed when MODE=after)
#   USE_LLM_JUDGE 1 to enable OpenAI judge for next_subquestion
#   OPENAI_API_KEY

set -euo pipefail

cd /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

MODEL_SHORT="${MODEL_NAME##*/}"
OUT_PATH="out/results/${MODEL_SHORT}_${TASK}_${MODE}.csv"

EXTRA_ARGS=""

if [[ "${MODE}" == "after" && -n "${CHECKPOINT:-}" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --checkpoint $CHECKPOINT"
fi

if [[ "${USE_LLM_JUDGE:-0}" == "1" && "${TASK}" == "next_subquestion" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-llm-judge"
fi

echo "=== Evaluation ==="
echo "  model      : $MODEL_NAME"
echo "  task       : $TASK"
echo "  mode       : $MODE"
echo "  checkpoint : ${CHECKPOINT:-<none>}"
echo "  out        : $OUT_PATH"

python evaluate.py \
    --model-name "$MODEL_NAME" \
    --task "$TASK" \
    --out-path "$OUT_PATH" \
    $EXTRA_ARGS

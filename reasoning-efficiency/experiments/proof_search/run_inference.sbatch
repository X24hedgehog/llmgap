#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40000m
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Required env vars (passed via --export from submit_all.sh):
#   MODEL_NAME   e.g. Qwen/Qwen2.5-0.5B-Instruct
#   TASK         correct_answer | next_subquestion
#   MODE         before | after
#
# Optional:
#   CHECKPOINT   path to fine-tuned LoRA adapter (required when MODE=after)
#   OPENAI_API_KEY  (required when TASK=next_subquestion)

set -euo pipefail

cd /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

MODEL_SHORT="${MODEL_NAME##*/}"

EXTRA_ARGS=""
if [[ "${MODE}" == "after" && -n "${CHECKPOINT:-}" ]]; then
    EXTRA_ARGS="--checkpoint ${CHECKPOINT}"
fi

echo "=== Inference ==="
echo "  model      : $MODEL_NAME"
echo "  task       : $TASK"
echo "  mode       : $MODE"
echo "  checkpoint : ${CHECKPOINT:-<none>}"

python run_inference.py \
    --model-name "$MODEL_NAME" \
    --task "$TASK" \
    --mode "$MODE" \
    $EXTRA_ARGS

#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40000m
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Required env vars (passed via --export from submit_all.sh):
#   MODEL_NAME   e.g. Qwen/Qwen2.5-0.5B-Instruct
#   TASK         correct_answer | next_subquestion

set -euo pipefail

cd /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

MODEL_SHORT="${MODEL_NAME##*/}"
OUT_DIR="out/checkpoints/${MODEL_SHORT}_${TASK}"

echo "=== Fine-tuning ==="
echo "  model : $MODEL_NAME"
echo "  task  : $TASK"
echo "  out   : $OUT_DIR"

python finetune.py \
    --model-name "$MODEL_NAME" \
    --task "$TASK" \
    --out-dir "$OUT_DIR"

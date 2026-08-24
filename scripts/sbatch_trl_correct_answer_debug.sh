#!/bin/bash
#SBATCH --job-name=trl_correct_ans
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20000m
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=/cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs/trl-correct-answer-%j.out
#SBATCH --error=/cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs/trl-correct-answer-%j.err

set -euo pipefail

cd /cluster/home/tunguyen1/llmgap
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

mkdir -p /cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs

export HF_HOME=/cluster/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/cluster/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/trl_grpo_gsm8k_correct_answer.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --train-csv /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs_gsm8k.csv \
  --out-dir /cluster/scratch/tunguyen1/llmgap/output/trl_grpo_correct_answer_debug \
  --question-col question \
  --answer-col target_answer \
  --split-col split \
  --train-split train \
  --max-rows 100 \
  --num-generations 4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --epochs 1 \
  --lr 1e-6 \
  --max-prompt-length 512 \
  --max-completion-length 128 \
  --temperature 0.8 \
  --top-p 0.95 \
  --beta 0.0 \
  --format-reward-weight 0.0 \
  --use-peft \
  --lora-r 16 \
  --lora-alpha 32 \
  --load-in-4bit \
  --gradient-checkpointing \
  --logging-steps 1 \
  --save-steps 50 \
  --debug-print-rewards 5

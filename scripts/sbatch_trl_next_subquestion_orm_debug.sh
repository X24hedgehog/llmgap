#!/bin/bash
#SBATCH --job-name=trl_nextsubq_orm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20000m
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=/cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs/trl-nextsubq-orm-%j.out
#SBATCH --error=/cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs/trl-nextsubq-orm-%j.err

set -euo pipefail

cd /cluster/home/tunguyen1/llmgap
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

mkdir -p /cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs

export HF_HOME=/cluster/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/cluster/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/trl_grpo_next_subquestion_orm.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --train-csv /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search/out/next_subquestion_pairs_gsm8k.csv \
  --out-dir /cluster/scratch/tunguyen1/llmgap/output/trl_grpo_next_subquestion_orm_debug \
  --prompt-col prompt \
  --question-col question \
  --subquestion-col next_subquestion \
  --reasoning-trace-col reasoning_trace \
  --tree-col tree \
  --split-col split \
  --train-split train \
  --max-rows 100 \
  --num-generations 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --epochs 1 \
  --lr 1e-6 \
  --max-prompt-length 512 \
  --max-completion-length 64 \
  --temperature 0.8 \
  --top-p 0.95 \
  --beta 0.0 \
  --orm-model-name RLHFlow/Llama3.1-8B-ORM-Mistral-Data \
  --orm-load-in-4bit \
  --orm-batch-size 1 \
  --use-peft \
  --lora-r 16 \
  --lora-alpha 32 \
  --load-in-4bit \
  --gradient-checkpointing \
  --logging-steps 1 \
  --save-steps 50 \
  --debug-print-rewards 5
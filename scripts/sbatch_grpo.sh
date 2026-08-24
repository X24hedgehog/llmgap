#!/bin/bash
#SBATCH --job-name=rl_grpo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20000m
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=/cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs/grpo-%j.out
#SBATCH --error=/cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs/grpo-%j.err

set -euo pipefail

cd /cluster/home/tunguyen1/llmgap
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

export HF_HOME=/cluster/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/cluster/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/rl_gsm8k_grpo.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --train-csv /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs_gsm8k.csv \
  --out-dir /cluster/scratch/tunguyen1/llmgap/output/benchmark_grpo \
  --prompt-style step_by_step \
  --max-rows 50 \
  --num-samples 4 \
  --batch-size 1 \
  --epochs 3 \
  --lr 1e-7 \
  --max-new-tokens 128 \
  --clip-eps 0.2

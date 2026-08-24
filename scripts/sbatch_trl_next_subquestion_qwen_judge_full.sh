#!/bin/bash
#SBATCH --job-name=trl_nextsubq_qwen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40000m
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=/cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs/trl-nextsubq-qwen-judge-full-%j.out
#SBATCH --error=/cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs/trl-nextsubq-qwen-judge-full-%j.err

set -euo pipefail

cd /cluster/home/tunguyen1/llmgap
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

mkdir -p /cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs

export HF_HOME=/cluster/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/cluster/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/cluster/scratch/$USER/hf_cache/datasets
export TORCH_HOME=/cluster/scratch/$USER/torch_cache
export PIP_CACHE_DIR=/cluster/scratch/$USER/pip_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/trl_grpo_next_subquestion_qwen_judge.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --train-csv /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search/out/next_subquestion_pairs_gsm8k.csv \
  --out-dir /cluster/scratch/tunguyen1/llmgap/output/trl_grpo_next_subquestion_qwen_judge_full \
  --prompt-col prompt \
  --question-col question \
  --subquestion-col next_subquestion \
  --reasoning-trace-col reasoning_trace \
  --tree-col tree \
  --split-col split \
  --train-split train \
  --num-generations 2 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --epochs 3 \
  --lr 5e-7 \
  --max-prompt-length 512 \
  --max-completion-length 40 \
  --temperature 0.8 \
  --top-p 0.95 \
  --beta 0.0 \
  --judge-model-name Qwen/Qwen2.5-7B-Instruct \
  --judge-load-in-4bit \
  --judge-batch-size 1 \
  --judge-max-input-length 2048 \
  --judge-max-new-tokens 8 \
  --include-policy-prompt-in-judge \
  --judge-reward-scale 1.0 \
  --format-reward-weight 0.1 \
  --use-peft \
  --lora-r 64 \
  --lora-alpha 128 \
  --lora-dropout 0.05 \
  --load-in-4bit \
  --gradient-checkpointing \
  --logging-steps 1 \
  --save-steps 50 \
  --debug-print-rewards 20

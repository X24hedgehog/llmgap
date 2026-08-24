#!/bin/bash
#SBATCH --job-name=gsm8k_rlvr_full
#SBATCH --output=/cluster/scratch/%u/llmgap/logs/gsm8k_rlvr_full_%j.out
#SBATCH --error=/cluster/scratch/%u/llmgap/logs/gsm8k_rlvr_full_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40000m
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=8192

set -euo pipefail

mkdir -p /cluster/scratch/$USER/llmgap/logs
mkdir -p /cluster/scratch/$USER/hf_cache/hub
mkdir -p /cluster/scratch/$USER/hf_cache/transformers
mkdir -p /cluster/scratch/$USER/hf_cache/datasets
mkdir -p /cluster/scratch/$USER/torch_cache
mkdir -p /cluster/scratch/$USER/pip_cache
mkdir -p /cluster/scratch/$USER/llmgap/output

export HF_HOME=/cluster/scratch/$USER/hf_cache
export HUGGINGFACE_HUB_CACHE=/cluster/scratch/$USER/hf_cache/hub
export TRANSFORMERS_CACHE=/cluster/scratch/$USER/hf_cache/transformers
export HF_DATASETS_CACHE=/cluster/scratch/$USER/hf_cache/datasets
export TORCH_HOME=/cluster/scratch/$USER/torch_cache
export PIP_CACHE_DIR=/cluster/scratch/$USER/pip_cache
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmgap

cd /cluster/home/$USER/llmgap

python src/trl_grpo_gsm8k_correct_answer_rlvr.py \
  --out-dir /cluster/scratch/$USER/llmgap/output/gsm8k_rlvr_qwen3b_formatfix_full \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --train-csv /cluster/home/$USER/llmgap/reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs_gsm8k.csv \
  --split-col split \
  --train-split train \
  --train-mode lora \
  --load-in-4bit \
  --use-chat-template \
  --num-generations 6 \
  --per-device-train-batch-size 6 \
  --gradient-accumulation-steps 8 \
  --max-prompt-length 384 \
  --max-completion-length 256 \
  --lr 2e-5 \
  --epochs 1 \
  --attn-implementation eager \
  --logging-steps 1 \
  --save-steps 200 \
  --debug-print-rewards 20
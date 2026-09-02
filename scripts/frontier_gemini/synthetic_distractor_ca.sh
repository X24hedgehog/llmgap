#!/bin/bash
#SBATCH --job-name=gem-dist-ca
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40000m
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=/cluster/home/tunguyen1/llmgap/out/frontier_gemini/logs/dist-ca-%j.out
#SBATCH --error=/cluster/home/tunguyen1/llmgap/out/frontier_gemini/logs/dist-ca-%j.err

set -euo pipefail

cd /cluster/home/tunguyen1/llmgap

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

mkdir -p out/frontier_gemini/logs
mkdir -p /cluster/scratch/$USER/hf_cache
mkdir -p /cluster/scratch/$USER/torch_cache

export HF_HOME=/cluster/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/cluster/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/cluster/scratch/$USER/hf_cache/datasets
export TORCH_HOME=/cluster/scratch/$USER/torch_cache
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${GEMINI_API_KEY:?GEMINI_API_KEY is not set}"

python src/run_inference_frontier_gemini.py \
  --api-model gemini-3.6-flash \
  --model-name gemini-3.6-flash \
  --task correct_answer \
  --mode before \
  --data-csv colm-paper-code-cleaned/experiments/csm_mwps/out/correct_answer_distractor_pairs.csv \
  --out-dir out/frontier_gemini \
  --suffix _distractor

#!/bin/bash
#SBATCH --job-name=ft_llama8b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20000m
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=jobs/slurm-%j.out
#SBATCH --error=jobs/slurm-%j.err

set -euo pipefail
cd /cluster/home/tunguyen1/llmgap/colm-paper-code-cleaned/experiments/csm_mwps
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

python finetune.py \
    --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --task distractor \
    --out-dir out/ft_distractor/Meta-Llama-3.1-8B-Instruct \
    --load-in-4bit

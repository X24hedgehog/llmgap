#!/bin/bash
#SBATCH --job-name=ft_qwen15b_ca_cot_g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20000m
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=logs/gsm8k/slurm-%j.out
#SBATCH --error=logs/gsm8k/slurm-%j.err

set -euo pipefail
cd /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

python finetune.py \
    --model-name "Qwen/Qwen2.5-1.5B-Instruct" \
    --task correct_answer \
    --train-csv out/correct_answer_pairs_gsm8k.csv \
    --out-dir out/checkpoints/Qwen2.5-1.5B-Instruct_correct_answer_cot_gsm8k/final

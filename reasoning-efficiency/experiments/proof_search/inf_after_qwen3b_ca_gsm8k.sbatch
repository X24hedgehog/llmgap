#!/bin/bash
#SBATCH --job-name=ia_qwen3b_ca_g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20000m
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=logs/gsm8k/slurm-%j.out
#SBATCH --error=logs/gsm8k/slurm-%j.err

set -euo pipefail
cd /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

python run_inference.py \
    --model-name "Qwen/Qwen2.5-3B-Instruct" \
    --task correct_answer \
    --mode after \
    --data-csv out/correct_answer_pairs_gsm8k.csv \
    --suffix _gsm8k \
    --checkpoint out/checkpoints/Qwen2.5-3B-Instruct_correct_answer_gsm8k/final

#!/bin/bash
#SBATCH --job-name=inf_aft_ca_qwen15b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20000m
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=jobs/slurm-%j.out
#SBATCH --error=jobs/slurm-%j.err

set -euo pipefail
cd /cluster/home/tunguyen1/llmgap/colm-paper-code-cleaned/experiments/csm_mwps
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

python run_inference.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --task correct_answer \
    --mode after \
    --checkpoint out/ft_correct_answer/Qwen2.5-1.5B-Instruct \
    --data-csv out/correct_answer_distractor_pairs.csv

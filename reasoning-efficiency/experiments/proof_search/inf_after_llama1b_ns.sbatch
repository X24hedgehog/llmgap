#!/bin/bash
#SBATCH --job-name=inf_a_llama1b_ns
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20000m
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

python run_inference.py \
    --model-name "meta-llama/Llama-3.2-1B-Instruct" \
    --task next_subquestion \
    --mode after \
    --checkpoint out/checkpoints/Llama-3.2-1B-Instruct_next_subquestion/final

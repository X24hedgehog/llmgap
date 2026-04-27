#!/bin/bash
#SBATCH --job-name=inf_a_qwen05b_ca_cot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20000m
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

python run_inference.py \
    --model-name "Qwen/Qwen2.5-0.5B-Instruct" \
    --task correct_answer \
    --mode after \
    --suffix _cot \
    --checkpoint out/checkpoints/Qwen2.5-0.5B-Instruct_correct_answer_cot/final

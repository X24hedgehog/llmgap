#!/bin/bash
#SBATCH --job-name=eedi_traces
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=logs/eedi_traces-%j.out
#SBATCH --error=logs/eedi_traces-%j.err

set -euo pipefail

cd /cluster/home/tunguyen1/llmgap/eedi
mkdir -p logs

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

/cluster/home/tunguyen1/miniconda3/envs/llmgap/bin/python generate_reasoning_traces_openai.py \
    --model gpt-5.4 \
    --write-batch-size 10
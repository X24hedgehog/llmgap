#!/bin/bash
#SBATCH --job-name=gem-eedi-ca
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --output=/cluster/home/tunguyen1/llmgap/out/frontier_gemini/logs/eedi-ca-%j.out
#SBATCH --error=/cluster/home/tunguyen1/llmgap/out/frontier_gemini/logs/eedi-ca-%j.err

set -euo pipefail

cd /cluster/home/tunguyen1/llmgap

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

mkdir -p out/frontier_gemini/logs

: "${GEMINI_API_KEY:?GEMINI_API_KEY is not set}"

python src/run_inference_frontier_gemini.py \
  --api-model gemini-3.6-flash \
  --model-name gemini-3.6-flash \
  --task correct_answer \
  --mode before \
  --data-csv eedi/data/processed/correct_answer_pairs_eedi_split_control.csv \
  --out-dir out/frontier_gemini \
  --suffix _eedi_split_control

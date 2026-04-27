#!/bin/bash
# submit_inf_before_all.sh — submit inference-before for all models except Qwen 0.5B

cd ~/llmgap/reasoning-efficiency/experiments/proof_search

MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-2-2b-it"
    "google/gemma-2-7b-it"
)

TASKS=("correct_answer" "next_subquestion")

for MODEL in "${MODELS[@]}"; do
    SHORT="${MODEL##*/}"
    for TASK in "${TASKS[@]}"; do
        JID=$(sbatch \
            --job-name="inf_b_${SHORT:0:10}_${TASK:0:2}" \
            --ntasks=1 --cpus-per-task=4 --gpus=1 \
            --gres=gpumem:20000m --time=05:00:00 --mem-per-cpu=8192 \
            --output="slurm-%j.out" --error="slurm-%j.err" \
            --wrap="set -euo pipefail
cd /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search
eval \"\$(\$HOME/miniconda3/bin/conda shell.bash hook)\"
conda activate llmgap
python run_inference.py --model-name \"${MODEL}\" --task ${TASK} --mode before" \
        | awk '{print $NF}')
        echo "Submitted $SHORT / $TASK → job $JID"
    done
done
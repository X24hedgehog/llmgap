#!/bin/bash
# Submit full pipeline for old models on gsm8k
# Run inf_before and ft first; submit inf_after after checkpoints are ready.

cd /cluster/home/tunguyen1/llmgap
DIR=out/sbatch/gsm8k

MODELS="gemma2b_old qwen4b qwen05b_old llama2_7b"
TASKS="ca ns"

echo "=== Submitting inf_before and finetune ==="
for m in $MODELS; do
  for t in $TASKS; do
    echo "sbatch $DIR/inf_before_${t}_${m}.sbatch"
    sbatch $DIR/inf_before_${t}_${m}.sbatch
    echo "sbatch $DIR/ft_${t}_${m}.sbatch"
    sbatch $DIR/ft_${t}_${m}.sbatch
  done
done

echo ""
echo "=== inf_after (run manually after finetuning completes) ==="
for m in $MODELS; do
  for t in $TASKS; do
    echo "sbatch $DIR/inf_after_${t}_${m}.sbatch"
  done
done

#!/bin/bash
# Submit cross-task jobs: NS-finetuned models → CA inference
# Output CSVs will be named *_correct_answer_after_ns_ckpt.csv
set -euo pipefail
cd /cluster/home/tunguyen1/llmgap

echo "=== reasoning_efficiency: NS ckpt → CA inference (7 models) ==="
for f in out/sbatch/reasoning_efficiency/inf_after_ca_ns_ckpt_*.sbatch; do
  name=$(basename "$f" .sbatch)
  id=$(sbatch "$f" | awk '{print $4}')
  echo "  $id  $name"
done

echo ""
echo "=== gsm8k: NS ckpt → CA inference (8 models) ==="
for f in out/sbatch/gsm8k/inf_after_ca_ns_ckpt_*.sbatch; do
  name=$(basename "$f" .sbatch)
  id=$(sbatch "$f" | awk '{print $4}')
  echo "  $id  $name"
done

echo ""
echo "Done. Results will appear as *_correct_answer_after_ns_ckpt.csv"

#!/bin/bash
# Submit all 8 inference-after jobs for the correct_answer task.
# All correct_answer finetunes are complete.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Submitting inference-after (correct_answer) ==="
sbatch inf_after_ca_qwen05b.sbatch
sbatch inf_after_ca_qwen15b.sbatch
sbatch inf_after_ca_qwen3b.sbatch
sbatch inf_after_ca_llama1b.sbatch
sbatch inf_after_ca_llama3b.sbatch
sbatch inf_after_ca_llama8b.sbatch
sbatch inf_after_ca_gemma2b.sbatch
sbatch inf_after_ca_gemma7b.sbatch
echo "Submitted 8 jobs."

#!/bin/bash
# Resubmit all finetune jobs for the distractor task.
#
# Usage: bash resubmit_ft_distractor.sh

set -euo pipefail
cd "$(dirname "$0")"

MODELS=(qwen05b qwen15b qwen3b llama1b llama3b llama8b gemma2b gemma7b)

echo "Submitting distractor finetune jobs..."
for m in "${MODELS[@]}"; do
    jid=$(sbatch --parsable "ft_${m}.sbatch")
    echo "  ft_${m}: job ${jid}"
done

echo "All distractor finetune jobs submitted. Check with: squeue -u \$USER"

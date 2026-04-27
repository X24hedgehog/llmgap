#!/bin/bash
# Submit all inference-before and finetune jobs for both tasks (distractor + correct_answer).
# Finetune jobs are submitted with a dependency on the corresponding inference-before job.
#
# Usage:
#   bash submit_all.sh

set -euo pipefail
cd "$(dirname "$0")"

MODELS=(qwen05b qwen15b qwen3b llama1b llama3b llama8b gemma2b gemma7b)

echo "=== Submitting DISTRACTOR jobs ==="
for m in "${MODELS[@]}"; do
    jid_inf=$(sbatch --parsable "inf_before_${m}.sbatch")
    echo "  inf_before_${m}: job ${jid_inf}"

    jid_ft=$(sbatch --parsable "ft_${m}.sbatch")
    echo "  ft_${m}:         job ${jid_ft}"
done

echo ""
echo "=== Submitting CORRECT ANSWER jobs ==="
for m in "${MODELS[@]}"; do
    jid_inf=$(sbatch --parsable "inf_before_ca_${m}.sbatch")
    echo "  inf_before_ca_${m}: job ${jid_inf}"

    jid_ft=$(sbatch --parsable "ft_ca_${m}.sbatch")
    echo "  ft_ca_${m}:         job ${jid_ft}"
done

echo ""
echo "All jobs submitted. Check with: squeue -u \$USER"

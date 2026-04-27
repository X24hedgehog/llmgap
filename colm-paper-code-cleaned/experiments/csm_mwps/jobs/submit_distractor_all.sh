#!/bin/bash
# Submit all inference-before AND finetune jobs for the distractor task.
# Both run in parallel (no dependency).
#
# Usage: bash submit_distractor_all.sh

set -euo pipefail
cd "$(dirname "$0")"

MODELS=(qwen05b qwen15b qwen3b llama1b llama3b llama8b gemma2b gemma7b)

echo "=== Submitting DISTRACTOR inference-before jobs ==="
for m in "${MODELS[@]}"; do
    jid=$(sbatch --parsable "inf_before_${m}.sbatch")
    echo "  inf_before_${m}: job ${jid}"
done

echo ""
echo "=== Submitting DISTRACTOR finetune jobs ==="
for m in "${MODELS[@]}"; do
    jid=$(sbatch --parsable "ft_${m}.sbatch")
    echo "  ft_${m}: job ${jid}"
done

echo ""
echo "All 16 distractor jobs submitted. Check with: squeue -u \$USER"

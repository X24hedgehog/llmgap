#!/bin/bash
# Cancel all running/pending finetune jobs for the distractor task.
# Matches job names: ft_qwen05b, ft_qwen15b, ft_qwen3b, ft_llama1b, ft_llama3b, ft_llama8b, ft_gemma2b, ft_gemma7b
#
# Usage: bash cancel_ft_distractor.sh

set -euo pipefail

MODELS=(qwen05b qwen15b qwen3b llama1b llama3b llama8b gemma2b gemma7b)

echo "Cancelling distractor finetune jobs..."
for m in "${MODELS[@]}"; do
    job_name="ft_${m}"
    jids=$(squeue -u "$USER" -n "$job_name" -h -o "%i" 2>/dev/null || true)
    if [[ -n "$jids" ]]; then
        for jid in $jids; do
            scancel "$jid"
            echo "  Cancelled ${job_name}: job ${jid}"
        done
    else
        echo "  ${job_name}: no running/pending job found"
    fi
done

echo "Done."

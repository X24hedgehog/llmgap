#!/bin/bash
# Cancel finetune jobs that will exceed 20h, then resubmit with 30h limit.
# Jobs: ft_qwen3b, ft_llama3b, ft_llama8b, ft_gemma7b (running but too slow)
#        ft_gemma2b (OOM, now fixed with --load-in-4bit)
# All sbatch files have been updated to --time=30:00:00.
# finetune.py auto-resumes from checkpoints, so jobs >30h will need
# a second submission after timeout.

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Cancelling slow finetune jobs ==="
scancel 63693668   # ft_qwen3b   (~21h est)
scancel 63693670   # ft_llama3b  (~21h est)
scancel 63693671   # ft_llama8b  (~46h est)
scancel 63693673   # ft_gemma7b  (~45h est)
echo "Cancelled 4 jobs."

# Wait a moment for SLURM to process cancellations
sleep 3

echo ""
echo "=== Resubmitting with 30h time limit ==="
sbatch ft_gemma2b.sbatch   # was OOM, now fixed with --load-in-4bit
sbatch ft_gemma7b.sbatch
sbatch ft_qwen3b.sbatch
sbatch ft_llama3b.sbatch
sbatch ft_llama8b.sbatch
echo ""
echo "Submitted 5 jobs. ft_llama8b and ft_gemma7b (~45h) will need a second"
echo "submission after they time out at 30h — they auto-resume from checkpoints."

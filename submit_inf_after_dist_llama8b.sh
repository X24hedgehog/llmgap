#!/bin/bash
set -euo pipefail

cd /cluster/home/tunguyen1/llmgap
sbatch out/sbatch/distractor/inf_after_dist_llama8b.sbatch

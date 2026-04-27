#!/bin/bash
# Submit all 10 tree-generation shards.
# Usage: bash submit_all_shards.sh
#
# After all jobs finish, run:
#   python merge_shards.py
#   python build_ns_targets.py

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs

echo "Submitting 10 shards..."
for i in $(seq 0 9); do
    jid=$(sbatch --parsable "prepare_gsm8k_shard_$(printf "%02d" $i).sbatch")
    echo "  shard $(printf "%02d" $i) → job $jid"
done

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "After all complete, run:"
echo "  python merge_shards.py"
echo "  python build_ns_targets.py"

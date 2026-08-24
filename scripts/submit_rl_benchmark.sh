#!/bin/bash
set -euo pipefail

echo "Submitting RL benchmark jobs..."

jid_basic=$(sbatch --parsable scripts/sbatch_basic.sh)
echo "Submitted BASIC REINFORCE: $jid_basic"

jid_grpo=$(sbatch --parsable scripts/sbatch_grpo.sh)
echo "Submitted GRPO: $jid_grpo"

jid_sft=$(sbatch --parsable scripts/sbatch_grpo_sft.sh)
echo "Submitted GRPO+SFT: $jid_sft"

jid_orm=$(sbatch --parsable scripts/sbatch_grpo_orm_sft.sh)
echo "Submitted GRPO+ORM+SFT: $jid_orm"

echo "All jobs submitted."

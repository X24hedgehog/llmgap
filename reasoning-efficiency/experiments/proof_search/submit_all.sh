#!/bin/bash
# submit_all.sh — Submit the full experiment pipeline to Slurm.
#
# Pipeline (per model × task):
#   0. split_data.py        — add 'split' column to both main CSVs (local, once)
#   1. inference BEFORE     — run base model on test split  (no dependency)
#   2. finetune             — LoRA fine-tuning on train split (no dependency)
#   3. inference AFTER      — run fine-tuned model on test split (depends on 2)
#   4. merge_results        — write model columns to main CSVs (depends on all 1+3)
#   5. compute_gain         — print and save gain table (depends on 4)
#
# Usage:
#   bash submit_all.sh
#   bash submit_all.sh --dry-run   # print sbatch commands without submitting
#
# Qwen/Qwen2.5-7B-Instruct is reserved as the local judge for next_subquestion
# scoring and is therefore excluded from the list of models being tested.

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-2-2b-it"
    "google/gemma-2-7b-it"
)

TASKS=("correct_answer" "next_subquestion")

# ── helper: submit or dry-run ─────────────────────────────────────────────────
sbatch_submit() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY-RUN] sbatch $*" >&2
        echo "0"   # fake job id
    else
        sbatch "$@" | awk '{print $NF}'
    fi
}

# ── Step 0: add split column to both main CSVs (runs locally, fast) ──────────
echo ">>> Step 0: split_data.py"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] python split_data.py --seed 42 --test-ratio 0.2"
else
    cd "$SCRIPT_DIR"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda activate llmgap
    python split_data.py --seed 42 --test-ratio 0.2
fi

mkdir -p "$SCRIPT_DIR/out/interim"
mkdir -p "$SCRIPT_DIR/out/checkpoints"
mkdir -p "$SCRIPT_DIR/out/results"

# ── Submit jobs per (model, task) ─────────────────────────────────────────────
ALL_BEFORE_JOB_IDS=()
ALL_AFTER_JOB_IDS=()

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT="${MODEL##*/}"

    for TASK in "${TASKS[@]}"; do
        CHECKPOINT_DIR="out/checkpoints/${MODEL_SHORT}_${TASK}/final"

        echo ""
        echo "── $MODEL_SHORT / $TASK ──────────────────────────────────────────"

        # 1. Inference BEFORE fine-tuning (base model, no dependency)
        BEFORE_JID=$(sbatch_submit \
            --job-name="inf_b_${MODEL_SHORT:0:18}_${TASK:0:5}" \
            --export="ALL,MODEL_NAME=${MODEL},TASK=${TASK},MODE=before" \
            "$SCRIPT_DIR/run_inference.sbatch")
        echo "  inference-before job: $BEFORE_JID"
        ALL_BEFORE_JOB_IDS+=("$BEFORE_JID")

        # 2. Fine-tune (can run in parallel with BEFORE inference)
        FT_JID=$(sbatch_submit \
            --job-name="ft_${MODEL_SHORT:0:20}_${TASK:0:5}" \
            --export="ALL,MODEL_NAME=${MODEL},TASK=${TASK}" \
            "$SCRIPT_DIR/run_finetune.sbatch")
        echo "  finetune         job: $FT_JID"

        # 3. Inference AFTER fine-tuning (depends on fine-tune completing)
        AFTER_JID=$(sbatch_submit \
            --dependency="afterok:${FT_JID}" \
            --job-name="inf_a_${MODEL_SHORT:0:18}_${TASK:0:5}" \
            --export="ALL,MODEL_NAME=${MODEL},TASK=${TASK},MODE=after,CHECKPOINT=${CHECKPOINT_DIR}" \
            "$SCRIPT_DIR/run_inference.sbatch")
        echo "  inference-after  job: $AFTER_JID  (depends on ft $FT_JID)"
        ALL_AFTER_JOB_IDS+=("$AFTER_JID")
    done
done

# ── Step 4: merge_results (depends on ALL before + after inference jobs) ──────
ALL_INF_JOB_IDS=("${ALL_BEFORE_JOB_IDS[@]}" "${ALL_AFTER_JOB_IDS[@]}")
INF_DEPS=$(printf ":%s" "${ALL_INF_JOB_IDS[@]}")
INF_DEPS="afterok${INF_DEPS}"

echo ""
echo "── merge_results ────────────────────────────────────────────────────────"
MERGE_JID=$(sbatch_submit \
    --dependency="${INF_DEPS}" \
    --job-name="merge_results" \
    --ntasks=1 --cpus-per-task=2 --mem-per-cpu=8192 --time=00:30:00 \
    --output="slurm-%j.out" --error="slurm-%j.err" \
    --wrap="cd $SCRIPT_DIR && eval \"\$($HOME/miniconda3/bin/conda shell.bash hook)\" && conda activate llmgap && python merge_results.py")
echo "  merge_results  job: $MERGE_JID  (depends on all inference jobs)"

# ── Step 5: compute_gain (depends on merge) ───────────────────────────────────
echo ""
echo "── compute_gain ─────────────────────────────────────────────────────────"
GAIN_JID=$(sbatch_submit \
    --dependency="afterok:${MERGE_JID}" \
    --job-name="compute_gain" \
    --ntasks=1 --cpus-per-task=1 --mem-per-cpu=4096 --time=00:10:00 \
    --output="slurm-%j.out" --error="slurm-%j.err" \
    --wrap="cd $SCRIPT_DIR && eval \"\$($HOME/miniconda3/bin/conda shell.bash hook)\" && conda activate llmgap && python compute_gain.py")
echo "  compute_gain   job: $GAIN_JID  (depends on merge $MERGE_JID)"

echo ""
echo "=================================================================="
echo "All jobs submitted. Final outputs:"
echo "  out/correct_answer_pairs.csv     <- model columns written here"
echo "  out/next_subquestion_pairs.csv   <- model columns written here"
echo "  out/results/gain_summary.csv     <- performance gain table"
echo "=================================================================="

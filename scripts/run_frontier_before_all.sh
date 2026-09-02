#!/bin/bash
# Run --mode before inference for all frontier models across every (setting, task).
#
# Fill in the real API model ids / base URLs / key env vars for Kimi K3 and
# Muse Spark 1.2 below (placeholders marked TODO) before running.
#
# Usage:
#   MAX_ROWS=100 bash scripts/run_frontier_before_all.sh   # cheap smoke test
#   bash scripts/run_frontier_before_all.sh                # full test split

set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="out/frontier_eval"
MAX_ROWS="${MAX_ROWS:-}"          # set env var MAX_ROWS to cap rows per run
MAX_ROWS_FLAG=()
if [[ -n "$MAX_ROWS" ]]; then
    MAX_ROWS_FLAG=(--max-rows "$MAX_ROWS")
fi

mkdir -p "$OUT_DIR"

# ── models: name | provider | api_model | api_key_env | base_url (or "") ────
MODELS=(
    "gpt-5.6|openai|gpt-5.6|OPENAI_API_KEY|"
    "opus-5|anthropic|claude-opus-5|ANTHROPIC_API_KEY|"
    "kimi-k3|openai_compatible|kimi-k3|MOONSHOT_API_KEY|https://api.moonshot.cn/v1"          # TODO: confirm exact model id / base URL
    "muse-spark-1.2|openai_compatible|muse-spark-1.2|MUSESPARK_API_KEY|https://TODO-base-url" # TODO: confirm provider endpoint
)

# ── (setting, task, csv) triples — before-mode only, matches SFT dataset paths ──
DATASETS=(
    "reasoning_efficiency|correct_answer|reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs.csv"
    "reasoning_efficiency|next_subquestion|reasoning-efficiency/experiments/proof_search/out/next_subquestion_pairs.csv"
    "gsm8k|correct_answer|reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs_gsm8k.csv"
    "gsm8k|next_subquestion|reasoning-efficiency/experiments/proof_search/out/next_subquestion_pairs_gsm8k_v2.csv"
    "distractor|correct_answer|colm-paper-code-cleaned/experiments/csm_mwps/out/correct_answer_distractor_pairs.csv"
    "distractor|distractor|colm-paper-code-cleaned/experiments/csm_mwps/out/distractor_pairs.csv"
    "eedi_split_control|correct_answer|eedi/data/processed/correct_answer_pairs_eedi_split_control.csv"
    "eedi_split_control|distractor|eedi/data/processed/distractor_pairs_eedi_split_control.csv"
)

for model_row in "${MODELS[@]}"; do
    IFS='|' read -r name provider api_model key_env base_url <<< "$model_row"

    for ds_row in "${DATASETS[@]}"; do
        IFS='|' read -r setting task csv <<< "$ds_row"

        echo ""
        echo "=== ${name} | ${setting}/${task} ==="

        base_url_flag=()
        if [[ "$provider" == "openai_compatible" ]]; then
            base_url_flag=(--api-base-url "$base_url")
        fi

        python src/run_inference_frontier.py \
            --provider "$provider" \
            --api-model "$api_model" \
            --model-name "$name" \
            --api-key-env "$key_env" \
            "${base_url_flag[@]}" \
            --task "$task" \
            --mode before \
            --data-csv "$csv" \
            --out-dir "$OUT_DIR" \
            --suffix "_${setting}" \
            "${MAX_ROWS_FLAG[@]}"
    done
done

echo ""
echo "All before-mode runs finished. Aggregate with:"
echo "  python src/aggregate_frontier_before_results.py --results-dir $OUT_DIR"

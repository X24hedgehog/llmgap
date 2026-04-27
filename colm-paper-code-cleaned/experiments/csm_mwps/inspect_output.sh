#!/usr/bin/env bash
# Generate a small dataset (5 problems × 2 instantiations) and inspect the output.
# Run from: colm-paper-code-cleaned/experiments/csm_mwps/
set -e

OUT_DIR="out/inspect_test"
rm -rf "$OUT_DIR"

echo "=== Generating small dataset ==="
python prepare_distractor_dataset.py \
    --n-problems 5 \
    --n-inst 2 \
    --seed 42 \
    --out-dir "$OUT_DIR"

echo ""
echo "=== distractor_pairs.csv columns ==="
head -1 "$OUT_DIR/distractor_pairs.csv" | tr ',' '\n' | cat -n

echo ""
echo "=== First 3 distractor rows (key fields) ==="
python3 -c "
import pandas as pd, json, textwrap

df = pd.read_csv('$OUT_DIR/distractor_pairs.csv')
print(f'Total distractor rows: {len(df)}')
print()

for i, row in df.head(3).iterrows():
    print(f'--- Row {i} ---')
    print(f'Problem:            {row[\"problem\"][:120]}...')
    print(f'Correct answer:     {row[\"correct_answer\"]}')
    print(f'Misconception type: {row[\"misconception_type\"]}')
    print(f'Distractor answers: {row[\"target_distractor_answers\"]}')
    print()

    rts = json.loads(row['target_distractor_reasoning_traces'])
    explained = json.loads(row['target_distractor_explained_traces'])

    for j, (rt, exp) in enumerate(zip(rts, explained)):
        print(f'  Trace {j}:')
        print(textwrap.indent(rt[:300], '    '))
        print()
        print(f'  Explained trace {j}:')
        print(textwrap.indent(exp[:400], '    '))
        print()

    # Show where twist markers were inserted
    for j, exp in enumerate(explained):
        if 'Applying the twist' in exp:
            markers = [s.strip() for s in exp.split('.') if 'Applying the twist' in s]
            print(f'  Twist markers in trace {j}: {markers}')
    print()
"

echo ""
echo "=== correct_answer_distractor_pairs.csv stats ==="
python3 -c "
import pandas as pd
df = pd.read_csv('$OUT_DIR/correct_answer_distractor_pairs.csv')
print(f'Total rows: {len(df)}')
print(f'Train: {(df[\"split\"]==\"train\").sum()}, Test: {(df[\"split\"]==\"test\").sum()}')
print(f'Columns: {list(df.columns)}')
"

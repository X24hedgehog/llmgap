"""Print average scores for all CSV files in out/interim/.

For models with all 4 scores (ca_before, ca_after, ns_before, ns_after),
shows a single row per model with gains. Partial models listed separately.

Usage:
  python print_scores.py                  # original dataset
  python print_scores.py --dataset gsm8k  # gsm8k dataset
"""
import argparse
import re
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["original", "gsm8k"], default="original",
                    help="Which dataset results to show (default: original)")
args = parser.parse_args()

INTERIM_DIR = Path("out/interim")

# suffix patterns per dataset:
#   original: ca -> _cot,  ns -> (empty)
#   gsm8k:    ca -> _cot_gsm8k,  ns -> _gsm8k
FILENAME_RE = re.compile(
    r"^(?P<model>.+?)_(?P<task>correct_answer|next_subquestion)_(?P<mode>before|after)"
    r"(?P<suffix>_cot_gsm8k|_cot|_gsm8k|)\.csv$"
)

if args.dataset == "gsm8k":
    ca_suffix = "_cot_gsm8k"
    ns_suffix = "_gsm8k"
else:
    ca_suffix = "_cot"
    ns_suffix = ""

records = []
for path in sorted(INTERIM_DIR.glob("*.csv")):
    m = FILENAME_RE.match(path.name)
    if not m:
        continue
    model, task, mode, suffix = m.group("model"), m.group("task"), m.group("mode"), m.group("suffix")
    # Keep only files matching the chosen dataset
    if task == "correct_answer" and suffix != ca_suffix:
        continue
    if task == "next_subquestion" and suffix != ns_suffix:
        continue
    df = pd.read_csv(path)
    records.append({"model": model, "task": task, "mode": mode, "avg_score": df["score"].mean(), "n": len(df)})

if not records:
    print("No CSV files found in", INTERIM_DIR)
    raise SystemExit(0)

df_all = pd.DataFrame(records)

# Pivot to one row per model: columns = (task, mode)
pivot = df_all.pivot_table(index="model", columns=["task", "mode"], values="avg_score", aggfunc="first")
pivot.columns = [f"{t}_{m}" for t, m in pivot.columns]

FULL_COLS = ["correct_answer_before", "correct_answer_after", "next_subquestion_before", "next_subquestion_after"]
present_full = [c for c in FULL_COLS if c in pivot.columns]
complete = pivot.dropna(subset=present_full).copy() if len(present_full) == len(FULL_COLS) else pivot.iloc[0:0].copy()
partial  = pivot[~pivot.index.isin(complete.index)].copy()

# Compute gains only when both before/after exist
for col in FULL_COLS:
    if col not in complete.columns:
        complete[col] = float("nan")
complete["ca_gain"] = complete["correct_answer_after"]  - complete["correct_answer_before"]
complete["ns_gain"] = complete["next_subquestion_after"] - complete["next_subquestion_before"]

# ── helpers ───────────────────────────────────────────────────────────────────
def fmt(v):
    return f"{v:.4f}" if pd.notna(v) else "  —   "

def gain_str(v):
    if pd.isna(v):
        return "   —   "
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.4f}"

# ── complete table ────────────────────────────────────────────────────────────
MW = 34
print(f"\n{'═'*110}")
print("  MODELS WITH ALL 4 SCORES")
print(f"{'═'*110}")
header = (
    f"{'Model':<{MW}}"
    f"{'CA before':>10}  {'CA after':>10}  {'CA gain':>9}  "
    f"{'NS before':>10}  {'NS after':>10}  {'NS gain':>9}"
)
print(header)
print("-" * len(header))
for model, row in complete.sort_values("model").iterrows():
    print(
        f"{model:<{MW}}"
        f"{fmt(row.get('correct_answer_before')):>10}  "
        f"{fmt(row.get('correct_answer_after')):>10}  "
        f"{gain_str(row.get('ca_gain')):>9}  "
        f"{fmt(row.get('next_subquestion_before')):>10}  "
        f"{fmt(row.get('next_subquestion_after')):>10}  "
        f"{gain_str(row.get('ns_gain')):>9}"
    )

# ── partial table ─────────────────────────────────────────────────────────────
if not partial.empty:
    print(f"\n{'─'*110}")
    print("  PARTIAL RESULTS (missing some scores)")
    print(f"{'─'*110}")
    print(header)
    print("-" * len(header))
    for model, row in partial.sort_values("model").iterrows():
        print(
            f"{model:<{MW}}"
            f"{fmt(row.get('correct_answer_before')):>10}  "
            f"{fmt(row.get('correct_answer_after')):>10}  "
            f"{'':>9}  "
            f"{fmt(row.get('next_subquestion_before')):>10}  "
            f"{fmt(row.get('next_subquestion_after')):>10}  "
            f"{'':>9}"
        )
print()

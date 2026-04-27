"""Compare common columns between correct_answer_pairs.csv and correct_answer_pairs_gsm8k.csv."""
import textwrap
import pandas as pd

ORIG = "out/correct_answer_pairs.csv"
GSM  = "out/correct_answer_pairs_gsm8k.csv"
N_SAMPLES = 3
WRAP_WIDTH = 100

orig = pd.read_csv(ORIG)
gsm  = pd.read_csv(GSM)

common = [c for c in gsm.columns if c in orig.columns]
print(f"Original : {len(orig)} rows, {orig.columns.tolist()}")
print(f"GSM8K    : {len(gsm)}  rows, {gsm.columns.tolist()}")
print(f"\nCommon columns: {common}\n")
print("=" * 120)

for col in common:
    print(f"\n{'=' * 120}")
    print(f"  COLUMN: '{col}'")
    print(f"{'=' * 120}")

    orig_nulls = orig[col].isna().sum()
    gsm_nulls  = gsm[col].isna().sum()
    orig_unique = orig[col].nunique()
    gsm_unique  = gsm[col].nunique()
    print(f"  Nulls   — original: {orig_nulls}/{len(orig)}   gsm8k: {gsm_nulls}/{len(gsm)}")
    print(f"  Unique  — original: {orig_unique}              gsm8k: {gsm_unique}")

    orig_sample = orig[col].dropna().iloc[:N_SAMPLES].tolist()
    gsm_sample  = gsm[col].dropna().iloc[:N_SAMPLES].tolist()

    for i, (ov, gv) in enumerate(zip(orig_sample, gsm_sample)):
        ov_str = str(ov)
        gv_str = str(gv)
        print(f"\n  -- Sample {i+1} --")
        print(f"  [original] {textwrap.shorten(ov_str, width=WRAP_WIDTH, placeholder='...')}")
        print(f"  [gsm8k  ] {textwrap.shorten(gv_str, width=WRAP_WIDTH, placeholder='...')}")

print(f"\n{'=' * 120}")

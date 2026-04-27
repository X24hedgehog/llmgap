#!/usr/bin/env python3
"""Remove rows from distractor_pairs.csv where the gold distractor set has more than 7 answers."""
import json
import pandas as pd

CSV_PATH = "out/distractor_pairs.csv"

df = pd.read_csv(CSV_PATH)
before = len(df)

lengths = df["target_distractor_answers"].apply(lambda x: len(json.loads(x)))
df = df[lengths <= 7].reset_index(drop=True)

after = len(df)
print(f"Removed {before - after} rows with >7 distractors ({before} -> {after})")

df.to_csv(CSV_PATH, index=False)
print(f"Saved -> {CSV_PATH}")

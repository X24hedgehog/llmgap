"""
merge_shards.py

Merges shard CSVs into one final CSV per task.

Shard naming convention:
  out/gsm8k_trees_<start>_<end>.csv   (produced by prepare_gsm8k_v2.py)
  out/correct_answer_pairs_gsm8k_<start>_<end>.csv
  out/next_subquestion_pairs_gsm8k_<start>_<end>.csv

Outputs:
  out/gsm8k_trees.csv                 ← feed into build_ns_targets.py
  out/correct_answer_pairs_gsm8k.csv
  out/next_subquestion_pairs_gsm8k.csv
"""

from pathlib import Path
import pandas as pd

OUT_DIR = Path("out")


def merge(prefix: str, output_name: str | None = None) -> None:
    def _shard_start(p: Path) -> int:
        parts = p.stem.split("_")
        try:
            return int(parts[-2])
        except (ValueError, IndexError):
            return -1

    shards = sorted(
        (
            p for p in OUT_DIR.glob(f"{prefix}_*_*.csv")
            if p.stem.split("_")[-1].isdigit() and p.stem.split("_")[-2].isdigit()
        ),
        key=_shard_start,
    )
    if not shards:
        print(f"No shard files found for prefix '{prefix}', skipping.")
        return

    print(f"Merging {len(shards)} shards for '{prefix}':")
    dfs = []
    for s in shards:
        df = pd.read_csv(s)
        print(f"  {s.name}: {len(df):,} rows")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    out_path = OUT_DIR / f"{output_name or prefix}.csv"
    merged.to_csv(out_path, index=False)
    print(f"  → Saved {len(merged):,} rows to {out_path}\n")


# Tree shards from prepare_gsm8k_v2.py  (feed into build_ns_targets.py next)
merge("gsm8k_trees")

# Legacy v1 shards (if present)
merge("correct_answer_pairs_gsm8k")
merge("next_subquestion_pairs_gsm8k")

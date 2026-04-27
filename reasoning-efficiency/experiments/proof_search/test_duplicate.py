import pandas as pd

for path in [
    "out/correct_answer_pairs.csv",
    "out/next_subquestion_pairs.csv",
]:
    df = pd.read_csv(path)
    total = len(df)
    unique = df["prompt"].nunique()
    dupes = df[df.duplicated(subset=["prompt"], keep=False)]

    print(f"\n{'='*60}")
    print(f"File   : {path}")
    print(f"Rows   : {total}")
    print(f"Unique prompts: {unique}")
    print(f"Duplicated rows: {len(dupes)}")

    if len(dupes) > 0:
        counts = df.groupby("prompt").size().sort_values(ascending=False)
        repeated = counts[counts > 1]
        print(f"Prompts appearing >1 time: {len(repeated)}")
        print(f"\nTop 5 most repeated:")
        for prompt, n in repeated.head(5).items():
            print(f"  x{n} | {prompt[:120]}")

        # ── inspect 2 rows sharing the most common duplicated prompt ──────
        most_common_prompt = repeated.index[0]
        example_rows = df[df["prompt"] == most_common_prompt].head(2)
        print(f"\n--- Example: 2 rows with the same prompt ---")
        for i, pos in enumerate(example_rows.index):
            print(f"  [Row {i+1}] file position={pos}")
    else:
        print("No duplicate prompts found.")
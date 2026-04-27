#!/usr/bin/env python3
"""Inspect interim result CSVs and print a summary table of scores."""

import argparse
import ast
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "out"
EPOCHS = 3  # expected number of checkpoints

SETTINGS = {
    "distractor": {"tasks": ["correct_answer", "distractor"]},
    "gsm8k": {"tasks": ["correct_answer", "next_subquestion"]},
    "reasoning_efficiency": {"tasks": ["correct_answer", "next_subquestion"]},
}

# Parse filename: {model}_{task}_{before|after}.csv
FILENAME_RE = re.compile(r"^(.+?)_(correct_answer|next_subquestion|distractor)_(before|after)\.csv$")


def read_score(path: Path, distractor_golds=None) -> tuple[float, int]:
    """Read a result CSV and return (accuracy, n).

    If distractor_golds is provided (list of gold strings, one per test row),
    compute regex-based score (last number match) instead of using stored score.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if distractor_golds is not None:
        n = min(len(rows), len(distractor_golds))
        scores = []
        for i in range(n):
            gold_nums = set(ast.literal_eval(distractor_golds[i]))
            all_nums = re.findall(r'-?\b\d+\b', rows[i]["prediction"])
            if all_nums and int(all_nums[-1]) in gold_nums:
                scores.append(1)
            else:
                scores.append(0)
    else:
        scores = [int(row["score"]) for row in rows]

    return sum(scores) / len(scores), len(scores)


def get_ft_status(ckpt_dir: Path) -> str:
    """Return finetune status string like '3/3 done (best=0.963)' or '1/3'."""
    if not ckpt_dir.exists():
        return "---"
    checkpoints = sorted(ckpt_dir.glob("checkpoint-*"))
    n_ckpt = len(checkpoints)
    best_file = ckpt_dir / "best_checkpoint.json"
    if best_file.exists():
        try:
            info = json.loads(best_file.read_text())
            acc = info.get("accuracy", "?")
            return f"{n_ckpt}/{EPOCHS} done (best={acc:.1%})"
        except Exception:
            return f"{n_ckpt}/{EPOCHS} done"
    if n_ckpt == 0:
        return "---"
    return f"{n_ckpt}/{EPOCHS}"


def main():
    # Collect: results[setting][task][model][phase] = (accuracy, n)
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # Collect: ft_status[setting][task][model] = status string
    ft_status = defaultdict(lambda: defaultdict(dict))

    # Preload distractor gold labels for regex scoring
    _distractor_golds_cache = {}
    for task in ["correct_answer", "distractor"]:
        key = ("distractor", task)
        data_csv = DATA_CSVS.get(key)
        if data_csv and data_csv.exists() and task == "distractor":
            with open(data_csv, newline="") as f:
                reader = csv.DictReader(f)
                _distractor_golds_cache[key] = [
                    r["target_distractor_answers"]
                    for r in reader if r.get("split") == "test"
                ]

    for setting in SETTINGS:
        interim = OUT_DIR / setting / "results" / "interim"
        if interim.exists():
            for csv_file in sorted(interim.glob("*.csv")):
                m = FILENAME_RE.match(csv_file.name)
                if not m:
                    continue
                model, task, phase = m.groups()
                # Use regex scoring for distractor task
                golds = _distractor_golds_cache.get((setting, task))
                acc, n = read_score(csv_file, distractor_golds=golds)
                results[setting][task][model][phase] = (acc, n)

        # Scan checkpoint dirs
        ckpt_root = OUT_DIR / setting / "results" / "checkpoints"
        if ckpt_root.exists():
            for ckpt_dir in sorted(ckpt_root.iterdir()):
                if not ckpt_dir.is_dir():
                    continue
                # dir name: {model}_{task}
                name = ckpt_dir.name
                for task in SETTINGS[setting]["tasks"]:
                    if name.endswith(f"_{task}"):
                        model = name[: -len(f"_{task}")]
                        ft_status[setting][task][model] = get_ft_status(ckpt_dir)
                        # ensure model appears in results
                        if model not in results[setting][task]:
                            results[setting][task][model] = {}
                        break

    if not results and not ft_status:
        print("No result CSVs found.")
        return

    # Print
    for setting in ["reasoning_efficiency", "gsm8k", "distractor"]:
        if setting not in results:
            continue
        print(f"\n{'=' * 90}")
        print(f"  Setting: {setting}")
        print(f"{'=' * 90}")

        for task in SETTINGS[setting]["tasks"]:
            if task not in results[setting]:
                continue
            print(f"\n  Task: {task}")
            print(f"  {'Model':<30} {'Before':>8} {'After':>8} {'Delta':>8}  {'Finetune':>22}")
            print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 22}")

            for model in sorted(results[setting][task]):
                data = results[setting][task][model]
                before = data.get("before")
                after = data.get("after")

                before_str = f"{before[0]:.1%}" if before else "---"
                after_str = f"{after[0]:.1%}" if after else "---"

                if before and after:
                    delta = after[0] - before[0]
                    sign = "+" if delta >= 0 else ""
                    delta_str = f"{sign}{delta:.1%}"
                else:
                    delta_str = "---"

                ft_str = ft_status.get(setting, {}).get(task, {}).get(model, "---")

                print(f"  {model:<30} {before_str:>8} {after_str:>8} {delta_str:>8}  {ft_str:>22}")

    # Overall counts
    total_before = sum(
        1 for s in results.values() for t in s.values()
        for m in t.values() if "before" in m
    )
    total_after = sum(
        1 for s in results.values() for t in s.values()
        for m in t.values() if "after" in m
    )
    total_ft_done = sum(
        1 for s in ft_status.values() for t in s.values()
        for st in t.values() if "done" in st
    )
    total_ft = sum(
        1 for s in ft_status.values() for t in s.values()
        for st in t.values() if st != "---"
    )
    print(f"\n{'=' * 90}")
    print(f"  Total: {total_before} before, {total_after} after, {total_ft_done}/{total_ft} finetune completed")
    print(f"{'=' * 90}")


# ── Data CSV paths for joining predictions with source data ───────────────────

BASE = Path(__file__).resolve().parent

DATA_CSVS = {
    ("distractor", "correct_answer"): BASE / "colm-paper-code-cleaned/experiments/csm_mwps/out/correct_answer_distractor_pairs.csv",
    ("distractor", "distractor"): BASE / "colm-paper-code-cleaned/experiments/csm_mwps/out/distractor_pairs.csv",
    ("gsm8k", "correct_answer"): BASE / "reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs_gsm8k.csv",
    ("gsm8k", "next_subquestion"): BASE / "reasoning-efficiency/experiments/proof_search/out/next_subquestion_pairs_gsm8k_v2.csv",
    ("reasoning_efficiency", "correct_answer"): BASE / "reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs.csv",
    ("reasoning_efficiency", "next_subquestion"): BASE / "reasoning-efficiency/experiments/proof_search/out/next_subquestion_pairs.csv",
}

TASK_TARGET_COL = {
    "correct_answer": "target_answer",
    "next_subquestion": "next_subquestion",
    "distractor": "target_distractor_answers",
}


def inspect_wrong(setting, task, model, phase="before", n=20, seed=42):
    """Print n random wrong predictions with full context from the source data."""
    result_csv = OUT_DIR / setting / "results" / "interim" / f"{model}_{task}_{phase}.csv"
    if not result_csv.exists():
        print(f"Result CSV not found: {result_csv}")
        return

    data_csv = DATA_CSVS.get((setting, task))
    if not data_csv or not data_csv.exists():
        print(f"Source data CSV not found for {setting}/{task}")
        return

    # Read result CSV
    with open(result_csv, newline="") as f:
        reader = csv.DictReader(f)
        result_rows = list(reader)

    # Read source CSV (test split only, same order as inference)
    with open(data_csv, newline="") as f:
        reader = csv.DictReader(f)
        source_rows = [r for r in reader if r.get("split") == "test"]

    if len(result_rows) != len(source_rows):
        print(f"WARNING: result rows ({len(result_rows)}) != source test rows ({len(source_rows)})")
        # Truncate to min length
        min_len = min(len(result_rows), len(source_rows))
        result_rows = result_rows[:min_len]
        source_rows = source_rows[:min_len]

    # Find wrong predictions
    target_col = TASK_TARGET_COL[task]
    wrong_indices = [i for i, r in enumerate(result_rows) if int(r["score"]) == 0]

    if not wrong_indices:
        print(f"No wrong predictions found for {model} on {setting}/{task}/{phase}!")
        return

    # Try to find a "false negative": wrong prediction that actually contains a gold answer
    smart_pick = None
    if task == "distractor":
        for idx in wrong_indices:
            src = source_rows[idx]
            pred = result_rows[idx]["prediction"]
            gold_str = src.get(target_col, "")
            # Parse gold distractor list like "[50, 66, 84]"
            gold_nums = re.findall(r"[\d.]+", gold_str)
            for gn in gold_nums:
                # Check if gold number appears as standalone number in prediction
                if re.search(r'(?<!\d)' + re.escape(gn) + r'(?!\d)', pred):
                    smart_pick = idx
                    break
            if smart_pick is not None:
                break

    if smart_pick is not None:
        sample = [smart_pick]
        pick_reason = "Smart pick: prediction contains a gold distractor number but was scored 0"
    else:
        random.seed(seed)
        sample = random.sample(wrong_indices, min(n, len(wrong_indices)))
        pick_reason = "Random sample (no false-negative candidate found)"

    print(f"\n{'#' * 90}")
    print(f"  Inspecting {len(sample)} WRONG predictions")
    print(f"  Setting: {setting} | Task: {task} | Model: {model} | Phase: {phase}")
    print(f"  Total wrong: {len(wrong_indices)}/{len(result_rows)}")
    print(f"  Selection: {pick_reason}")
    print(f"{'#' * 90}")

    for rank, idx in enumerate(sample, 1):
        src = source_rows[idx]
        res = result_rows[idx]

        print(f"\n{'─' * 90}")
        print(f"  [{rank}/{len(sample)}]  Row index: {idx}")
        print(f"{'─' * 90}")

        # Show relevant source fields
        if "problem" in src:
            print(f"  PROBLEM:          {src['problem'][:300]}")
        if "correct_answer" in src:
            print(f"  CORRECT ANSWER:   {src['correct_answer']}")
        if target_col in src:
            gold = src[target_col]
            print(f"  GOLD ({target_col}): {gold[:300]}")
        if "misconception_type" in src:
            print(f"  MISCONCEPTION:    {src['misconception_type']}")

        print()
        pred = res["prediction"]
        print(f"  PREDICTION:")
        print(pred)
        print(f"\n  SCORE:            {res['score']}")

    print(f"\n{'#' * 90}\n")


def regex_score_distractor(prediction: str, gold_str: str) -> int:
    """Return 1 if the last number in the prediction matches any gold distractor."""
    gold_nums = set(ast.literal_eval(gold_str))  # e.g. {91} or {50, 66, 84}
    all_nums = re.findall(r'-?\b\d+\b', prediction)
    if not all_nums:
        return 0
    last_num = int(all_nums[-1])
    return 1 if last_num in gold_nums else 0


def rescore():
    """Re-score all distractor task results with regex matching and compare to LLM judge."""
    setting = "distractor"
    task = "distractor"
    target_col = TASK_TARGET_COL[task]

    data_csv = DATA_CSVS.get((setting, task))
    if not data_csv or not data_csv.exists():
        print(f"Source data CSV not found for {setting}/{task}")
        return

    # Read source CSV (test split only)
    with open(data_csv, newline="") as f:
        reader = csv.DictReader(f)
        source_rows = [r for r in reader if r.get("split") == "test"]

    interim = OUT_DIR / setting / "results" / "interim"
    if not interim.exists():
        print("No interim results directory found.")
        return

    # Find all distractor result CSVs
    csvs = sorted(interim.glob(f"*_{task}_*.csv"))
    if not csvs:
        print("No distractor result CSVs found.")
        return

    print(f"\n{'=' * 100}")
    print(f"  Distractor Re-scoring: LLM Judge vs Regex Number Match")
    print(f"{'=' * 100}")
    print(f"  {'Model':<30} {'Phase':<8} {'Judge':>8} {'Regex':>8} {'Delta':>8}  {'Flipped':>10}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 10}")

    for csv_file in csvs:
        m = FILENAME_RE.match(csv_file.name)
        if not m:
            continue
        model, t, phase = m.groups()
        if t != task:
            continue

        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            result_rows = list(reader)

        n = min(len(result_rows), len(source_rows))

        judge_correct = 0
        regex_correct = 0
        flipped_0_to_1 = 0  # judge=0 but regex=1 (false negatives)
        flipped_1_to_0 = 0  # judge=1 but regex=0 (false positives)

        for i in range(n):
            judge_s = int(result_rows[i]["score"])
            regex_s = regex_score_distractor(
                result_rows[i]["prediction"],
                source_rows[i][target_col],
            )
            judge_correct += judge_s
            regex_correct += regex_s
            if judge_s == 0 and regex_s == 1:
                flipped_0_to_1 += 1
            elif judge_s == 1 and regex_s == 0:
                flipped_1_to_0 += 1

        j_acc = judge_correct / n
        r_acc = regex_correct / n
        delta = r_acc - j_acc
        sign = "+" if delta >= 0 else ""
        flip_str = f"+{flipped_0_to_1}/-{flipped_1_to_0}"

        print(f"  {model:<30} {phase:<8} {j_acc:>7.1%} {r_acc:>7.1%} {sign}{delta:>7.1%}  {flip_str:>10}")

    print(f"\n  Flipped column: +N = judge missed (false neg), -N = judge hallucinated (false pos)")
    print(f"{'=' * 100}\n")


def show_generations(setting, task, models, phase="before", row_idx=0):
    """Print full generations from 1-2 models side by side for a single data point."""
    data_csv = DATA_CSVS.get((setting, task))
    if not data_csv or not data_csv.exists():
        print(f"Source data CSV not found for {setting}/{task}")
        return

    target_col = TASK_TARGET_COL[task]

    # Read source CSV (test split only)
    with open(data_csv, newline="") as f:
        reader = csv.DictReader(f)
        source_rows = [r for r in reader if r.get("split") == "test"]

    if row_idx >= len(source_rows):
        print(f"Row index {row_idx} out of range (max {len(source_rows) - 1})")
        return

    src = source_rows[row_idx]

    # Print question context
    print(f"\n{'=' * 100}")
    print(f"  Data Point: row {row_idx} (test split)")
    print(f"{'=' * 100}")
    print(f"\n  PROBLEM:")
    print(f"  {src.get('problem', 'N/A')}")
    print(f"\n  CORRECT ANSWER: {src.get('correct_answer', 'N/A')}")
    print(f"\n  GOLD DISTRACTOR ANSWERS: {src.get(target_col, 'N/A')}")
    if "misconception_type" in src:
        print(f"  MISCONCEPTION TYPE:      {src['misconception_type']}")
    print()

    # Print each model's generation
    for model in models:
        result_csv = OUT_DIR / setting / "results" / "interim" / f"{model}_{task}_{phase}.csv"
        if not result_csv.exists():
            print(f"  [{model}] — result CSV not found: {result_csv.name}")
            continue

        with open(result_csv, newline="") as f:
            reader = csv.DictReader(f)
            result_rows = list(reader)

        if row_idx >= len(result_rows):
            print(f"  [{model}] — row {row_idx} out of range")
            continue

        res = result_rows[row_idx]
        judge_score = int(res["score"])
        regex_s = regex_score_distractor(res["prediction"], src[target_col]) if task == "distractor" else "N/A"

        print(f"{'─' * 100}")
        print(f"  MODEL: {model}  |  Phase: {phase}  |  Judge score: {judge_score}  |  Regex score: {regex_s}")
        print(f"{'─' * 100}")
        print()
        print(res["prediction"])
        print()

    print(f"{'=' * 100}\n")


def false_negatives(setting, task, model, phase="before", idx=None):
    """Find datapoints where judge=0 but regex=1.
    
    If idx is None: print the list of all false-negative indices.
    If idx is an int: print full details for that index in the false-neg list.
    """
    data_csv = DATA_CSVS.get((setting, task))
    if not data_csv or not data_csv.exists():
        print(f"Source data CSV not found for {setting}/{task}")
        return

    target_col = TASK_TARGET_COL[task]

    result_csv = OUT_DIR / setting / "results" / "interim" / f"{model}_{task}_{phase}.csv"
    if not result_csv.exists():
        print(f"Result CSV not found: {result_csv}")
        return

    with open(result_csv, newline="") as f:
        result_rows = list(csv.DictReader(f))
    with open(data_csv, newline="") as f:
        source_rows = [r for r in csv.DictReader(f) if r.get("split") == "test"]

    rows_n = min(len(result_rows), len(source_rows))

    # Build list of false-negative rows (judge=0, regex=1)
    fn_rows = []
    for i in range(rows_n):
        judge_s = int(result_rows[i]["score"])
        regex_s = regex_score_distractor(result_rows[i]["prediction"], source_rows[i][target_col])
        if judge_s == 0 and regex_s == 1:
            fn_rows.append(i)

    print(f"\n  False Negatives: judge=0 but regex=1")
    print(f"  Model: {model} | Setting: {setting} | Task: {task} | Phase: {phase}")
    print(f"  Total: {len(fn_rows)} / {rows_n}\n")

    if not fn_rows:
        print("  No false negatives found.\n")
        return

    if idx is None:
        # Print compact table of all false-negative indices
        print(f"  {'FN#':<6} {'Row':<8} {'Gold Distractors':<30} {'Misconception'}")
        print(f"  {'-'*6} {'-'*8} {'-'*30} {'-'*30}")
        for fn_i, row_i in enumerate(fn_rows):
            src = source_rows[row_i]
            gold = src.get(target_col, "")
            misc = src.get("misconception_type", "")
            print(f"  {fn_i:<6} {row_i:<8} {gold:<30} {misc}")
        print(f"\n  Use --idx <FN#> to inspect a specific row.\n")
    else:
        if idx < 0 or idx >= len(fn_rows):
            print(f"  Index {idx} out of range (0-{len(fn_rows)-1})")
            return
        row_i = fn_rows[idx]
        src = source_rows[row_i]
        res = result_rows[row_i]

        print(f"{'─' * 100}")
        print(f"  FN#{idx}  (test row {row_i})")
        print(f"{'─' * 100}")
        print(f"  PROBLEM:        {src.get('problem', 'N/A')}")
        print(f"  CORRECT ANSWER: {src.get('correct_answer', 'N/A')}")
        print(f"  GOLD DISTRACTORS: {src.get(target_col, 'N/A')}")
        if "misconception_type" in src:
            print(f"  MISCONCEPTION:  {src['misconception_type']}")
        print(f"\n  GENERATION (judge=0, regex=1):")
        print(res["prediction"])
        print(f"\n{'─' * 100}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true",
                        help="Inspect wrong predictions instead of showing summary")
    parser.add_argument("--rescore", action="store_true",
                        help="Re-score distractor results with regex matching")
    parser.add_argument("--show", action="store_true",
                        help="Show full generations for 1-2 models on a single data point")
    parser.add_argument("--false-neg", action="store_true",
                        help="Show cases where judge=0 but regex=1 (false negatives)")
    parser.add_argument("--idx", type=int, default=None,
                        help="Index into the false-neg list to inspect (--false-neg mode)")
    parser.add_argument("--models", nargs="+",
                        default=["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct"],
                        help="Model(s) to show generations for (--show mode)")
    parser.add_argument("--row", type=int, default=0,
                        help="Row index in the test split to display (--show mode)")
    parser.add_argument("--setting", default="distractor")
    parser.add_argument("--task", default="distractor")
    parser.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--phase", default="before")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.inspect:
        inspect_wrong(args.setting, args.task, args.model, args.phase, args.n, args.seed)
    elif args.rescore:
        rescore()
    elif args.show:
        show_generations(args.setting, args.task, args.models, args.phase, args.row)
    elif args.false_neg:
        false_negatives(args.setting, args.task, args.model, args.phase, args.idx)
    else:
        main()

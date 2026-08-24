#!/usr/bin/env python3
"""Inspect interim result CSVs and print a summary table of scores."""

import argparse
import ast
import csv
import json
import os
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional

BASE = Path(__file__).resolve().parent
LEGACY_OUT_DIR = BASE / "out"
SCRATCH_OUTPUT_DIR = BASE / "output"
EPOCHS = 3  # expected number of checkpoints

SETTINGS = {
    "eedi": {"tasks": ["correct_answer", "distractor"]},
    "eedi_split_control": {"tasks": ["correct_answer", "distractor"]},
    "distractor": {"tasks": ["correct_answer", "distractor"]},
    "gsm8k": {"tasks": ["correct_answer", "next_subquestion"]},
    "reasoning_efficiency": {"tasks": ["correct_answer", "next_subquestion"]},
}

SETTING_PRINT_ORDER = ["eedi", "eedi_split_control", "reasoning_efficiency", "gsm8k", "distractor"]

SETTING_ROOTS = {
    "eedi": SCRATCH_OUTPUT_DIR,
    "eedi_split_control": SCRATCH_OUTPUT_DIR,
    "reasoning_efficiency": SCRATCH_OUTPUT_DIR,
    "gsm8k": SCRATCH_OUTPUT_DIR,
    "distractor": SCRATCH_OUTPUT_DIR,
}

MODEL_ALIASES = {
    "Qwen2.5-0.5B-Instruct": "qwen05b",
    "Qwen2.5-1.5B-Instruct": "qwen15b",
    "Qwen2.5-3B-Instruct": "qwen3b",
    "Llama-3.2-1B-Instruct": "llama1b",
    "Llama-3.2-3B-Instruct": "llama3b",
    "Meta-Llama-3.1-8B-Instruct": "llama8b",
    "gemma-2-2b-it": "gemma2b",
    "gemma-7b-it": "gemma7b",
}

MODEL_ORDER = [
    "qwen05b",
    "qwen15b",
    "qwen3b",
    "llama1b",
    "llama3b",
    "llama8b",
    "gemma2b",
    "gemma7b",
]

TASK_SHORT = {
    "correct_answer": "ca",
    "next_subquestion": "ns",
    "distractor": "dist",
}

TASK_LONG = {value: key for key, value in TASK_SHORT.items()}
SEMANTIC_DISTRACTOR_SETTINGS = {"eedi", "eedi_split_control"}

# Parse filename: {model}_{task}_{before|after}.csv
FILENAME_RE = re.compile(r"^(.+?)_(correct_answer|next_subquestion|distractor)_(before|after)\.csv$")
SBATCH_RE = re.compile(r"^(inf_before|ft|inf_after)_([^_]+)_([^\.]+)\.sbatch$")


def get_setting_root(setting: str) -> Path:
    return SETTING_ROOTS[setting]


def canonical_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def ordered_models(models):
    extras = sorted(m for m in models if m not in MODEL_ORDER)
    return [m for m in MODEL_ORDER if m in models] + extras


def print_aligned_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))

    def format_row(row_values):
        cells = [f" {str(value):<{widths[idx]}} " for idx, value in enumerate(row_values)]
        return "|" + "|".join(cells) + "|"

    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    print(separator)
    print(format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))
    print(separator)


def build_setting_table_rows(rows, setting: str):
    task_names = SETTINGS[setting]["tasks"]
    task_shorts = [TASK_SHORT[name] for name in task_names]

    model_names = {
        model for (row_setting, _, model) in rows
        if row_setting == setting
    }

    headers = ["Model"]
    for task_short in task_shorts:
        headers.extend([
            f"{task_short}_before",
            f"{task_short}_ft",
            f"{task_short}_after",
        ])

    table_rows = []
    for model in ordered_models(model_names):
        row_values = [model]
        for task_short in task_shorts:
            task_row = rows[(setting, task_short, model)]
            row_values.extend([
                task_row["inf_before"],
                task_row["ft"],
                task_row["inf_after"],
            ])
        table_rows.append(row_values)

    return headers, table_rows


def get_live_jobs() -> dict[str, dict[str, str]]:
    jobs = {}
    try:
        result = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%T|%R"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return jobs

    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        jobid, state, reason = line.split("|", 2)
        try:
            details = subprocess.run(
                ["scontrol", "show", "job", jobid],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            continue
        match = re.search(r"Command=(\S+)", details.stdout)
        if match:
            jobs[str(Path(match.group(1)).resolve())] = {
                "state": state,
                "reason": reason,
            }
    return jobs


def get_failed_jobs() -> dict[str, str]:
    failed = {}
    try:
        result = subprocess.run(
            ["sacct", "-n", "-X", "-u", os.environ.get("USER", ""), "--format=JobIDRaw,State"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return failed

    interesting = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 2:
            continue
        jobid, state = parts
        if not jobid or "." in jobid or state not in interesting:
            continue
        try:
            details = subprocess.run(
                ["scontrol", "show", "job", jobid],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            continue
        match = re.search(r"Command=(\S+)", details.stdout)
        if match:
            failed[str(Path(match.group(1)).resolve())] = state.lower()
    return failed


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


def get_ft_accuracy(ckpt_dir: Path) -> Optional[str]:
    best_file = ckpt_dir / "best_checkpoint.json"
    if not best_file.exists():
        return None
    try:
        info = json.loads(best_file.read_text())
    except Exception:
        return "failed"
    acc = info.get("accuracy")
    if acc is None:
        return "failed"
    return f"{acc:.1%}"


def summarize_state(live_job, failed_state: Optional[str]) -> str:
    if live_job:
        return "running" if live_job["state"] == "RUNNING" else "pending"
    if failed_state:
        return "failed"
    return "not started"


def infer_step_value(setting: str, step: str, sbatch_path: Path, distractor_golds_cache) -> str:
    text = sbatch_path.read_text()
    live_jobs = infer_step_value.live_jobs
    failed_jobs = infer_step_value.failed_jobs
    live_job = live_jobs.get(str(sbatch_path.resolve()))
    failed_state = failed_jobs.get(str(sbatch_path.resolve()))

    if step in {"inf_before", "inf_after"}:
        model_match = re.search(r'--model-name "([^"]+)"', text)
        task_match = re.search(r"--task (\S+)", text)
        mode_match = re.search(r"--mode (\S+)", text)
        out_match = re.search(r"--out-dir (\S+)", text)
        if not (model_match and task_match and mode_match and out_match):
            return summarize_state(live_job, failed_state)
        model_name = model_match.group(1).split("/")[-1]
        task = task_match.group(1)
        mode = mode_match.group(1)
        result_path = Path(out_match.group(1)) / f"{model_name}_{task}_{mode}.csv"
        if result_path.exists():
            golds = (
                None
                if setting in SEMANTIC_DISTRACTOR_SETTINGS
                else distractor_golds_cache.get((setting, task))
            )
            acc, _ = read_score(result_path, distractor_golds=golds)
            return f"{acc:.1%}"
        return summarize_state(live_job, failed_state)

    out_match = re.search(r"--out-dir (\S+)", text)
    if not out_match:
        return summarize_state(live_job, failed_state)
    ft_acc = get_ft_accuracy(Path(out_match.group(1)))
    if ft_acc is not None:
        return ft_acc
    return summarize_state(live_job, failed_state)


infer_step_value.live_jobs = {}
infer_step_value.failed_jobs = {}


# ── PRM (RLHFlow) rerank + RL post-training summary ───────────────────────────

PRM_ROOT = SCRATCH_OUTPUT_DIR / "prm_rlhflow"
PRM_SBATCH_DIR = SCRATCH_OUTPUT_DIR / "sbatch" / "prm_rlhflow"
PRM_DATASETS = ["gsm8k", "eedi"]
PRM_PRMS = [("deepseek", "ds"), ("mistral", "mi")]
PRM_MODELS = ["qwen05b", "qwen15b", "llama1b"]


def _prm_status(mode: str, tag: str) -> str:
    """Fallback status (running/pending/failed/not started) for a PRM job."""
    sbatch_path = PRM_SBATCH_DIR / f"{mode}_{tag}.sbatch"
    if not sbatch_path.exists():
        return "not started"
    key = str(sbatch_path.resolve())
    live_job = infer_step_value.live_jobs.get(key)
    failed_state = infer_step_value.failed_jobs.get(key)
    return summarize_state(live_job, failed_state)


def _prm_rerank_cell(tag: str) -> str:
    selected = PRM_ROOT / "results" / f"{tag}_rerank_selected.csv"
    if selected.exists():
        acc, _ = read_score(selected)
        return f"{acc:.1%}"
    return _prm_status("rerank", tag)


def _prm_rl_cell(tag: str) -> str:
    ckpt_dir = PRM_ROOT / "results" / "checkpoints" / f"{tag}_prm_rl"
    ft_acc = get_ft_accuracy(ckpt_dir)
    if ft_acc is not None:
        return ft_acc
    return _prm_status("train", tag)


def prm_summary() -> None:
    """Print rerank (best-of-N) and RL post-training accuracy for the RLHFlow PRMs."""
    if not PRM_ROOT.exists():
        return

    headers = ["Model"]
    for _, prm_short in PRM_PRMS:
        headers.extend([f"{prm_short}_rr", f"{prm_short}_rl"])

    for dataset in PRM_DATASETS:
        table_rows = []
        for model in PRM_MODELS:
            row_values = [model]
            for prm_name, _ in PRM_PRMS:
                tag = f"{dataset}_{prm_name}_{model}"
                row_values.append(_prm_rerank_cell(tag))
                row_values.append(_prm_rl_cell(tag))
            table_rows.append(row_values)
        print(f"\n[prm_rlhflow: {dataset}]  (rr=rerank best-of-N, rl=RL best ckpt)")
        print_aligned_table(headers, table_rows)


# ── ORM (RLHFlow) rerank + RL post-training summary ───────────────────────────

ORM_ROOT = SCRATCH_OUTPUT_DIR / "orm_rlhflow"
ORM_SBATCH_DIR = SCRATCH_OUTPUT_DIR / "sbatch" / "orm_rlhflow"
ORM_DATASETS = ["gsm8k", "eedi"]
ORM_MODELS = ["qwen05b", "qwen15b", "qwen1b"]
ORM_RMS = [("deepseek", "ds"), ("mistral", "mi")]


def _orm_status(mode: str, tag: str) -> str:
    """Fallback status (running/pending/failed/not started) for an ORM job."""
    sbatch_path = ORM_SBATCH_DIR / f"{mode}_{tag}.sbatch"
    if not sbatch_path.exists():
        return "not started"
    key = str(sbatch_path.resolve())
    live_job = infer_step_value.live_jobs.get(key)
    failed_state = infer_step_value.failed_jobs.get(key)
    return summarize_state(live_job, failed_state)


def _orm_rerank_cell(tag: str) -> str:
    selected = ORM_ROOT / "results" / f"{tag}_rerank_selected.csv"
    if selected.exists():
        acc, _ = read_score(selected)
        return f"{acc:.1%}"
    return _orm_status("rerank", tag)


def _orm_rl_cell(tag: str) -> str:
    ckpt_dir = ORM_ROOT / "results" / "checkpoints" / f"{tag}_orm_rl"
    ft_acc = get_ft_accuracy(ckpt_dir)
    if ft_acc is not None:
        return ft_acc
    return _orm_status("train", tag)


def orm_summary() -> None:
    """Print rerank (best-of-N) and RL post-training accuracy for RLHFlow ORMs."""
    if not ORM_ROOT.exists():
        return

    headers = ["Model"]
    for _, rm_short in ORM_RMS:
        headers.extend([f"{rm_short}_rr", f"{rm_short}_rl"])

    for dataset in ORM_DATASETS:
        table_rows = []
        for model in ORM_MODELS:
            row_values = [model]
            for rm_name, _ in ORM_RMS:
                tag = f"{dataset}_{rm_name}_{model}"
                row_values.append(_orm_rerank_cell(tag))
                row_values.append(_orm_rl_cell(tag))
            # Suppress rows that are completely missing for cleaner output.
            if any(cell != "not started" for cell in row_values[1:]):
                table_rows.append(row_values)
        if not table_rows:
            continue
        print(f"\n[orm_rlhflow: {dataset}]  (rr=rerank best-of-N, rl=RL best ckpt)")
        print_aligned_table(headers, table_rows)


def main():
    infer_step_value.live_jobs = get_live_jobs()
    infer_step_value.failed_jobs = get_failed_jobs()

    distractor_golds_cache = {}
    for key, data_csv in DATA_CSVS.items():
        setting, task = key
        if task == "distractor" and data_csv.exists():
            with open(data_csv, newline="") as f:
                reader = csv.DictReader(f)
                distractor_golds_cache[(setting, task)] = [
                    row["target_distractor_answers"]
                    for row in reader if row.get("split") == "test"
                ]

    rows = defaultdict(lambda: {
        "inf_before": "not started",
        "ft": "not started",
        "inf_after": "not started",
    })

    for setting in SETTINGS:
        root = get_setting_root(setting)
        sbatch_dir = root / "sbatch" / setting
        if sbatch_dir.exists():
            for sbatch_file in sorted(sbatch_dir.glob("*.sbatch")):
                match = SBATCH_RE.match(sbatch_file.name)
                if not match:
                    continue
                step, task_short, model_short = match.groups()
                rows[(setting, task_short, model_short)][step] = infer_step_value(
                    setting,
                    step,
                    sbatch_file,
                    distractor_golds_cache,
                )

        # EEDI has historical results in out/ without the new sbatch structure being authoritative.
        interim = root / setting / "results" / "interim"
        if interim.exists():
            for csv_file in sorted(interim.glob("*.csv")):
                match = FILENAME_RE.match(csv_file.name)
                if not match:
                    continue
                model_name, task_name, phase = match.groups()
                model_short = canonical_model_name(model_name)
                task_short = TASK_SHORT[task_name]
                golds = (
                    None
                    if setting in SEMANTIC_DISTRACTOR_SETTINGS
                    else distractor_golds_cache.get((setting, task_name))
                )
                acc, _ = read_score(csv_file, distractor_golds=golds)
                rows[(setting, task_short, model_short)][f"inf_{phase}"] = f"{acc:.1%}"

        ckpt_root = root / setting / "results" / "checkpoints"
        if ckpt_root.exists():
            for ckpt_dir in sorted(ckpt_root.iterdir()):
                if not ckpt_dir.is_dir():
                    continue
                for task_name in SETTINGS[setting]["tasks"]:
                    suffix = f"_{task_name}"
                    if ckpt_dir.name.endswith(suffix):
                        model_name = ckpt_dir.name[: -len(suffix)]
                        model_short = canonical_model_name(model_name)
                        task_short = TASK_SHORT[task_name]
                        ft_acc = get_ft_accuracy(ckpt_dir)
                        if ft_acc is not None:
                            rows[(setting, task_short, model_short)]["ft"] = ft_acc
                        break

    if not rows:
        print("No result CSVs or job files found.")
        return

    for setting in SETTING_PRINT_ORDER:
        headers, table_rows = build_setting_table_rows(rows, setting)
        if not table_rows:
            continue
        print(f"\n[{setting}]")
        print_aligned_table(headers, table_rows)

    prm_summary()
    orm_summary()


# ── Data CSV paths for joining predictions with source data ───────────────────

DATA_CSVS = {
    ("eedi", "correct_answer"): BASE / "eedi/data/processed/correct_answer_pairs_eedi.csv",
    ("eedi", "distractor"): BASE / "eedi/data/processed/distractor_pairs_eedi.csv",
    ("eedi_split_control", "correct_answer"): BASE / "eedi/data/processed/correct_answer_pairs_eedi_split_control.csv",
    ("eedi_split_control", "distractor"): BASE / "eedi/data/processed/distractor_pairs_eedi_split_control.csv",
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
    result_csv = get_setting_root(setting) / setting / "results" / "interim" / f"{model}_{task}_{phase}.csv"
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

    interim = get_setting_root(setting) / setting / "results" / "interim"
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
        result_csv = get_setting_root(setting) / setting / "results" / "interim" / f"{model}_{task}_{phase}.csv"
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

    result_csv = get_setting_root(setting) / setting / "results" / "interim" / f"{model}_{task}_{phase}.csv"
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

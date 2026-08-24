#!/usr/bin/env python3
"""Download and inspect the EEDI misconceptions dataset from Kaggle.

This script:
1. Downloads the competition files via the Kaggle API.
2. Prints dataset sizes and useful summary counts.
3. Pretty-prints one full datapoint from train/test/misconception mapping.

Prerequisites:
- You must have accepted the competition rules on Kaggle.
- You must have Kaggle credentials configured, e.g. ~/.kaggle/kaggle.json.
- The `kaggle` Python package must be installed.

Example:
    python eedi/download_and_inspect_eedi.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import requests


COMPETITION = "eedi-mining-misconceptions-in-mathematics"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIRNAME = "raw"

QUESTION_COLUMNS = [
    "QuestionId",
    "ConstructId",
    "ConstructName",
    "CorrectAnswer",
    "SubjectId",
    "SubjectName",
    "QuestionText",
    "AnswerAText",
    "AnswerBText",
    "AnswerCText",
    "AnswerDText",
    "MisconceptionAId",
    "MisconceptionBId",
    "MisconceptionCId",
    "MisconceptionDId",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where the dataset should be downloaded",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the files already exist locally",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Row index to pretty-print from each table",
    )
    return parser.parse_args()


def ensure_kaggle_api():
    home = Path.home()
    config_candidates = [
        home / ".config" / "kaggle" / "kaggle.json",
        home / ".kaggle" / "kaggle.json",
    ]

    # Kaggle defaults to ~/.config/kaggle, but many users keep credentials in ~/.kaggle.
    if not os.environ.get("KAGGLE_CONFIG_DIR"):
        legacy_dir = home / ".kaggle"
        if (legacy_dir / "kaggle.json").exists():
            os.environ["KAGGLE_CONFIG_DIR"] = str(legacy_dir)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "The `kaggle` package is not installed. Install it with `pip install kaggle`, "
            "make sure you accepted the competition rules, and configure ~/.kaggle/kaggle.json."
        ) from exc
    except Exception as exc:
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")
        if username and key:
            raise SystemExit(
                "The Kaggle package was found, but import-time authentication still failed. "
                "Verify that KAGGLE_USERNAME and KAGGLE_KEY are correct and that you accepted "
                "the competition rules."
            ) from exc

        found_paths = [str(path) for path in config_candidates if path.exists()]
        path_hint = (
            f"Found kaggle.json at: {found_paths}. "
            if found_paths else
            "No kaggle.json was found in ~/.config/kaggle or ~/.kaggle. "
        )
        raise SystemExit(
            "Failed to initialize the Kaggle API. "
            + path_hint
            + "Create /cluster/home/tunguyen1/.config/kaggle/kaggle.json or "
            "set KAGGLE_USERNAME and KAGGLE_KEY in the environment."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - depends on local auth setup
        raise SystemExit(
            "Failed to authenticate with Kaggle. Make sure ~/.kaggle/kaggle.json exists "
            "and that you accepted the competition rules."
        ) from exc
    return api


def download_competition_files(api, raw_dir: Path, force: bool) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    expected_files = [
        raw_dir / "train.csv",
        raw_dir / "test.csv",
        raw_dir / "misconception_mapping.csv",
        raw_dir / "sample_submission.csv",
    ]
    if not force and all(path.exists() for path in expected_files):
        print(f"Dataset already present in {raw_dir}")
        return

    print(f"Downloading Kaggle competition files to {raw_dir}...")
    try:
        api.competition_download_files(
            COMPETITION,
            path=str(raw_dir),
            quiet=False,
            force=force,
        )
    except requests.exceptions.HTTPError as exc:
        response = getattr(exc, "response", None)
        if response is not None and response.status_code == 403:
            raise SystemExit(
                "Kaggle returned 403 Forbidden while downloading the competition data. "
                "Your API credentials are being recognized, but this account does not currently "
                "have permission to download the files. The usual causes are: "
                "1) you have not accepted the competition rules in the browser, or "
                "2) this Kaggle account does not have access to the competition data. "
                "Open the competition page, sign in with the same account, accept the rules, "
                "and then rerun the script."
            ) from exc
        raise

    zip_path = raw_dir / f"{COMPETITION}.zip"
    if zip_path.exists():
        print(f"Extracting {zip_path.name}...")
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(raw_dir)
        zip_path.unlink()


def load_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {
        "train": pd.read_csv(raw_dir / "train.csv"),
        "test": pd.read_csv(raw_dir / "test.csv"),
        "misconception_mapping": pd.read_csv(raw_dir / "misconception_mapping.csv"),
        "sample_submission": pd.read_csv(raw_dir / "sample_submission.csv"),
    }
    return tables


def count_labeled_distractors(df: pd.DataFrame) -> int:
    total = 0
    for answer in ["A", "B", "C", "D"]:
        mask = df["CorrectAnswer"] != answer
        misconception_col = f"Misconception{answer}Id"
        if misconception_col in df.columns:
            mask &= df[misconception_col].notna()
        total += int(mask.sum())
    return total


def print_table_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name}")
    print(f"  rows: {len(df):,}")
    print(f"  columns: {len(df.columns)}")
    print(f"  column names: {', '.join(df.columns)}")

    if name in {"train", "test"}:
        missing_cols = [col for col in QUESTION_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"  missing expected columns: {missing_cols}")
        else:
            print("  expected question columns: present")

        print(f"  unique questions: {df['QuestionId'].nunique():,}")
        print(f"  unique constructs: {df['ConstructId'].nunique():,}")
        print(f"  unique subjects: {df['SubjectId'].nunique():,}")

        if name == "train":
            print(
                "  labeled distractor datapoints: "
                f"{count_labeled_distractors(df):,}"
            )

    if name == "misconception_mapping":
        id_col = "MisconceptionId"
        name_col = "MisconceptionName"
        if id_col in df.columns:
            print(f"  unique misconception ids: {df[id_col].nunique():,}")
        if name_col in df.columns:
            print(f"  unique misconception names: {df[name_col].nunique():,}")


def pretty_print_row(name: str, df: pd.DataFrame, sample_index: int) -> None:
    if df.empty:
        print(f"\n{name} sample: table is empty")
        return

    if sample_index < 0 or sample_index >= len(df):
        sample_index = 0

    row = df.iloc[sample_index].to_dict()
    print(f"\n{name} sample row at index {sample_index}")
    print(json.dumps(row, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    args = parse_args()
    raw_dir = args.data_dir / RAW_DIRNAME

    api = ensure_kaggle_api()
    download_competition_files(api, raw_dir, args.force)

    tables = load_tables(raw_dir)

    print("\nEEDI dataset inspection")
    print(f"  competition: {COMPETITION}")
    print(f"  raw data dir: {raw_dir}")

    for name in ["train", "test", "misconception_mapping", "sample_submission"]:
        print_table_summary(name, tables[name])

    for name in ["train", "test", "misconception_mapping"]:
        pretty_print_row(name, tables[name], args.sample_index)


if __name__ == "__main__":
    main()
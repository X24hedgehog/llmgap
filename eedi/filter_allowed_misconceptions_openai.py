#!/usr/bin/env python3
"""Classify EEDI misconceptions and save the non-geometry ones.

This script reads the raw misconception mapping, asks an OpenAI model which
misconceptions are geometry-related, and writes out the complementary allowed
list for downstream dataset filtering.

Default outputs:
  - data/processed/allowed_misconceptions.csv
  - data/processed/misconception_geometry_labels.csv

API key lookup order:
    1. --api-key
    2. --api-key-file (defaults to eedi/openai_api.txt)
    3. OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The openai package is required. Install it with: pip install openai"
    ) from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = ROOT / "data" / "raw" / "misconception_mapping.csv"
DEFAULT_ALLOWED_CSV = ROOT / "data" / "processed" / "allowed_misconceptions.csv"
DEFAULT_LABELS_CSV = ROOT / "data" / "processed" / "misconception_geometry_labels.csv"
DEFAULT_API_KEY_FILE = ROOT / "openai_api.txt"


def write_csv_atomic(df: pd.DataFrame, output_path: Path) -> None:
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.4", help="OpenAI model name")
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. If omitted, the script will try a local key file.",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=DEFAULT_API_KEY_FILE,
        help="Path to a file containing only the OpenAI API key",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to misconception_mapping.csv",
    )
    parser.add_argument(
        "--allowed-output",
        type=Path,
        default=DEFAULT_ALLOWED_CSV,
        help="Where to write the allowed non-geometry misconceptions CSV",
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help="Where to write the full geometry/non-geometry classification CSV",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="How many misconceptions to classify per API call",
    )
    return parser.parse_args()


def load_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()

    if args.api_key_file.exists():
        return args.api_key_file.read_text(encoding="utf-8").strip()

    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    raise SystemExit(
        "No OpenAI API key found. Put your key in eedi/openai_api.txt, "
        "or pass --api-key, or set OPENAI_API_KEY."
    )


def build_messages(batch: pd.DataFrame) -> list[dict[str, str]]:
    lines = []
    for row in batch.itertuples(index=False):
        lines.append(f"{row.MisconceptionId}: {row.MisconceptionName}")

    developer_message = (
        "You are classifying math misconceptions by topic.\n\n"
        "Task:\n"
        "For each misconception, decide whether it is geometry-related.\n\n"
        "Label as geometry-related if it is mainly about:\n"
        "- geometry\n"
        "- shapes\n"
        "- angles\n"
        "- polygons\n"
        "- circles\n"
        "- perimeter\n"
        "- area\n"
        "- volume\n"
        "- nets\n"
        "- symmetry\n"
        "- transformations\n"
        "- bearings\n"
        "- loci\n"
        "- coordinates or graphs used as geometry\n"
        "- trigonometry\n"
        "- Pythagoras\n"
        "- constructions\n"
        "- spatial reasoning\n\n"
        "Label as non-geometry if it is mainly about:\n"
        "- arithmetic\n"
        "- algebra\n"
        "- number\n"
        "- fractions\n"
        "- decimals\n"
        "- percentages\n"
        "- ratio\n"
        "- probability\n"
        "- statistics\n"
        "- place value\n"
        "- general equation manipulation\n\n"
        "Rules:\n"
        "- Use the main mathematical topic, not incidental wording.\n"
        "- If a misconception is about interpreting a diagram, angle fact, shape property, transformation, graph-as-geometry, trigonometry, or measurement of shapes, mark it as geometry.\n"
        "- If it is about number operations, fractions, algebraic manipulation, probability, statistics, or general algebra, mark it as non-geometry.\n"
        "- Use every provided id exactly once.\n"
        "- Return strict JSON only.\n"
        "- Keep each reason short.\n\n"
        "Return exactly this schema:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "misconception_id": 0,\n'
        '      "is_geometry": true,\n'
        '      "reason": "short reason"\n'
        "    }\n"
        "  ]\n"
        "}"
    )
    user_message = "Classify these misconceptions.\n\n" + "\n".join(lines)

    return [
        {"role": "developer", "content": developer_message},
        {"role": "user", "content": user_message},
    ]


def parse_response(text: str, expected_ids: set[int]) -> list[dict[str, object]]:
    data = json.loads(text)
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("Response JSON did not contain a list under 'items'.")

    parsed_items = []
    seen_ids: set[int] = set()
    for item in items:
        misconception_id = int(item["misconception_id"])
        is_geometry = bool(item["is_geometry"])
        reason = str(item.get("reason", "")).strip()
        parsed_items.append(
            {
                "MisconceptionId": misconception_id,
                "is_geometry": is_geometry,
                "geometry_reason": reason,
            }
        )
        seen_ids.add(misconception_id)

    if seen_ids != expected_ids:
        missing = sorted(expected_ids - seen_ids)
        extra = sorted(seen_ids - expected_ids)
        raise ValueError(f"Response ids mismatch. Missing={missing[:5]} Extra={extra[:5]}")

    return parsed_items


def classify_batch(client: OpenAI, model: str, batch: pd.DataFrame) -> list[dict[str, object]]:
    messages = build_messages(batch)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )

    text = (response.choices[0].message.content or "").strip()

    if not text:
        raise ValueError("Model returned empty output.")

    expected_ids = set(batch["MisconceptionId"].astype(int).tolist())
    return parse_response(text, expected_ids)


def persist_outputs(
    mapping_df: pd.DataFrame,
    classified_batches: list[dict[str, object]],
    labels_output: Path,
    allowed_output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    classified_df = pd.DataFrame(classified_batches)
    labels_df = mapping_df.merge(
        classified_df,
        on="MisconceptionId",
        how="inner",
        validate="one_to_one",
    ).sort_values("MisconceptionId")
    allowed_df = labels_df.loc[~labels_df["is_geometry"]].copy()

    write_csv_atomic(labels_df, labels_output)
    write_csv_atomic(
        allowed_df[["MisconceptionId", "MisconceptionName"]],
        allowed_output,
    )
    return labels_df, allowed_df


def main() -> None:
    args = parse_args()
    api_key = load_api_key(args)

    mapping_df = pd.read_csv(args.input_csv)
    client = OpenAI(api_key=api_key)
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    args.allowed_output.parent.mkdir(parents=True, exist_ok=True)

    classified_batches = []
    labels_df = pd.DataFrame()
    allowed_df = pd.DataFrame()
    for start in range(0, len(mapping_df), args.batch_size):
        batch = mapping_df.iloc[start : start + args.batch_size].copy()
        classified_batches.extend(classify_batch(client, args.model, batch))
        labels_df, allowed_df = persist_outputs(
            mapping_df=mapping_df,
            classified_batches=classified_batches,
            labels_output=args.labels_output,
            allowed_output=args.allowed_output,
        )
        end = start + len(batch)
        print(
            f"Classified misconceptions {start + 1}-{end} of {len(mapping_df)} "
            f"and saved partial results to disk"
        )

    print(f"Total misconceptions: {len(labels_df):,}")
    print(f"Geometry misconceptions removed: {int(labels_df['is_geometry'].sum()):,}")
    print(f"Allowed misconceptions kept: {len(allowed_df):,}")
    print(f"Saved labels to: {args.labels_output}")
    print(f"Saved allowed list to: {args.allowed_output}")


if __name__ == "__main__":
    main()
import json
import textwrap
from pathlib import Path

import pandas as pd


CSV_PATH = Path("out/next_subquestion_pairs.csv")
ROW_INDEX = 0
TEXT_WIDTH = 100


def wrap(text: str, indent: str = "") -> str:
    if text is None:
        return ""
    return textwrap.fill(
        str(text),
        width=TEXT_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
    )


def parse_jsonish(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if not isinstance(value, str):
        return value

    value = value.strip()
    if not value:
        return None

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))


def print_json_field(name: str, value) -> None:
    parsed = parse_jsonish(value)
    print(f"\n{name}:")
    if parsed is None:
        print("  <empty>")
    elif isinstance(parsed, (dict, list)):
        print(textwrap.indent(json.dumps(parsed, indent=2, ensure_ascii=True), prefix="  "))
    else:
        print(wrap(parsed, indent="  "))


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if ROW_INDEX < 0 or ROW_INDEX >= len(df):
        raise IndexError(f"ROW_INDEX={ROW_INDEX} out of range for {len(df)} rows")

    row = df.iloc[ROW_INDEX]

    print_section(f"Row {ROW_INDEX}")

    scalar_fields = [
        "example_idx",
        "problem_type",
        "complexity",
        "overlap_type",
        "answer",
        "depth",
        "width",
    ]
    for field in scalar_fields:
        if field in row.index:
            print(f"{field}: {row[field]}")

    text_fields = [
        "problem",
        "question",
        "prompt",
        "groundquery",
        "reasoning_trace",
    ]
    for field in text_fields:
        if field in row.index:
            print(f"\n{field}:")
            print(wrap(row[field], indent="  "))

    for field in ["subquestions", "subquestions_by_node_id", "subquestion_tree", "trees"]:
        if field in row.index:
            print_json_field(field, row[field])


if __name__ == "__main__":
    main()
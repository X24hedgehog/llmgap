import json
import textwrap
from pathlib import Path

import pandas as pd


TEXT_WIDTH = 100
CSV_PATH = "out/next_subquestion_pairs.csv"
ROW_INDEX = 215
EXAMPLE_IDX = None


def _wrap(text: str, indent: str = "") -> str:
	if text is None:
		return ""
	return textwrap.fill(str(text), width=TEXT_WIDTH, initial_indent=indent, subsequent_indent=indent)


def _maybe_parse_json(value):
	if value is None:
		return None

	if isinstance(value, float) and pd.isna(value):
		return None

	if not isinstance(value, str):
		return value

	value = value.strip()
	if value == "":
		return None

	try:
		return json.loads(value)
	except json.JSONDecodeError:
		return value


def _print_section(title: str) -> None:
	print(f"\n{title}")
	print("=" * len(title))


def _print_json_field(name: str, value) -> None:
	parsed = _maybe_parse_json(value)
	print(f"{name}:")
	if parsed is None:
		print("  <empty>")
		return
	if isinstance(parsed, (dict, list)):
		print(textwrap.indent(json.dumps(parsed, indent=2, ensure_ascii=True), prefix="  "))
		return
	print(_wrap(str(parsed), indent="  "))


def print_row(row_index: int, row: pd.Series) -> None:
	_print_section(f"Row {row_index}")

	scalar_fields = [
		"example_idx",
		"problem_type",
		"complexity",
		"overlap_type",
		"answer",
		"depth",
		"width",
		"pair_index",
		"target_node_id",
		"num_next_subquestion_examples",
	]
	for field in scalar_fields:
		if field in row.index:
			print(f"{field}: {row.get(field, '')}")

	text_fields = [
		"problem",
		"question",
		"groundquery",
		"reasoning_trace",
		"next_subquestion",
		"prompt",
	]
	for field in text_fields:
		if field in row.index:
			print(f"\n{field}:")
			print(_wrap(str(row.get(field, "")), indent="  "))

	json_fields = [
		"subquestions",
		"subquestions_by_node_id",
		"subquestion_tree",
		"trees",
		"solved_node_ids",
		"solved_steps",
		"next_subquestion_examples",
		"solved_steps_to_next_subquestion",
	]
	for field in json_fields:
		if field in row.index:
			print()
			_print_json_field(field, row.get(field))


def load_dataframe(csv_path: str) -> pd.DataFrame:
	path = Path(csv_path)
	if not path.exists():
		raise FileNotFoundError(f"CSV file not found: {path}")
	return pd.read_csv(path)


def main() -> None:
	if ROW_INDEX is None and EXAMPLE_IDX is None:
		raise ValueError("Set either ROW_INDEX or EXAMPLE_IDX in the file before running")

	df = load_dataframe(CSV_PATH)
	print(f"Columns: {df.columns.tolist()}")

	if ROW_INDEX is not None:
		if ROW_INDEX < 0 or ROW_INDEX >= len(df):
			raise IndexError(f"ROW_INDEX {ROW_INDEX} is out of range for dataset of size {len(df)}")
		print_row(ROW_INDEX, df.iloc[ROW_INDEX])
		return

	matches = df[df["example_idx"] == EXAMPLE_IDX]
	if len(matches) == 0:
		raise ValueError(f"No rows found with EXAMPLE_IDX={EXAMPLE_IDX}")

	for row_index, (_, row) in enumerate(matches.iterrows()):
		print_row(row_index, row)


if __name__ == "__main__":
	main()

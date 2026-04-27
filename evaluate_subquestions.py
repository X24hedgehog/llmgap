import ast
import gc
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET_PATH = Path("reasoning-efficiency/experiments/proof_search/out/data.csv")
# DATASET_PATH = Path("mathgap-experiments/experiments/opedal24_ood_eval/out/nonlinear_comparison_100.csv")
OUTPUT_PATH = Path("subquestion_eval_results.csv")

GENERATOR_MODELS = [
    ("qwen05", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen15", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen3", "Qwen/Qwen2.5-3B-Instruct"),
]
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

DTYPE = "auto"
LIMIT = None
FILTERS: Dict[str, Any] = {}
SUBSET_BASE_PROBLEMS = 30
SUBSET_RANDOM_SEED = 14
CHECKPOINT_BASE_PROBLEMS = 2

GENERATOR_MAX_NEW_TOKENS = 256
JUDGE_MAX_NEW_TOKENS = 128
GENERATOR_TEMPERATURE = 0.0
GENERATOR_TOP_P = 1.0
JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 1.0

RESULT_COLUMNS = [
    "generated_subquestions",
    "sufficiency",
    "efficiency",
    "order",
    "score_summary",
    "sufficiency_details",
    "efficiency_details",
    "order_details",
]


@dataclass
class ProblemRecord:
    row_id: int
    row_data: Dict[str, Any]
    problem: str
    gold_subquestions: List[str]
    gold_subquestion_tree: Any


class HuggingFaceQwenModel:
    def __init__(self, model_name: str, dtype: Optional[torch.dtype], max_new_tokens: int,
                 temperature: float, top_p: float):
        self.model_name = model_name
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if self.dtype is not None:
            model_kwargs["dtype"] = self.dtype

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_text(self, prompt_text: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(f"Model '{self.model_name}' is not loaded")

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        do_sample = self.temperature > 0.0
        generation_kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class SubquestionGenerator:
    def __init__(self, model: HuggingFaceQwenModel):
        self.model = model

    def _build_prompt(self, problem: str) -> str:
        return (
            "You are decomposing a math word problem into the intermediate subquestions needed to solve it.\n"
            "Return ONLY valid JSON with exactly this schema: {\"subquestions\": [str, ...]}.\n"
            "Generate only the necessary intermediate subquestions.\n"
            "Do not include explanations.\n"
            "Do not include the final question unless it is itself an intermediate requirement.\n"
            "Avoid redundancy.\n\n"
            f"Problem:\n{problem}\n"
        )

    def generate(self, problem: str) -> List[str]:
        return _parse_subquestions_json(self.model.generate_text(self._build_prompt(problem)))


class SubquestionJudge:
    def __init__(self, model: HuggingFaceQwenModel):
        self.model = model

    def judge_gold_against_generated(self, problem: str, gold_subquestion: str,
                                     generated_subquestions: List[str]) -> Dict[str, Any]:
        prompt = (
            "You are judging whether a gold intermediate subquestion is covered by a model-generated list of subquestions.\n"
            "Semantic equivalence counts as a match even if wording differs.\n"
            "Return ONLY valid JSON with exactly this schema:\n"
            "{\"covered\": bool, \"matched_generated_subquestion\": str|null, \"reason\": str}\n\n"
            "Decision rules:\n"
            "1) A match requires the same meaning and target quantity.\n"
            "2) Broader, narrower, or different-quantity questions are not matches.\n"
            "3) If covered is true, return the best matching generated subquestion verbatim.\n"
            "4) Keep reason short.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Gold subquestion:\n{gold_subquestion}\n\n"
            f"Generated subquestions:\n{json.dumps(generated_subquestions, ensure_ascii=True)}\n"
        )
        parsed = _parse_json_object(self.model.generate_text(prompt))
        return {
            "gold_subquestion": gold_subquestion,
            "covered": bool(parsed.get("covered", False)),
            "matched_generated_subquestion": parsed.get("matched_generated_subquestion"),
            "reason": str(parsed.get("reason", "")),
        }

    def judge_generated_against_gold(self, problem: str, generated_subquestion: str,
                                     gold_subquestions: List[str]) -> Dict[str, Any]:
        prompt = (
            "You are judging whether a generated intermediate subquestion is part of the gold set of valid subquestions.\n"
            "Semantic equivalence counts as a match even if wording differs.\n"
            "Return ONLY valid JSON with exactly this schema:\n"
            "{\"included\": bool, \"matched_gold_subquestion\": str|null, \"reason\": str}\n\n"
            "Decision rules:\n"
            "1) A match requires the same meaning and target quantity.\n"
            "2) Broader, narrower, or different-quantity questions are not matches.\n"
            "3) If included is true, return the best matching gold subquestion verbatim.\n"
            "4) Keep reason short.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Generated subquestion:\n{generated_subquestion}\n\n"
            f"Gold subquestions:\n{json.dumps(gold_subquestions, ensure_ascii=True)}\n"
        )
        parsed = _parse_json_object(self.model.generate_text(prompt))
        return {
            "generated_subquestion": generated_subquestion,
            "included": bool(parsed.get("included", False)),
            "matched_gold_subquestion": parsed.get("matched_gold_subquestion"),
            "reason": str(parsed.get("reason", "")),
        }


class OrderScorer:
    @staticmethod
    def _is_valid_tree_node(node: Any) -> bool:
        return isinstance(node, dict) and "node_id" in node and "children" in node

    @staticmethod
    def _collect_ancestors(node: Dict[str, Any], ancestors: List[int], ancestors_by_id: Dict[int, Set[int]],
                           question_to_id: Dict[str, int]) -> None:
        node_id = int(node["node_id"])
        ancestors_by_id[node_id] = set(ancestors)

        question = node.get("question")
        if isinstance(question, str) and question.strip():
            question_to_id[question] = node_id

        next_ancestors = ancestors + [node_id]
        for child in node.get("children", []):
            OrderScorer._collect_ancestors(child, next_ancestors, ancestors_by_id, question_to_id)

    @staticmethod
    def score(matched_generated_details: List[Dict[str, Any]], gold_tree: Any) -> Dict[str, Any]:
        if not OrderScorer._is_valid_tree_node(gold_tree):
            return {
                "order": 1.0,
                "correct_pair_count": 0,
                "comparable_pair_count": 0,
            }

        ancestors_by_id: Dict[int, Set[int]] = {}
        question_to_id: Dict[str, int] = {}
        OrderScorer._collect_ancestors(gold_tree, [], ancestors_by_id, question_to_id)

        matched_gold_sequence = []
        for detail in matched_generated_details:
            if not detail.get("included"):
                continue
            matched_gold = detail.get("matched_gold_subquestion")
            if isinstance(matched_gold, str) and matched_gold.strip() in question_to_id:
                matched_gold_sequence.append(matched_gold.strip())

        correct_pair_count = 0
        comparable_pair_count = 0

        for left_idx in range(len(matched_gold_sequence)):
            for right_idx in range(left_idx + 1, len(matched_gold_sequence)):
                left_gold = matched_gold_sequence[left_idx]
                right_gold = matched_gold_sequence[right_idx]
                if left_gold == right_gold:
                    continue

                left_id = question_to_id[left_gold]
                right_id = question_to_id[right_gold]

                if left_id in ancestors_by_id.get(right_id, set()):
                    comparable_pair_count += 1
                    correct_pair_count += 1
                elif right_id in ancestors_by_id.get(left_id, set()):
                    comparable_pair_count += 1

        order = 1.0 if comparable_pair_count == 0 else correct_pair_count / comparable_pair_count
        return {
            "order": order,
            "correct_pair_count": correct_pair_count,
            "comparable_pair_count": comparable_pair_count,
            "matched_gold_sequence": matched_gold_sequence,
        }


class MetricScorer:
    @staticmethod
    def sufficiency(gold_to_generated_details: List[Dict[str, Any]], gold_count: int) -> Dict[str, Any]:
        covered_count = sum(1 for detail in gold_to_generated_details if detail.get("covered"))
        score = covered_count / gold_count if gold_count else 1.0
        return {
            "sufficiency": score,
            "covered_gold_count": covered_count,
            "gold_count": gold_count,
        }

    @staticmethod
    def efficiency(generated_to_gold_details: List[Dict[str, Any]], generated_count: int) -> Dict[str, Any]:
        included_count = sum(1 for detail in generated_to_gold_details if detail.get("included"))
        score = included_count / generated_count if generated_count else 1.0
        return {
            "efficiency": score,
            "included_generated_count": included_count,
            "generated_count": generated_count,
        }


class EvaluationPipeline:
    REQUIRED_COLUMNS = {"problem", "subquestions", "subquestion_tree"}

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def _select_reasoning_efficiency_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        required_subset_columns = {"example_idx", "problem_type", "complexity", "overlap_type"}
        if not required_subset_columns.issubset(df.columns):
            return df

        base_rows = df[df["problem_type"] == "base"].sort_values("example_idx")
        selected_example_ids = list(base_rows["example_idx"].head(SUBSET_BASE_PROBLEMS))
        rng = random.Random(SUBSET_RANDOM_SEED)

        selected_indices: List[int] = []
        difficulties = ["simple", "complex", "more_complex"]

        for example_idx in selected_example_ids:
            example_rows = df[df["example_idx"] == example_idx]

            base_match = example_rows[example_rows["problem_type"] == "base"]
            if len(base_match) == 0:
                continue
            selected_indices.append(int(base_match.index[0]))

            for problem_type in ["connected", "disconnected"]:
                for difficulty in difficulties:
                    candidates = example_rows[
                        (example_rows["problem_type"] == problem_type)
                        & (example_rows["complexity"] == difficulty)
                    ]
                    if len(candidates) == 0:
                        continue

                    candidate_indices = list(candidates.index)
                    selected_indices.append(rng.choice(candidate_indices))

        selected_df = df.loc[selected_indices].copy()
        return selected_df.reset_index(drop=True)

    def load_records(self, limit: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, List[ProblemRecord]]:
        df = pd.read_csv(self.dataset_path)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {self.dataset_path}: {sorted(missing)}")

        if filters:
            for key, value in filters.items():
                if key not in df.columns:
                    raise ValueError(f"Filter column '{key}' not found in dataset")
                df = df[df[key] == value]

        df = self._select_reasoning_efficiency_subset(df)

        if limit is not None:
            df = df.head(limit)

        records = []
        for row_id, row in df.reset_index(drop=True).iterrows():
            row_data = row.to_dict()
            records.append(
                ProblemRecord(
                    row_id=row_id,
                    row_data=row_data,
                    problem=str(row_data.get("problem", "")),
                    gold_subquestions=_coerce_list_field(row_data.get("subquestions")),
                    gold_subquestion_tree=_coerce_obj_field(row_data.get("subquestion_tree")),
                )
            )

        return df.reset_index(drop=True), records

    def initialize_output(self, source_df: pd.DataFrame) -> pd.DataFrame:
        return source_df.copy()

    def make_batches(self, source_df: pd.DataFrame, base_problems_per_batch: int) -> List[List[int]]:
        if "example_idx" in source_df.columns:
            example_ids = list(dict.fromkeys(source_df["example_idx"].tolist()))
            batches: List[List[int]] = []
            for start in range(0, len(example_ids), base_problems_per_batch):
                batch_example_ids = set(example_ids[start:start + base_problems_per_batch])
                batch_indices = [
                    idx for idx, example_idx in enumerate(source_df["example_idx"].tolist())
                    if example_idx in batch_example_ids
                ]
                batches.append(batch_indices)
            return batches

        row_indices = list(range(len(source_df)))
        return [
            row_indices[start:start + base_problems_per_batch]
            for start in range(0, len(row_indices), base_problems_per_batch)
        ]


def get_torch_dtype(dtype_str: str) -> Optional[torch.dtype]:
    if dtype_str == "auto":
        return None
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _coerce_list_field(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v).strip()]
        except (SyntaxError, ValueError):
            pass
        return [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    return []


def _coerce_obj_field(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return {"raw": text}
    return {"raw": value}


def _parse_json_object(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {}


def _parse_subquestions_json(text: str) -> List[str]:
    parsed = _parse_json_object(text)
    value = parsed.get("subquestions", []) if isinstance(parsed, dict) else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    return [line for line in lines if "{" not in line and "}" not in line]


def _error_placeholder(error: str) -> Dict[str, Any]:
    return {
        "generated_subquestions": [],
        "sufficiency": None,
        "efficiency": None,
        "order": None,
        "score_summary": {"error": error},
        "sufficiency_details": {"error": error},
        "efficiency_details": {"error": error},
        "order_details": {"error": error},
    }


def _evaluate_record(record: ProblemRecord, generator: SubquestionGenerator, judge: SubquestionJudge) -> Dict[str, Any]:
    try:
        generated_subquestions = generator.generate(record.problem)
    except Exception as error:
        return _error_placeholder(f"generation_error: {error}")

    try:
        gold_to_generated_details = [
            judge.judge_gold_against_generated(record.problem, gold_subquestion, generated_subquestions)
            for gold_subquestion in record.gold_subquestions
        ]
        generated_to_gold_details = [
            judge.judge_generated_against_gold(record.problem, generated_subquestion, record.gold_subquestions)
            for generated_subquestion in generated_subquestions
        ]

        sufficiency = MetricScorer.sufficiency(gold_to_generated_details, len(record.gold_subquestions))
        efficiency = MetricScorer.efficiency(generated_to_gold_details, len(generated_subquestions))
        order = OrderScorer.score(generated_to_gold_details, record.gold_subquestion_tree)
    except Exception as error:
        placeholder = _error_placeholder(f"judge_or_score_error: {error}")
        placeholder["generated_subquestions"] = generated_subquestions
        return placeholder

    return {
        "generated_subquestions": generated_subquestions,
        "sufficiency": float(sufficiency["sufficiency"]),
        "efficiency": float(efficiency["efficiency"]),
        "order": float(order["order"]),
        "score_summary": {**sufficiency, **efficiency, **order},
        "sufficiency_details": gold_to_generated_details,
        "efficiency_details": generated_to_gold_details,
        "order_details": order,
    }


def _parse_result_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _initialize_output_columns(output_df: pd.DataFrame) -> None:
    for column in RESULT_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = json.dumps({}, ensure_ascii=True)
        else:
            output_df[column] = output_df[column].apply(lambda value: json.dumps(_parse_result_dict(value), ensure_ascii=True))


def _update_row_result(output_df: pd.DataFrame, row_id: int, generator_alias: str, result: Dict[str, Any]) -> None:
    for column in RESULT_COLUMNS:
        existing = _parse_result_dict(output_df.at[row_id, column])
        existing[generator_alias] = result[column]
        output_df.at[row_id, column] = json.dumps(existing, ensure_ascii=True)


def _write_checkpoint(output_df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)


def _load_generators(dtype: Optional[torch.dtype]) -> Tuple[
    List[Tuple[str, HuggingFaceQwenModel, SubquestionGenerator]],
    Dict[str, str],
]:
    loaded_generators: List[Tuple[str, HuggingFaceQwenModel, SubquestionGenerator]] = []
    load_errors: Dict[str, str] = {}

    for generator_alias, generator_model_name in tqdm(GENERATOR_MODELS, desc="Loading generators"):
        generator_model = HuggingFaceQwenModel(
            model_name=generator_model_name,
            dtype=dtype,
            max_new_tokens=GENERATOR_MAX_NEW_TOKENS,
            temperature=GENERATOR_TEMPERATURE,
            top_p=GENERATOR_TOP_P,
        )
        try:
            generator_model.load()
            loaded_generators.append((generator_alias, generator_model, SubquestionGenerator(generator_model)))
        except Exception as error:
            generator_model.unload()
            load_errors[generator_alias] = f"model_load_error: {error}"

    return loaded_generators, load_errors


def evaluate_questions_in_batches(records: List[ProblemRecord], batch_indices: List[List[int]], output_df: pd.DataFrame,
                                 judge: SubquestionJudge, dtype: Optional[torch.dtype]) -> None:
    _initialize_output_columns(output_df)
    loaded_generators, load_errors = _load_generators(dtype)

    try:
        for batch_number, indices in enumerate(
            tqdm(batch_indices, desc="Batches"),
            start=1,
        ):
            batch_records = [records[index] for index in indices]
            for record in tqdm(batch_records, desc=f"Questions batch {batch_number}", leave=False):
                for generator_alias, error_message in load_errors.items():
                    _update_row_result(output_df, record.row_id, generator_alias, _error_placeholder(error_message))
                for generator_alias, _generator_model, generator in loaded_generators:
                    try:
                        result = _evaluate_record(record, generator, judge)
                    except Exception as error:
                        result = _error_placeholder(f"record_level_error: {error}")
                    _update_row_result(output_df, record.row_id, generator_alias, result)

            _write_checkpoint(output_df)
    finally:
        for _alias, generator_model, _generator in loaded_generators:
            generator_model.unload()


def print_summary(output_df: pd.DataFrame) -> None:
    print(f"Loaded {len(output_df)} rows from {DATASET_PATH}")
    for alias, _model_name in GENERATOR_MODELS:
        suff_values = []
        eff_values = []
        order_values = []
        for _, row in output_df.iterrows():
            suff_dict = _parse_result_dict(row.get("sufficiency", ""))
            eff_dict = _parse_result_dict(row.get("efficiency", ""))
            order_dict = _parse_result_dict(row.get("order", ""))
            if isinstance(suff_dict.get(alias), (int, float)):
                suff_values.append(float(suff_dict[alias]))
            if isinstance(eff_dict.get(alias), (int, float)):
                eff_values.append(float(eff_dict[alias]))
            if isinstance(order_dict.get(alias), (int, float)):
                order_values.append(float(order_dict[alias]))

        suff = sum(suff_values) / len(suff_values) if suff_values else float("nan")
        eff = sum(eff_values) / len(eff_values) if eff_values else float("nan")
        order = sum(order_values) / len(order_values) if order_values else float("nan")
        print(
            f"{alias}: sufficiency={suff:.4f}, efficiency={eff:.4f}, order={order:.4f}"
        )


def main() -> None:
    dtype = get_torch_dtype(DTYPE)

    pipeline = EvaluationPipeline(DATASET_PATH)
    source_df, records = pipeline.load_records(limit=LIMIT, filters=FILTERS)
    output_df = pipeline.initialize_output(source_df)
    _initialize_output_columns(output_df)
    batch_indices = pipeline.make_batches(source_df, CHECKPOINT_BASE_PROBLEMS)

    judge_model = HuggingFaceQwenModel(
        model_name=JUDGE_MODEL,
        dtype=dtype,
        max_new_tokens=JUDGE_MAX_NEW_TOKENS,
        temperature=JUDGE_TEMPERATURE,
        top_p=JUDGE_TOP_P,
    )

    try:
        judge_model.load()
        judge = SubquestionJudge(judge_model)
        evaluate_questions_in_batches(records, batch_indices, output_df, judge, dtype)
    finally:
        judge_model.unload()

    _write_checkpoint(output_df)
    print_summary(output_df)
    print(f"Saved evaluation results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
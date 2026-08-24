import argparse
import gc
from typing import Any, List, Optional, Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# Edit this list to control which example indices (0-based) are printed.
# Example: INDICES = [0, 2, 5]
INDICES = [2,3,4,5]


# Qwen models to test (change if you want a subset)
DEFAULT_MODELS = [
	"Qwen/Qwen2.5-0.5B-Instruct",
	"Qwen/Qwen2.5-1.5B-Instruct",
	"Qwen/Qwen2.5-3B-Instruct",
	"Qwen/Qwen2.5-7B-Instruct",
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


def load_model_and_tokenizer(model_name: str, dtype: Optional[torch.dtype]):
	tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

	model_kwargs: Dict[str, Any] = {
		"trust_remote_code": True,
		"device_map": "auto",
	}
	if dtype is not None:
		model_kwargs["torch_dtype"] = dtype

	model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
	model.eval()
	return model, tokenizer


def cleanup_model(model, tokenizer) -> None:
	del model
	del tokenizer
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


def generate_one(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	prompt_text: str,
	max_new_tokens: int,
	temperature: float,
	top_p: float,
) -> str:
	# Some tokenizers (chat-style) provide helpers, but fall back to plain tokenization.
	inputs = tokenizer(prompt_text, return_tensors="pt")
	inputs = {k: v.to(model.device) for k, v in inputs.items()}

	do_sample = temperature > 0.0

	with torch.no_grad():
		output_ids = model.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			do_sample=do_sample,
			temperature=temperature if do_sample else None,
			top_p=top_p if do_sample else None,
			pad_token_id=tokenizer.eos_token_id,
		)

	generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
	return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def build_skeleton_prompt(problem: str) -> str:
	return (
		"You are an expert Python programmer.\n"
		"Given the following problem description, produce ONLY a minimal SKELETON of the solution in valid Python code.\n"
		"The SKELETON should include the function signature, a concise docstring, and commented placeholders or control-flow outlines for each step.\n"
		"Do NOT implement the full logic or return working internals — use comments like '# TODO: ...' for implementation points.\n"
		"Do NOT include explanations, tests, or any text outside the Python code block.\n\n"
		f"Problem:\n{problem}\n\n"
		"Provide the SKELETON now:\n"
	)


def safe_get(example: dict, keys: list) -> Any:
	for k in keys:
		if k in example and example[k] is not None:
			return example[k]
	return None


def pretty_print_example(idx: int, ex: dict) -> None:
	prompt = safe_get(ex, ["prompt", "task", "problem", "input"]) or "<MISSING PROMPT>"
	# common field names for canonical solution
	solution = safe_get(ex, ["canonical_solution", "solution", "canonical_output"]) or "<MISSING SOLUTION>"
	tests = safe_get(ex, ["tests", "test", "raw_tests"]) or safe_get(ex, ["testcases"]) or "<MISSING TESTS>"

	print("=" * 80)
	print(f"EXAMPLE INDEX: {idx}")
	print("=" * 80)

	print("1, prompt")
	print()
	# prompt may be multi-line; print as-is
	print(prompt)
	print()

	print("2, solution")
	print()
	# solution can be code or list; handle common cases
	if isinstance(solution, list):
		for line in solution:
			print(line)
	else:
		print(solution)
	print()

	print("3, tests")
	print()
	if isinstance(tests, list):
		for t in tests:
			print(t)
	else:
		print(tests)

	print("\n")


def main() -> None:
	parser = argparse.ArgumentParser(description="Inspect OpenAI HumanEval dataset examples and print prompt/solution/tests for selected indices.")
	parser.add_argument(
		"--split",
		type=str,
		default="test",
		help="Which split to use from the dataset (default: test).",
	)
	parser.add_argument(
		"--models",
		nargs="+",
		default=DEFAULT_MODELS,
		help="List of model names to use (default: Qwen variants).",
	)
	parser.add_argument(
		"--max_new_tokens",
		type=int,
		default=256,
		help="Max new tokens to generate for skeletons.",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.0,
		help="Sampling temperature (0.0 for greedy).",
	)
	parser.add_argument(
		"--top_p",
		type=float,
		default=1.0,
		help="Top-p sampling value.",
	)
	parser.add_argument(
		"--dtype",
		type=str,
		default="auto",
		choices=["auto", "bfloat16", "float16", "float32"],
		help="Torch dtype for model loading.",
	)

	args = parser.parse_args()

	ds = load_dataset("openai_humaneval")
	if args.split not in ds:
		raise ValueError(f"Split '{args.split}' not found in dataset. Available: {list(ds.keys())}")

	split = ds[args.split]

	# Use the INDICES variable declared at the top of the file. Edit that variable
	# to control which examples are printed.
	idxs = INDICES

	torch_dtype = get_torch_dtype(args.dtype)

	for model_name in args.models:
		print("\n" + "#" * 80)
		print(f"LOADING MODEL: {model_name}")
		print("#" * 80 + "\n")

		try:
			model, tokenizer = load_model_and_tokenizer(model_name, torch_dtype)
		except Exception as e:
			print(f"Failed to load model {model_name}: {e}")
			continue

		for i in idxs:
			if i < 0 or i >= len(split):
				print(f"Index {i} out of range for split '{args.split}' (size={len(split)}). Skipping.")
				continue
			ex = split[i]
			prompt = safe_get(ex, ["prompt", "task", "problem", "input"]) or "<MISSING PROMPT>"

			# Print the original prompt for context
			print("=" * 80)
			print(f"EXAMPLE INDEX: {i}")
			print("=" * 80)
			print("PROMPT:")
			print(prompt)
			print("-" * 80)

			skeleton_prompt = build_skeleton_prompt(prompt)

			try:
				skeleton = generate_one(
					model=model,
					tokenizer=tokenizer,
					prompt_text=skeleton_prompt,
					max_new_tokens=args.max_new_tokens,
					temperature=args.temperature,
					top_p=args.top_p,
				)
			except Exception as e:
				print(f"Generation failed for model {model_name} on example {i}: {e}")
				skeleton = "<GENERATION FAILED>"

			print(f"MODEL: {model_name}")
			print("SKELETON OUTPUT:")
			print(skeleton)
			print("\n" + "~" * 80 + "\n")

		cleanup_model(model, tokenizer)


if __name__ == "__main__":
	main()
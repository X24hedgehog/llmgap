"""
prepare_gsm8k_v2.py

Builds a tree-with-subquestions CSV from GSM8K using a controlled algorithm
with minimal LLM use (Qwen 7B only in step 3 for question phrasing).

Steps:
  1. Parse <<expr=result>> annotations from reasoning trace → computation steps
  2. Build dependency tree algorithmically by matching numeric operands to
     previous step results — no LLM needed
  3. LLM converts step explanation text → natural-language subquestion
     (one call per problem, Qwen 7B)

Output schema: question, reasoning_trace, tree, split
  (tree nodes have content = natural-language subquestion)

Target-node selection and final NS-prompt construction are handled
separately by build_ns_targets.py.

Usage:
  python prepare_gsm8k_v2.py                          # full dataset
  python prepare_gsm8k_v2.py --start 0 --end 880      # shard [0, 880)
  python prepare_gsm8k_v2.py --max-rows 10             # smoke test
  python prepare_gsm8k_v2.py --steps-only              # debug steps 1 & 2
"""

import argparse
import ast
import json
import re
import time
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
OUT_DIR   = Path("out")
TRAIN_CSV = OUT_DIR / "gsm8k_train.csv"
TEST_CSV  = OUT_DIR / "gsm8k_test.csv"

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_CACHE_DIR  = "/cluster/scratch/tunguyen1/hf_cache"
MAX_RETRIES   = 3
RETRY_DELAY   = 5

# ── prompts ───────────────────────────────────────────────────────────────────
CONVERT_SYSTEM = (
    "You are an expert annotation assistant for GSM8K-style math reasoning traces."
)

# Sentinels __PROBLEM__ and __STEPS__ avoid .format() brace-escaping issues.
CONVERT_PROMPT_TEMPLATE = """\
You will receive:
1. a PROBLEM (math word problem)
2. a list of STEPS, each with an id (n1, n2, ...) and an explanation text

Your task is to rephrase each step explanation as a one-sentence natural-language subquestion.

It should:
- ask for the quantity introduced by that node,
- mention the relevant object / unit / entity,
- be understandable without copying raw equations only,
- reflect the semantic meaning, not merely the arithmetic expression.

==================================================
RULES
==================================================

1. Do NOT include the numerical result of the step in the subquestion.
2. The subquestion MUST end with "?".
3. Reference specific entities from the PROBLEM (names, objects, units, prices, etc.).
4. Ask what quantity is being computed — do not echo the arithmetic formula.
5. One short sentence per subquestion.
6. Use the PROBLEM context to recover the real semantic meaning of a step,
   especially when the step explanation contains only a bare equation.

==================================================
WHAT COUNTS AS A GOOD SUBQUESTION
==================================================

Good examples:
- "How many days are in December?"
- "How many cups of bird food does Herman use per day?"
- "How much do the burgers cost in total?"
- "What is the total number of paintings sold by the first 4 customers?"

Bad examples (never output these):
- "What is 31+31+28?"           ← raw arithmetic formula, no semantic meaning
- "What is 1/2+1/2?"            ← same issue
- "What is the next step?"      ← meaningless
- "How much is $300?"           ← contains the numeric result
- "There are 120 - 36 = 8484 pages left to be read." ← Bare copy from explanation, no question format

==================================================
OUTPUT FORMAT
==================================================

Return exactly one valid Python dictionary mapping each step id to its subquestion.

{"n1": "...", "n2": "...", ...}

Requirements:
- All step ids must appear as keys.
- Values must be natural-language subquestions ending with "?".
- No explanations. No markdown. No code fences. Output the dictionary only.

==================================================
EXAMPLE
==================================================

PROBLEM:
Herman likes to feed the birds in December, January and February. He feeds them
1/2 cup in the morning and 1/2 cup in the afternoon. How many cups of food will
he need for all three months?

STEPS:
n1: December has 31 days
n2: January has 31 days
n3: February has 28 days
n4: 31+31+28 = 90 days total
n5: 1/2+1/2 = 1 cup per day
n6: 1*90 = 90 cups total

OUTPUT:
{"n1": "How many days are in December?", "n2": "How many days are in January?", "n3": "How many days are in February?", "n4": "How many days are there in December, January, and February altogether?", "n5": "How many cups of bird food does Herman use per day?", "n6": "How many cups of bird food will Herman need for all three months?"}

==================================================
NOW ANNOTATE THE FOLLOWING INSTANCE
==================================================

PROBLEM:
__PROBLEM__

STEPS:
__STEPS__

OUTPUT:"""

# ── regex ─────────────────────────────────────────────────────────────────────
ANNOTATION_RE = re.compile(r'<<([^>]+?)=(-?[\d.]+)>>')


# ── step 1: parse steps from reasoning trace ──────────────────────────────────

def clean_explanation(text: str) -> str:
    """Remove <<expr=result>> wrappers, keeping only the visible result number."""
    return re.sub(r'<<[^>]+=(-?[\d.]+)>>', r'\1', text).strip()


def parse_steps(reasoning_trace: str) -> list[dict]:
    """
    Extract computation steps from a GSM8K reasoning trace.

    Each <<expr=result>> annotation becomes one step.
    Returns a list of dicts: step_id, result (int or float), expression, explanation.
    Lines starting with '####' (final answer) are ignored.
    """
    steps = []
    step_num = 1
    for line in reasoning_trace.split('\n'):
        line = line.strip()
        if not line or line.startswith('####'):
            continue
        for m in ANNOTATION_RE.finditer(line):
            expr       = m.group(1)
            result_str = m.group(2)
            try:
                result = float(result_str)
                if result.is_integer():
                    result = int(result)
            except (ValueError, OverflowError):
                continue
            steps.append({
                "step_id":     f"n{step_num}",
                "result":      result,
                "expression":  expr,
                "explanation": line,
            })
            step_num += 1
    return steps


# ── step 2: build dependency tree algorithmically ─────────────────────────────

def extract_numbers_from_expr(expr: str) -> set[float]:
    """Return all non-negative numeric values present in an expression string."""
    nums: set[float] = set()
    for m in re.finditer(r'\d+\.?\d*', expr):
        try:
            nums.add(float(m.group()))
        except ValueError:
            pass
    return nums


def build_tree_from_steps(steps: list[dict]) -> dict:
    """
    Step 2: Build a dependency tree purely from numeric matching.

    Rule: step A is a child of step B if:
      - A appears before B in the trace (i.e. i_A < i_B), AND
      - A's result value appears as a numeric token in B's expression.

    Returns tree dict with same format as prepare_gsm8k.py:
      {nid: {"node_id": nid, "content": explanation_text, "children": [...]}}
    The "content" field will be replaced by a subquestion in step 3.
    """
    tree = {
        s["step_id"]: {
            "node_id":  s["step_id"],
            "content":  s["explanation"],  # overwritten in step 3
            "children": [],
        }
        for s in steps
    }

    for i, step in enumerate(steps):
        expr_nums = extract_numbers_from_expr(step["expression"])
        for prev in steps[:i]:
            prev_result = float(prev["result"])
            if prev_result == 0:
                continue  # 0 is too common to be a reliable dependency signal
            if any(abs(n - prev_result) < 1e-6 for n in expr_nums):
                cid = prev["step_id"]
                children = tree[step["step_id"]]["children"]
                if cid not in children:
                    children.append(cid)

    return tree


# ── step 3: llm converts explanations to subquestions ─────────────────────────

def call_llm(model, tokenizer, user_prompt: str, system: str) -> str | None:
    for attempt in range(MAX_RETRIES):
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    [retry {attempt+1}/{MAX_RETRIES}] {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    [failed] {e}")
                return None


def parse_questions_dict(raw: str) -> dict | None:
    """Parse LLM output → dict[node_id → subquestion_string], or None on failure."""
    if not raw:
        return None
    raw = re.sub(r'^```(?:python|json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw.strip())
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(raw)
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None
    return obj


def generate_subquestions(problem: str, steps: list[dict], model, tokenizer) -> dict:
    """
    Step 3: One LLM call per problem to convert all step explanations into
    natural-language subquestions.

    Falls back to cleaned explanation text for any node the LLM misses or
    if the LLM output cannot be parsed.
    """
    steps_text = "\n".join(
        f"{s['step_id']}: {clean_explanation(s['explanation'])}"
        for s in steps
    )
    user_prompt = (
        CONVERT_PROMPT_TEMPLATE
        .replace("__PROBLEM__", problem)
        .replace("__STEPS__", steps_text)
    )
    raw       = call_llm(model, tokenizer, user_prompt, CONVERT_SYSTEM)
    questions = parse_questions_dict(raw) if raw else None

    # Build result dict, falling back to cleaned explanation for any missing key
    result: dict[str, str] = {}
    for s in steps:
        nid = s["step_id"]
        if questions and nid in questions and isinstance(questions[nid], str) and questions[nid].strip():
            q = questions[nid].strip()
            if not q.endswith("?"):
                q += "?"
            result[nid] = q
        else:
            result[nid] = clean_explanation(s["explanation"])
    return result


def update_tree_content(tree: dict, questions: dict) -> dict:
    """Replace each node's 'content' with the LLM-generated subquestion."""
    for nid, q in questions.items():
        if nid in tree:
            tree[nid]["content"] = q
    return tree


# ── main pipeline ─────────────────────────────────────────────────────────────

def build_tree_csv(df: pd.DataFrame, model, tokenizer) -> pd.DataFrame:
    """
    Steps 1-3 only: parse steps, build tree, fill content with LLM subquestions.
    Output schema: question, reasoning_trace, tree, split
    """
    rows    = []
    n_empty = 0

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="build_trees")):
        problem = str(row["question"])
        answer  = str(row["answer"])

        empty_row = {
            "question":        problem,
            "reasoning_trace": answer,
            "tree":            json.dumps({}),
            "split":           row["split"],
        }

        # ── step 1 ────────────────────────────────────────────────────────────
        steps = parse_steps(answer)
        if len(steps) < 2:
            n_empty += 1
            rows.append(empty_row)
            continue

        # ── step 2 ────────────────────────────────────────────────────────────
        tree = build_tree_from_steps(steps)

        # ── step 3 ────────────────────────────────────────────────────────────
        print(f"  [{i+1}/{len(df)}] Generating subquestions ({len(steps)} steps)...")
        questions = generate_subquestions(problem, steps, model, tokenizer)
        tree = update_tree_content(tree, questions)

        rows.append({
            "question":        problem,
            "reasoning_trace": answer,
            "tree":            json.dumps(tree),
            "split":           row["split"],
        })

    print(f"  Done. empty rows (skipped): {n_empty}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help="HuggingFace model id (default: %(default)s)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit rows per split (smoke test)")
    parser.add_argument("--start",      type=int, default=None,
                        help="Start index (inclusive, 0-based)")
    parser.add_argument("--end",        type=int, default=None,
                        help="End index (exclusive, 0-based)")
    parser.add_argument("--steps-only", action="store_true",
                        help="Run only steps 1 & 2, save debug CSV, then exit")
    args = parser.parse_args()

    for p in (TRAIN_CSV, TEST_CSV):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run inspect_gsm8k.py first")

    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    train["split"] = "train"
    test["split"]  = "test"

    if args.max_rows:
        train = train.head(args.max_rows)
        test  = test.head(args.max_rows)

    df_all     = pd.concat([train, test], ignore_index=True)
    total_rows = len(df_all)
    print(f"Loaded {len(train):,} train + {len(test):,} test = {total_rows:,} total rows")

    start = args.start if args.start is not None else 0
    end   = args.end   if args.end   is not None else total_rows
    if start != 0 or end != total_rows:
        df_all = df_all.iloc[start:end].reset_index(drop=True)
        print(f"Processing shard [{start}:{end}] → {len(df_all):,} rows")
    shard_suffix = f"_{start}_{end}" if (args.start is not None or args.end is not None) else ""

    tree_out = OUT_DIR / f"gsm8k_trees{shard_suffix}.csv"
    OUT_DIR.mkdir(exist_ok=True)

    # ── steps-only mode (debug step 1 & 2) ───────────────────────────────────
    if args.steps_only:
        debug_out = OUT_DIR / f"gsm8k_v2_steps_debug{shard_suffix}.csv"
        print(f"\n── Steps-only mode: running steps 1 & 2, saving → {debug_out.name} ──")
        rows = []
        n_short = 0
        for _, row in tqdm(df_all.iterrows(), total=len(df_all), desc="steps1+2"):
            problem = str(row["question"])
            answer  = str(row["answer"])
            steps   = parse_steps(answer)
            if len(steps) < 2:
                n_short += 1
            tree = build_tree_from_steps(steps) if steps else {}
            rows.append({
                "question":       problem,
                "reasoning_trace": answer,
                "steps_json":     json.dumps(steps),
                "tree_json":      json.dumps(tree),
                "split":          row["split"],
            })
        debug_df = pd.DataFrame(rows)
        debug_df.to_csv(debug_out, index=False)
        print(f"Saved {len(debug_df):,} rows → {debug_out}  (skipped <2 steps: {n_short})")
        return

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=HF_CACHE_DIR, local_files_only=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    hf_model.eval()
    print("Model loaded.")

    print(f"\n── Building {tree_out.name} ──")
    tree_df = build_tree_csv(df_all, hf_model, tokenizer)
    tree_df.to_csv(tree_out, index=False)
    print(f"Saved {len(tree_df):,} rows → {tree_out}")


if __name__ == "__main__":
    main()

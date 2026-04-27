"""
prepare_gsm8k.py

Reads out/gsm8k_train.csv and out/gsm8k_test.csv and produces two
pipeline-ready CSVs compatible with run_inference.py and finetune.py:

  out/correct_answer_pairs_gsm8k.csv   -- one row per problem, no API needed
  out/next_subquestion_pairs_gsm8k.csv -- one row per problem, uses OpenRouter

For the correct_answer task:
  - prompt = problem text  (run_inference.py appends "Solve step by step.")
  - target_question_reasoning_trace = full GSM8K answer (CoT gold label)
  - target_answer = numeric answer extracted after ####

For the next_subquestion task:
  - Uses OpenRouter to extract numbered subquestions from the answer trace
  - Target = the LAST subquestion (directly answers the overall problem)
  - Prompt = problem + context of all preceding subquestions + their answers

Requirements:
  pip install openai
  export OPENROUTER_API_KEY=<your key>

Usage:
  python prepare_gsm8k.py
  python prepare_gsm8k.py --model anthropic/claude-3-haiku
  python prepare_gsm8k.py --max-rows 10     # smoke test
  python prepare_gsm8k.py --skip-ns         # only build correct_answer CSV
"""

import argparse
import ast
import json
import os
import random
import re
import time
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
OUT_DIR   = Path("out")
TRAIN_CSV = OUT_DIR / "gsm8k_train.csv"
TEST_CSV  = OUT_DIR / "gsm8k_test.csv"
CA_OUT    = OUT_DIR / "correct_answer_pairs_gsm8k_sample.csv"
NS_OUT    = OUT_DIR / "next_subquestion_pairs_gsm8k_sample.csv"

# ── defaults ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_CACHE_DIR  = "/cluster/scratch/tunguyen1/hf_cache"
MAX_RETRIES   = 3
RETRY_DELAY   = 5   # seconds between retries

# ── prompts ────────────────────────────────────────────────────────────────────
EXTRACT_SYSTEM = (
    "You are an expert annotation assistant for GSM8K-style math reasoning traces."
)

# Uses __QUESTION__ and __ANSWER__ as sentinels (no .format() brace escaping needed).
EXTRACT_PROMPT_TEMPLATE = """\
You will receive:
1. a QUESTION
2. an ANSWER_REASONING_TRACE

Your task is to transform the reasoning trace into a tree of subquestions.

IMPORTANT SCOPE RESTRICTION

We only handle relatively simple arithmetic reasoning for now.

If the problem or reasoning trace requires non-trivial algebraic setup or solving
equations with variables, such as:
- introducing variables like x, y, p, S, etc. and solving equations,
- combining like terms such as 8m + m = 27,
- deriving equations such as 7S = 847,
- solving equations such as x = 2350 or p = 1000 after symbolic manipulation,
- systems of equations,
- any non-trivial variable instantiation / algebraic solving,

then do NOT build the tree.

Instead output exactly:
{"status": "non-trivial"}

Do not output anything else in that case.

==================================================
MAIN OBJECTIVE
==================================================

Build a tree of subquestions from the reasoning trace.

A subquestion corresponds to one newly introduced numeric quantity that is needed
in the reasoning.

The key principle is:

  DO NOT use "one sentence = one step".
  DO NOT use "one line = one step".

  Instead:
  LOOK FOR NEW NUMERIC QUANTITIES introduced in the reasoning.

Each newly introduced numeric quantity should usually correspond to one node.

A newly introduced numeric quantity is a quantity that:
- is not explicitly stated in the original QUESTION,
- and appears in the ANSWER_REASONING_TRACE through either:
  1. a mathematical operation, or
  2. an inference / assumption / common-sense fact / conversion / factual lookup.

Examples:
- 31 days in December
- 28 days in February
- 90 days total
- 1 cup per day from 1/2 + 1/2
- 90 cups total

==================================================
WHAT COUNTS AS A VALID STEP / NODE
==================================================

Create one node for each new numeric quantity introduced by:

A. Arithmetic operation
Examples:
- 31 + 31 + 28 = 90
- 1/2 + 1/2 = 1
- 5 * 7 = 35
- 100 - 40 = 60
- 3 * 12 = 36

B. Inference or world knowledge not explicitly given in the question
Examples:
- February has 28 days
- A year has 12 months
- 1 week has 7 days
- 1 hour has 60 minutes
- 1 dozen means 12

If such information is needed and not explicitly stated in the question, you may
infer it. This inference itself counts as a valid subquestion node.

When needed, you may make common-sense assumptions, but only if the information
is not already explicitly present in the question.

For example:
  Subquestion: "How many days are in February?"
  This may be answered by assuming February has 28 days.

==================================================
WHAT DOES NOT COUNT AS A NODE
==================================================

Do NOT create a node for:
- numbers already explicitly present in the QUESTION,
- mere variable declarations with no numeric result,
- restatements of the same quantity without introducing a new quantity,
- symbolic setup steps with no new numeric quantity,
- algebraic solving steps (those should trigger {"status": "non-trivial"}).

==================================================
SENTENCE / CHUNK HANDLING
==================================================

You must analyze the reasoning semantically.

Important:
- Do NOT naively split only on periods.
- Do NOT assume each line is a single step.
- Do NOT assume each sentence is a single step.

A single sentence may contain multiple subquestions if it introduces multiple
new numeric quantities.

Example:
  "December has 31 days, January has 31 days and February has 28 days for a
  total of 31+31+28 = 90 days"

  This should produce FOUR nodes:
  1. days in December = 31
  2. days in January = 31
  3. days in February = 28
  4. days in December+January+February = 90

You must use semantic understanding to break the reasoning into minimal
subquestions.

Be careful with abbreviations and semantic sentence splitting.
Do not incorrectly split on abbreviations such as:
- Mr.
- Mrs.
- Dr.
- lbs.
- P.E.

==================================================
QUESTION CONTEXT IS REQUIRED
==================================================

You must use BOTH:
- the QUESTION
- the ANSWER_REASONING_TRACE
- and earlier parts of the reasoning trace

to determine the meaning of a numeric quantity.

This is especially important when the reasoning trace contains only equations
or very short fragments without explanations.

Example:
  If the reasoning trace contains:
    "1/2 + 1/2 = 1"
  then you must use the QUESTION context to determine what this 1 refers to.

  For example, if the question is about milk consumption, the node content should
  be something like:
    "How many cups of milk does Alex drink in one day?"
  and NOT just:
    "What is 1/2 + 1/2?"

Always recover the real semantic subquestion.

==================================================
CHILD RELATION / TREE STRUCTURE
==================================================

You must output a tree.

A node X is a child of node Y if:
  the numeric quantity introduced in node X is used to compute the numeric
  quantity introduced in node Y.

Important:
- Child relations are based on quantity dependency, not sentence order alone.
- Use semantic understanding to determine dependency.
- Two nodes may have the same numeral but represent different quantities; treat
  them separately if the meanings differ.

Example:
  - 90 days
  - 90 cups
  These are DIFFERENT quantities and therefore DIFFERENT nodes.

Also:
  If the same numeric value is repeated merely as a restatement of the same
  quantity, it should usually remain ONE node.

  Example:
    "3/5 x 100 = 60"
    "60 tennis balls Ralph did not hit"
  This is usually one node for "60 tennis balls Ralph did not hit", not two.

If a node is computed only from numbers directly given in the QUESTION, then its
children list may be empty.

Example:
  Question gives 1/2 cup in the morning and 1/2 cup in the afternoon.
  Then the node:
    "How many cups of bird food does Herman use per day?"
  introduces the new value 1, but its children list should be empty because the
  inputs 1/2 and 1/2 came directly from the question, not from earlier nodes.

==================================================
OUTPUT FORMAT
==================================================

Return exactly one valid Python dictionary.

Case 1: simple arithmetic problem
Return a dictionary of dictionaries.

Use node ids:
- n1
- n2
- n3
- ...

Each node must be of the form:

  "nX": {
      "node_id": "nX",
      "content": "<natural language subquestion>",
      "children": [list of child node ids]
  }

Requirements:
- The outer dictionary key is the node id.
- The "node_id" field must repeat the same id.
- "content" must be a natural-language subquestion.
- "children" must be a list of node ids.
- Use [] for leaf nodes.
- Prefer assigning ids bottom-up:
    leaves first, final target node last.

Case 2: non-trivial algebra / equation solving
Return exactly:
  {"status": "non-trivial"}

Do not include explanations.
Do not include markdown.
Do not include code fences.
Do not include any text before or after the dictionary.

==================================================
HOW TO WRITE EACH NODE CONTENT
==================================================

The "content" field must be a natural-language subquestion.

It should:
- ask for the quantity introduced by that node,
- mention the relevant object / unit / entity,
- be understandable without copying raw equations only,
- reflect the semantic meaning, not merely the arithmetic expression.

Good examples:
- "How many days are in December?"
- "How many days are in January?"
- "How many days are in February?"
- "How many days are there in December, January, and February altogether?"
- "How many cups of bird food does Herman use per day?"
- "How many cups of bird food will Herman need for all three months?"

Bad examples:
- "What is 31+31+28?"
- "What is 1/2+1/2?"
- "What is the next step?"

Prefer semantic subquestions over raw arithmetic questions.

==================================================
DECISION RULES SUMMARY
==================================================

1. First decide whether the problem is non-trivial due to algebra / equation-solving.
   - If yes, return {"status": "non-trivial"}.

2. Otherwise, read the QUESTION carefully.

3. Read the ANSWER_REASONING_TRACE semantically.

4. Identify every newly introduced numeric quantity needed in the reasoning.
   - A single sentence may create multiple nodes.
   - A single line may create multiple nodes.
   - A chunk containing only an equation still requires semantic interpretation
     using the question context.

5. Create one node per introduced quantity.

6. Build dependency edges:
   - node X is a child of node Y if X's quantity is used to compute Y's quantity.

7. Distinguish same numeral / different quantity from same numeral / same quantity.
   - same numeral + different meaning => different nodes
   - same numeral + same meaning restated => same node

8. Output only the final dictionary.

==================================================
EXAMPLE
==================================================

QUESTION:
Herman likes to feed the birds in December, January and February. He feeds them
1/2 cup in the morning and 1/2 cup in the afternoon. How many cups of food will
he need for all three months?

ANSWER_REASONING_TRACE:
December has 31 days, January has 31 days and February has 28 days for a total
of 31+31+28 = 90 days
He feeds them 1/2 cup in the morning and 1/2 cup in the afternoon for a total
of 1/2+1/2 = 1 cup per day
If he feeds them 1 cup per day for 90 days then he will need 1*90 = 90 cups of
birdseed
#### 90

OUTPUT:
{
    "n1": {
        "node_id": "n1",
        "content": "How many days are in December?",
        "children": []
    },
    "n2": {
        "node_id": "n2",
        "content": "How many days are in January?",
        "children": []
    },
    "n3": {
        "node_id": "n3",
        "content": "How many days are in February?",
        "children": []
    },
    "n4": {
        "node_id": "n4",
        "content": "How many days are there in December, January, and February altogether?",
        "children": ["n1", "n2", "n3"]
    },
    "n5": {
        "node_id": "n5",
        "content": "How many cups of bird food does Herman use per day?",
        "children": []
    },
    "n6": {
        "node_id": "n6",
        "content": "How many cups of bird food will Herman need for all three months?",
        "children": ["n4", "n5"]
    }
}

==================================================
NOW ANNOTATE THE FOLLOWING INSTANCE
==================================================

QUESTION:
__QUESTION__

ANSWER_REASONING_TRACE:
__ANSWER__"""

NS_INSTRUCTION = (
    "You are given a math word problem and some intermediate reasoning steps that are already "
    "solved.\n  Your task is to produce exactly one next subquestion that should be asked to "
    "continue solving the\n  problem. Do not solve the full problem. Do not output anything "
    "except the next subquestion."
)

NS_CONTEXT_TEMPLATE = (
    "{instruction}\n  Problem: {problem}  Given that we know the answer to the following "
    "questions: {context}  Next subquestion:"
)


# ── helpers ────────────────────────────────────────────────────────────────────

def extract_numeric_answer(answer_text: str) -> str:
    """Extract the number after #### in the GSM8K answer field."""
    m = re.search(r"####\s*(.+)", answer_text)
    if m:
        return m.group(1).strip().replace(",", "")
    return ""


def strip_solution(answer_text: str) -> str:
    """Return answer text without the #### line for cleaner CoT."""
    return re.sub(r"\s*####.*", "", answer_text).strip()


def call_api(model, tokenizer, user_prompt: str, system: str) -> str | None:
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
                    max_new_tokens=2048,
                    do_sample=False,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    [retry {attempt + 1}/{MAX_RETRIES}] {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    [failed] {e}")
                return None


def make_extract_prompt(question: str, answer: str) -> str:
    """Fill the tree-extraction prompt template with the actual question and answer."""
    return EXTRACT_PROMPT_TEMPLATE.replace("__QUESTION__", question).replace("__ANSWER__", answer)


def parse_tree(raw: str) -> dict | None:
    """
    Parse LLM response → tree dict or {"status": "non-trivial"} or None on parse error.
    Accepts both JSON and Python-style single-quoted dicts.
    """
    if not raw:
        return None
    # strip markdown fences
    raw = re.sub(r"^```(?:python|json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
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


def is_non_trivial(tree: dict) -> bool:
    return tree.get("status") == "non-trivial"


# ── tree navigation ────────────────────────────────────────────────────────────

def get_root_node(tree: dict) -> str | None:
    """Return the node id that is not a child of any other node (= final answer node)."""
    all_children: set[str] = set()
    for node in tree.values():
        if isinstance(node, dict):
            all_children.update(node.get("children", []))
    for nid in tree:
        if nid not in all_children:
            return nid
    return None


def topological_order(tree: dict, root_id: str) -> list[str]:
    """DFS post-order: leaves first, root last."""
    order: list[str] = []
    visited: set[str] = set()

    def dfs(nid: str) -> None:
        if nid in visited or nid not in tree:
            return
        visited.add(nid)
        for child_id in tree[nid].get("children", []):
            dfs(child_id)
        order.append(nid)

    dfs(root_id)
    return order


def get_transitive_deps(tree: dict, node_id: str) -> list[str]:
    """Return all node ids reachable via children edges from node_id
    (its transitive dependencies), in topological order (leaves first).
    Does NOT include node_id itself."""
    order: list[str] = []
    visited: set[str] = set()

    def dfs(nid: str) -> None:
        if nid in visited or nid not in tree:
            return
        visited.add(nid)
        for child_id in tree[nid].get("children", []):
            dfs(child_id)
        order.append(nid)

    for child_id in tree[node_id].get("children", []):
        dfs(child_id)
    return order


def build_ns_prompt_from_tree(problem: str, tree: dict) -> tuple[str, str] | None:
    """
    Build NS prompt + target from tree.
    Picks a random intermediate node (has children, is not root) as target.
    Context = transitive dependencies of the target node (its subtree), leaves first.
    Falls back to root if no intermediate nodes exist.
    Returns (prompt, target_subquestion) or None if unusable.
    """
    if len(tree) < 2:
        return None
    root_id = get_root_node(tree)
    if root_id is None:
        return None

    # Intermediate nodes: have at least one child AND are not the root
    intermediate_ids = [
        nid for nid, node in tree.items()
        if nid != root_id
        and isinstance(node, dict)
        and len(node.get("children", [])) > 0
    ]

    if intermediate_ids:
        target_id = random.choice(intermediate_ids)
        dep_ids = get_transitive_deps(tree, target_id)
        if not dep_ids:
            # Intermediate node has children but none resolved — skip
            return None
    else:
        # No intermediate nodes (flat tree): fall back to root, all leaves as context
        target_id = root_id
        topo = topological_order(tree, root_id)
        dep_ids = topo[:-1]
        if not dep_ids:
            return None

    context_questions = [tree[nid]["content"] for nid in dep_ids if nid in tree]
    context = " ".join(q if q.endswith("?") else q + "?" for q in context_questions)
    target = tree[target_id]["content"]
    prompt = NS_CONTEXT_TEMPLATE.format(
        instruction=NS_INSTRUCTION,
        problem=problem,
        context=context,
    )
    return prompt, target


# ── main ───────────────────────────────────────────────────────────────────────

def build_correct_answer_csv(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="correct_answer"):
        answer_full = str(row["answer"])
        rows.append({
            "question":                      row["question"],
            "reasoning_trace":               answer_full,
            "target_answer":                 extract_numeric_answer(answer_full),
            "target_question_reasoning_trace": strip_solution(answer_full),
            "prompt":                        row["question"],
            "split":                         row["split"],
        })
    return pd.DataFrame(rows)


def build_next_subquestion_csv(
    df: pd.DataFrame,
    model,
    tokenizer,
) -> pd.DataFrame:
    """
    For each problem, call the API to extract a tree of subquestions, then build
    one row where the target is the root node (final answer subquestion) and the
    prompt provides context of all preceding nodes in topological order.
    Problems flagged as non-trivial are skipped.
    """
    rows = []
    n_nontrivial = 0
    n_errors = 0

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="next_subquestion")):
        problem = str(row["question"])
        answer  = str(row["answer"])

        # ── default empty row (written on any failure) ──────────────────────
        empty_row = {
            "question":         problem,
            "reasoning_trace":  answer,
            "tree":             json.dumps({}),
            "next_subquestion": "",
            "prompt":           "",
            "split":            row["split"],
        }

        print(f"  [{i+1}/{len(df)}] Running inference...")
        user_prompt = make_extract_prompt(problem, answer)
        raw = call_api(model, tokenizer, user_prompt, EXTRACT_SYSTEM)
        tree = parse_tree(raw)
        if tree is None:
            n_errors += 1
            print(f"    Warning: could not parse tree for row {i}, writing empty row")
            rows.append(empty_row)
            continue

        if is_non_trivial(tree):
            n_nontrivial += 1
            rows.append(empty_row)
            continue

        result = build_ns_prompt_from_tree(problem, tree)
        if result is None:
            print(f"    Warning: tree has <2 nodes for row {i}, writing empty row")
            rows.append(empty_row)
            continue
        prompt, target = result

        rows.append({
            "question":         problem,
            "reasoning_trace":  answer,
            "tree":             json.dumps(tree),
            "next_subquestion": target,
            "prompt":           prompt,
            "split":            row["split"],
        })

    print(f"  Done. non-trivial skipped: {n_nontrivial}, parse errors: {n_errors}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help="HuggingFace model id (default: %(default)s)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit rows per split (smoke test)")
    parser.add_argument("--sample",   type=int, default=None,
                        help="Randomly sample N rows from the full dataset (quality check)")
    parser.add_argument("--skip-ns",  action="store_true",
                        help="Skip next_subquestion CSV (no model needed)")
    parser.add_argument("--start",    type=int, default=None,
                        help="Start index (inclusive) for slicing df_all (0-based)")
    parser.add_argument("--end",      type=int, default=None,
                        help="End index (exclusive) for slicing df_all (0-based)")
    args = parser.parse_args()

    # ── load data ──────────────────────────────────────────────────────────────
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

    df_all = pd.concat([train, test], ignore_index=True)
    total_rows = len(df_all)

    if args.sample:
        df_all = df_all.sample(n=min(args.sample, len(df_all)), random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df_all)} rows for quality check.")
    print(f"Loaded {len(train):,} train + {len(test):,} test = {total_rows:,} total rows")

    # ── index-range slicing ───────────────────────────────────────────────────
    start = args.start if args.start is not None else 0
    end   = args.end   if args.end   is not None else total_rows
    if start != 0 or end != total_rows:
        df_all = df_all.iloc[start:end].reset_index(drop=True)
        print(f"Processing shard [{start}:{end}] → {len(df_all):,} rows")
    shard_suffix = f"_{start}_{end}" if (args.start is not None or args.end is not None) else ""

    ca_out = OUT_DIR / f"correct_answer_pairs_gsm8k{shard_suffix}.csv"
    ns_out = OUT_DIR / f"next_subquestion_pairs_gsm8k{shard_suffix}.csv"

    OUT_DIR.mkdir(exist_ok=True)

    # ── correct_answer CSV ─────────────────────────────────────────────────────
    print(f"\n── Building {ca_out.name} ──")
    ca_df = build_correct_answer_csv(df_all)
    ca_df.to_csv(ca_out, index=False)
    print(f"Saved {len(ca_df):,} rows → {ca_out}")

    # ── next_subquestion CSV ───────────────────────────────────────────────────
    if args.skip_ns:
        print("\nSkipping next_subquestion CSV (--skip-ns).")
        return

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=HF_CACHE_DIR, local_files_only=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    hf_model.eval()
    print("Model loaded.")

    print(f"\n── Building {ns_out.name} (model: {args.model}) ──")
    ns_df = build_next_subquestion_csv(df_all, hf_model, tokenizer)
    ns_df.to_csv(ns_out, index=False)
    print(f"Saved {len(ns_df):,} rows → {ns_out}")


if __name__ == "__main__":
    main()

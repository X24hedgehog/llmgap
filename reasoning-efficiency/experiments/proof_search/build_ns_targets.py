"""
build_ns_targets.py

Second stage of the pipeline. Takes the merged tree CSV produced by
prepare_gsm8k_v2.py (after shard merging) and generates the final
next-subquestion dataset for fine-tuning.

Algorithm for target-node selection and context construction:
  For a target node X (any non-root node):
    context = transitive_deps(X)                              -- what X depends on
            ∪ full_subtree(Y) for each sibling Y of X
              WHERE target X is NOT in Y's subtree            -- no circular deps

  Equivalent intuition: "given that you already know these sub-results,
                         what is X?" — makes X the logical next step.

Special cases:
  - Zero-result nodes (result == 0) are excluded from dependency matching,
    so the root is detected correctly even when they are disconnected.
  - Solo leaf (only non-root node, no siblings, empty context) → falls back to
    NS_SINGLE_TEMPLATE (no context list, just the problem).

Output schema: question, reasoning_trace, tree, next_subquestion, prompt, split

Input:  out/gsm8k_trees.csv            (produced by prepare_gsm8k_v2.py + merge)
Output: out/next_subquestion_pairs_gsm8k_v2.csv

Usage:
  python build_ns_targets.py
  python build_ns_targets.py --input  out/gsm8k_trees.csv
  python build_ns_targets.py --output out/next_subquestion_pairs_gsm8k_v2.csv
  python build_ns_targets.py --seed 42
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = Path("out/gsm8k_trees.csv")
DEFAULT_OUTPUT = Path("out/next_subquestion_pairs_gsm8k_v2.csv")

# ── prompt templates ──────────────────────────────────────────────────────────
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

NS_SINGLE_TEMPLATE = (
    "You are given a math word problem.\n  Your task is to produce the one subquestion "
    "you need to answer in order to solve the problem. Do not solve the full problem. "
    "Do not output anything except the subquestion.\n  Problem: {problem}  "
    "Only subquestion:"
)


# ── tree helpers ──────────────────────────────────────────────────────────────

def get_root_node(tree: dict) -> str | None:
    """Node not listed as a child of any other node = root.

    When multiple such nodes exist (e.g. disconnected zero-result nodes),
    prefer the one that has children — that is the true answer root.
    """
    all_children: set[str] = set()
    for node in tree.values():
        if isinstance(node, dict):
            all_children.update(node.get("children", []))
    candidates = [nid for nid in tree if nid not in all_children]
    if not candidates:
        return None
    # Prefer candidates that have at least one child (true root, not isolated leaf)
    with_children = [
        nid for nid in candidates
        if isinstance(tree[nid], dict) and tree[nid].get("children")
    ]
    if with_children:
        return with_children[0]
    return candidates[0]


def build_parent_map(tree: dict) -> dict[str, str | None]:
    """Map node_id → parent node_id (None for root / disconnected nodes)."""
    parent: dict[str, str | None] = {nid: None for nid in tree}
    for nid, node in tree.items():
        if not isinstance(node, dict):
            continue
        for child_id in node.get("children", []):
            if child_id in tree:
                parent[child_id] = nid
    return parent


def get_subtree(tree: dict, node_id: str) -> list[str]:
    """All nodes in subtree rooted at node_id (including node_id),
    topological order: leaves first."""
    order:   list[str] = []
    visited: set[str]  = set()

    def dfs(nid: str) -> None:
        if nid in visited or nid not in tree:
            return
        visited.add(nid)
        for child in tree[nid].get("children", []):
            dfs(child)
        order.append(nid)

    dfs(node_id)
    return order


def get_transitive_deps(tree: dict, node_id: str) -> list[str]:
    """Subtree of node_id EXCLUDING node_id itself, leaves first."""
    order:   list[str] = []
    visited: set[str]  = set()

    def dfs(nid: str) -> None:
        if nid in visited or nid not in tree:
            return
        visited.add(nid)
        for child in tree[nid].get("children", []):
            dfs(child)
        order.append(nid)

    if isinstance(tree.get(node_id), dict):
        for child in tree[node_id].get("children", []):
            dfs(child)
    return order


def topological_order_full(tree: dict, root_id: str) -> list[str]:
    """Full topological order of the tree (leaves first, root last)."""
    order:   list[str] = []
    visited: set[str]  = set()

    def dfs(nid: str) -> None:
        if nid in visited or nid not in tree:
            return
        visited.add(nid)
        for child in tree[nid].get("children", []):
            dfs(child)
        order.append(nid)

    dfs(root_id)
    return order


# ── target-selection algorithm ────────────────────────────────────────────────

def build_context_for_target(
    tree: dict,
    target_id: str,
    parent_map: dict[str, str | None],
) -> list[str]:
    """
    Returns the context node list (topological order) for target node X:

      context = transitive_deps(X)                             -- what X depends on
              ∪ full_subtree(Y) for each sibling Y of X
                WHERE target X is NOT in Y's subtree           -- no circular deps

    target_id is always excluded from the returned list.
    """
    parent_id = parent_map.get(target_id)
    if parent_id is None:
        return []

    context_set: set[str] = set()

    # Part 1: what X depends on
    for nid in get_transitive_deps(tree, target_id):
        context_set.add(nid)

    # Part 2: full subtrees of siblings (skipping circular ones)
    siblings = [
        c for c in tree[parent_id].get("children", [])
        if c != target_id
    ]
    for sibling in siblings:
        sibling_subtree = get_subtree(tree, sibling)
        if target_id in sibling_subtree:
            continue  # sibling's subtree contains target → circular → skip
        for nid in sibling_subtree:
            context_set.add(nid)

    context_set.discard(target_id)  # safety

    if not context_set:
        return []

    root_id = get_root_node(tree)
    if root_id is None:
        return []
    topo = topological_order_full(tree, root_id)
    return [nid for nid in topo if nid in context_set]


def get_eligible_targets(
    tree: dict,
    parent_map: dict[str, str | None],
    root_id: str | None,
) -> list[str]:
    """Non-root nodes that have a non-empty context."""
    eligible = []
    for nid in tree:
        if nid == root_id:
            continue
        if parent_map.get(nid) is None:
            continue  # disconnected / no parent
        if build_context_for_target(tree, nid, parent_map):
            eligible.append(nid)
    return eligible


def build_ns_prompt(problem: str, tree: dict) -> tuple[str, str] | None:
    """
    Pick a random eligible target and build (prompt, target_subquestion).

    Normal case:  eligible target with non-empty context → NS_CONTEXT_TEMPLATE.
    Fallback case: single leaf directly under root with no siblings →
                   NS_SINGLE_TEMPLATE (no context list).

    Returns None if tree is too small or no suitable target exists.
    """
    if len(tree) < 2:
        return None

    root_id    = get_root_node(tree)
    parent_map = build_parent_map(tree)
    eligible   = get_eligible_targets(tree, parent_map, root_id)

    if eligible:
        target_id   = random.choice(eligible)
        context_ids = build_context_for_target(tree, target_id, parent_map)

        context_qs = [tree[nid]["content"] for nid in context_ids if nid in tree]
        context    = " ".join(q if q.endswith("?") else q + "?" for q in context_qs)
        target     = tree[target_id]["content"]
        prompt     = NS_CONTEXT_TEMPLATE.format(
            instruction=NS_INSTRUCTION,
            problem=problem,
            context=context,
        )
        return prompt, target

    # Fallback: solo leaf (no context possible) → use single-subquestion template
    solo_nodes = [
        nid for nid in tree
        if nid != root_id and parent_map.get(nid) is not None
    ]
    if not solo_nodes:
        return None

    target_id = random.choice(solo_nodes)
    target    = tree[target_id]["content"]
    prompt    = NS_SINGLE_TEMPLATE.format(problem=problem)
    return prompt, target


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(DEFAULT_INPUT),
                        help="Input CSV with tree column (default: %(default)s)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output CSV (default: %(default)s)")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found — run prepare_gsm8k_v2.py first")

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df):,} rows from {in_path}")

    n_rebuilt = 0
    n_empty   = 0
    new_targets: list[str] = []
    new_prompts: list[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="build_ns"):
        tree_raw = row.get("tree", "{}")
        try:
            tree = json.loads(tree_raw) if isinstance(tree_raw, str) else {}
        except Exception:
            tree = {}

        if not tree:
            new_targets.append("")
            new_prompts.append("")
            n_empty += 1
            continue

        problem = str(row["question"])
        result  = build_ns_prompt(problem, tree)

        if result is None:
            new_targets.append("")
            new_prompts.append("")
            n_empty += 1
        else:
            prompt, target = result
            new_targets.append(target)
            new_prompts.append(prompt)
            n_rebuilt += 1

    df["next_subquestion"] = new_targets
    df["prompt"]           = new_prompts

    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Done. rebuilt: {n_rebuilt:,}  empty: {n_empty:,}")
    print(f"Saved {len(df):,} rows → {out_path}")


if __name__ == "__main__":
    main()

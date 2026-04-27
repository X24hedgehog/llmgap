"""
rebuild_ns_targets.py

Re-generates the `next_subquestion` and `prompt` columns in a
next_subquestion_pairs CSV using a corrected target-selection algorithm.

The new algorithm:
  For a target node X (any non-root node with non-empty context):
    context = (subtree of X minus X itself)          -- what X is built from
            ∪ (full subtree of each sibling of X)    -- peer-results needed alongside X

  Intuition: "given that you already know these sub-results,
              what is X?" makes X the logical next step.

All other columns (question, reasoning_trace, tree, split) are unchanged.

Usage:
  python rebuild_ns_targets.py
  python rebuild_ns_targets.py --input  out/next_subquestion_pairs_gsm8k_v2.csv
  python rebuild_ns_targets.py --output out/next_subquestion_pairs_gsm8k_v2_fixed.csv
  python rebuild_ns_targets.py --seed 42
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = Path("out/next_subquestion_pairs_gsm8k_v2.csv")
DEFAULT_OUTPUT = Path("out/next_subquestion_pairs_gsm8k_v2_fixed.csv")

# ── prompt templates (must match prepare_gsm8k_v2.py) ────────────────────────
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
    prefer the one that has children — that is the true answer root,
    not an isolated leaf.
    """
    all_children: set[str] = set()
    for node in tree.values():
        if isinstance(node, dict):
            all_children.update(node.get("children", []))
    candidates = [nid for nid in tree if nid not in all_children]
    if not candidates:
        return None
    # Prefer nodes that have at least one child (true root over disconnected leaf)
    with_children = [nid for nid in candidates
                     if isinstance(tree[nid], dict) and tree[nid].get("children")]
    if with_children:
        return with_children[0]
    return candidates[0]


def build_parent_map(tree: dict) -> dict[str, str | None]:
    """Map node_id → parent node_id (None for root / disconnected nodes)."""
    parent: dict[str, str | None] = {nid: None for nid in tree}
    for nid, node in tree.items():
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


# ── new target-selection algorithm ───────────────────────────────────────────

def build_context_for_target(tree: dict, target_id: str, parent_map: dict) -> list[str]:
    """
    Returns the context node list (in topological order) for a given target node X:

      context = transitive_deps(X)                             -- what X depends on
              ∪ full_subtree(Y) for each sibling Y of X
                WHERE target X is NOT in Y's subtree           -- no circular deps

    After union, target_id is always removed from context_set (safety).

    Intuition: "given that you already know these sub-results,
                what is X?" makes X the logical next step.
    """
    parent_id = parent_map.get(target_id)
    if parent_id is None:
        return []

    context_set: set[str] = set()

    # Part 1: transitive deps of X (what X depends on)
    for nid in get_transitive_deps(tree, target_id):
        context_set.add(nid)

    # Part 2: full subtrees of siblings of X — only if target is NOT in sibling's subtree
    siblings = [c for c in tree[parent_id].get("children", []) if c != target_id]
    for sibling in siblings:
        sibling_subtree = get_subtree(tree, sibling)
        if target_id in sibling_subtree:
            continue  # sibling depends on target → circular, skip
        for nid in sibling_subtree:
            context_set.add(nid)

    # Safety: target must never appear in its own context
    context_set.discard(target_id)

    if not context_set:
        return []

    # Return in topological order of the full tree
    root_id = get_root_node(tree)
    if root_id is None:
        return []
    topo = topological_order_full(tree, root_id)
    return [nid for nid in topo if nid in context_set]


def get_eligible_targets(tree: dict) -> list[str]:
    """
    All nodes that can serve as target (non-root, non-empty context).
    """
    root_id    = get_root_node(tree)
    parent_map = build_parent_map(tree)
    eligible   = []
    for nid in tree:
        if nid == root_id:
            continue
        if parent_map.get(nid) is None:
            continue  # disconnected / no parent
        context = build_context_for_target(tree, nid, parent_map)
        if context:
            eligible.append(nid)
    return eligible


def build_ns_prompt(problem: str, tree: dict) -> tuple[str, str] | None:
    """
    Pick a random eligible target and build (prompt, target_subquestion).

    Normal case: target has non-empty context → NS_CONTEXT_TEMPLATE.
    Fallback case: only one non-root node exists with empty context
                   (single leaf directly under root, no siblings) →
                   NS_SINGLE_TEMPLATE with no context list.

    Returns None if tree is too small or no eligible target exists.
    """
    if len(tree) < 2:
        return None

    eligible = get_eligible_targets(tree)
    if eligible:
        parent_map  = build_parent_map(tree)
        target_id   = random.choice(eligible)
        context_ids = build_context_for_target(tree, target_id, parent_map)

        context_questions = [tree[nid]["content"] for nid in context_ids]
        context = " ".join(q if q.endswith("?") else q + "?" for q in context_questions)
        target  = tree[target_id]["content"]
        prompt  = NS_CONTEXT_TEMPLATE.format(
            instruction=NS_INSTRUCTION,
            problem=problem,
            context=context,
        )
        return prompt, target

    # Fallback: find non-root nodes that have a parent but empty context
    # (leaf nodes that are the sole child of their parent)
    root_id    = get_root_node(tree)
    parent_map = build_parent_map(tree)
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
                        help="Input CSV (default: %(default)s)")
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
        raise FileNotFoundError(f"{in_path} not found")

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df):,} rows from {in_path}")

    n_rebuilt = 0
    n_empty   = 0
    new_targets = []
    new_prompts = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="rebuild"):
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

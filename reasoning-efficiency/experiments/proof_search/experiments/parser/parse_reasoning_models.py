import pandas as pd
import json
import re
import os
import operator
from typing import List, Dict, Tuple, Set
from collections import defaultdict, deque

ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.floordiv}

def extract_equations(text: str) -> List[Dict]:
    '''extracts equations from text'''
    equations = []
    
    # "25 + 2 = 27", "(25 + 2) = 27"
    pattern1 = r'[\(\s]*([\d\s+\-*/()]+?)\s*=\s*(\d+)'
    matches = re.findall(pattern1, text)
    
    for lhs_raw, rhs in matches:
        lhs_raw = lhs_raw.replace('(', '').replace(')', '').strip()
        tokens = re.findall(r'\d+|[+\-*/]', lhs_raw)
        
        if tokens:
            equations.append({
                'lhs_tokens': tokens,
                'rhs': int(rhs),
                'lhs_raw': lhs_raw
            })
    
    return equations


def eval_expression(tokens: List[str]) -> int:
    '''evaluates arithmetic expression left to right'''
    try:
        result = int(tokens[0])
        for i in range(1, len(tokens) - 1, 2):
            op = ops.get(tokens[i])
            if not op:
                return None
            val = int(tokens[i + 1])
            result = op(result, val)
        return result
    except:
        return None


def equations_match(eq1: Dict, eq2: Dict) -> bool:
    '''checks if two equations are equivalent'''
    if eq1['rhs'] != eq2['rhs']:
        return False
    
    eval1 = eval_expression(eq1['lhs_tokens'])
    eval2 = eval_expression(eq2['lhs_tokens'])
    
    return eval1 is not None and eval1 == eval2 and eval1 == eq1['rhs']


def extract_agents_entities(gt_step: Dict) -> Tuple[Set[str], Set[str]]:
    '''extracts agent and entity forms from gt step'''
    inst = gt_step.get('inst', {})
    
    agents = set()
    for agent in inst.get('agents', []):
        agents.add(agent.lower())
    
    entities = set()
    for entity_list in inst.get('entities', []):
        for entity in entity_list:
            entities.add(entity.lower())
    
    return agents, entities


def contains_required_terms(text: str, agents: Set[str], entities: Set[str]) -> bool:
    '''checks if text contains agent and entity'''
    text_lower = text.lower()
    
    has_agent = any(agent in text_lower for agent in agents) if agents else True
    has_entity = any(entity in text_lower for entity in entities) if entities else True
    
    return has_agent and has_entity


def extract_reasoning_steps(text: str) -> List[str]:
    '''extracts numbered reasoning steps'''
    # "1. step\n2. step"
    pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)'
    steps = re.findall(pattern, str(text), flags=re.DOTALL)
    return [step.strip() for step in steps]


def match_model_step_to_gt(model_step: str, gt_steps: List[Dict]) -> List[int]:
    '''matches model step to gt steps'''
    matched_indices = []
    
    model_equations = extract_equations(model_step)
    
    if not model_equations:
        return matched_indices
    
    for idx, gt_step in enumerate(gt_steps):
        if gt_step.get('type') != 'conclusion':
            continue
        
        gt_equations = extract_equations(gt_step['text'])
        
        if not gt_equations:
            continue
        
        gt_eq = gt_equations[0]
        
        equation_matched = any(equations_match(model_eq, gt_eq) for model_eq in model_equations)
        
        if not equation_matched:
            continue
        
        agents, entities = extract_agents_entities(gt_step)
        
        if contains_required_terms(model_step, agents, entities):
            matched_indices.append(idx + 1)
    
    return matched_indices


def match_all_reasoning_steps(model_steps: List[str], gt_steps: List[Dict]) -> Tuple[List[List[int]], List[int]]:
    '''matches all model steps to gt steps'''
    all_matches = []
    seen = set()
    unique_matches = []
    
    for model_step in model_steps:
        matches = match_model_step_to_gt(model_step, gt_steps)
        all_matches.append(matches)
        
        for match_idx in matches:
            if match_idx not in seen:
                seen.add(match_idx)
                unique_matches.append(match_idx)
    
    return all_matches, unique_matches


def get_ground_truth_steps(gt_data: List[Dict], index: int, filename: str) -> List[Dict]:
    '''extracts rt steps from gt json'''
    if filename.endswith("_nonground.csv"):
        fname_core = filename[:-len("_nonground.csv")]
    elif filename.endswith("_ground.csv"):
        fname_core = filename[:-len("_ground.csv")]
    else:
        raise ValueError(f"Unsupported suffix: {filename}")
    
    parts = fname_core.split("_")
    if len(parts) < 2:
        raise ValueError(f"Filename too short: {filename}")
    
    source = parts[0]
    
    if source == 'base':
        try:
            rt_list = gt_data[index][source]['rt']
            assert rt_list
            return rt_list
        except (KeyError, AssertionError):
            raise KeyError(f"Missing base rt: {filename}")
    
    for i in range(2, len(parts)):
        complexity = "_".join(parts[1:i])
        key = "_".join(parts[i:])
        try:
            rt_list = gt_data[index][source][complexity][key]['rt']
            assert rt_list
            return rt_list
        except (KeyError, AssertionError):
            continue
    
    raise KeyError(f"No valid rt: {filename}")


def compute_search_orders(gt_steps: List[Dict]) -> Tuple[List[int], List[int], List[int]]:
    '''computes dfs, bfs, efficient search orders'''
    from collections import defaultdict, deque
    
    premise_to_conclusion = {}
    axioms = set()
    efficient_order = []
    
    for i, step in enumerate(gt_steps):
        if step.get('relevant'):
            efficient_order.append(i + 1)
        
        if step.get('premise_sent_indices'):
            premise_to_conclusion[frozenset(step['premise_sent_indices'])] = i
        else:
            axioms.add(i)
    
    dfs_order = list(range(1, len(gt_steps) + 1))
    
    bfs_order = bfs_hypergraph(premise_to_conclusion, axioms)
    
    return dfs_order, bfs_order, efficient_order


def bfs_hypergraph(edges: Dict, leaves: Set) -> List[int]:
    '''bfs traversal on hypergraph'''
    edge_wait_count = {}
    node_to_edges = defaultdict(list)
    inputset_to_output = {}
    
    for input_set, output_node in edges.items():
        edge_wait_count[input_set] = len(input_set)
        inputset_to_output[input_set] = output_node
        for node in input_set:
            node_to_edges[node].append(input_set)
    
    queue = deque(sorted(leaves))
    visited_nodes = set()
    visited_edges = set()
    traversal_order = []
    
    while queue:
        node = queue.popleft()
        if node in visited_nodes:
            continue
        visited_nodes.add(node)
        traversal_order.append(node)
        
        for input_set in sorted(node_to_edges[node], key=lambda s: sorted(s)):
            edge_wait_count[input_set] -= 1
            if edge_wait_count[input_set] == 0 and input_set not in visited_edges:
                visited_edges.add(input_set)
                output_node = inputset_to_output[input_set]
                queue.append(output_node)
    
    return [node + 1 for node in traversal_order]


def bullet_point_steps(steps: List[str]) -> str:
    '''formats steps as numbered bullet points'''
    return "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])


def process_all_experiments(csv_dir: str, ground_truth_path: str, output_dir: str):
    '''main pipeline to process all experiment csvs'''
    os.makedirs(output_dir, exist_ok=True)
    
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)
    
    for fname in sorted(os.listdir(csv_dir)):
        if not fname.endswith(".csv"):
            continue
        
        path = os.path.join(csv_dir, fname)
        print(f"Processing {fname}...")
        
        df = pd.read_csv(path)
        df["model_reasoning_steps"] = df["model_cot"].apply(extract_reasoning_steps)
        
        results = {
            'gt_reasoning_steps': [],
            'matched_indices': [],
            'step_matches': [],
            'num_matches': [],
            'num_steps': [],
            'match_ratio': [],
            'gt_cot_enumerated': [],
            'dfs_search_order': [],
            'bfs_search_order': [],
            'efficient_search_order': []
        }
        
        for idx, row in df.iterrows():
            try:
                gt_steps = get_ground_truth_steps(ground_truth, idx, fname)
                gt_text = [step['text'] for step in gt_steps]
            except KeyError:
                gt_steps = []
                gt_text = []
            
            all_matches, unique_matches = match_all_reasoning_steps(
                row['model_reasoning_steps'], gt_steps
            )
            
            step_matches = [len(matches) > 0 for matches in all_matches]
            
            num_matches = len(unique_matches)
            num_steps = len(row['model_reasoning_steps'])
            match_ratio = num_matches / num_steps if num_steps > 0 else 0
            
            if gt_steps:
                dfs, bfs, efficient = compute_search_orders(gt_steps)
            else:
                dfs, bfs, efficient = [], [], []
            
            results['gt_reasoning_steps'].append(gt_text)
            results['matched_indices'].append(all_matches)
            results['step_matches'].append(step_matches)
            results['num_matches'].append(num_matches)
            results['num_steps'].append(num_steps)
            results['match_ratio'].append(match_ratio)
            results['gt_cot_enumerated'].append(bullet_point_steps(gt_text))
            results['dfs_search_order'].append(dfs)
            results['bfs_search_order'].append(bfs)
            results['efficient_search_order'].append(efficient)
        
        for key, values in results.items():
            df[key] = values
        
        base_cols = [col for col in df.columns if col not in {
            'model_reasoning_steps', 'gt_reasoning_steps', 'model_cot', 
            'gt_cot_enumerated', 'matched_indices', 'dfs_search_order',
            'bfs_search_order', 'efficient_search_order', 'step_matches',
            'num_matches', 'num_steps', 'match_ratio'
        }]
        
        eval_cols = [
            'model_reasoning_steps',
            'gt_reasoning_steps',
            'model_cot',
            'gt_cot_enumerated',
            'matched_indices',
            'dfs_search_order',
            'bfs_search_order',
            'efficient_search_order',
            'step_matches',
            'num_matches',
            'num_steps',
            'match_ratio'
        ]
        
        df = df[base_cols + eval_cols]
        
        output_path = os.path.join(output_dir, fname)
        df.to_csv(output_path, index=False)


# csv_dir = "experiments/proof_search/experiments/results/qwq-32b"
# ground_truth_path = "experiments/proof_search/experiments/data/test500.json"
# output_dir = "qwq-32b_matched_outputs_search_order_v3"

if __name__ == "__main__":
    process_all_experiments(
        csv_dir=csv_dir,
        ground_truth_path=ground_truth_path,
        output_dir=output_dir
    )
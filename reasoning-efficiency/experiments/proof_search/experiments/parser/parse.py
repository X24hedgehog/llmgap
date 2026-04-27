import pandas as pd
import re
import json
import operator
import os
import ast
import glob
import argparse
from itertools import permutations
from collections import defaultdict, deque

def extract_reasoning_steps(text):
    pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)'
    steps = re.findall(pattern, text, flags=re.DOTALL)
    return [step.strip() for step in steps]

def normalize(text):
    text is text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def extract_equation(text):
    match = re.search(r'([\d\s+\-*/]+)=\s*(\d+)', text)
    if not match:
        return None
    lhs_raw, rhs = match.groups()
    rhs = int(rhs)
    tokens = re.findall(r'\d+|[+\-*/]', lhs_raw)
    return {'lhs': tokens, 'rhs': rhs}

def extract_equations(text):
    matches = re.findall(r'([\d\s+\-*/]+)=\s*(\d+)', text)
    equations = []
    for lhs_raw, rhs in matches:
        rhs = int(rhs)
        tokens = re.findall(r'\d+|[+\-*/]', lhs_raw)
        equations.append({'lhs': tokens, 'rhs': rhs})
    return equations if equations else None

ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.floordiv}

def eval_expression(tokens):
    try:
        result = int(tokens[0])
        for i in range(1, len(tokens)-1, 2):
            op = ops.get(tokens[i])
            val = int(tokens[i+1])
            result = op(result, val)
        return result
    except:
        return None

def extract_rate(text):
    text = text.lower()
    name_pattern = r'([\w’\'\-]+(?:\s+[\w’\'\-]+)*)'
    pattern = fr'{name_pattern}\s+possesses\s+(\d+)\s+([\w\s]+?)\s+per\s+([\w\s]+)\.?'

    match = re.search(pattern, text)
    if match:
        agent, quantity, sub_entity, super_entity = match.groups()
        return agent.strip(), quantity, sub_entity.strip(), super_entity.strip()
    return None, None, None, None

def extract_comparison(text):
    text = text.lower()
    name_pattern = r'([\w’\'\-]+(?:\s+[\w’\'\-]+)*)'
    pattern1 = fr'{name_pattern}\s+has\s+(\d+)\s+(?:[\w\s]+?)\s+(more|less|fewer)\s+than\s+{name_pattern}'
    pattern2 = fr'{name_pattern}\s+has\s+(\d+)\s+(more|less|fewer)\s+(?:[\w\s]+?)\s+than\s+{name_pattern}'

    for pattern in [pattern1, pattern2]:
        match = re.search(pattern, text)
        if match:
            subj, quantity, comp_word, obj = match.groups()
            comp_word = comp_word.strip()

            if comp_word == "more":
                return subj.strip(), obj.strip(), "more", quantity
            elif comp_word in {"less", "fewer"}:
                return obj.strip(), subj.strip(), "more", quantity

    return None, None, None, None

def extract_transfer_roles(text):
    text = text.lower()
    name_pattern = r'([\w’\'\-]+(?:\s+[\w’\'\-]+)*)'

    patterns = [
        ("direct", fr'{name_pattern}\s+gave\s+{name_pattern}\s+(\d+)'),
        ("direct", fr'{name_pattern}\s+gave\s+(\d+)[\w\s]*?\s+to\s+{name_pattern}'),
        ("inverse", fr'{name_pattern}\s+got\s+(\d+)[\w\s]*?\s+from\s+{name_pattern}'),
        ("inverse", fr'{name_pattern}\s+then\s+gets\s+(\d+)[\w\s]*?\s+from\s+{name_pattern}'),
        ("inverse", fr'{name_pattern}\s+then\s+receives\s+(\d+)[\w\s]*?\s+from\s+{name_pattern}'),
        ("inverse", fr'{name_pattern}\s+then\s+gets\s+another\s+(\d+)[\w\s]*?\s+from\s+{name_pattern}'),
        ("direct", fr'{name_pattern}\s+then\s+gives\s+(\d+)[\w\s]*?\s+more\s+to\s+{name_pattern}'),
        ("direct", fr'{name_pattern}\s+then\s+gives\s+{name_pattern}\s+(\d+)'),
        ("direct", fr'{name_pattern}\s+then\s+donates\s+(\d+)[\w\s]*?\s+to\s+{name_pattern}'),
        ("sender_only", fr'{name_pattern}\s+then\s+loses\s+(\d+)[\w\s]*?'),
        ("receiver_only", fr'{name_pattern}\s+then\s+gets\s+(\d+)[\w\s]*?')
    ]

    for direction, pattern in patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            if direction == "direct":
                return groups[0].strip(), groups[1].strip(), groups[2]
            elif direction == "inverse":
                return groups[2].strip(), groups[0].strip(), groups[1]
            elif direction == "sender_only":
                return groups[0].strip(), None, groups[1]
            elif direction == "receiver_only":
                return None, groups[0].strip(), groups[1]

    return None, None, None

def match_comparison_step(model_step, gt_step, lf_type="conclusion"):
    if lf_type == "conclusion":
        eq_model = extract_equation(model_step)
        eq_gt = extract_equation(gt_step['text'])

        if eq_gt and (not eq_model or eq_model['rhs'] != eq_gt['rhs'] or eval_expression(eq_model['lhs']) != eval_expression(eq_gt['lhs'])):
            return False

    subj_gt, obj_gt, comp_gt, q_gt = extract_comparison(gt_step['text'])
    subj_model, obj_model, comp_model, q_model = extract_comparison(model_step)

    if not all([subj_gt, obj_gt, comp_gt, subj_model, obj_model, comp_model]):
        return False

    roles_ok = (subj_model == subj_gt and obj_model == obj_gt)
    comp_type_ok = (comp_model == comp_gt)

    if lf_type != 'conclusion':
        quantity_ok = (q_model == q_gt)
    else:
        quantity_ok = True

    entity_forms = [e for ent in gt_step['inst'].get('entities', []) for e in ent]
    entity_ok = any(e in model_step for e in entity_forms)

    return roles_ok and comp_type_ok and quantity_ok and entity_ok

def match_transfer_step(model_step, gt_step, lf_type="conclusion"):
    if lf_type == "conclusion":
        eq_model = extract_equation(model_step)
        eq_gt = extract_equation(gt_step['text'])

        if eq_gt and (not eq_model or eq_model['rhs'] != eq_gt['rhs'] or eval_expression(eq_model['lhs']) != eval_expression(eq_gt['lhs'])):
            return False
    
    sender_gt, receiver_gt, q_gt = extract_transfer_roles(gt_step['text'])
    sender_model, receiver_model, q_model = extract_transfer_roles(model_step)

    if (sender_gt and not sender_model) or (receiver_gt and not receiver_model) or (q_gt and not q_model):
        return False

    if sender_gt and receiver_gt:
        roles_ok = (sender_model == sender_gt and receiver_model == receiver_gt)
    elif sender_gt and not receiver_gt:
        roles_ok = (sender_model == sender_gt)
    elif receiver_gt and not sender_gt:
        roles_ok = (receiver_model == receiver_gt)
    else:
        roles_ok = False

    if lf_type != 'conclusion':
        quantity_ok = (q_model == q_gt)
    else:
        quantity_ok = True

    entity_forms = [e for ent in gt_step['inst'].get('entities', []) for e in ent]
    entity_ok = any(e in model_step for e in entity_forms)

    return roles_ok and quantity_ok and entity_ok

def match_container_step(model_step, gt_step, lf_type="conclusion"):
    model_step_lower = model_step.lower()
    gt_text = gt_step['text'].lower()
    inst = gt_step.get("inst", {})

    agents = inst.get("agents", [])
    entities = [e for ent in inst.get("entities", []) for e in ent]
    quantities = inst.get("quantities", [])

    agent_ok = all(agent.lower() in model_step_lower for agent in agents)
    entity_ok = any(ent.lower() in model_step_lower for ent in entities)
    quantity_ok = any(q in model_step for q in quantities)

    if not (agent_ok and entity_ok and quantity_ok):
        return False

    eq_model = extract_equation(model_step)
    eq_gt = extract_equation(gt_text)
    gt_rhs = eq_gt['rhs'] if eq_gt else None
    if lf_type == "conclusion" and eq_model:
        if eq_gt and (
            not eq_model or
            eq_model['rhs'] != eq_gt['rhs'] or
            eval_expression(eq_model['lhs']) != eval_expression(eq_gt['lhs'])
        ):
            return False
        return True
    
    elif lf_type == "conclusion" and not eq_model:
        verbs = ["has", "owns", "possesses"]
        verb_pattern = r'(?:' + '|'.join(re.escape(v) for v in verbs) + r')'

        pattern = (
            r'\b' + re.escape(agents[0].lower()) + r'\b\s+' +
            verb_pattern + r'\s+' + re.escape(str(gt_rhs)) + r'\s+(?:[\w\s]*?)' +
            r'\b(?:' + '|'.join(re.escape(e.lower()) for e in entities) + r')\b'
        )

        return bool(re.search(pattern, model_step_lower))

    else:
        verbs = ["has", "owns", "possesses"]
        verb_pattern = r'(?:' + '|'.join(re.escape(v) for v in verbs) + r')'

        pattern = (
            r'\b' + re.escape(agents[0].lower()) + r'\b\s+' +
            verb_pattern + r'\s+' + re.escape(quantities[0].lower()) + r'\s+(?:[\w\s]*?)' +
            r'\b(?:' + '|'.join(re.escape(e.lower()) for e in entities) + r')\b'
        )

        return bool(re.search(pattern, model_step_lower))
    
def match_rate_step(model_step, gt_step, lf_type="statement"):
    model_step_lower = model_step.lower()
    gt_text = gt_step['text'].lower()
    inst = gt_step.get('inst', {})

    if lf_type == "conclusion":
        eq_model_list = extract_equation(model_step_lower)
        eq_gt_list = extract_equation(gt_text)

        if not eq_gt_list or not eq_model_list:
            return False

        eq_model = eq_model_list[0]
        eq_gt = eq_gt_list[0]

        if eq_model['rhs'] != eq_gt['rhs']:
            return False
        if eval_expression(eq_model['lhs']) != eval_expression(eq_gt['lhs']):
            return False

    agent_model, q_model, sub_ent_model, super_ent_model = extract_rate(model_step)
    agent_gt, q_gt, sub_ent_gt, super_ent_gt = extract_rate(gt_text)

    if not all([agent_model, q_model, sub_ent_model, super_ent_model,
                agent_gt, q_gt, sub_ent_gt, super_ent_gt]):
        return False

    agent_ok = (agent_model == agent_gt)
    quantity_ok = True if lf_type == "conclusion" else (q_model == q_gt)

    entity_forms = [e for ent in inst.get('entities', []) for e in ent]
    sub_entity_ok = any(e in sub_ent_model for e in entity_forms)
    super_entity_ok = any(e in super_ent_model for e in entity_forms)

    return agent_ok and quantity_ok and sub_entity_ok and super_entity_ok

def match_partwhole_step(model_step, gt_step, lf_type="conclusion"):
    if lf_type == "conclusion":
        eq_model = extract_equations(model_step)
        eq_gt = extract_equation(gt_step['text'])

        if eq_gt and not eq_model:
            return False
        elif eq_gt:
            eq_match = False
            for eq in eq_model:
                if eq['rhs'] == eq_gt['rhs'] and eval_expression(eq['lhs']) == eval_expression(eq_gt['lhs']):
                    eq_match = True
            if not eq_match:
                return False

    model_step_lower = model_step.lower()
    inst = gt_step.get("inst", {})

    entity_forms = [e for ent in inst.get("entities", []) for e in ent]
    entity_ok = any(e in model_step_lower for e in entity_forms)

    quantities = inst.get("quantities", [])
    quantity_ok = all(q in model_step for q in quantities)

    aggregation_keywords = ["together", "sum", "combined", "total", "altogether"]
    aggregation_ok = any(word in model_step_lower for word in aggregation_keywords)

    return entity_ok and quantity_ok and aggregation_ok

def match_compeq_step(model_step, gt_step, lf_type="statement"):
    model_step_lower = model_step.lower()
    inst = gt_step.get("inst", {})
    agents = [a.lower() for a in inst.get("agents", [])]
    quantities = inst.get("quantities", [])
    entity_forms = [e.lower() for ent in inst.get("entities", []) for e in ent]

    if len(agents) < 4:
        return False

    subj_agent, obj_agent, other_subj_agent, other_obj_agent = agents[:4]

    entity_ok = any(e in model_step_lower for e in entity_forms)
    quantity_ok = any(q in model_step for q in quantities)
    linking_ok = any(phrase in model_step_lower for phrase in ["is the same as", "equal to", "equals"])

    comp1_ok = (
        subj_agent in model_step_lower and
        obj_agent in model_step_lower and
        ("more than" in model_step_lower or "difference" in model_step_lower)
    )

    comp2_ok = (
        other_subj_agent in model_step_lower and
        other_obj_agent in model_step_lower and
        ("compared to" in model_step_lower or "difference" in model_step_lower)
    )

    return all([entity_ok, quantity_ok, linking_ok, comp1_ok, comp2_ok])

def match(step, gt_step_text, gt_step):
    lf = gt_step.get('lf')
    lf_type = gt_step.get('type')
    if normalize(gt_step_text) in normalize(step):
        return True
    elif lf == 'PartWhole':
        return match_partwhole_step(step, gt_step, lf_type)
    elif lf == 'Compeq':
        return match_compeq_step(step, gt_step, lf_type)
    elif lf == 'Comp':
        return match_comparison_step(step, gt_step, lf_type)
    elif lf == 'Rate':
        return match_rate_step(step, gt_step, lf_type)
    elif lf == 'Transfer':
        return match_transfer_step(step, gt_step, lf_type)
    elif lf == 'Container':
        return match_container_step(step, gt_step, lf_type)
    return False

def match_reasoning_steps(row, gt_steps):
    matched_flags = []
    matched_ids = []
    for step in row['model_reasoning_steps']:
        model_step = step.strip()
        matched = False
        matched_idx = -1
        for idx, gt_step in enumerate(gt_steps):
            gt_step_text = gt_step['text']
            if match(model_step, gt_step_text, gt_step):
                matched = True
                matched_idx = idx + 1
        matched_flags.append(matched)
        matched_ids.append(matched_idx)
    return matched_flags, matched_ids

def get_ground_truth_steps(gt_data, index, filename):
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

def bullet_point_steps(steps):
    return "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])

def dfs_hypergraph_ordered(edges: dict, leaves: set):
    node_to_outputs = defaultdict(list)
    for input_set, output_node in edges.items():
        for node in input_set:
            node_to_outputs[node].append((input_set, output_node))

    for lst in node_to_outputs.values():
        lst.sort(key=lambda x: (sorted(x[0]), x[1]))

    visited_nodes = set()
    visited_edges = set()
    traversal_order = []

    def dfs(node):
        if node in visited_nodes:
            return
        visited_nodes.add(node)
        traversal_order.append(node)

        for input_set, output_node in node_to_outputs.get(node, []):
            if input_set.issubset(visited_nodes):
                if (input_set, output_node) not in visited_edges:
                    visited_edges.add((input_set, output_node))
                    dfs(output_node)

    for leaf in sorted(leaves):
        dfs(leaf)

    return [node + 1 for node in traversal_order]

def bfs_hypergraph_ordered(edges: dict, leaves: set):
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

def process_all_experiments(csv_dir, ground_truth_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue

        path = os.path.join(csv_dir, fname)
        print(f"Processing {fname}...")

        df = pd.read_csv(path)
        df["model_reasoning_steps"] = df["model_cot"].apply(extract_reasoning_steps)

        matched_flags_list = []
        matched_ids_list = []
        num_matches = []
        match_ratios = []
        gt_reasoning = []

        for idx, row in df.iterrows():
            try:
                gt_steps = get_ground_truth_steps(ground_truth, idx, fname)
                gt_reasoning.append([step['text'] for step in gt_steps])
            except KeyError:
                gt_steps = []
                gt_reasoning.append([])

            flags, indices = match_reasoning_steps(row, gt_steps)
            matched_flags_list.append(flags)
            matched_ids_list.append(indices)
            num_matches.append(sum(flags))
            match_ratios.append(sum(flags) / len(row['model_reasoning_steps']) if row['model_reasoning_steps'] else 0)

        df['gt_reasoning_steps'] = gt_reasoning
        df['step_matches'] = matched_flags_list
        df['matched_indices'] = matched_ids_list
        df['num_matches'] = num_matches
        df['num_steps'] = df['model_reasoning_steps'].apply(len)
        df['match_ratio'] = match_ratios
        df['gt_cot_enumerated'] = df['gt_reasoning_steps'].apply(bullet_point_steps)

        base_cols = [col for col in df.columns if col not in {'gt_reasoning_steps', 'step_matches', 'matched_indices', 'num_matches', 'num_steps', 'match_ratio', 'model_reasoning_steps', 'model_cot', 'gt_cot_enumerated'}]
        eval_cols = ['model_reasoning_steps', 'gt_reasoning_steps', 'model_cot', 'gt_cot_enumerated', 'matched_indices', 'step_matches' , 'num_matches', 'num_steps', 'match_ratio']
        df = df[base_cols + eval_cols]

        output_path = os.path.join(output_dir, fname)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

def process_all_experiments_with_search(csv_dir, ground_truth_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue

        path = os.path.join(csv_dir, fname)
        print(f"Processing {fname}...")

        df = pd.read_csv(path)
        df["model_reasoning_steps"] = df["model_cot"].apply(extract_reasoning_steps)

        matched_flags_list = []
        matched_ids_list = []
        num_matches = []
        match_ratios = []
        gt_reasoning = []
        dfs_search_orders = []
        bfs_search_orders = []
        efficient_search_orders = []

        for idx, row in df.iterrows():
            try:
                gt_steps = get_ground_truth_steps(ground_truth, idx, fname)
                gt_reasoning.append([step['text'] for step in gt_steps])
            except KeyError:
                gt_steps = []
                gt_reasoning.append([])

            flags, indices = match_reasoning_steps(row, gt_steps)
            matched_flags_list.append(flags)
            matched_ids_list.append(indices)
            num_matches.append(sum(flags))
            match_ratios.append(sum(flags) / len(row['model_reasoning_steps']) if row['model_reasoning_steps'] else 0)

            premise_to_conclusion = {}
            axioms = set()
            efficient_search_order = []
            for i in range(len(gt_steps)):
                    if gt_steps[i]['relevant']:
                         efficient_search_order.append(i+1)
                    if gt_steps[i]['premise_sent_indices']:
                            premise_to_conclusion[frozenset(gt_steps[i]['premise_sent_indices'])] = i
                    else:
                            axioms.add(i)

            dfs_order = [i+1 for i in list(range(len(gt_steps)))]
            bfs_order = bfs_hypergraph_ordered(premise_to_conclusion, axioms)
            dfs_search_orders.append(dfs_order)
            bfs_search_orders.append(bfs_order)
            efficient_search_orders.append(efficient_search_order)

        df['gt_reasoning_steps'] = gt_reasoning
        df['step_matches'] = matched_flags_list
        df['matched_indices'] = matched_ids_list
        df['num_matches'] = num_matches
        df['num_steps'] = df['model_reasoning_steps'].apply(len)
        df['match_ratio'] = match_ratios
        df['gt_cot_enumerated'] = df['gt_reasoning_steps'].apply(bullet_point_steps)
        df['dfs_search_order'] = dfs_search_orders
        df['bfs_search_order'] = bfs_search_orders
        df['efficient_search_order'] = efficient_search_orders

        base_cols = [col for col in df.columns if col not in {'gt_reasoning_steps', 'step_matches', 'matched_indices', 'num_matches', 'num_steps', 'match_ratio', 'model_reasoning_steps', 'model_cot', 'gt_cot_enumerated', 'dfs_search_order', 'bfs_search_order', 'efficient_search_order'}]
        eval_cols = ['model_reasoning_steps', 'gt_reasoning_steps', 'model_cot', 'gt_cot_enumerated', 'matched_indices', 'dfs_search_order', 'bfs_search_order', 'efficient_search_order', 'step_matches' , 'num_matches', 'num_steps', 'match_ratio']
        df = df[base_cols + eval_cols]

        output_path = os.path.join(output_dir, fname)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")







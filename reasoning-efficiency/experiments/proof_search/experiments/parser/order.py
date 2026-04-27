import json
from collections import defaultdict, deque

def get_gt(gt_data, index, fname):
    '''retrieves ground truth steps for problem'''
    if fname.endswith("_nonground.csv"):
        fname_core = fname[:-len("_nonground.csv")]
    elif fname.endswith("_ground.csv"):
        fname_core = fname[:-len("_ground.csv")]
    else:
        raise ValueError(f"Unsupported suffix: {fname}")

    parts = fname_core.split("_")
    if len(parts) < 2:
        raise ValueError(f"Filename too short: {fname}")

    source = parts[0]

    if source == 'base':
        try:
            rt_list = gt_data[index][source]['rt']
            assert rt_list
            return rt_list
        except (KeyError, AssertionError):
            raise KeyError(f"Missing base rt: {fname}")

    for i in range(2, len(parts)):
        complexity = "_".join(parts[1:i])
        key = "_".join(parts[i:])
        try:
            rt_list = gt_data[index][source][complexity][key]['rt']
            assert rt_list
            return rt_list
        except (KeyError, AssertionError):
            continue

    raise KeyError(f"No valid rt: {fname}")


def dfs_hypergraph_ordered(edges: dict, leaves: set):
    '''dfs traversal on hypergraph'''
    from collections import defaultdict

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
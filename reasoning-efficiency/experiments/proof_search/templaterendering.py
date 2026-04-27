from typing import Generator, List, Tuple
import random
from mathgap.instantiate.instantiation import Instantiation
from mathgap.logicalforms.logicalform import LogicalForm
from mathgap.natlang.templates.metadata import RenderingMetadata
from mathgap.natlang.templates.sampling import ProblemStructureSampler, ReasoningTraceSampler, TemplateSampler, TemplateSelection
from mathgap.natlang.templates.template import NEW_LINE, WHITESPACE, Template, TemplatePart, TemplateType, TextPart
from mathgap.natlang.templates.templaterenderer import TemplateRenderer
from mathgap.problemsample import ProblemOrder
from mathgap.trees.prooftree import ProofTree, TraversalOrder, TreeNode


def interleave_lists(lists: List[List], seed: int = 14) -> List:
    """ Interleaves multiple lists while preserving the order of elements inside each list """
    random.seed(seed)

    choices = [lst_id for lst_id,lst in enumerate(lists) for i in range(len(lst))]
    random.shuffle(choices)
    
    active_iterators = [iter(lst) for lst in lists]
    result = []
    for choice in choices:
        result.append(next(active_iterators[choice]))
    return result

def interleaved_template_selections(base_mwp: Tuple[ProofTree, Instantiation, ProblemOrder, ProblemStructureSampler],
                                    irrelevant_mwps: List[Tuple[ProofTree, Instantiation, ProblemOrder, ProblemStructureSampler]],
                                    connected_nodes: List[Tuple[Tuple[ProofTree,TreeNode],Tuple[ProofTree,TreeNode]]] = [],
                                    seed: int = 14) -> List[Tuple[ProofTree, Instantiation, TemplateSelection]]:
    """ Selects templates for nodes of MWPs in an interleaved manner """
    # 1. generate all template-selections independently
    base_tree, base_instantiation, base_order, base_ps_sampler = base_mwp
    base_ts = base_ps_sampler.sample(tree=base_tree, problem=base_order, seed=seed)
    base_ts_body = [
        (base_tree, base_instantiation, ts, base_order) 
        for ts in base_ts
        if ts.primary_node_id in base_order.body_node_ids
    ]
    base_ts_question = [
        (base_tree, base_instantiation, ts, base_order) 
        for ts in base_ts
        if ts.primary_node_id in base_order.question_node_ids
    ]
        
    # 2. start with base and interleave irrelevant template selections one after the other
    def _filter_rendered(tss):
        return [(tree, instantiation, ts) for (tree, instantiation, ts, order) in tss if order.is_rendered(ts.primary_node_id)]
    tss_body = _filter_rendered(base_ts_body)
    tss_question = _filter_rendered(base_ts_question)

    _seed = seed
    for tree, instantiation, order, ps_sampler in irrelevant_mwps:
        irrelevant_ts = ps_sampler.sample(tree=tree, problem=order, seed=seed)

        irrelevant_ts_body = _filter_rendered([(tree, instantiation, ts, order) for ts in irrelevant_ts if ts.primary_node_id in order.body_node_ids])
        irrelevant_ts_question = _filter_rendered([(tree, instantiation, ts, order) for ts in irrelevant_ts if ts.primary_node_id in order.question_node_ids])

        relevant_connections = [
            ((base_tree,base_node),(irrelevant_tree,irrelevant_node))
            for (base_tree,base_node),(irrelevant_tree,irrelevant_node) in connected_nodes
            if tree == irrelevant_tree
        ]
        assert len(relevant_connections) <= 1, "Multiple connections for the same irrelevant tree not supported"

        if len(relevant_connections) == 0:
            # no barrier, interleave freely
            tss_body = interleave_lists([tss_body, irrelevant_ts_body], _seed)
        else:
            # connection introduces barrier (i.e., only after the connected node of base can be inferred, can it be considered rendered)
            (base_tree,base_node),(irrelevant_tree,irrelevant_node) = relevant_connections[0]
            required_base_leaf_nids = set([base_tree.id_by_node[l] for l in base_node.get_leaves()])

            irrelevant_split_index = 0
            for i,(tree,instantiation,ts) in enumerate(irrelevant_ts_body):
                if ts.primary_node_id == irrelevant_tree.id_by_node[irrelevant_node]:
                    irrelevant_split_index = i
                    break
            
            irrelevant_body_pre = irrelevant_ts_body[:irrelevant_split_index] # up to connecting node
            irrelevant_body_rest = irrelevant_ts_body[irrelevant_split_index:] # starting from connected node

            # find the point in the existing templateselections, where all the required base nodes have been rendered for the connected irrelevant node
            available_base_leaf_nids = set([])
            existing_split_index = len(tss_body)
            for i,(tree,instantiation,ts) in enumerate(tss_body):
                if tree == base_tree:                
                    available_base_leaf_nids.add(ts.primary_node_id)
                    
                    if required_base_leaf_nids.issubset(available_base_leaf_nids):
                        existing_split_index = i+1 # i should still be in pre
                        break

            assert required_base_leaf_nids.issubset(available_base_leaf_nids), "Requires all base leaf nodes required to deduce the connecting node to be rendered before considering the irrelevant connected node to be rendered"
            tss_body_pre = tss_body[:existing_split_index]
            tss_body_rest = tss_body[existing_split_index:]
            tss_body = interleave_lists([tss_body_pre, irrelevant_body_pre], _seed) + interleave_lists([tss_body_rest, irrelevant_body_rest], _seed+1)

        # NOTE: we do not support barriers on questions
        tss_question = interleave_lists([tss_question, irrelevant_ts_question], _seed+2)
        
        _seed += 3

    all_ts = tss_body + tss_question # asking questions at the end
    return all_ts

def render_interleaved(template_renderer: TemplateRenderer, template_selections: List[Tuple[ProofTree, Instantiation, TemplateSelection]], append: List[TemplatePart] = []) -> Tuple[str, RenderingMetadata]:
    """ 
        Renders a selection of templates from different (tree, instantiation)s into a natural language problem formulation.

        Parameters:
        - template_renderer: renderer for a single template
        - template_selections: specifies which tree, instantiation, tree-nodes (by index) should be rendered in which order
        - append: appended to the end of each selected template (e.g. separator of nodes/sentences)
    """
    all_text = ""
    all_metadata = RenderingMetadata(template_selections=template_selections)
    for i,(tree,instantiation,selection) in enumerate(template_selections):
        is_last_selection = (len(template_selections) - 1 == i)
        assert len(selection.selection) == 1, "Only supporting rendering of a single template per node in problem-structure rendering."

        node_id, template = selection.selection[0]
        appendix = append if is_last_selection else [WHITESPACE]
        txt, metadata = template_renderer.render(tree.node_by_id[node_id].logicalform, instantiation, template, append=appendix, parent_unit=[i])
        metadata.node_ids.append((node_id, len(txt)))

        all_text += txt
        all_metadata += metadata

    return all_text, all_metadata

def interleaved_template_selection_rt(trees: List[ProofTree], instantiations: List[Instantiation], order: List[Tuple[ProofTree, int]], 
           template_sampler: TemplateSampler,
           identical_nodes: List[Tuple[Tuple[ProofTree,TreeNode], Tuple[ProofTree,TreeNode]]] = [],
           preselected_templates: List[TemplateSelection] = None, 
           enforce_premise_axiom_consistency: bool = True,
           enforce_same_axiom_order: bool = True,
           seed: int = 14) -> List[Tuple[ProofTree, Instantiation, TemplateSelection]]:
        """ 
            Picks a template to express each conclusion and all premises in natural language. 

            - trees: the underlying prooftrees
            - order: the underlying problem(-order)
            - template_sampler: sampler to use
            - identical_nodes: pairs of tree nodes that should be considered equivalent (i.e. unlocking one also unlocks the other)
            - preselected_templates: for any conclusion-node or premise-node for which a template has been preselected, said template will be used 
                (only for non-preselected nodes, new templates will be selected)
            - enforce_premise_axiom_consistency: if true, then will try to enforce consistency between rendering of axioms as premises or standalone axioms in the problem formulation
            - enforce_same_axiom_order: if true, will render the axioms in the same order as they are given in the problem text
            - seed
        """
        assert all(nb[1].is_leaf for na,nb in identical_nodes), "Convention to always pass the irrelevant axiom node as the second in the tuple of identical nodes"

        preselected_templates_by_primary_node_id = {} if preselected_templates is None else {(str(tree), str(inst), s.primary_node_id): {i:t for i,t in s.selection} for tree,inst,s in preselected_templates}
        template_selections = []

        for tree,node in traverse_reasoning_trace_multitree(trees, order, identical_nodes):
            if node.is_leaf and (tree, tree.id_by_node[node]) not in order: continue # skip axioms the original order did not intend to render

            instantiation = instantiations[trees.index(tree)]
            selection: List[Tuple[int, Template]] = []
            if node.is_leaf and not enforce_same_axiom_order: continue # no need to provide reasoning for axioms
            
            node_id = tree.id_by_node[node]
            lf = node.logicalform
            
            # leaf nodes
            if node.is_leaf and enforce_same_axiom_order:
                if enforce_premise_axiom_consistency and ((str(tree), str(instantiation), node_id) in preselected_templates_by_primary_node_id):
                    # enforce consistency between premise and axiom
                    template = preselected_templates_by_primary_node_id[(str(tree), str(instantiation), node_id)][node_id]
                else:
                    # sample new template
                    template = template_sampler.choose_template(lf, TemplateType.STATEMENT, tree, seed)
                assert template.template_type == TemplateType.STATEMENT, f"Template should be a statement and not {template.template_type.name}. Template={template}. Tree={tree}"
                assert template.condition.is_satisified(lf=lf, tree=tree), "Template should still be valid!"
                selection.append((node_id, template))
                template_selections.append((tree, instantiation, TemplateSelection(node_id, selection)))
                seed += 1

                continue

            # premises
            if not enforce_same_axiom_order:
                for premise_node in node.child_nodes:
                    premise_node_id = tree.id_by_node[premise_node]
                    premise_lf = premise_node.logicalform
                    if ((str(tree), str(instantiation), node_id) in preselected_templates_by_primary_node_id) and (premise_node_id in preselected_templates_by_primary_node_id[(str(tree), str(instantiation), node_id)]):
                        # try using preselected template
                        template = preselected_templates_by_primary_node_id[(str(tree), str(instantiation), node_id)][premise_node_id]
                    elif premise_node.is_leaf and enforce_premise_axiom_consistency and ((str(tree), str(instantiation), premise_node_id) in preselected_templates_by_primary_node_id):
                        # enforce consistency between premise and axiom
                        template = preselected_templates_by_primary_node_id[(str(tree), str(instantiation), premise_node_id)][premise_node_id]
                    else:
                        # sample new template
                        template = template_sampler.choose_template(premise_lf, TemplateType.STATEMENT, tree, seed)
                    assert template.template_type == TemplateType.STATEMENT, f"Template should be a statement and not {template.template_type.name}"
                    assert template.condition.is_satisified(lf=premise_lf, tree=tree), "Template should still be valid!"
                    selection.append((premise_node_id, template))
                    seed += 1

            # conclusion
            if (str(tree), str(instantiation), node_id) in preselected_templates_by_primary_node_id and node_id in preselected_templates_by_primary_node_id[(str(tree), str(instantiation), node_id)]:
                # try using preselected template
                template = preselected_templates_by_primary_node_id[(str(tree), str(instantiation), node_id)][node_id]
                assert template.template_type == TemplateType.CONCLUSION, f"Preselected template should be a conclusion and not {template.template_type.name}"
                assert template.condition.is_satisified(lf=lf, tree=tree), "Preselected template should still be valid!"
            else:
                # sample new template
                template = template_sampler.choose_template(lf, TemplateType.CONCLUSION, tree, seed)
            selection.append((node_id, template))

            template_selections.append((tree, instantiation, TemplateSelection(node_id, selection)))
            seed += 1

        return template_selections

def render_interleaved_rt(template_renderer: TemplateRenderer, template_selections: List[Tuple[ProofTree, Instantiation, TemplateSelection]], eods_separator: TextPart = NEW_LINE) -> Tuple[str, RenderingMetadata]:
    """ 
        Renders a selection of templates from different (tree, instantiation)s into a natural language reasoning trace.

        Parameters:
        - template_renderer: renderer for a single template
        - template_selections: specifies which tree, instantiation, tree-nodes (by index) should be rendered in which order
        - eods_separator: appended to the end of each selected template (e.g. separator of nodes/sentences)
    """
    all_text = ""
    all_metadata = RenderingMetadata(template_selections=template_selections)
    visited_nodes_by_tree = {}
    for i,(tree,instantiation,selection) in enumerate(template_selections):
        is_last_selection = (len(template_selections) - 1 == i)
        for j, (node_id, template) in enumerate(selection.selection):
            if node_id in visited_nodes_by_tree.get(tree, []): continue # skip nodes/facts we already stated

            visited_nodes_by_tree[tree] = visited_nodes_by_tree.get(tree, []) + [node_id]
            is_last_template = (len(selection.selection) - 1 == j)
            node = tree.node_by_id[node_id]

            assert template.condition.is_satisified(node.logicalform, tree), "Template-selection must be valid on the rendered instance!"
            
            appendix = [WHITESPACE] if not is_last_template else ([] if is_last_selection else [eods_separator])
            text, metadata = template_renderer.render(node.logicalform, instantiation, template, append=appendix, parent_unit=[i,j])
            all_text += text
            all_metadata += metadata
    return all_text, all_metadata

def traverse_reasoning_trace_multitree(trees: List[ProofTree], order: List[Tuple[ProofTree, int]], identical_nodes: List[Tuple[Tuple[ProofTree,TreeNode], Tuple[ProofTree,TreeNode]]]) -> Generator[TreeNode, None, None]:
    """     
        Visits all trees in a DFS/max-inferences manner based on availability of axioms:
        Meaning, as soon as all required premises have been discovered, come to the conclusion 
        and add it to the list of available facts, iterate until no more facts can be derived
        before consuming the next axiom etc.
    """
    rest_of_leaves = order.copy()
    all_facts: set[LogicalForm] = set([lf for tree in trees for lf in tree.nodes_by_lf.keys()])
    known_facts: set[LogicalForm] = []
    while not all(f in known_facts for f in all_facts) and len(rest_of_leaves) > 0:
        # add leaf as new fact
        tree,leaf_id = rest_of_leaves.pop(0)
        leaf_node = tree.node_by_id[leaf_id]

        id_nodes = [(tree, leaf_node)] + [na for na,nb in identical_nodes if nb[1] == leaf_node] + [nb for na,nb in identical_nodes if na[1] == leaf_node]
        for t,n in id_nodes:
            known_facts.append(n.logicalform)
            # return only if we actually intend to render it
            if (t,t.id_by_node[n]) in order:
                yield t,n
        
        # while we can conclude new facts:
        new_fact_inferred = True
        while new_fact_inferred:
            new_fact_inferred = False
            for tree in trees:
                for node in tree.traverse(TraversalOrder.POST):
                    if node.logicalform in known_facts: continue # skip known facts
                    if node.is_leaf: continue # leaves should be consumed from the provided order
                    
                    # if all premises are known to derive the node
                    if all(node.logicalform in known_facts for node in node.child_nodes):
                        # add the conclusion as a new fact
                        id_nodes = [(tree, node)] + [na for na,nb in identical_nodes if nb[1] == node] + [nb for na,nb in identical_nodes if na[1] == node]
                        for t,n in id_nodes:
                            known_facts.append(n.logicalform)
                            yield t,n
                        new_fact_inferred = True
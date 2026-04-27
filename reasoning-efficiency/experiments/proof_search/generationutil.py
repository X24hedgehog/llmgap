from itertools import product
from typing import List, Dict
from collections import Counter, defaultdict
import re

from mathgap.expressions import Variable
from mathgap.logicalforms.logicalform import LogicalForm
from mathgap.natlang.templates import WHITESPACE, Template, render_problem, render_reasoning_trace, RenderingMetadata, TemplateType
from mathgap.problemsample import ProblemOrder
from mathgap.trees.generators.multi import MultiGenerator
from mathgap.trees.sampling.order import OrderSampler
from mathgap.generation_util import *
from mathgap.instantiate.instantiation import Instantiation, delete_all_of_type
from mathgap.mathwordproblems import MathWordProblem
from mathgap.properties import PropertyKey
from mathgap.trees.prooftree import ProofTree, TreeNode

from controlgenerator import ControlGenerator
from templaterendering import interleaved_template_selections, render_interleaved, interleaved_template_selection_rt, render_interleaved_rt

def sum_abs_rule_diff(rules1: List[InferenceRule], rules2: List[InferenceRule]) -> int:
    """ Get the number of rules that would need to be added/removed to make the two lists equal """
    cnt_by_rule1 = Counter(rules1)
    cnt_by_rule2 = Counter(rules2)
    return sum(abs(cnt_by_rule1.get(k, 0) - cnt_by_rule2.get(k, 0)) for k in set(cnt_by_rule1.keys()) | set(cnt_by_rule2.keys()))

def instantiate_irrelevant_rest_overlap(instantiator: Instantiator, entities_by_topic: Dict[Tuple[str,str], List[Tuple[str,str]]], lexical_overlaps: List[str], 
                                        base_tree: ProofTree, base_instantiation: Instantiation, 
                                        irrelevant_trees: List[ProofTree], partial_irrelevant_instantiations: List[Instantiation],
                                        overlap_type: str) -> List[Instantiation]:
    """ Instantiates the irrelevant trees starting from a given list of partial instantiations """

    # establish the set of agent/entity instantiations that are already used
    taken_agents = set(base_instantiation.get_instantiations_of_type(PropertyType.AGENT).values())
    taken_entities = set(base_instantiation.get_instantiations_of_type(PropertyType.ENTITY).values())
    for irrelevant_instantiation in partial_irrelevant_instantiations:
        taken_agents = taken_agents.union(irrelevant_instantiation.get_instantiations_of_type(PropertyType.AGENT).values())
        taken_entities = taken_entities.union(irrelevant_instantiation.get_instantiations_of_type(PropertyType.ENTITY).values())

    # get all the agents/entities the problem can overlap with
    # base_entities = [base_instantiation[PropertyKey(PropertyType.ENTITY, pid)] for pid in base_tree.property_tracker.get_by_type(PropertyType.ENTITY)]
    # base_agents = [base_instantiation[PropertyKey(PropertyType.AGENT, pid)] for pid in base_tree.property_tracker.get_by_type(PropertyType.AGENT)]

    # (restrict to agents/entities asked about in question)
    base_agents = set([])
    base_entities = set([])
    for vk in base_tree.root_node.logicalform.get_variable_keys():
        agent,entity,_,_ = vk.variable_key
        base_agents.add(base_instantiation[agent])
        base_entities.add(base_instantiation[entity])

    base_agents = list(base_agents)
    base_entities = list(base_entities)

    base_lexical_agents = [f"{ag}{lex}" for ag,lex in product(base_agents, lexical_overlaps)]

    # instantiate all irrelevant trees one after the other
    irrelevant_instantiations: List[Instantiation] = []
    for irrelevant_tree,irrelevant_instantiation in zip(irrelevant_trees,partial_irrelevant_instantiations):
        _seed = random.randint(0, 2**32-1)
        random.seed(_seed)

        # avoid agent/entity overlap by adding all agents/entities as pseudo-entries to the instantiation
        for i,agent_name in enumerate(taken_agents, start=1):
            irrelevant_instantiation[PropertyKey(PropertyType.AGENT, -i)] = agent_name
        for i,entity_name in enumerate(taken_entities, start=1):
            irrelevant_instantiation[PropertyKey(PropertyType.ENTITY, -i)] = entity_name

        # instantiate according to overlap_type
        if overlap_type == "entity_overlap":
            # Ensure irrelevant trees do not cause ambguitiy by making sure agent names differ between trees
            available_base_entities = base_entities.copy() 
            for p in irrelevant_tree.property_tracker.get_by_type(PropertyType.ENTITY):
                pk = PropertyKey(PropertyType.ENTITY, p)
                if pk in irrelevant_instantiation: continue

                irrelevant_entity = random.choice(available_base_entities)
                irrelevant_instantiation[pk] = irrelevant_entity
            
                # ensure the same entity cannot be chosen multiple times for the same tree 
                # NOTE: Can lead to problems if the irrelevant tree has more entities than the base tree. 
                #       If this is the case, we might want to allow other entities than the ones in the question to be chosen too.
                #       However, should then check that there is no two (agent1,entity1), (agent2,entity2) that are equivalent in terms of instantiation
                available_base_entities.remove(irrelevant_entity) 

        elif overlap_type == "agent_overlap":
            # Ensure irrelevant trees do not cause ambguitiy by making sure entities differ between trees
            available_base_entities = base_entities.copy()
            available_base_agents = base_agents.copy()
            available_base_lexical_agents = base_lexical_agents.copy()
            agent_pids = irrelevant_tree.property_tracker.get_by_type(PropertyType.AGENT)
            random.shuffle(agent_pids) # shuffle s.t. available_base_agents not used late on in tree (properties are sorted by when they are introduced)
            for p in agent_pids:
                pk = PropertyKey(PropertyType.AGENT, p)
                if pk in irrelevant_instantiation: continue

                if len(available_base_agents) == 0 or isinstance(base_tree.root_node.logicalform, PartWhole):
                    # lexical overlap
                    irrelevant_agent = random.choice(available_base_lexical_agents)
                    available_base_lexical_agents.remove(irrelevant_agent)
                else:
                    # overlap 100% with agent
                    irrelevant_agent = random.choice(available_base_agents)
                    available_base_agents.remove(irrelevant_agent)
            
                irrelevant_instantiation[pk] = irrelevant_agent

            # ensure entities are of the same type as base_entities (also make sure there are no conflicts between irrelevant trees)
            for p in irrelevant_tree.property_tracker.get_by_type(PropertyType.ENTITY):
                pk = PropertyKey(PropertyType.ENTITY, p)
                if pk in irrelevant_instantiation: continue

                entity = random.choice(available_base_entities)
                topics_of_entity = [t for t,ents in entities_by_topic.items() if entity in ents]
                assert len(topics_of_entity) > 0, f"Expecting entity to be associated with some topic, s.t. we can match that in agent_overlap! Got entity {entity}"
                topic = random.choice(topics_of_entity)
                
                available_entities = [e for e in entities_by_topic[topic] if e not in taken_entities]
                assert len(available_entities) > 0, f"Ran out of entities of topic {topic}"
                irrelevant_entity = random.choice(available_entities)
                taken_entities.add(irrelevant_entity)

                irrelevant_instantiation[pk] = irrelevant_entity
        elif overlap_type == "agent_entity_overlap":
            # Ensure irrelevant trees do not cause ambguitiy by making sure agents differ between trees

            # overlap entities
            available_base_entities = base_entities.copy()
            for p in irrelevant_tree.property_tracker.get_by_type(PropertyType.ENTITY):
                pk = PropertyKey(PropertyType.ENTITY, p)
                if pk in irrelevant_instantiation: continue

                irrelevant_entity = random.choice(available_base_entities)
                irrelevant_instantiation[pk] = irrelevant_entity
                available_base_entities.remove(irrelevant_entity)

            # overlap agents-lexically
            available_base_lexical_agents = [agent for agent in base_lexical_agents if agent not in taken_agents]
            for p in irrelevant_tree.property_tracker.get_by_type(PropertyType.AGENT):
                pk = PropertyKey(PropertyType.AGENT, p)
                if pk in irrelevant_instantiation: continue

                irrelevant_agent = random.choice(available_base_lexical_agents)
                available_base_lexical_agents.remove(irrelevant_agent)
                irrelevant_instantiation[pk] = irrelevant_agent

        # instantiate the remaining properties with no-overlap
        irrelevant_instantiation = instantiator.instantiate(irrelevant_tree, irrelevant_instantiation, skip_existing=True, seed=_seed)
        irrelevant_instantiations.append(irrelevant_instantiation)
        
        taken_agents = taken_agents.union(irrelevant_instantiation.get_instantiations_of_type(PropertyType.AGENT).values())
        taken_entities = taken_entities.union(irrelevant_instantiation.get_instantiations_of_type(PropertyType.ENTITY).values())

    return irrelevant_instantiations

def get_single_assign_cont(tree: ProofTree, only_axioms: bool = True, require_agent_entity_overlap: bool = True, require_premise_of_binary_rule: bool = True) -> List[Container]:
        """ 
            Gets all containers where the agent,entity is only assigned once (i.e. no transfer) 
            - only_axioms: if true, then will only select from leaf nodes of the tree
            - require_agent_entity_overlap: if true, will require that the other premises that are used jointly with the selected container overlap in at least one agent,entity (i.e. no cont of contcontcomp)
            - require_premise_of_binary_rule: if true, will require that there is exactly one other premise used jointly with the selected container
        """
        def _overlap_other_premises_agent_entity(tree: ProofTree, node: TreeNode) -> bool:
            return all([len(set(p.get_variable_keys()).intersection(set(node.logicalform.get_variable_keys()))) > 0 for p in tree.parent_by_node[node].premises])
    
        def _is_premise_of_binary_rule(tree: ProofTree, node: TreeNode) -> bool:
            return len(tree.parent_by_node[node].premises) == 2

        agent_entity_by_lf = {
            n.logicalform: vk.variable_key[:2] 
            for n in tree.nodes 
            for vk in n.logicalform.get_variable_keys() 
            if isinstance(n.logicalform, Container) 
                and (not only_axioms or n.is_leaf) 
                and (not require_agent_entity_overlap or _overlap_other_premises_agent_entity(tree, n))
                and (not require_premise_of_binary_rule or _is_premise_of_binary_rule(tree, n))
        }
        times_by_agent_entity = {}
        for vk,time in tree.traverse_writes():
            agent_entity = vk.variable_key[:2]
            times_by_agent_entity[agent_entity] = times_by_agent_entity.get(agent_entity, []) + [time]

        for agent_entity,times in times_by_agent_entity.items():
            if len(times) > 1:
                agent_entity_by_lf = {lf:ae for lf,ae in agent_entity_by_lf.items() if ae != agent_entity}

        return list(agent_entity_by_lf.keys())

def drop_one_axiom(orig_tree: ProofTree, seed: int) -> ProofTree:
    """ Simplifies the proof tree by removing one axiom """
    tree = orig_tree.copy()
    random.seed(seed)

    node: TreeNode = random.choice([tree.parent_by_node[n] for n in tree.leaf_nodes if len(tree.parent_by_node[n].premises) == 2 and all(tree.nodes_by_lf[p].is_leaf for p in tree.parent_by_node[n].premises)])
    assert isinstance(node.logicalform, Container) or isinstance(node.logicalform, Comp), f"Expecting Container or Comp but got {type(node.logicalform)} instead"

    # remove all premises from the tree
    for p in node.premises:
        pn = tree.nodes_by_lf[p]

        tree.node_by_id.pop(tree.id_by_node[pn])
        tree.id_by_node.pop(pn)
        tree.nodes_by_type[type(pn.logicalform)].remove(pn)
        tree.times_by_node.pop(pn)
        tree.nodes_by_lf.pop(p)
        tree.leaf_nodes.remove(pn)
        tree.depth = max([n.depth for n in tree.nodes])

    # fix the conclusion node
    tree.leaf_nodes.append(node)
    node.logicalform.quantity = Variable(tree.property_tracker.request_key(PropertyType.QUANTITY))
    node.rule = None
    node.child_nodes = []

    tree._refresh_complete_variable_times()
    tree.compute_symbolically()

    # update the tree's property_tracker as some props might not be needed any longer
    used_ids = defaultdict(set)
    for n in tree.nodes:
        available_props = n.logicalform.get_available_properties()
        for prop in available_props.values():
            if isinstance(prop, PropertyKey):
                if isinstance(prop.identifier, int):
                    used_ids[prop.property_type].add(prop.identifier)
                elif isinstance(prop.identifier, Variable):
                    used_ids[prop.property_type].add(prop.identifier.identifier.identifier)
            elif isinstance(prop, list):
                for pk in prop:
                    if isinstance(pk.identifier, int):
                        used_ids[pk.property_type].add(pk.identifier)
                    elif isinstance(pk.identifier, Variable):
                        used_ids[pk.property_type].add(pk.identifier.identifier.identifier)
    tree.property_tracker.used_ids = {
        t: list(used_ids.get(t, []))
        for t in list(PropertyType)
    }

    return tree


def get_base_subquestion_data(tree: ProofTree, instantiation: Instantiation, problem_order: ProblemOrder,
                              ps_template_sampler: ProblemStructureSampler, ps_renderer: ProblemStructureRenderer,
                              rt_template_sampler: ReasoningTraceSampler, rt_renderer: ReasoningTraceRenderer,
                              seed: int = 14) -> Dict:
    if not tree.is_symbolically_computed:
        tree.compute_symbolically()

    mwp = MathWordProblem(
        tree=tree,
        instantiation=instantiation,
        ps_template_sampler=ps_template_sampler,
        ps_renderer=ps_renderer,
        rt_template_sampler=rt_template_sampler,
        rt_renderer=rt_renderer,
    )
    mwp.problem_order = problem_order
    mwp.problem_as_nl(seed=seed)
    mwp.subquestions_as_nl(seed=seed)

    subquestion_answers_by_node_id = {
        str(node_id): tree.instantiated_quantity(tree.node_by_id[node_id].logicalform, instantiation)
        for node_id in (mwp.subquestions_by_node_id or {}).keys()
    }
    subproblems_by_node_id = {
        str(node_id): mwp._render_subproblem_for_node(node_id=node_id, seed=seed + node_id)
        for node_id in (mwp.subquestions_by_node_id or {}).keys()
    }

    return {
        "subquestions": mwp.subquestions_nl or [],
        "subquestions_by_node_id": mwp.subquestions_by_node_id or {},
        "subquestion_answers_by_node_id": subquestion_answers_by_node_id,
        "subproblems_by_node_id": subproblems_by_node_id,
        "subquestion_tree": mwp.subquestion_tree.model_dump() if mwp.subquestion_tree is not None else None,
    }

def get_next_problem(base_generator: MultiGenerator, irrelevant_generator: MultiGenerator, instantiator: Instantiator, topic_instantiator: Instantiator, 
                     entities_by_topic: Dict[Tuple[str,str], List[Tuple[str,str]]], lexical_overlaps: List[str],
                     num_irrelevant_trees: int, problem_order_sampler: OrderSampler, ps_template_sampler: ProblemStructureSampler, ps_renderer: ProblemStructureRenderer, 
                     rt_template_sampler: ReasoningTraceSampler, rt_renderer: ReasoningTraceRenderer, template_renderer: TemplateRenderer, max_attempts: int = 100, control_pop_size: int = 5, seed: int = 14) -> Dict:
    """
        - instantiator: used by default to instantiate mwps
        - topic_instantiator: used to instantiate mwps where all entities should be of a topic
        - entities_by_topic: maps some entities to a topic
        - lexical_overlap: list of suffixes that can be appended to agent names to make them overlap lexically
        - max_attempts: max attemts to find a tree that fits specification when looking for trees of some exact width
        - control_pop_size: how many control trees should be considered when searching for a close one?
    """
    data = {}    

    def _generate_control(base_tree: ProofTree, control_rule_stack: List[InferenceRule], base_generator: MultiGenerator, base_instantiation: Instantiation, desired_width: int) -> Dict[str, str]:    
        # 1. generate a selection of random trees that try to match the width and types of rules
        control_trees = [ControlGenerator(base_tree, control_rule_stack, base_generator.sample_generator(seed+i), desired_width).generate(seed+i) for i in range(control_pop_size)]

        # 2. select the tree that matches most importantly the desired width and but then also optimize for matching rules, i.e. match rules(base_tree) + control stack
        control_trees.sort(key=(lambda t: (abs(t.width - desired_width), sum_abs_rule_diff(control_rule_stack, [n.rule for n in t.nodes]))))
        control_tree = control_trees[0]

        # 3. instantiate the best control tree
        control_instantiation = instantiator.instantiate(control_tree, base_instantiation, skip_existing=True, seed=seed)
        control_order = problem_order_sampler.sample_order(control_tree, seed=seed)

        # 4. render
        # 4.1 render problembody + question
        problem,meta = render_problem(control_tree, control_instantiation, control_order, ps_template_sampler, ps_renderer, seed=seed)
        preselected_templates = [ts for ts in meta.template_selections if control_tree.node_by_id[ts.primary_node_id].is_leaf]
        
        # 4.2 render groundquery
        ps_template_sampler.use_groundquery = True
        problem_gq,meta_gq = render_problem(control_tree, control_instantiation, control_order, ps_template_sampler, ps_renderer, seed=seed)
        ps_template_sampler.use_groundquery = False

        rt,rt_meta = render_reasoning_trace(control_tree, control_instantiation, control_order, rt_template_sampler, rt_renderer, 
                                      preselected_templates=preselected_templates, enforce_premise_axiom_consistency=True,
                                      enforce_same_axiom_order=True, seed=seed)
        
        # 5. extract answer
        question_lfs = [control_tree.node_by_id[i].logicalform for i in control_order.question_node_ids]
        answer = control_tree.instantiated_quantities(question_lfs, control_instantiation)[0]
        
        # 6. convert for storage
        problem_sent = extract_per_sent_data(problem, meta, control_tree, [], control_instantiation, [])
        return { 
            "problembody": problem_sent[:-1], 
            "question": problem_sent[-1],
            "groundquery": extract_per_sent_data(problem_gq, meta_gq, control_tree, [], control_instantiation, [])[-1],
            "rt": extract_per_sent_data(rt, rt_meta, control_tree, [], control_instantiation, []), 
            "answer": answer, 
            "trees": {
                "control": {
                    "rule_counts": dict(Counter([type(n.rule).__qualname__ for n in control_tree.nodes if not n.is_leaf])),
                    "width": control_tree.width,
                    "depth": control_tree.depth
                }
            }
        }

    def _generate_connected(base_tree: ProofTree, base_instantiation: Instantiation, base_problem_order: ProblemOrder, base_answer: int, seed: int) -> Tuple[Dict, List[ProofTree]]:
        data_connected = {
            "simple": {},
            "complex": {},
            "more_complex": {}
        }
        
        # 1. generate the irrelevant trees
        irrelevant_trees: List[ProofTree] = []
        _seed = seed
            
        for _ in range(num_irrelevant_trees):
            irrelevant_tree = irrelevant_generator.generate(seed=seed)
            for i in range(max_attempts):
                if len(get_single_assign_cont(irrelevant_tree, only_axioms=True, require_agent_entity_overlap=True, require_premise_of_binary_rule=True)) > 0: break
                _seed = random.randint(0, 2**32-1)
                irrelevant_tree = irrelevant_generator.generate(seed=_seed)
            if len(get_single_assign_cont(irrelevant_tree, only_axioms=True, require_agent_entity_overlap=True, require_premise_of_binary_rule=True)) == 0: raise ValueError(f"Could not find a valid irrelevant tree after {max_attempts}")
            irrelevant_trees.append(irrelevant_tree)

        def _render_instantiation(irrelevant_instantiations: List[Instantiation], overlap: str, seed: int):
            # they are random already, so we just pick the first one
            irrelevant_tree_idx = 0
            irrelevant_tree, irrelevant_instantiation, connection = irrelevant_trees[irrelevant_tree_idx], irrelevant_instantiations[irrelevant_tree_idx], connected_nodes[irrelevant_tree_idx]
            base_node, irrelevant_node = connection[0][1], connection[1][1]

            # render groundquery (we only care about the final sentence => just render it once and extract the groundquery)
            ps_template_sampler.use_groundquery = True
            problem_gq,meta_gq = render_problem(base_tree, base_instantiation, base_problem_order, ps_template_sampler, ps_renderer, seed=seed)
            ps_template_sampler.use_groundquery = False
            groundquery_sent = extract_per_sent_data(problem_gq, meta_gq, base_tree, [], base_instantiation, [])[-1]

            # 3.1 simple case (one connected axiom that allows for one proof irrelevant axiom to be deduced)
            irrelevant_problem_order = problem_order_sampler.sample_order(irrelevant_tree, seed=seed)

            # render only the premises used together with the selected container, which are also not overlapping with the base-tree
            irrelevant_cont_premises_nid = [irrelevant_tree.id_by_node[irrelevant_tree.nodes_by_lf[lf]] for lf in irrelevant_tree.parent_by_node[irrelevant_node].premises]
            irrelevant_cont_premises_nid.remove(irrelevant_tree.id_by_node[irrelevant_node])
            irrelevant_problem_order.show_only_body_ids(set(irrelevant_cont_premises_nid)).hide_questions()

            interleaved_ts = interleaved_template_selections(
                base_mwp=(base_tree, base_instantiation, base_problem_order, ps_template_sampler),
                irrelevant_mwps=[(irrelevant_tree, irrelevant_instantiation, irrelevant_problem_order, ps_template_sampler)], 
                connected_nodes=connected_nodes, seed=seed)

            problem_interleaved,problem_interleaved_meta = render_interleaved(template_renderer, interleaved_ts)
            preselected_templates = [(t,i,ts) for t,i,ts in problem_interleaved_meta.template_selections if t.node_by_id[ts.primary_node_id].is_leaf]
            rt_interleaved, rt_interleaved_meta = render_interleaved_rt(template_renderer, 
                    interleaved_template_selection_rt([base_tree, irrelevant_tree], [base_instantiation, irrelevant_instantiation], 
                                                    [(tree,ts.primary_node_id) for tree,inst,ts in interleaved_ts], 
                                                    rt_template_sampler.sampler, 
                                                    identical_nodes=[connection],
                                                    preselected_templates=preselected_templates,
                                                    seed=seed), eods_separator=WHITESPACE)

            control_rule_stack = [irrelevant_tree.parent_by_node[irrelevant_node].rule]
            problem_sent = extract_per_sent_data(problem_interleaved, problem_interleaved_meta, base_tree, irrelevant_trees, base_instantiation, irrelevant_instantiations, connected_nodes)
            data_connected["simple"][overlap] = {
                "problembody": problem_sent[:-1],
                "question": problem_sent[-1],
                "groundquery": groundquery_sent,
                "rt": extract_per_sent_data(rt_interleaved, rt_interleaved_meta, base_tree, irrelevant_trees, base_instantiation, irrelevant_instantiations, connected_nodes),
                "answer": base_answer,
                "trees": {
                    "base": {
                        "rule_counts": dict(Counter([type(n.rule).__qualname__ for n in base_tree.nodes if not n.is_leaf])),
                        "width": base_tree.width,
                        "depth": base_tree.depth
                    },
                    f"irrelevant_{irrelevant_tree_idx}": {
                        "rule_counts": dict(Counter([type(n.rule).__qualname__ for n in [irrelevant_tree.parent_by_node[irrelevant_node]] if not n.is_leaf])),
                        "width": 2,
                        "depth": 1
                    }
                    
                }
            }

            if "control" not in data_connected["simple"]:
                data_connected["simple"]["control"] = _generate_control(base_tree, control_rule_stack=control_rule_stack, base_generator=irrelevant_generator, base_instantiation=base_instantiation, desired_width=base_tree.width+1)

            # 3.2 complex case (one connected irrelevant tree)
            irrelevant_problem_order = problem_order_sampler.sample_order(irrelevant_tree, seed=seed)

            # do not render the overlapping axiom
            irrelevant_cont_nid = irrelevant_tree.id_by_node[irrelevant_node]
            irrelevant_problem_order.hide_body_ids({irrelevant_cont_nid}).hide_questions()

            interleaved_ts = interleaved_template_selections(
                base_mwp=(base_tree, base_instantiation, base_problem_order, ps_template_sampler),
                irrelevant_mwps=[(irrelevant_tree, irrelevant_instantiation, irrelevant_problem_order, ps_template_sampler)], 
                connected_nodes=connected_nodes, seed=seed)

            problem_interleaved,problem_interleaved_meta = render_interleaved(template_renderer, interleaved_ts)
            preselected_templates = [(t,i,ts) for t,i,ts in problem_interleaved_meta.template_selections if t.node_by_id[ts.primary_node_id].is_leaf]
            rt_interleaved, rt_interleaved_meta = render_interleaved_rt(template_renderer, 
                    interleaved_template_selection_rt([base_tree, irrelevant_tree], [base_instantiation, irrelevant_instantiation], 
                                                    [(tree,ts.primary_node_id) for tree,inst,ts in interleaved_ts], 
                                                    rt_template_sampler.sampler, 
                                                    identical_nodes=[connection],
                                                    preselected_templates=preselected_templates,
                                                    seed=seed), eods_separator=WHITESPACE)
            
            control_rule_stack = [node.rule for node in irrelevant_tree.nodes if not node.is_leaf]
            problem_sent = extract_per_sent_data(problem_interleaved, problem_interleaved_meta, base_tree, irrelevant_trees, base_instantiation, irrelevant_instantiations, connected_nodes)
            data_connected["complex"][overlap] = {
                "problembody": problem_sent[:-1],
                "question": problem_sent[-1],
                "groundquery": groundquery_sent,
                "rt": extract_per_sent_data(rt_interleaved, rt_interleaved_meta, base_tree, irrelevant_trees, base_instantiation, irrelevant_instantiations, connected_nodes),
                "answer": base_answer,
                "trees": {
                    "base": {
                        "rule_counts": dict(Counter([type(n.rule).__qualname__ for n in base_tree.nodes if not n.is_leaf])),
                        "width": base_tree.width,
                        "depth": base_tree.depth
                    },
                    f"irrelevant_{irrelevant_tree_idx}": {
                        "rule_counts": dict(Counter([type(n.rule).__qualname__ for n in irrelevant_tree.nodes if not n.is_leaf])),
                        "width": irrelevant_tree.width,
                        "depth": irrelevant_tree.depth
                    }
                }
            }

            if "control" not in data_connected["complex"]:
                data_connected["complex"]["control"] = _generate_control(base_tree, control_rule_stack=control_rule_stack, base_generator=irrelevant_generator, base_instantiation=base_instantiation, desired_width=base_tree.width+irrelevant_tree.width-1) # -1 width because of overlap

            # 3.3. more complex case (multiple connected irrelevant trees)
            irrelevant_problem_orders = []
            for irrelevant_tree, connection in zip(irrelevant_trees, connected_nodes):
                base_node, irrelevant_node = connection[0][1], connection[1][1]

                irrelevant_problem_order = problem_order_sampler.sample_order(irrelevant_tree, seed=seed)

                # remove the overlapping axiom
                irrelevant_cont_nid = irrelevant_tree.id_by_node[irrelevant_node]
                irrelevant_problem_order.hide_body_ids({irrelevant_cont_nid}).hide_questions()

                irrelevant_problem_orders.append(irrelevant_problem_order)

            interleaved_ts = interleaved_template_selections(
                base_mwp=(base_tree, base_instantiation, base_problem_order, ps_template_sampler),
                irrelevant_mwps=[(it, ii, ip, ps_template_sampler) for it,ii,ip in zip(irrelevant_trees, irrelevant_instantiations, irrelevant_problem_orders)], 
                connected_nodes=connected_nodes, seed=seed)

            problem_interleaved,problem_interleaved_meta = render_interleaved(template_renderer, interleaved_ts)
            preselected_templates = [(t,i,ts) for t,i,ts in problem_interleaved_meta.template_selections if t.node_by_id[ts.primary_node_id].is_leaf]
            rt_interleaved, rt_interleaved_meta = render_interleaved_rt(template_renderer, 
                    interleaved_template_selection_rt([base_tree, *irrelevant_trees], [base_instantiation, *irrelevant_instantiations], 
                                                    [(tree,ts.primary_node_id) for tree,inst,ts in interleaved_ts], 
                                                    rt_template_sampler.sampler, 
                                                    identical_nodes=connected_nodes,
                                                    preselected_templates=preselected_templates,
                                                    seed=seed), eods_separator=WHITESPACE)
            
            control_desired_width = base_tree.width + sum([it.width - 1 for it in irrelevant_trees]) # -1 width per tree because of overlap
            control_rule_stack = [node.rule for irrelevant_tree in irrelevant_trees for node in irrelevant_tree.nodes if not node.is_leaf]
            problem_sent = extract_per_sent_data(problem_interleaved, problem_interleaved_meta, base_tree, irrelevant_trees, base_instantiation, irrelevant_instantiations, connected_nodes)
            data_connected["more_complex"][overlap] = {
                "problembody": problem_sent[:-1],
                "question": problem_sent[-1],
                "groundquery": groundquery_sent,
                "rt": extract_per_sent_data(rt_interleaved, rt_interleaved_meta, base_tree, irrelevant_trees, base_instantiation, irrelevant_instantiations, connected_nodes),
                "answer": base_answer,
                "trees": {
                    "base": {
                        "rule_counts": dict(Counter([type(n.rule).__qualname__ for n in base_tree.nodes if not n.is_leaf])),
                        "width": base_tree.width,
                        "depth": base_tree.depth
                    },
                    **{
                        f"irrelevant_{i}": {
                            "rule_counts": dict(Counter([type(n.rule).__qualname__ for n in irrelevant_trees[i].nodes if not n.is_leaf])),
                            "width": irrelevant_trees[i].width,
                            "depth": irrelevant_trees[i].depth
                            }
                        for i in range(num_irrelevant_trees)
                    }
                }
            }

            if "control" not in data_connected["more_complex"]:
                data_connected["more_complex"]["control"] = _generate_control(base_tree, control_rule_stack=control_rule_stack, base_generator=irrelevant_generator, base_instantiation=base_instantiation, desired_width=control_desired_width)

        # 2. connect & pre-instantiate the irrelevant problems
        irrelevant_instantiations: List[Instantiation] = []
        connected_nodes = [] # ((base_tree, base_node), (irrelevant_tree, irrelevant_node))
        available_base_sacs = set(get_single_assign_cont(base_tree, only_axioms=False, require_agent_entity_overlap=False, require_premise_of_binary_rule=False))

        for irrelevant_tree in irrelevant_trees:
            _seed = random.randint(0, 2**32-1)
            random.seed(_seed)

            # find two containers on which to connect the trees
            irrelevant_sacs = get_single_assign_cont(irrelevant_tree, only_axioms=True, require_agent_entity_overlap=True, require_premise_of_binary_rule=True)
            base_cont = random.choice(list(available_base_sacs))
            available_base_sacs.remove(base_cont) # make sure no two trees connect on the same agent,entity!
            
            irrelevant_cont = random.choice(irrelevant_sacs)
            base_node = base_tree.nodes_by_lf[base_cont]
            irrelevant_node = irrelevant_tree.nodes_by_lf[irrelevant_cont]

            # match quantity & instantiate numerically
            irrelevant_instantiation = Instantiation({})
            irrelevant_instantiation[irrelevant_cont.quantity_prop.identifier.identifier] = base_cont.get_quantities()[0].eval(base_instantiation)

            irrelevant_instantiation = instantiator.instantiate(irrelevant_tree, irrelevant_instantiation, skip_existing=True, seed=_seed)
            irrelevant_instantiation = delete_all_of_type(irrelevant_instantiation, PropertyType.AGENT)
            irrelevant_instantiation = delete_all_of_type(irrelevant_instantiation, PropertyType.ENTITY)

            # set the agent, entity to match
            irrelevant_instantiation[irrelevant_cont.agent_prop] = base_instantiation[base_cont.agent_prop]
            irrelevant_instantiation[irrelevant_cont.entity_prop] = base_instantiation[base_cont.entity_prop]

            connected_nodes.append(((base_tree, base_node), (irrelevant_tree, irrelevant_node)))
            irrelevant_instantiations.append(irrelevant_instantiation)

        # 3. instantiate agent/entities and render
        for overlap_type in ["no_overlap", "entity_overlap", "agent_overlap", "agent_entity_overlap"]:
            partial_irrelevant_insts = instantiate_irrelevant_rest_overlap(instantiator, entities_by_topic, lexical_overlaps, base_tree, base_instantiation, irrelevant_trees, [i.copy() for i in irrelevant_instantiations], overlap_type=overlap_type)
            _render_instantiation(partial_irrelevant_insts, overlap_type, seed=_seed) # NOTE: use the same seed s.t. it will be the exact same problems apart from the instantiation

        return data_connected, irrelevant_trees

    def _generate_disconnected(base_tree: ProofTree, base_instantiation: Instantiation, base_problem_order: ProblemOrder, connected_trees: List[ProofTree], base_answer: int, seed: int) -> Tuple[Dict]:
        _seed = seed
        data_disconnected = {
            "simple": {},
            "complex": {},
            "more_complex": {}
        }

        # 1. generate the irrelevant trees
        # ensure the number of axioms matches that of the connected cases
        irrelevant_trees: List[ProofTree] = [drop_one_axiom(t, _seed) for t in connected_trees]

        def _render_instantiation(irrelevant_instantiations: List[Instantiation], overlap: str, seed: int):
            irrelevant_tree_idx = 0
            irrelevant_tree, irrelevant_instantiation = irrelevant_trees[irrelevant_tree_idx], irrelevant_instantiations[irrelevant_tree_idx] # they are random already, so we just pick the first one
            
            # render groundquery (we only care about the final sentence => just render it once and extract the groundquery)
            ps_template_sampler.use_groundquery = True
            problem_gq,meta_gq = render_problem(base_tree, base_instantiation, base_problem_order, ps_template_sampler, ps_renderer, seed=seed)
            ps_template_sampler.use_groundquery = False
            groundquery_sent = extract_per_sent_data(problem_gq, meta_gq, base_tree, [], base_instantiation, [])[-1]

            # 3.1 simple case (one disconnected axiom)
            problem_order_irrelevant = problem_order_sampler.sample_order(irrelevant_tree, seed=seed)
            
            # show only one axiom
            random.seed(seed)
            irrelevant_nid = random.choice(problem_order_irrelevant.body_node_ids)
            problem_order_irrelevant.show_only_body_ids({irrelevant_nid}).hide_questions()

            interleaved_ts = interleaved_template_selections(
                base_mwp=(base_tree, base_instantiation, base_problem_order, ps_template_sampler),
                irrelevant_mwps=[(irrelevant_tree, irrelevant_instantiation, problem_order_irrelevant, ps_template_sampler)], 
                seed=seed)

            problem_interleaved,problem_interleaved_meta = render_interleaved(template_renderer, interleaved_ts)
            preselected_templates = [(t,i,ts) for t,i,ts in problem_interleaved_meta.template_selections if t.node_by_id[ts.primary_node_id].is_leaf]
            rt_interleaved, rt_interleaved_meta = render_interleaved_rt(template_renderer, 
                interleaved_template_selection_rt([base_tree, irrelevant_tree], [base_instantiation, irrelevant_instantiation], 
                                                [(tree,ts.primary_node_id) for tree,inst,ts in interleaved_ts], 
                                                rt_template_sampler.sampler, preselected_templates=preselected_templates,
                                                seed=seed), eods_separator=WHITESPACE) 

            problem_sent = extract_per_sent_data(problem_interleaved, problem_interleaved_meta, base_tree, irrelevant_trees, base_instantiation, irrelevant_instantiations)
            data_disconnected["simple"][overlap] = {
                "problembody": problem_sent[:-1],
                "question": problem_sent[-1],
                "groundquery": groundquery_sent,
                "rt": extract_per_sent_data(rt_interleaved, rt_interleaved_meta, base_tree, irrelevant_trees, base_instantiation, irrelevant_instantiations),
                "answer": base_answer,
                "trees": {
                    "base": {
                        "rule_counts": dict(Counter([type(n.rule).__qualname__ for n in base_tree.nodes if not n.is_leaf])),
                        "width": base_tree.width,
                        "depth": base_tree.depth
                    },
                    f"irrelevant_{irrelevant_tree_idx}": {
                        "rule_counts": dict(Counter([type(n.rule).__qualname__ for n in irrelevant_tree.nodes if not n.is_leaf])),
                        "width": 1,
                        "depth": 0
                    }
                }
            }

            if "control" not in data_disconnected["simple"]:
                data_disconnected["simple"]["control"] = _generate_control(base_tree, control_rule_stack=[], base_generator=irrelevant_generator, base_instantiation=base_instantiation, desired_width=base_tree.width+1)

            # 3.2 complex case (one irrelevant disconnected tree)
            problem_order_irrelevant = problem_order_sampler.sample_order(irrelevant_tree, seed=seed).hide_questions()

            interleaved_ts = interleaved_template_selections(
                base_mwp=(base_tree, base_instantiation, base_problem_order, ps_template_sampler),
                irrelevant_mwps=[(irrelevant_tree, irrelevant_instantiation, problem_order_irrelevant, ps_template_sampler)],
                seed=seed)

            problem_interleaved,problem_interleaved_meta = render_interleaved(template_renderer, interleaved_ts)
            preselected_templates = [(t,i,ts) for t,i,ts in problem_interleaved_meta.template_selections if t.node_by_id[ts.primary_node_id].is_leaf]
            rt_interleaved, rt_interleaved_meta = render_interleaved_rt(template_renderer, 
                interleaved_template_selection_rt([base_tree, irrelevant_tree], [base_instantiation, irrelevant_instantiation], 
                                                [(tree,ts.primary_node_id) for tree,inst,ts in interleaved_ts], 
                                                rt_template_sampler.sampler, preselected_templates=preselected_templates,
                                                seed=seed), eods_separator=WHITESPACE) 

            control_rule_stack = [node.rule for node in irrelevant_tree.nodes if not node.is_leaf]
            problem_sent = extract_per_sent_data(problem_interleaved, problem_interleaved_meta, base_tree, irrelevant_trees, base_instantiation, irrelevant_instantiations)
            data_disconnected["complex"][overlap] = {
                "problembody": problem_sent[:-1],
                "question": problem_sent[-1],
                "groundquery": groundquery_sent,
                "rt": extract_per_sent_data(rt_interleaved, rt_interleaved_meta, base_tree, irrelevant_trees, base_instantiation, irrelevant_instantiations),
                "answer": base_answer,
                "trees": {
                    "base": {
                        "rule_counts": dict(Counter([type(n.rule).__qualname__ for n in base_tree.nodes if not n.is_leaf])),
                        "width": base_tree.width,
                        "depth": base_tree.depth
                    },
                    f"irrelevant_{irrelevant_tree_idx}": {
                        "rule_counts": dict(Counter([type(n.rule).__qualname__ for n in irrelevant_tree.nodes if not n.is_leaf])),
                        "width": irrelevant_tree.width,
                        "depth": irrelevant_tree.depth
                    }
                }
            }

            if "control" not in data_disconnected["complex"]:
                data_disconnected["complex"]["control"] = _generate_control(base_tree, control_rule_stack=control_rule_stack, base_generator=irrelevant_generator, base_instantiation=base_instantiation, desired_width=base_tree.width+irrelevant_tree.width)


            # 3.3 more complex case (multiple irrelevant disconnected trees)
            interleaved_ts = interleaved_template_selections(
                base_mwp=(base_tree, base_instantiation, base_problem_order, ps_template_sampler),
                irrelevant_mwps=[(it, ii, problem_order_sampler.sample_order(it, seed=seed).hide_questions(), ps_template_sampler) for it,ii in zip(irrelevant_trees, irrelevant_instantiations)], 
                seed=seed)

            assert set([t for t,i,ts in interleaved_ts]) == set([base_tree, *irrelevant_trees]), "Expecting all trees to be present in template selection in more_complex case"
            assert len(set([t for t,i,ts in interleaved_ts])) == num_irrelevant_trees + 1, f"Expecting there to be {num_irrelevant_trees} + 1 trees to be present in template selection for more_complex case"

            problem_interleaved,problem_interleaved_meta = render_interleaved(template_renderer, interleaved_ts)
            preselected_templates = [(t,i,ts) for t,i,ts in problem_interleaved_meta.template_selections if t.node_by_id[ts.primary_node_id].is_leaf]
            rt_interleaved, rt_interleaved_meta = render_interleaved_rt(template_renderer, 
                interleaved_template_selection_rt([base_tree, *irrelevant_trees], [base_instantiation, *irrelevant_instantiations], 
                                                [(tree,ts.primary_node_id) for tree,inst,ts in interleaved_ts], 
                                                rt_template_sampler.sampler, preselected_templates=preselected_templates,
                                                seed=seed), eods_separator=WHITESPACE) 

            control_desired_width = base_tree.width + sum([it.width for it in irrelevant_trees])
            control_rule_stack = [node.rule for irrelevant_tree in irrelevant_trees for node in irrelevant_tree.nodes if not node.is_leaf]
            problem_sent = extract_per_sent_data(problem_interleaved, problem_interleaved_meta, base_tree, irrelevant_trees, base_instantiation, irrelevant_instantiations)
            data_disconnected["more_complex"][overlap] = {
                "problembody": problem_sent[:-1],
                "question": problem_sent[-1],
                "groundquery": groundquery_sent,
                "rt": extract_per_sent_data(rt_interleaved, rt_interleaved_meta, base_tree, irrelevant_trees, base_instantiation, irrelevant_instantiations),
                "answer": base_answer,
                "trees": {
                    "base": {
                        "rule_counts": dict(Counter([type(n.rule).__qualname__ for n in base_tree.nodes if not n.is_leaf])),
                        "width": base_tree.width,
                        "depth": base_tree.depth
                    },
                    **{
                        f"irrelevant_{i}": {
                            "rule_counts": dict(Counter([type(n.rule).__qualname__ for n in irrelevant_trees[i].nodes if not n.is_leaf])),
                            "width": irrelevant_trees[i].width,
                            "depth": irrelevant_trees[i].depth
                            }
                        for i in range(num_irrelevant_trees)
                    }
                }
            }

            if "control" not in data_disconnected["more_complex"]:
                data_disconnected["more_complex"]["control"] = _generate_control(base_tree, control_rule_stack=control_rule_stack, base_generator=irrelevant_generator, base_instantiation=base_instantiation, desired_width=control_desired_width)

        # 2. pre-instantiate the irrelevant problems numerically
        irrelevant_instantiations: List[Instantiation] = []
        for irrelevant_tree in irrelevant_trees:
            _seed = random.randint(0, 2**32-1)
            irrelevant_instantiation = instantiator.instantiate(irrelevant_tree, skip_existing=True, seed=_seed)
            irrelevant_instantiation = delete_all_of_type(irrelevant_instantiation, PropertyType.AGENT)
            irrelevant_instantiation = delete_all_of_type(irrelevant_instantiation, PropertyType.ENTITY)
            irrelevant_instantiations.append(irrelevant_instantiation)

        # 3. instantiate agent/entities and render
        for overlap_type in ["no_overlap", "entity_overlap", "agent_overlap", "agent_entity_overlap"]:
            partial_irrelevant_insts = instantiate_irrelevant_rest_overlap(instantiator, entities_by_topic, lexical_overlaps, base_tree, base_instantiation, irrelevant_trees, [i.copy() for i in irrelevant_instantiations], overlap_type=overlap_type)
            _render_instantiation(partial_irrelevant_insts, overlap_type, seed=_seed) # NOTE: use the same seed s.t. it will be the exact same problems apart from the instantiation

        return data_disconnected
    
    # 1. generate relevant and irrelevant trees that can be connected to at least num_irrelevant_trees times
    base_tree = base_generator.generate(seed=seed)
    while len(get_single_assign_cont(base_tree, only_axioms=False, require_agent_entity_overlap=False, require_premise_of_binary_rule=False)) < num_irrelevant_trees:
        seed = random.randint(0, 2**32-1)
        base_tree = base_generator.generate(seed=seed)

    # 2. instantiate the base problem
    # make sure all base-question-entities are part of a topic
    base_instantiation_topic = topic_instantiator.instantiate(base_tree, seed=seed) 
    base_instantiation = Instantiation({})
    for vk in base_tree.root_node.logicalform.get_variable_keys():
        agent,entity,attribute,unit = vk.variable_key
        base_instantiation[agent] = base_instantiation_topic[agent]
        base_instantiation[entity] = base_instantiation_topic[entity]
        if attribute.identifier is not None:
            base_instantiation[attribute] = base_instantiation_topic[attribute]
        if unit.identifier is not None:
            base_instantiation[unit] = base_instantiation_topic[unit]

    # copy numerical instantiation
    for quantity_id in base_tree.property_tracker.get_by_type(PropertyType.QUANTITY):
        pk = PropertyKey(PropertyType.QUANTITY, quantity_id)
        base_instantiation[pk] = base_instantiation_topic[pk]

    # instantiate rest
    base_instantiation = instantiator.instantiate(base_tree, base_instantiation, skip_existing=True, seed=seed)
    

    # 3. render the base problem
    base_problem_order = problem_order_sampler.sample_order(base_tree, seed=seed)
    base_problem,base_problem_meta = render_problem(base_tree, base_instantiation, base_problem_order, ps_template_sampler, ps_renderer, seed=seed)

    # render with groundquery
    ps_template_sampler.use_groundquery = True
    problem_gq,meta_gq = render_problem(base_tree, base_instantiation, base_problem_order, ps_template_sampler, ps_renderer, seed=seed)
    ps_template_sampler.use_groundquery = False

    preselected_templates = [ts for ts in base_problem_meta.template_selections if base_tree.node_by_id[ts.primary_node_id].is_leaf]
    base_rt, base_rt_meta = render_reasoning_trace(base_tree, base_instantiation, base_problem_order, rt_template_sampler, rt_renderer, preselected_templates=preselected_templates, seed=seed)
    base_question_lfs = [base_tree.node_by_id[i].logicalform for i in base_problem_order.question_node_ids]
    problem_sent = extract_per_sent_data(base_problem, base_problem_meta, base_tree, [], base_instantiation, [])
    base_answer = base_tree.instantiated_quantities(base_question_lfs, base_instantiation)[0]
    base_subquestion_data = get_base_subquestion_data(
        base_tree,
        base_instantiation,
        base_problem_order,
        ps_template_sampler,
        ps_renderer,
        rt_template_sampler,
        rt_renderer,
        seed=seed,
    )
    data["base"] = {
        "problembody": problem_sent[:-1],
        "question": problem_sent[-1],
        "groundquery": extract_per_sent_data(problem_gq, meta_gq, base_tree, [], base_instantiation, [])[-1],
        "rt": extract_per_sent_data(base_rt, base_rt_meta, base_tree, [], base_instantiation, []),
        "answer": base_answer,
        **base_subquestion_data,
        "trees": {
            "base": {
                "rule_counts": dict(Counter([type(n.rule).__qualname__ for n in base_tree.nodes if not n.is_leaf])),
                "width": base_tree.width,
                "depth": base_tree.depth
            }
        }
    }

    # 5. Connected Case
    data["connected"], connected_trees = _generate_connected(base_tree, base_instantiation, base_problem_order, base_answer, seed=seed)

    # 4. Disconnected Case
    data["disconnected"] = _generate_disconnected(base_tree, base_instantiation, base_problem_order, connected_trees, base_answer, seed=seed)

    return data

def extract_per_sent_data(nl: str, meta: RenderingMetadata, base_tree: ProofTree, irrelevant_trees: List[ProofTree], 
                          base_instantiation: Instantiation, irrelevant_instantiations: List[Instantiation], connected_nodes: List[Tuple[Tuple[ProofTree,TreeNode],Tuple[ProofTree,TreeNode]]] = []) -> List:
    assert len(irrelevant_trees) == len(irrelevant_instantiations), "Expecting trees and instantiations to match"

    per_character = list(zip(nl, meta.lf_per_character, meta.template_per_character))

    per_character_per_sent = []
    sent = []
    
    for item in per_character:
        if len(sent) > 0 or item[0] != " ": # skip trailing whitespaces
            sent.append(item)

        if item[0] == "." or item[0] == "?": # check for end of sentence
            if sent:
                per_character_per_sent.append(sent)
            sent = []

    sent_idx_by_treeandnode = {}
    sent_data = []
    for i,sent in enumerate(per_character_per_sent):
        text = ''.join([c for c,lf,templ in sent])
        lf: LogicalForm = Counter([lf for c,lf,templ in sent]).most_common(1)[0][0]
        templ: Template = Counter([templ for c,lf,templ in sent]).most_common(1)[0][0]

        equations = re.findall(r'\b(-?\d+)\s*([+-])\s*(-?\d+)\s*(=)\s*(-?\d+)\b', text)
        quantities = re.findall(r'\b(\d+)\b', text)

        instantiation = None
        relevant = False
        source_tree: ProofTree = None
        source_tree_id: str = None
        if lf in base_tree.nodes_by_lf.keys():
            instantiation = base_instantiation
            relevant = True
            source_tree = base_tree
            source_tree_id = "base"
        else:
            for tree, inst in zip(irrelevant_trees, irrelevant_instantiations):
                if lf in tree.nodes_by_lf.keys():
                    instantiation = inst
                    source_tree = tree
                    source_tree_id = f"irrelevant_{irrelevant_trees.index(tree)}"
                    break
        
        source_node = source_tree.nodes_by_lf[lf]
        sent_idx_by_treeandnode[(source_tree, source_node)] = i
        
        dependencies = []
        if templ.template_type == TemplateType.CONCLUSION:
            # mark dependence on premises
            for premise_node in source_node.child_nodes:
                candidates = [(bat,ban) for (bat,ban),(irt,irn) in connected_nodes if irt == source_tree and irn == premise_node]
                assert len(candidates) <= 1, "Irrelevant tree cannot be connected on more than one node with base-tree"
                if len(candidates) == 1:
                    bat,ban = candidates[0]
                    dependencies.append(sent_idx_by_treeandnode[(bat,ban)])
                else:
                    assert (source_tree,premise_node) in sent_idx_by_treeandnode, f"Reference to sentence that has not been rendered yet {(source_tree,source_tree.id_by_node[premise_node])}"
                    dependencies.append(sent_idx_by_treeandnode[(source_tree,premise_node)])

        sent_data.append({
            "text": text,
            "inst": {
                "agents": list(set([instantiation[lf.get_available_properties()[p]] for p in templ.get_required_properties() if lf.get_available_properties()[p].property_type == PropertyType.AGENT])),
                "attributes": list(set([instantiation[lf.get_available_properties()[p]] for p in templ.get_required_properties() if lf.get_available_properties()[p].property_type == PropertyType.ATTRIBUTE])),
                "entities": list(set([instantiation[lf.get_available_properties()[p]] for p in templ.get_required_properties() if lf.get_available_properties()[p].property_type == PropertyType.ENTITY])),
                "units": list(set([instantiation[lf.get_available_properties()[p]] for p in templ.get_required_properties() if lf.get_available_properties()[p].property_type == PropertyType.UNIT])),
                "quantities": quantities,
                "equations": equations,
            },
            "lf": type(lf).__name__,
            "type": templ.template_type.value,
            "relevant": relevant,
            "source": source_tree_id,
            "premise_sent_indices": dependencies
        })

    return sent_data

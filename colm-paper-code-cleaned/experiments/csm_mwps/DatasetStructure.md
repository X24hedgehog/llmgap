# High Level
The dataset consists of math word problem structures as the top-most hierarchy (mwp-level). For each math word problem structure, it contains multiple instantiations (inst-level). All instantiations share the same structure but names, entities, quantities and where distractors are applicable might vary (depending on the parameters used to generate the dataset).
For each instantiation, the problem, a consistent formulation of the problem (where no comparison inconsistencies are applicable), the correct answer and all the possible misconception answers (made by applying the error wherever applicable) are listed.
Metadata about the structure of the problem is exported at the mwp-level, metadata about the instantiation at the inst-level.

# Detail
We have a list of problem structures, for each of which we have (MWP-level):
- metadata
    - depth: the depth of the proof tree / problem structure
    - width: the width "
    - rule_count: how many steps of each type are required to solve the problem? (we use "Premise...PremiseConclusion" notation, e.g., "Alice has 5 apples. Bob has 3 apples. Therefore, Bob has 2 apples less than Alice" corresponds to ContContComp)
    - orig_rules_by_conclusion: a dictionary (node_id, rule-type) specifying which is the correct rule to use to derive each conclusion (node_id is just a unique identifier for the node in the proof)
    - depth_by_conclusion: a dictionary (node_id, depth) specifying for each conclusion, how far from the solution it still is
    - correct_expression: a symbolic expression for the correct solution to the problem in terms of all axiom quantities
- instantiations: list of instantiations for mwp structure, for each instantiation we have (INST-level):
    - inconsistent_conclusions: list of (node_id), specifying which conclusions are formulated inconsistently / where comparison inconsistency is applicable
    - problem: natural language represenation of the problem with inconsistent phrasings as specified
    - cons_problem: natural language representation of the problem without any inconsistent phrasings
    - instantiation: dictionary (axiom quantity, value) of how each axiom value is chosen
    - all_op_perurbation_answers: null if not generated, otherwise: a list of all the answers possible by flipping + to - or vice versa
    - correct_answer
        - answer: numerical value if solved correctly
        - rt: natural language representation of the correct solution
    - misconception_answers: List of all possible misconception answers (made by applying the error in one or more applicable steps)
        - answer: numerical value if solved incorrectly
        - expression: symbolic expression for the incorrect solution in terms of all axiom quantities
        - misconceptions: dictionary (node_id, rule-type) specifying which rules have been incorrectly applied
        - plausible: true iff no intermediate values required to compute the solution are negative or too large
        - rt: the natural language reasoning trace corresponding to the solution making the inconsistency error as specified



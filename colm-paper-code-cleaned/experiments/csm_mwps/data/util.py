import os
import re
import random

DATA_FOLDER = os.path.join("data")

def select_fewshot_examples(data, mwp_id, inst_id, shots, selection_strategy: str, file_data = None, miscon_selection_strategy: str = "all"):
    """ 
        Selects fewshot examples for an instantiation of an mwp that is part of data, 
        using one of the following selection_strategies:
        - MWP: choose first instantiation from other mwps (cannot select mwp_id)
        - INSTSA: choose from other instantiations of the same mwp_id (cannot select (mwp_id,inst_id)), ensuring the same errors are applied for all examples
        - INSTVA: choose from other instantiations of the same mwp_id (cannot select (mwp_id,inst_id)) but different errors may be applied between examples
        - FILE: choose first instantiation from other mwps of a different file (i.e. provided as file_data)

        - miscon_selection_strategy: how to select the misconception example 
            (all = where all applicable errors are done, random = where a random subset of errors is performed)

        Returns dict with ids, problems, rts, answers, distractor rts and distractor answers
    """
    assert not selection_strategy == "FILE" or file_data is not None, "file_data must be provided when sampling from a file"
    mwp_data = data[mwp_id]
    inst_data = data[mwp_id]["instantiations"][inst_id]

    if selection_strategy == "MWP":
        # sample n examples from other problems (always the first num-inst)
        example_ids = random.sample([(i, "0") for i in data.keys() if i != mwp_id], shots)
        example_data = [data[mid]["instantiations"][iid] for mid,iid in example_ids]
        example_problems = [e["problem"] for e in example_data]
        example_problems_cons = [e.get("cons_problem", None) for e in example_data]
        example_rts = [e["correct_answer"]["rt"] for e in example_data]
        example_answers = [e["correct_answer"]["answer"] for e in example_data]
        
        if miscon_selection_strategy == "all":
            miscon_answers = [sorted(e["misconception_answers"], key=(lambda x: len(x["misconceptions"])), reverse=True)[0] for e in example_data]
        elif miscon_selection_strategy == "random":
            miscon_answers = [random.sample(e["misconception_answers"]) for e in example_data]
        else:
            raise ValueError(f"Unknown misconception selection strategy {miscon_selection_strategy}! (options: all, random)")
        
        example_misconception_rts = [e["rt"] for e in miscon_answers]
        example_misconception_answers = [e["answer"] for e in miscon_answers]
    elif selection_strategy == "INSTSA":
        # sample n examples from other instantiations of the same problem, while keeping the application of misconceptions the same too
        assert miscon_selection_strategy == "all", "random miscon-selection not currently supported with INSTSA"
        example_ids = random.sample([(mwp_id,i) for i in mwp_data["instantiations"].keys() if i != inst_id], shots)
        example_data = [data[mid]["instantiations"][iid] for mid,iid in example_ids]
        example_problems = [e["problem"] for e in example_data]
        example_problems_cons = [e.get("cons_problem", None) for e in example_data]
        example_rts = [e["correct_answer"]["rt"] for e in example_data]
        example_answers = [e["correct_answer"]["answer"] for e in example_data]

        example_misconception_rts = []
        example_misconception_answers = []
        inst_miscon_answer = sorted(inst_data["misconception_answers"], key=(lambda x: len(x["misconceptions"])), reverse=True)[0]
        for e in example_data:
            appended = False
            for ma in e["misconception_answers"]:
                # ensure the application of the misconceptions is the same for all (mwp,inst)
                if ma["misconceptions"] == inst_miscon_answer["misconceptions"]: 
                    example_misconception_rts.append(ma["rt"])
                    example_misconception_answers.append(ma["answer"])
                    appended = True
                    break # only add one misconception answer per (mwp,inst)
            assert appended, "Could not find misconception answer with same misconceptions that are also applicable to problem (make sure all instantiations have the same applicable misconceptions!)"
    elif selection_strategy == "INSTVA":
        # sample n examples from other instantiations of the same problem where the application of misconceptions may vary
        assert miscon_selection_strategy == "all", "random miscon-selection not currently supported with INSTVA"
        example_ids = random.sample([(mwp_id,i) for i in mwp_data["instantiations"].keys() if i != inst_id], shots)
        example_data = [data[mid]["instantiations"][iid] for mid,iid in example_ids]
        example_problems = [e["problem"] for e in example_data]
        example_problems_cons = [e.get("cons_problem", None) for e in example_data]
        example_rts = [e["correct_answer"]["rt"] for e in example_data]
        example_answers = [e["correct_answer"]["answer"] for e in example_data]

        if miscon_selection_strategy == "all":
            miscon_answers = [sorted(e["misconception_answers"], key=(lambda x: len(x["misconceptions"])), reverse=True)[0] for e in example_data]
        elif miscon_selection_strategy == "random":
            miscon_answers = [random.sample(e["misconception_answers"]) for e in example_data]
        else:
            raise ValueError(f"Unknown misconception selection strategy {miscon_selection_strategy}! (options: all, random)")
        
        example_misconception_rts = [e["rt"] for e in miscon_answers]
        example_misconception_answers = [e["answer"] for e in miscon_answers]
    elif selection_strategy.startswith("FILE"):
        # sample n examples from a different file
        example_ids = random.sample([(i, "0") for i in file_data.keys()], shots)
        example_data = [file_data[mid]["instantiations"][iid] for mid,iid in example_ids]
        example_problems = [e["problem"] for e in example_data]
        example_problems_cons = [e.get("cons_problem", None) for e in example_data]
        example_rts = [e["correct_answer"]["rt"] for e in example_data]
        example_answers = [e["correct_answer"]["answer"] for e in example_data]
        
        if miscon_selection_strategy == "all":
            miscon_answers = [sorted(e["misconception_answers"], key=(lambda x: len(x["misconceptions"])), reverse=True)[0] for e in example_data]
        elif miscon_selection_strategy == "random":
            miscon_answers = [random.sample(e["misconception_answers"]) for e in example_data]
        else:
            raise ValueError(f"Unknown misconception selection strategy {miscon_selection_strategy}! (options: all, random)")
        
        example_misconception_rts = [e["rt"] for e in miscon_answers]
        example_misconception_answers = [e["answer"] for e in miscon_answers]
    else:
        raise ValueError(f"Unsupported example-selection-strategy: {selection_strategy}! Can be one of: (MWP, INST)")
    
    return {
        "ids": example_ids,
        "problems": example_problems,
        "consistent_problems": example_problems_cons,
        "rts": example_rts,
        "answers": example_answers,
        "distractor_rts": example_misconception_rts,
        "distractor_answers": example_misconception_answers
    }

def first_diff_eq_op_idx(s1, s2):
    """ Return the indices of the first character of the first equation that is different in both strings """
    equation_pattern = re.compile(r'\w+\s*[\+\-]\s*\w+\s*=\s*\w+')
    equations_s1 = equation_pattern.findall(s1)
    equations_s2 = equation_pattern.findall(s2)
    for eq1, eq2 in zip(equations_s1, equations_s2):
        if eq1 != eq2:
            for m1, m2 in zip(re.finditer(r'[\+\-]', eq1), re.finditer(r'[\+\-]', eq2)):
                if m1.group() != m2.group():
                    i1 = s1.index(eq1) + m1.start()
                    i2 = s2.index(eq2) + m2.start()
                    return i1, i2
    return len(s1), len(s2)

def first_diff_quant_idx(s1, s2):
    """ Return the indices of the end of the first number that is different in both strings """
    equation_pattern = re.compile(r'\w+\s*[\+\-]\s*\w+\s*=\s*\w+')
    equations_s1 = equation_pattern.findall(s1)
    equations_s2 = equation_pattern.findall(s2)
    for eq1, eq2 in zip(equations_s1, equations_s2):
        if eq1 != eq2:
            for m1, m2 in zip(re.finditer(r'\d+', eq1), re.finditer(r'\d+', eq2)):
                if m1.group() != m2.group():
                    i1 = s1.index(eq1) + m1.end()
                    i2 = s2.index(eq2) + m2.end()
                    return i1, i2    
    return len(s1), len(s2)
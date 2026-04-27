import json
import os

def load_json_to_dict(file_path: str):

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error: File at {file_path} is not valid JSON")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return {}

def retrieve_problem_data(problem_dict: dict, overlap_type: str = None, lexical_agent_overlap: str = None):
    body = []
    for sen in problem_dict['problembody']:
        body.append((sen['text'], sen['relevant']))
    query = problem_dict['question']['text']
    groundquery = problem_dict['groundquery']['text']
    rt = []
    for sen in problem_dict['rt']:
        rt.append((sen['text'], sen['relevant']))
    answer = problem_dict['answer']

    res = {'body': body, 'query': query, 'groundquery': groundquery, 'rt': rt, 'answer': answer}

    if overlap_type:
        res['overlap_type'] = overlap_type
    
    if lexical_agent_overlap:
        res['lexical_agent_overlap'] = lexical_agent_overlap

    return res

def format_problem(mwp: dict, ground: bool = False):
    body_text = " ".join([x for x, _ in mwp["body"]])
    query = mwp["groundquery"] if ground else mwp["query"]
    return body_text + " " + query

def format_cot(mwp: dict, ground: bool = False):
    """
    Uses the formatting for problems generated with MathGAP
    """
    import re

    sentences = re.split(r'(?<=[.!?])\s+', mwp['problem'].strip())
    body_text, question = " ".join(sentences[:-1]), sentences[-1]
    if ground:
        query = " Prove that " + mwp["answer_nl"]
    else:
        query = " " + question
    problem = body_text + query
    
    rt = re.split(r'(?<=[.!?])\s+', mwp["reasoning_trace"].strip())
    rt = "".join([f"{i+1}. {s}\n" for i, s in enumerate(rt)])

    return {"problem": problem, "cot": rt, "answer": mwp["answer"]}


def get_problems_by_type(file_path: str, problem_type: str = 'base', complexity: str = 'simple', instantiation: str = "no_overlap"):
    """
    type: 'base', 'connected', 'disconnected'
    complexity: 'simple', 'complex', 'more_complex'
    'no_overlap', 'entity_overlap', 'agent_overlap', 'agent_entity_overlap', 'control'
    """
    assert problem_type in ['base', 'connected', 'disconnected']
    assert complexity in ['simple', 'complex', 'more_complex']
    assert instantiation in ['no_overlap', 'entity_overlap', 'agent_overlap', 'agent_entity_overlap', 'control']

    problems = []

    dict = load_json_to_dict(file_path)

    if problem_type == 'base':
        for i in range(len(dict)):
            problems.append(retrieve_problem_data(dict[i][problem_type]))
    
    else:
        for i in range(len(dict)):
            problems.append(retrieve_problem_data(dict[i][problem_type][complexity][instantiation]))
    
    return problems
import gc
import click
from mathgap.properties import PropertyType
import numpy as np
import os
import re
import json
import torch
import random

import numpy as np
from tqdm import tqdm

from prompt_util import ask_to_solve, ask_for_distractor, get_logprobs, DETERMINISTIC_GENERATION, construct_ask_for_distractor_prompt_ci, construct_ask_to_solve_prompt
from data.util import select_fewshot_examples

from datageneration import *

def create_llm_config(model_id: str, library: str, quantize_4_bit: bool = False, max_new_tokens: int = 1024, max_model_len: int = None, **kwargs):
    """ 
        Parse the input parameters into a description of an LLM (e.g. load the model or create a client to query the model) 

        NOTE:
        - If you are using a gated hugginface repo, make sure to set the HF_TOKEN enviornment variable to a token that has read-access to gated repos
        - If you are using an azure openai model, make sure to set AZURE_OPENAI_KEY and AZURE_ENDPOINT
    """
    if library == "vllm":
        # VLLM models
        from transformers import AutoTokenizer
        from vllm import LLM

        quant_args = {
            "dtype": torch.bfloat16,
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes"
        } if quantize_4_bit else {}

        tokenizer = AutoTokenizer.from_pretrained(model_id) # still need the tokenizer for chat-templates
        
        if max_model_len is None:
            max_model_len = 8*max_new_tokens
        
        model = LLM(model=model_id, trust_remote_code=True, **quant_args, tensor_parallel_size=torch.cuda.device_count(), 
                    task="generate", max_model_len=max_model_len, **kwargs)
        
        print(f"Loaded vllm-model {model_id}...")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "library": library
        }
    elif library == "huggingface":
        # Huggingface Models
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        quant_conf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if quantize_4_bit else None


        # init the model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quant_conf)
        model.eval()
        print(f"Loaded huggingface-model {model_id} on {model.device}...")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "library": library
        }
    elif library == "unsloth":
        # Unsloth models
        from unsloth import FastLanguageModel
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_id,
            max_seq_length = 8*max_new_tokens,
            load_in_4bit = quantize_4_bit,
            device_map="auto"
        )
        model = FastLanguageModel.for_inference(model)
        
        print(f"Loaded unsloth-model {model_id} on {model.device}")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "library": library
        }
    elif library == "azureopenai":
        # Azure OpenAI models
        from openai import AzureOpenAI
        azure_endpoint = os.environ.get("AZURE_ENDPOINT")
        openai_client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_KEY"),
            api_version="2024-05-01-preview",
            azure_endpoint=azure_endpoint
        )
        print(f"Specified azure-oppenai-model {model_id} on {azure_endpoint}")

        return {
            "model": model_id,
            "openai_client": openai_client,
            "library": library
        }
    else:
        raise ValueError(f"Inference library {library} not supported!")

def extract_last_integer(text: str) -> None|int:
    numbers = [int(n) for n in re.findall(r'\d+', text)]
    return None if len(numbers) == 0 else numbers[-1]

def extract_first_integer(text: str) -> None|int:
    numbers = [int(n) for n in re.findall(r'\d+', text)]
    return None if len(numbers) == 0 else numbers[0]

@click.group()
def cli():
    pass

@cli.command()
@click.option("-n", "--numproblems", default=1000, help="The number of problems that should be generated")
@click.option("-i", "--numinst", default=5, help="The number of instantiations that should be generated for each mwp")
@click.option("-vi", "--varyinst", default="quantities", help="Different instantiations: quantities = only quantities are varied, all = all properties are varied")
@click.option("-vf", "--varyformulation", default=False, help="If true, different instantiations can be phrased differently")
@click.option("-va", "--varyapplicable", default=False, help="If true, different instantiations can have different applicable errors")
@click.option("-p", "--incons", default=0.1, help="The chance that conclusions will be chosen to become inconsistent (where applicable)")
@click.option("-f", "--file", default="data/datasets/distractors.json", help="Path to output file")
@click.option("--mindepth", default=0, help="The min depth the trees should have")
@click.option("--depthdecay", default=1.05, help="Base of exponentially decaying probability of continuing tree branches with depth")
@click.option("--partwholestartchance", default=0.33, help="Chance to start with a partwhole (not guaranteed to correspond to final mix bc we do rejection sampling)")
@click.option("--reducenonlinearity", default=True, help="Will apply techniques to reduce nonlinearity such as disallowing Containers of ContContComp to be expanded if the Comp is part of a ContCompCompeqCont")
@click.option("--nonlinearityuniformprob", default=16, help="Depth at which nonlinearity rules are equally likely as other rules")
@click.option("--offset", default=50, help="Offset for numbers (higher value reduces frequency of implausible distractors)")
@click.option("--leafmin", default=2, help="Minimum value for leaf nodes")
@click.option("--leafmax", default=25, help="Maximum value for leaf nodes (that are not offset)")
@click.option("--ruleset", default="full", help="Ruleset to be used to generate the proof tree, any of { full, comp, nocompeq, transpart }")
@click.option("--misconruleset", default="comp", help="Misconception ruleset to be used, any of { keyword, comp }")
@click.option("-opp", "--incloppert", is_flag=True, help="Whether to compute and include all op-pert answers (can be a lot for larger mwps as it grows exponentially)")
@click.option("-rt", "--inclrt", is_flag=True, help="Include the reasoning traces for the correct as well as all inconsistency-errors?")
@click.option("-cp", "--consprob", is_flag=True, help="Include the consistent formulation of the problem where no errors are applicable")
@click.option("-ona", "--omitnonapplicable", default=True, help="Omit problems where no misconception is applicable")
@click.option("-s", "--seed", default=14, help="The seed to be used")
def gen_ci_distractors(numproblems, numinst, varyinst, varyapplicable, varyformulation, incons, file, mindepth, depthdecay, 
                    partwholestartchance, reducenonlinearity, nonlinearityuniformprob, offset, leafmin, leafmax, 
                    ruleset, misconruleset, incloppert, inclrt, consprob, omitnonapplicable, seed):
    """ Generates a comparison inconcsistency dataset that contains multiple mwps, different instantiations for each one and a list of distractors choices """
    print("Generating dataset... (this might take a while)")
    rulesets = {
        "full": FULL_RULESET,
        "comp": COMP_RULESET,
        "nocompeq": NO_COMPEQ_RULESET,
        "transpart": TRANSFER_PARTWHOLE_RULESET
    }

    miscon_rulesets = {
        "keyword": FULL_MISCONCEPTIONS_BY_RULE_TYPE,
        "comp": COMP_MISCONCEPTIONS_BY_RULE_TYPE
    }

    cons_template_versions = {
        "keyword": "consistent_keywords",
        "comp": "consistent"
    } 
    
    incons_template_versions = {
        "keyword": "inconsistent_keywords",
        "comp": "inconsistent"
    } 

    vary_typesets = {
        "quantities": [PropertyType.QUANTITY],
        "all": [t for t in PropertyType]
    }

    dataset = generate_ci_problems_and_distractors(nr_problems=numproblems, nr_instantiations=numinst, vary_types=vary_typesets[varyinst], 
                vary_formulations=varyformulation, vary_applicable=varyapplicable, prob_misconcievable=incons, min_depth=mindepth, 
                expand_exp_decay_base=depthdecay, ruleset=rulesets[ruleset], miscon_ruleset=miscon_rulesets[misconruleset],
                cons_template_version=cons_template_versions[misconruleset], incons_template_version=incons_template_versions[misconruleset],
                partwhole_start_chance=partwholestartchance, reduce_nonlinearity=reducenonlinearity, 
                nonlinearity_uniform_prob_depth=nonlinearityuniformprob, offset=offset, leaf_min_value=leafmin, 
                leaf_max_value=leafmax, include_op_pert=incloppert, include_rt=inclrt, include_cons_form=consprob, 
                omit_non_applicable=omitnonapplicable, seed=seed)
    
    dir = os.path.dirname(file)
    if len(dir) > 0:
        os.makedirs(dir, exist_ok=True)
    
    with open(file, "w+") as f:
        json.dump(dataset, f)

    # print some stats
    num_mis_distr_grouped_by_inst = [[a["answer"] for a in mwp["instantiations"][i]["misconception_answers"]] for mwp in dataset.values() for i in mwp["instantiations"].keys()]
    all_mis_distr = [a["answer"] for mwp in dataset.values() for i in mwp["instantiations"].keys() for a in mwp["instantiations"][i]["misconception_answers"]]
    all_mis_distr_plausibilities = [a["plausible"] for mwp in dataset.values() for i in mwp["instantiations"].keys() for a in mwp["instantiations"][i]["misconception_answers"]]

    if len(all_mis_distr) > 0 and len(all_mis_distr_plausibilities) > 0:
        print("Some statistics of the generated dataset:")
        print(f"# total misconception distractors = {len(all_mis_distr)}")
        print(f"# unique misconception distractors = {len(np.unique(all_mis_distr))}")
        print(f"# misconception distractors < 2 = {len(list(filter(lambda x: x < 2, all_mis_distr)))}")
        print(f"# misconception distractors > 1000 = {len(list(filter(lambda x: x > 1000, all_mis_distr)))}")
        print(f"# fractional misconception distractors = {len(list(filter(lambda x: int(x) != x, all_mis_distr)))}")
        print(f"min misconception distractors = {min(all_mis_distr)}")
        print(f"max misconception distractors = {max(all_mis_distr)}")
        print(f"# instantiations = {len(num_mis_distr_grouped_by_inst)}")
        print(f"# instantiations with at least one plausible distractor = {len(list(filter(lambda x: any(a >= 2 and int(a) == a for a in x), num_mis_distr_grouped_by_inst)))}")
        print(f"# plausible misconception distractors = {len([p for p in all_mis_distr_plausibilities if p])}")
        print(f"# implausible misconception distractors = {len([p for p in all_mis_distr_plausibilities if not p])}")
        print(f"# partwhole mwps = {len([dataset[mwp_id] for mwp_id in dataset if dataset[mwp_id]['metadata']['rule_count'].get('ContPartWhole', 0) > 0])}")

@cli.command()
@click.option("-n", "--numproblems", default=1000, help="The number of problems that should be generated")
@click.option("-i", "--numinst", default=5, help="The number of instantiations that should be generated for each mwp")
@click.option("-vi", "--varyinst", default="quantities", help="Different instantiations: quantities = only quantities are varied, all = all properties are varied")
@click.option("-vf", "--varyformulation", default=False, help="If true, different instantiations can be phrased differently")
@click.option("-f", "--file", default="data/datasets/distractors.json", help="Path to output file")
@click.option("--mindepth", default=0, help="The min depth the trees should have")
@click.option("--depthdecay", default=1.05, help="Base of exponentially decaying probability of continuing tree branches with depth")
@click.option("--partwholestartchance", default=0.33, help="Chance to start with a partwhole (not guaranteed to correspond to final mix bc we do rejection sampling)")
@click.option("--reducenonlinearity", default=True, help="Will apply techniques to reduce nonlinearity such as disallowing Containers of ContContComp to be expanded if the Comp is part of a ContCompCompeqCont")
@click.option("--nonlinearityuniformprob", default=16, help="Depth at which nonlinearity rules are equally likely as other rules")
@click.option("--offset", default=50, help="Offset for numbers (higher value reduces frequency of implausible distractors)")
@click.option("--leafmin", default=10, help="Minimum value for leaf nodes")
@click.option("--leafmax", default=25, help="Maximum value for leaf nodes (that are not offset)")
@click.option("-rt", "--inclrt", is_flag=True, help="Include the reasoning traces for the correct as well as all inconsistency-errors?")
@click.option("-s", "--seed", default=14, help="The seed to be used")
def gen_am_distractors(numproblems, numinst, varyinst, varyformulation, file, mindepth, depthdecay, 
                    partwholestartchance, reducenonlinearity, nonlinearityuniformprob, offset, leafmin, leafmax, inclrt, seed):
    """ Generates a arithmetic misconception dataset that contains multiple mwps, different instantiations for each one and a list of distractors choices """
    print("Generating dataset... (this might take a while)")

    all_err_mental_models = ALL_ERRONEOUS_MENTAL_MODELS
    ruleset_factory = AM_NO_COMPEQ_RULESET
    template_version = "consistent"
    vary_typesets = {
        "quantities": [PropertyType.QUANTITY],
        "all": [t for t in PropertyType]
    }

    dataset = generate_am_problems_and_distractors(nr_problems=numproblems, nr_instantiations=numinst, vary_types=vary_typesets[varyinst], 
                vary_formulations=varyformulation, all_err_mental_models=all_err_mental_models, min_depth=mindepth, 
                expand_exp_decay_base=depthdecay, ruleset_factory=ruleset_factory,
                template_version=template_version, partwhole_start_chance=partwholestartchance, reduce_nonlinearity=reducenonlinearity, 
                nonlinearity_uniform_prob_depth=nonlinearityuniformprob, offset=offset, leaf_min_value=leafmin, 
                leaf_max_value=leafmax, include_rt=inclrt, seed=seed)
    
    dir = os.path.dirname(file)
    if len(dir) > 0:
        os.makedirs(dir, exist_ok=True)
    
    with open(file, "w+") as f:
        json.dump(dataset, f)

    # print some stats
    num_mis_distr_grouped_by_inst = [[a["answer"] for a in mwp["instantiations"][i]["misconception_answers"]] for mwp in dataset.values() for i in mwp["instantiations"].keys()]
    all_mis_distr = [a["answer"] for mwp in dataset.values() for i in mwp["instantiations"].keys() for a in mwp["instantiations"][i]["misconception_answers"]]

    if len(all_mis_distr) > 0:
        print("Some statistics of the generated dataset:")
        print(f"# total misconception distractors = {len(all_mis_distr)}")
        print(f"# unique misconception distractors = {len(np.unique(all_mis_distr))}")
        print(f"# misconception distractors < 2 = {len(list(filter(lambda x: x < 2, all_mis_distr)))}")
        print(f"# misconception distractors > 1000 = {len(list(filter(lambda x: x > 1000, all_mis_distr)))}")
        print(f"# fractional misconception distractors = {len(list(filter(lambda x: int(x) != x, all_mis_distr)))}")
        print(f"min misconception distractors = {min(all_mis_distr)}")
        print(f"max misconception distractors = {max(all_mis_distr)}")
        print(f"# instantiations = {len(num_mis_distr_grouped_by_inst)}")
        print(f"# instantiations with at least one plausible distractor = {len(list(filter(lambda x: any(a >= 2 and int(a) == a for a in x), num_mis_distr_grouped_by_inst)))}")
        print(f"# partwhole mwps = {len([dataset[mwp_id] for mwp_id in dataset if dataset[mwp_id]['metadata']['rule_count'].get('ContPartWhole', 0) > 0])}")

@cli.command()
@click.option("-f", "--file", default="data/datasets/sol/distractors.json", help="Path to output file")
@click.option("-o", "--outfile", default="data/datasets/sol/distractors_sol.json", help="Path to output file")
@click.option("-m", "--model", default=None, help="Name of model")
@click.option("-lib", "--library", default="vllm", help="Name of inference library (vllm, huggingface, unsloth, azureopenai)")
@click.option("-q4", "--quant4", is_flag=True, help="Quantize the model to 4 bit (only for non-API models)")
@click.option("-s", "--shots", multiple=True, default=(0,5), help="List specifying with how many fewshot examples the model should be prompted")
@click.option("-es", "--examplestrategy", default="FILE", help="How to pick the few-shot examples: FILE = pick from a different file")
@click.option("-ef", "--examplesfile", default=None, help="File from which the examples should be picke in case examplestrategy = FILE")
@click.option("-t", "--maxnewtokens", default=2048, help="Number of new tokens that can be generated during CoT steps (solving and distractor)")
@click.option("-ct", "--chattemplates", default=True, help="Use chat-templates / model fewshot as conversation")
@click.option("-det", "--deterministic", default=True, help="Do inference in deterministic way")
@click.option("-cot", "--chainofthought", default=True, help="Use chain of thought prompting ('Let's think step by step.') for both solving and distractor generation")
def solutions(file, outfile, model, library, quant4, shots, examplestrategy, examplesfile, maxnewtokens, chattemplates, deterministic, chainofthought):
    """
        Runs a model to solve multiple instantiations of MWPs on
        1) the consistent formulation
        2) the inconsistent formulation
    """
    max_model_len = 4*maxnewtokens
    if max(shots) >= 10:
        max_model_len = 16*maxnewtokens
    elif max(shots) >= 5:
        max_model_len = 8*maxnewtokens

    llm_config = create_llm_config(model_id=model, library=library, quantize_4_bit=quant4, max_new_tokens=maxnewtokens, max_model_len=max_model_len)
    llm_param_fn = DETERMINISTIC_GENERATION if deterministic else lambda tokenizer: { "do_sample": True, "top_p": 1.0, "temperature": 0.7, "pad_token_id": tokenizer.eos_token_id }
    
    # Load the dataset
    with open(file, "r+") as f:
        data = json.load(f)
        print(f"Loaded dataset of {len(data)} MWPs")

    example_file_data = None
    if examplestrategy == "FILE":
        with open(examplesfile) as f:
            example_file_data = json.load(f)

    # for each mwp
    index = 0
    for mwp_id in tqdm(data):
        index += 1
        mwp_data = data[mwp_id]

        # for the instantiations of the mwp that we want to run
        for inst_id in sorted(list(mwp_data["instantiations"].keys())):
            inst_data = mwp_data["instantiations"][inst_id]
            problem_con = inst_data["cons_problem"]
            problem_incon = inst_data["problem"]

            assert problem_con is not None, "Consistent problem is none. Did you forget to include them when generating the datasets?"

            inst_data["llm_answer"] = {}
            inst_data["cons_llm_answer"] = {}

            for s in shots:
                llm_answer_con = None
                llm_answer_incon = None

                # case: 0-shot
                if s == 0:
                    llm_answer_con = ask_to_solve(problem_con, llm_config, max_new_tokens=maxnewtokens, 
                                                    llm_param_fn=llm_param_fn, use_cot=chainofthought, 
                                                    use_chat_template=chattemplates, use_rts=True)
                            
                    llm_answer_incon = ask_to_solve(problem_incon, llm_config, max_new_tokens=maxnewtokens, 
                                                    llm_param_fn=llm_param_fn, use_cot=chainofthought, 
                                                    use_chat_template=chattemplates, use_rts=True)
                # case: fewshot
                else:
                    # 1. select as many few-shot examples as will be needed
                    examples = select_fewshot_examples(data, mwp_id, inst_id, s, examplestrategy, example_file_data)
            
                    # 2. prompt the model
                    llm_answer_con = ask_to_solve(problem_con, llm_config, 
                        example_problems=examples["problems"], example_rts=examples["rts"], example_answers=examples["answers"], 
                        max_new_tokens=maxnewtokens, llm_param_fn=llm_param_fn, use_cot=chainofthought, 
                        use_chat_template=chattemplates, use_rts=True)
                    llm_answer_con["example_ids"] = examples["ids"] # storing exactly which (mwp, inst)s have been used as examples

                    llm_answer_incon = ask_to_solve(problem_incon, llm_config, 
                        example_problems=examples["problems"], example_rts=examples["rts"], example_answers=examples["answers"], 
                        max_new_tokens=maxnewtokens, llm_param_fn=llm_param_fn, use_cot=chainofthought, 
                        use_chat_template=chattemplates, use_rts=True)
                    llm_answer_incon["example_ids"] = examples["ids"] # storing exactly which (mwp, inst)s have been used as examples
                
                # log outputs
                inst_data["cons_llm_answer"][s] = llm_answer_con
                inst_data["llm_answer"][s] = llm_answer_incon

        # save every 100 mwps
        if index % 100 == 0:
            with open(outfile, "w+") as f:
                json.dump(data, f)
        
    # final save
    with open(outfile, "w+") as f:
        json.dump(data, f)

    # cleanup
    if llm_config.get("tokenizer", None) is not None:
        del llm_config["tokenizer"]
    if llm_config.get("model", None) is not None:
        del llm_config["model"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@cli.command()
@click.option("-cat", "--category", default="ci", help="Misconception category that should be considered (ci = comparison inconsistency errors, am = arithmetic mistakes)")
@click.option("-f", "--file", default="data/datasets/ci/icl/distractors.json", help="Path to output file")
@click.option("-o", "--outfile", default="data/datasets/ci/icl/distractors_icl.json", help="Path to output file")
@click.option("-m", "--model", default=None, help="Name of model")
@click.option("-lib", "--library", default="vllm", help="Name of inference library (vllm, huggingface, unsloth, azureopenai)")
@click.option("-q4", "--quant4", is_flag=True, help="Quantize the model to 4 bit (only for non-API models)")
@click.option("-t", "--maxnewtokens", default=2048, help="Number of new tokens that can be generated during CoT steps (solving and distractor)")
@click.option("-s", "--shots", multiple=True, default=(0,5), help="List specifying with how many fewshot examples the model should be prompted")
@click.option("-es", "--examplestrategy", default="MWP", help="How to pick the few-shot examples: MWP = from other mwps, INSTSA = pick from instantiations (same applicable error), INSTVA = pick from instantiations (different applicable errors)")
@click.option("-ct", "--chattemplates", default=True, help="Use chat-templates / model fewshot as conversation")
@click.option("-tsm", "--tasksystemmessage", default=True, help="Put the task as system message")
@click.option("-det", "--deterministic", default=True, help="Do inference in deterministic way")
@click.option("-cot", "--chainofthought", default=True, help="Use chain of thought prompting ('Let's think step by step.') for both solving and distractor generation")
@click.option("-rt", "--reasoningtraces", default=True, help="Show the reasoning traces in few-shot examples")
@click.option("-rfi", "--runforinsts", default=1, help="Run for how many instantiations of each mwp?")
@click.option("-rtfe", "--rtinclfirsterr", default=False, help="Show the reasoning trace including the first error (i.e. put - instead of + or vice versa and ask to continue)")
@click.option("-hle", "--highlvlerr", default=False, help="Provide a high-level error message of the students misconception")
@click.option("-pp", "--perplexity", is_flag=True, help="Compute the perplexity of all generated output")
@click.option("-dgo", "--distgenonly", is_flag=True, help="Only do distractor generation")
def icl(category, file, outfile, model, library, quant4, maxnewtokens, shots, examplestrategy, chattemplates, tasksystemmessage, deterministic, chainofthought, reasoningtraces, runforinsts, rtinclfirsterr, highlvlerr, perplexity, distgenonly):
    """
        Runs an LLM to perform the following two tasks (with and without ICL):
        1) Correctly solve an MWP
        2) Generate a conceptual distractor for the MWP
    """
    # scale max_model_len (limit on nr of tokens of prompt + output) with max number of shots
    max_model_len = 4*maxnewtokens
    if max(shots) >= 10:
        max_model_len = 16*maxnewtokens
    elif max(shots) >= 5:
        max_model_len = 8*maxnewtokens

    llm_config = create_llm_config(model_id=model, library=library, quantize_4_bit=quant4, max_new_tokens=maxnewtokens, max_model_len=max_model_len)
    llm_param_fn = DETERMINISTIC_GENERATION if deterministic else lambda tokenizer: { "do_sample": True, "top_p": 1.0, "temperature": 0.7, "pad_token_id": tokenizer.eos_token_id }
    
    # Load the dataset
    with open(file, "r+") as f:
        data = json.load(f)
        print(f"Loaded dataset of {len(data)} MWPs")

    # for each mwp
    index = 0
    for mwp_id in tqdm(data):
        index += 1
        mwp_data = data[mwp_id]

        # for the instantiations of the mwp that we want to run
        for inst_id in sorted(list(mwp_data["instantiations"].keys()))[:runforinsts]:
            inst_data = mwp_data["instantiations"][inst_id]
            problem = inst_data["problem"]
            correct_answer = inst_data["correct_answer"]["answer"]
            correct_rt = inst_data["correct_answer"]["rt"]
            distractor_idx = random.choice(range(len(inst_data["misconception_answers"])))
            distractor_rt = inst_data["misconception_answers"][distractor_idx]["rt"]
        
            inst_data["llm_answer"] = {}
            inst_data["llm_distractor"] = {}

            # foreach number of shots that we want to try
            for s in shots:
                llm_answer = None
                llm_distractor = None

                # case: 0-shot
                if s == 0:
                    llm_answer = None
                    if not distgenonly:
                        llm_answer = ask_to_solve(problem, llm_config, max_new_tokens=maxnewtokens, 
                                                llm_param_fn=llm_param_fn, use_cot=chainofthought, 
                                                use_rts=reasoningtraces, compute_perplexity=perplexity)
                    llm_distractor = ask_for_distractor(problem, correct_answer, llm_config, miscon_category=category, max_new_tokens=maxnewtokens, 
                                                        use_chat_template=chattemplates, task_as_system_message=tasksystemmessage, 
                                                        llm_param_fn=llm_param_fn, use_cot=chainofthought, incl_highlvl_err=highlvlerr,
                                                        use_rts=reasoningtraces, compute_perplexity=perplexity,
                                                        show_rt_inlc_first_error=rtinclfirsterr, distractor_rt=distractor_rt, corr_rt=correct_rt)
                # case: fewshot
                else:
                    # 1. select as many few-shot examples as will be needed
                    examples = select_fewshot_examples(data, mwp_id, inst_id, s, examplestrategy)
            
                    # 2. prompt the model
                    llm_answer = None
                    if not distgenonly:
                        llm_answer = ask_to_solve(problem, llm_config, 
                            example_problems=examples["problems"], example_rts=examples["rts"], example_answers=examples["answers"], 
                            max_new_tokens=maxnewtokens, llm_param_fn=llm_param_fn, use_cot=chainofthought, 
                            use_rts=reasoningtraces, compute_perplexity=perplexity)
                        llm_answer["example_ids"] = examples["ids"] # storing exactly which (mwp, inst)s have been used as examples
                    
                    llm_distractor = ask_for_distractor(problem, correct_answer, llm_config, miscon_category=category,
                        example_problems=examples["problems"], example_answers=examples["answers"], 
                        example_error_rts=examples["distractor_rts"], example_error_answers=examples["distractor_answers"], 
                        max_new_tokens=maxnewtokens, use_chat_template=chattemplates, task_as_system_message=tasksystemmessage, 
                        llm_param_fn=llm_param_fn, use_cot=chainofthought, incl_highlvl_err=highlvlerr, use_rts=reasoningtraces, compute_perplexity=perplexity,
                        show_rt_inlc_first_error=rtinclfirsterr, example_correct_rts=examples["rts"], distractor_rt=distractor_rt, corr_rt=correct_rt)
                    llm_distractor["example_ids"] = examples["ids"]

                # log outputs
                llm_distractor["example_distr_id"] = distractor_idx
                inst_data["llm_answer"][s] = llm_answer
                inst_data["llm_distractor"][s] = llm_distractor
            
        # save every 100
        if index % 100 == 0:
            with open(outfile, "w+") as f:
                json.dump(data, f)

    # final save
    with open(outfile, "w+") as f:
        json.dump(data, f)

    # cleanup
    if llm_config.get("tokenizer", None) is not None:
        del llm_config["tokenizer"]
    if llm_config.get("model", None) is not None:
        del llm_config["model"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@cli.command()
@click.option("-f", "--file", default="data/datasets/perplexity/distractors.json", help="Path to output file")
@click.option("-o", "--outfile", default="data/datasets/perplexity/distractors_logprobs.json", help="Path to output file")
@click.option("-m", "--model", default=None, help="Name of model")
@click.option("-lib", "--library", default="huggingface", help="Name of inference library (vllm, huggingface, unsloth, azureopenai)")
@click.option("-q4", "--quant4", is_flag=True, help="Quantize the model to 4 bit (only for non-API models)")
@click.option("-s", "--shots", multiple=True, default=(0,1,5), help="List specifying with how many fewshot examples the model should be prompted")
@click.option("-es", "--examplestrategy", default="MWP", help="How to pick the few-shot examples: MWP = from other mwps, INST = pick from instantiations")
@click.option("-cot", "--chainofthought", default=True, help="Use chain of thought prompting ('Let's think step by step.') for both solving and distractor generation")
@click.option("-hle", "--highlvlerr", default=False, help="Provide a high-level error message of the students misconception")
@click.option("-rfi", "--runforinsts", default=1, help="Run for how many instantiations of each mwp?")
def logprobs(file, outfile, model, library, quant4, shots, examplestrategy, chainofthought, highlvlerr, runforinsts):
    # TODO: add support for am-task
    """
        Computes the logprobs for:
        1) the correct reasoning trace
        2) one distractor reasoning trace
        3) the correct reasoning trace given the "solve-task-prompt" and the number of examples
        4) the correct reasoning trace given the "distractor-task-prompt" and the number of examples
        5) one distractor reasoning trace given the "correct-task-prompt" and the number of examples
        6) one distractor reasoning trace given the "distractor-task-prompt" and the number of examples
        

        TODO: maybe expand with options
        - 5) if available: the generated solution reasoning trace
        - 6) if available: the generated distractor reasoning trace
    """
    maxnewtokens = 1024
    top_k = 3

    max_model_len = 4*maxnewtokens
    if max(shots) >= 10:
        max_model_len = 16*maxnewtokens
    elif max(shots) >= 5:
        max_model_len = 8*maxnewtokens

    # set memory utilization low bc generating the logprobs will consume additional memory
    llm_config = create_llm_config(model_id=model, library=library, quantize_4_bit=quant4, max_new_tokens=1, max_model_len=max_model_len, gpu_memory_utilization=0.6)
    
    # Load the dataset
    with open(file, "r+") as f:
        data = json.load(f)
        print(f"Loaded dataset of {len(data)} MWPs")

    # for each mwp
    for mwp_id in tqdm(data):
        mwp_data = data[mwp_id]

        # for the instantiations of the mwp that we want to run
        for inst_id in sorted(list(mwp_data["instantiations"].keys()))[:runforinsts]:
            inst_data = mwp_data["instantiations"][inst_id]
            problem = inst_data["problem"]
            correct_answer = inst_data["correct_answer"]["answer"]
            correct_rt = inst_data["correct_answer"]["rt"]
            selectable_miscons = [(i,ma["rt"]) for i,ma in enumerate(inst_data["misconception_answers"]) if ma["plausible"]]
            
            if len(selectable_miscons) == 0: 
                print(f"Warning: No plausible misconception for mwp-id={mwp_id}, inst-id={inst_id}. Values = {[ma['answer'] for ma in inst_data['misconception_answers']]}")
                continue

            miscon_rt_idx, miscon_rt = random.choice(selectable_miscons)
            
            # no-context
            corr_cctx_prompt, corr_cctx_full_prompt, corr_cctx_logprobs = get_logprobs([{"content": correct_rt}], llm_conf=llm_config, use_chat_template=False, top_k=top_k)            
            dist_dctx_prompt, dist_dctx_full_prompt, dist_dctx_logprobs = get_logprobs([{"content": miscon_rt}], llm_conf=llm_config, use_chat_template=False, top_k=top_k)
            inst_data["correct_answer"]["logprobs"] = {
                "no_context": { "context": corr_cctx_prompt, "prompt": corr_cctx_full_prompt, "logprobs": corr_cctx_logprobs }
            }
            inst_data["misconception_answers"][miscon_rt_idx]["logprobs"] = {
                "no_context": { "context": dist_dctx_prompt, "prompt": dist_dctx_full_prompt, "logprobs": dist_dctx_logprobs }
            }

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # prompt and fewshot
            for s in shots:
                inst_data["correct_answer"]["logprobs"]["fewshot"] = inst_data["correct_answer"]["logprobs"].get("fewshot", {})
                inst_data["misconception_answers"][miscon_rt_idx]["logprobs"]["fewshot"] = inst_data["misconception_answers"][miscon_rt_idx]["logprobs"].get("fewshot", {})

                # case: 0-shot
                if s == 0:
                    solve_context = construct_ask_to_solve_prompt(problem, use_chat_template=True, use_cot=chainofthought, use_rts=True)
                    distractor_context = construct_ask_for_distractor_prompt_ci(problem, correct_answer, 
                                                        use_chat_template=True, task_as_system_message=True, 
                                                        incl_highlvl_err=highlvlerr, use_cot=chainofthought, use_rts=True)
                    
                    corr_cctx_prompt, corr_cctx_full_prompt, corr_cctx_logprobs = get_logprobs([{"role": "assistant", "content": correct_rt}], context_messages=solve_context, llm_conf=llm_config, use_chat_template=True, top_k=top_k)
                    corr_dctx_prompt, corr_dctx_full_prompt, corr_dctx_logprobs = get_logprobs([{"role": "assistant", "content": correct_rt}], context_messages=distractor_context, llm_conf=llm_config, use_chat_template=True, top_k=top_k)
                    dist_cctx_prompt, dist_cctx_full_prompt, dist_cctx_logprobs = get_logprobs([{"role": "assistant", "content": miscon_rt}], context_messages=solve_context, llm_conf=llm_config, use_chat_template=True, top_k=top_k)
                    dist_dctx_prompt, dist_dctx_full_prompt, dist_dctx_logprobs = get_logprobs([{"role": "assistant", "content": miscon_rt}], context_messages=distractor_context, llm_conf=llm_config, use_chat_template=True, top_k=top_k)

                    inst_data["correct_answer"]["logprobs"]["fewshot"][s] = {
                        "solve_context": { "context": corr_cctx_prompt, "prompt": corr_cctx_full_prompt, "logprobs": corr_cctx_logprobs },
                        "distractor_context": { "context": corr_dctx_prompt, "prompt": corr_dctx_full_prompt, "logprobs": corr_dctx_logprobs }
                    }
                    inst_data["misconception_answers"][miscon_rt_idx]["logprobs"]["fewshot"][s] = {
                        "solve_context": { "context": dist_cctx_prompt, "prompt": dist_cctx_full_prompt, "logprobs": dist_cctx_logprobs },
                        "distractor_context": { "context": dist_dctx_prompt, "prompt": dist_dctx_full_prompt, "logprobs": dist_dctx_logprobs }
                    }
                # case: fewshot
                else:
                    examples = select_fewshot_examples(data, mwp_id, inst_id, s, selection_strategy=examplestrategy)

                    solve_context = construct_ask_to_solve_prompt(problem, examples["problems"], examples["rts"], examples["answers"],
                                                                  use_chat_template=True, use_cot=chainofthought, use_rts=True)
                    distractor_context = construct_ask_for_distractor_prompt_ci(problem, correct_answer, 
                                                        examples["problems"], examples["answers"], examples["distractor_rts"], examples["distractor_answers"],
                                                        use_chat_template=True, task_as_system_message=True, 
                                                        incl_highlvl_err=highlvlerr, use_cot=chainofthought, use_rts=True)
                    
                    corr_cctx_prompt, corr_cctx_full_prompt, corr_cctx_logprobs = get_logprobs([{"role": "assistant", "content": correct_rt}], context_messages=solve_context, llm_conf=llm_config, use_chat_template=True, top_k=top_k)
                    corr_dctx_prompt, corr_dctx_full_prompt, corr_dctx_logprobs = get_logprobs([{"role": "assistant", "content": correct_rt}], context_messages=distractor_context, llm_conf=llm_config, use_chat_template=True, top_k=top_k)
                    dist_cctx_prompt, dist_cctx_full_prompt, dist_cctx_logprobs = get_logprobs([{"role": "assistant", "content": miscon_rt}], context_messages=solve_context, llm_conf=llm_config, use_chat_template=True, top_k=top_k)
                    dist_dctx_prompt, dist_dctx_full_prompt, dist_dctx_logprobs = get_logprobs([{"role": "assistant", "content": miscon_rt}], context_messages=distractor_context, llm_conf=llm_config, use_chat_template=True, top_k=top_k)

                    inst_data["correct_answer"]["logprobs"]["fewshot"][s] = {
                        "solve_context": { "context": corr_cctx_prompt, "prompt": corr_cctx_full_prompt, "logprobs": corr_cctx_logprobs, "example_ids": examples["ids"] },
                        "distractor_context": { "context": corr_dctx_prompt, "prompt": corr_dctx_full_prompt, "logprobs": corr_dctx_logprobs, "example_ids": examples["ids"] }
                    }
                    inst_data["misconception_answers"][miscon_rt_idx]["logprobs"]["fewshot"][s] = {
                        "solve_context": { "context": dist_cctx_prompt, "prompt": dist_cctx_full_prompt, "logprobs": dist_cctx_logprobs, "example_ids": examples["ids"] },
                        "distractor_context": { "context": dist_dctx_prompt, "prompt": dist_dctx_full_prompt, "logprobs": dist_dctx_logprobs, "example_ids": examples["ids"] }
                    }

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # save every mwp
        with open(outfile, "w+") as f:
            json.dump(data, f)


    # cleanup
    if llm_config.get("tokenizer", None) is not None:
        del llm_config["tokenizer"]
    if llm_config.get("model", None) is not None:
        del llm_config["model"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    cli()
import click
from tqdm import tqdm
import csv
import re

from data_processing import *

import wandb
import vllm
import torch

@click.group()
def cli():
    pass

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def create_llm_config(model_id: str, library: str = "vllm", **kwargs):
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

        tokenizer = AutoTokenizer.from_pretrained(model_id) # still need the tokenizer for chat-templates
        
        model = LLM(model=model_id, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count(), swap_space=1, task="generate", **kwargs)
        
        print(f"Loaded vllm-model {model_id}...")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "library": library
        }
    elif library == "huggingface":
        # Huggingface Models
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


        # init the model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        model.eval()
        print(f"Loaded huggingface-model {model_id} on {model.device}...")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "library": library
        }
    elif library == "openai": # api model
        from openai import OpenAI

        client = OpenAI(base_url="https://openrouter.ai/api/v1", 
                        api_key="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")
    
        return {
            "model": model_id,
            "tokenizer": client,
            "library": library
        }

    else:
        raise ValueError(f"Inference library {library} not supported!")

def get_llm_response(messages, text, llm_conf, max_new_tokens):
    """
    Chain-of-thought inference using the two-stage prompt method proposed by Kojima et al. (2022)
    """

    messages.append({"role": "user", "content": f"Q: {text}\n"})
    messages.append({"role": "assistant", "content": "A: Let's think step by step.\n"})

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prompt1, decoded_answer1 = prompt_and_decode(messages, llm_conf, max_new_tokens=max_new_tokens)

    # Step 2: ask for formatted output
    answer_prompt = "\nTherefore, the answer (arabic numerals) is "
    messages.append({"role": "assistant", "content": decoded_answer1 + answer_prompt})

    _, decoded_answer2 = prompt_and_decode(messages, llm_conf, max_new_tokens=5)
    
    return {
        "reasoning_prompt": prompt1, 
        "reasoning": decoded_answer1,
        "answer_prompt": answer_prompt,
        "answer": decoded_answer2,
    }

def prompt_and_decode(messages, llm_conf, max_new_tokens):
    """ 
        Prompts an llm with some messages (provide a single message if you want to use no chat-templates).
        Returns the prompt and answer
    """
    decoded_answer = None
    prompt = None

    library = llm_conf["library"]
    if library == "vllm":
        # vllm
        from vllm import SamplingParams
        tokenizer = llm_conf["tokenizer"]
        model = llm_conf["model"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)

        # convert hf params to vllm params
        sampling_params = None
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            skip_special_tokens=True,
        )

        # inference
        output = model.generate([prompt], sampling_params, use_tqdm=False)
        out = output[0].outputs[0]
        decoded_answer = out.text
    elif library == "huggingface" or library == "unsloth":
        # huggingface and unsloth
        tokenizer = llm_conf["tokenizer"]
        model = llm_conf["model"]

        if model.name_or_path == "facebook/opt-125m":
            prompt = "hello my name is"
            max_new_tokens = 3
        else:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # inference
        model_input = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            len_prompt = model_input['input_ids'].shape[1]
            output = model.generate(**model_input, max_new_tokens=max_new_tokens, do_sample=False)[0,:]
            decoded_answer = tokenizer.decode(output[len_prompt:], skip_special_tokens=True)
        
    elif library == "openai":
        client = llm_conf["tokenizer"]
        model = llm_conf["model"]
        prompt = build_openai_prompt(messages)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=0,
            max_tokens=max_new_tokens
        )
        decoded_answer = response.choices[0].message.content

    else:
        raise ValueError(f"Inference library {library} not supported!")
        
    return prompt, decoded_answer

def build_openai_prompt(messages):
    prompt = ""
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        prompt += f"{role}: {content}\n"
    return prompt

@cli.command()
@click.option("-o", "--out-path", default="experiments/proof_search/experiments/results/", type=str, help="Where the results will be stored")
@click.option("-i", "--in-path", default="experiments/proof_search/experiments/data/test500.json", type=str, help="Input problems")
@click.option("-icl", "--icl-in-path", default="experiments/proof_search/experiments/data/icl.csv", type=str, help="Input ICL problems")
@click.option("--model", default="Qwen/Qwen2.5-Math-7B-Instruct", type=str, help="Language model to run")
@click.option("-type", "--problem-type", default="disconnected", type=str, help="The type of problem set to evaluate")
@click.option("--complexity", default="more_complex", type=str, help="The complexity of problem set to evaluate")
@click.option("--instantiation", default="no_overlap", type=str, help="The type of problem set to evaluate")
@click.option("--ground", default=False, type=bool, help="The type of query")
@click.option("-length", "--max-new-tokens", default=4000, type=int, help="Limit on the context length")
def eval(out_path: str, in_path: str, icl_in_path: str, model: str, problem_type: str = 'disconnected', complexity: str = 'more_complex', instantiation: str = "no_overlap", 
         ground: bool = False, max_new_tokens: int = 4000, use_wandb: bool = True):

    # load problems
    problems = get_problems_by_type(in_path, problem_type, complexity, instantiation)

    # get LLM config
    if model == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        llm_config = create_llm_config(model_id=model, library="vllm")
    elif model in ["Qwen/Qwen2.5-Math-7B-Instruct", "Qwen/QwQ-32B"]:
        max_new_tokens = 2500 if max_new_tokens > 2500 else max_new_tokens # max context length for this model
        llm_config = create_llm_config(model_id=model, library="vllm")
    elif model == "facebook/opt-125m": # only used for local debugging
        llm_config = create_llm_config(model_id=model, library="huggingface")
        problems = problems[:10]
    elif model == "deepseek/deepseek-r1":
        max_new_tokens = 4000 if max_new_tokens > 4000 else max_new_tokens # reduce the cost
        llm_config = create_llm_config(model_id=model, library="openai")
    elif model == "qwen/qwq-32b":
        llm_config = create_llm_config(model_id=model, library="openai")
    else:
        raise ValueError(f"Model {model} not supported!")
    
    # prepare prompt
    messages = []
    system_prompt = "You are a helpful assistant tasked with solving math word problems. You follow the formatting of the problems and the solutions as given in the examples below.\n"
    messages.append({"role": "system", "content": system_prompt})
    if icl_in_path is not None:

        with open(icl_in_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            icl_data = [format_cot(row, ground) for row in reader]

        # do fewshot in-context learning
        for example in icl_data:
            messages.append({"role": "user", "content": f"Q: {example['problem']}\n"})
            messages.append({"role": "assistant", "content": f"A: Let's think step by step.\n{example['cot']}\nTherefore, the answer (arabic numerals) is {example['answer']}.\n"})
    print("Prepared prompt")

    file_name = f"{problem_type}_{complexity}_{instantiation}_ground" if ground else f"{problem_type}_{complexity}_{instantiation}_nonground"
    if use_wandb:
        wandb.login(key="769ab3ea3c509f66b3743d12bef5d8ef3ab6f6bc")
        wandb.init(project="math-proof-search", name=file_name)
        wandb.config.update({
        "model": model.split("/")[-1],
        })
    
    outputs = []
    accuracies = []
    print("Starting inference...")
    for i, mwp in tqdm(enumerate(problems)):

        text = format_problem(mwp, ground)
        
        # prompt the LM with problem
        res = get_llm_response(messages.copy(), text, llm_config, max_new_tokens)

        # retrieve prediction by regex match with the first number in the output
        try:
            match = re.findall(r'\d+', res["answer"])[0]
        except:
            match = None
        pred  = match if match else "-1"
        accuracies.append(int(int(pred) == mwp["answer"]) if is_int(pred) else 0)

        output = {"id": i, "problem": text, "true_efficient_cot": " ".join([x for x, y in mwp["rt"] if y]), "true_answer": mwp["answer"], 
                  "model_cot": res["reasoning"] + res["answer_prompt"] + res["answer"], "model_answer": pred}
        outputs.append(output)

        if use_wandb:
            wandb.log(output)
    
    print("Inference completed...")

    accuracy = sum(accuracies)/len(accuracies)
    for o in outputs: o["accuracy"] = accuracy
    os.makedirs(os.path.join(out_path, model.split("/")[-1]), exist_ok=True)
    with open(os.path.join(out_path, model.split("/")[-1], f"{file_name}.csv"), mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=outputs[0].keys())
        writer.writeheader()
        writer.writerows(outputs)
    
    if use_wandb:
        wandb.finish()

    # cleanup
    if llm_config.get("tokenizer", None) is not None:
        del llm_config["tokenizer"]
    if llm_config.get("model", None) is not None:
        del llm_config["model"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    eval()
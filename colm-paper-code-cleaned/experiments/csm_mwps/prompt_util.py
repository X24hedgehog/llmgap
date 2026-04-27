import math
from typing import Dict, List, Tuple, Optional
import torch
from wrapt_timeout_decorator import timeout
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.util import first_diff_eq_op_idx, first_diff_quant_idx
import time

DETERMINISTIC_GENERATION = lambda tokenizer: {
    "do_sample": False,
    "top_p": None,
    "temperature": None,
    "pad_token_id": tokenizer.eos_token_id,
    "top_k": None
}
    
# Function to make an API call
@timeout(60)
def prompt_api_model(client, messages, model='gpt-3.5-turbo', max_tokens=256, temperature=0, top_p=1):
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            break
        except Exception as e:
            print(f'(probably) OpenAI error : {e}\n retrying...')
            time.sleep(5)
            continue
    return response.choices[0].message.content

def render_messages(messages, tokenizer, use_chat_template) -> str:
    """ Renders a list of messages to text """
    if messages is None or len(messages) == 0: return ""

    prompt = None
    if use_chat_template:
        # render the messages into a prompt
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # just join all the messages (usually there will be only one)
        prompt = " ".join([m["content"] for m in messages]) 
    return prompt

def prompt_and_decode(messages, llm_conf, max_new_tokens, llm_param_fn, use_chat_template, compute_perplexity) -> Tuple[str,str,float]:
    """ 
        Prompts an llm with some messages (provide a single message if you want to use no chat-templates).
        Supports multiple inference methods (vllm, huggingface, azureopenai etc)
        Returns the prompt and answer
    """
    decoded_answer = None
    prompt = None
    perplexity = None

    library = llm_conf["library"]
    if library == "vllm":
        # vllm
        from vllm import SamplingParams
        tokenizer = llm_conf["tokenizer"]
        model = llm_conf["model"]
        prompt = render_messages(messages, tokenizer, use_chat_template)

        # convert hf params to vllm params
        sampling_params = None
        llm_params = llm_param_fn(tokenizer)
        if llm_params["do_sample"]:
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=llm_params["temperature"],
                top_p=llm_params["top_p"],
                top_k=llm_params["top_k"],
                skip_special_tokens=True,
                logprobs=1 if compute_perplexity else None
            )
        else:
            # do_sample being false means deterministic sampling
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                skip_special_tokens=True,
                logprobs=1 if compute_perplexity else None
            )

        # inference
        output = model.generate([prompt], sampling_params, use_tqdm=False)
        out = output[0].outputs[0]
        decoded_answer = out.text
        if compute_perplexity and len(out.token_ids) > 0:
            perplexity = math.exp(-out.cumulative_logprob / len(out.token_ids))
    elif library == "huggingface" or library == "unsloth":
        # huggingface and unsloth
        assert not compute_perplexity, f"Perplexity computation not yet supported with {library} models!"
        tokenizer = llm_conf["tokenizer"]
        model = llm_conf["model"]

        prompt = render_messages(messages, tokenizer, use_chat_template)
        
        # inference
        model_input = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            len_prompt = model_input['input_ids'].shape[1]
            output = model.generate(**model_input, max_new_tokens=max_new_tokens, **llm_param_fn(tokenizer))[0,:]
            decoded_answer = tokenizer.decode(output[len_prompt:], skip_special_tokens=True)

    elif library == "azureopenai":
        # we are using a an openai model through the APIs
        assert not compute_perplexity, "Perplexity computation not supported with Azure OpenAI models!"
        decoded_answer = prompt_api_model(client=llm_conf["openai_client"], messages=messages, model=llm_conf["model"], max_tokens=max_new_tokens, temperature=0, top_p=1)
        prompt = str(messages)

    else:
        raise ValueError(f"Inference library {library} not supported!")
        
    return prompt, decoded_answer, perplexity

def get_logprobs(messages, llm_conf, use_chat_template = True, context_messages = None, top_k = 1) -> Tuple[str, Dict]:
    """ 
        Computes the log probs of some messages

        - context_messages: additional list of messages that will be rendered before the messages themselves (only the logprobs of the messages are reported though)

        Returns 
            - the exact prompt that was used
            - all the tokens, their logprobs and also the logprobs of the top_k tokens
    """
    prompt_full = None
    logprobs = None

    library = llm_conf["library"]
    if library == "vllm":
        print(f"WARNING: vllm natively supports computing logprobs only on the generated output, computing it on some message given a context will use enormous amounts of VRAM. Consider switching to huggingface-transformers instead")
        # vllm
        from vllm import SamplingParams
        def lp_obj_to_dict(token_id, lp):
            return {"token_id": token_id, "logprob": lp.logprob, "rank": lp.rank, "decoded_token": lp.decoded_token}
        
        tokenizer = llm_conf["tokenizer"]
        model = llm_conf["model"]
        
        sampling_params = SamplingParams(
            max_tokens=1, # cannot set to 0 :/
            prompt_logprobs=top_k,
            temperature=0.0
        )

        with torch.no_grad():
            if context_messages is None:
                prompt_context = None
                prompt_full = render_messages(messages, tokenizer, use_chat_template)
                
                output_full = model.generate(prompt_full, sampling_params, use_tqdm=False)[0]

                prompt_token_ids = output_full.prompt_token_ids
                prompt_logprobs = output_full.prompt_logprobs
            else:
                prompt_context = render_messages(context_messages, tokenizer, use_chat_template)
                prompt_full = render_messages(context_messages + messages, tokenizer, use_chat_template)

                params_no_probs = sampling_params.clone()
                params_no_probs.prompt_logprobs = None
                output_context = model.generate(prompt_context, params_no_probs, use_tqdm=False)[0]
                torch.cuda.empty_cache()
                output_full = model.generate(prompt_full, sampling_params, use_tqdm=False)[0]
                
                nr_context_tokens = len(output_context.prompt_token_ids)
                prompt_token_ids = output_full.prompt_token_ids[nr_context_tokens:]
                prompt_logprobs = output_full.prompt_logprobs[nr_context_tokens:]

        # convert the logprobs data into a standardized dictionary format
        logprobs = []
        for tid,lps in zip(prompt_token_ids,prompt_logprobs):
            if lps is None: continue # some models prepend a None here as the first token
            tid_by_rank = {lp.rank:id for id,lp in lps.items()}
            logprobs.append({
                "given": lp_obj_to_dict(tid, lps[tid]),
                **{
                    f"rank_{rank}": lp_obj_to_dict(tid_by_rank[rank], lps[tid_by_rank[rank]])
                    for rank in range(1, top_k+1)
                }
            })
    elif library == "huggingface":
        tokenizer: AutoTokenizer = llm_conf["tokenizer"]
        model: AutoModelForCausalLM = llm_conf["model"]
        
        prompt_context = None
        if context_messages:
            prompt_context = render_messages(context_messages, tokenizer, use_chat_template)
            prompt_full = render_messages(context_messages + messages, tokenizer, use_chat_template)
        else:
            prompt_full = render_messages(messages, tokenizer, use_chat_template)

        input_ids = tokenizer(prompt_full, return_tensors="pt").input_ids.to(model.device)
        if context_messages:
            context_ids = tokenizer(prompt_context, return_tensors="pt").input_ids
            context_length = context_ids.shape[1]
            del context_ids
        else:
            context_length = 0

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        log_probs = torch.log_softmax(logits, dim=-1)
        message_token_ids = input_ids[0, context_length:]
        message_log_probs = log_probs[0, context_length:]

        del logits, input_ids, log_probs

        def lp_obj_to_dict(token_id, lp, rank):
            return {
                "token_id": token_id.item(),
                "logprob": lp.item(),
                "rank": rank,
                "decoded_token": tokenizer.decode([token_id])
            }

        logprobs = []
        for i, token_id in enumerate(message_token_ids):
            if i == 0: continue # we need to shift by one (i.e. the first logprob will be the logprob after token 0)
            token_probs = message_log_probs[i-1]
            sorted_logprobs, sorted_indices = torch.sort(token_probs, descending=True)
            top_logprobs, top_indices = sorted_logprobs[:top_k], sorted_indices[:top_k]
            
            token_rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item()

            logprobs.append({
                "given": lp_obj_to_dict(token_id, token_probs[token_id], rank=token_rank),
                **{
                    f"rank_{rank}": lp_obj_to_dict(top_indices[rank], top_logprobs[rank], rank+1)
                    for rank in range(top_k)
                }
            })
        del message_log_probs, message_token_ids
    else:
        raise ValueError(f"Inference library {library} currently not supported for logprob computation!")
        
    return prompt_context, prompt_full, logprobs

def construct_ask_to_solve_prompt(problem, example_problems = None, example_rts = None, example_answers = None, 
                 use_chat_template: bool = True, use_cot = True, use_rts = True):
    cot_instr = "Let's think step by step." if use_cot else ""
    
    # Step 1: ask for reasoning
    if use_chat_template:
        # use messages with chat template
        messages = []
        if example_problems is not None:
            # do fewshot in-context learning
            if use_rts:
                # show reasoning traces
                for ex_prob, ex_rt, ex_answer in zip(example_problems, example_rts, example_answers):
                    messages.append({"role": "user", "content": f"Q: {ex_prob}\nA: {cot_instr} "})
                    messages.append({"role": "assistant", "content": f"{ex_rt}"})
                    messages.append({"role": "user", "content": f"Therefore, the answer (arabic numerals) is "})
                    messages.append({"role": "assistant", "content": f"{ex_answer}"})
            else:
                # no reasoning traces
                for ex_prob, ex_answer in zip(example_problems, example_answers):
                    messages.append({"role": "user", "content": f"Q: {ex_prob}\nA: "})
                    messages.append({"role": "assistant", "content": f"{ex_answer}"})
        
        # prompt the problem
        messages.append({"role": "user", "content": f"Q: {problem}\nA: {cot_instr} "})
    else:
        # basically use plain-text
        prompt = ""
        if example_problems is not None:
            # do fewshot in-context learning
            if use_rts:
                # show reasoning traces
                for ex_prob, ex_rt, ex_answer in zip(example_problems, example_rts, example_answers):
                    prompt += f"Q: {ex_prob}\nA: {cot_instr} {ex_rt}\nTherefore, the answer (arabic numerals) is {ex_answer}.\n\n"
            else:
                # no reasoning traces
                for ex_prob, ex_answer in zip(example_problems, example_answers):
                    prompt += f"Q: {ex_prob}\nA: {ex_answer}.\n\n"

        prompt += f"Q: {problem}\nA: {cot_instr} " 
        messages = [{"role": "user", "content": prompt}]

    return messages


def ask_to_solve(problem, llm_conf, max_new_tokens=512, example_problems = None, example_rts = None, example_answers = None, 
                 use_chat_template: bool = True, llm_param_fn = DETERMINISTIC_GENERATION, use_cot = True, use_rts = True, compute_perplexity = False):
    """
        Chain-of-thought inference using the two-stage prompt method proposed by Kojima et al. (2022)

        To do fewshot, set (0-shot if left None):
        - example_problems: examples of problems
        - example_rts: correct reasoning traces for each example problem
        - example_answers: correct answers for each example problem
    
        - use_chat_template: if true, will model the fewshot conversation as a back and forth between user and model, making use of chat-templates
        - llm_param_fn: function that creates params when given a tokenizer (e.g. do_sample, temperature etc)
        - use_cot: if true, will put a "Let's think step by step" at the beginning of every answer (NOTE: reasoning traces will still be shown)
        - use_rts: if true, will put the reasoning traces of the examples, otherwise only the distractor will be shown
        - compute_perplexity: if true, will compute the perplexity of both the reasoning output and extracted output
        
        NOTE: we're always doing Q: A: format for consistency

        Prompt:
        Q: {question 1}
        A: Let's think step by step. {reasoning trace 1}
        Therefore, the answer (arabic numerals) is {answer 1}.

        [... \n\n inbetween examples]

        Q: {question n}
        A: Let's think step by step. {reasoning trace n}
        Therefore, the answer (arabic numerals) is {answer n}.

        Q: {problem}
        A: Let's think step by step. [... let the model generate ...]
        Therefore, the answer (arabic numerals) is [... let the model generate ...]
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    messages = construct_ask_to_solve_prompt(problem, example_problems, example_rts, example_answers, use_chat_template, use_cot, use_rts)

    prompt1, decoded_answer1, perplexity1 = prompt_and_decode(messages, llm_conf, max_new_tokens=max_new_tokens, 
                                                llm_param_fn=llm_param_fn, use_chat_template=use_chat_template, 
                                                compute_perplexity=compute_perplexity)

    # Step 2: ask for formatted output
    if use_chat_template:
        messages.append({"role": "assistant", "content": decoded_answer1})
        messages.append({"role": "user", "content": "Therefore, the answer (arabic numerals) is "})
    else:
        messages[0]["content"] += decoded_answer1 + "\nTherefore, the answer (arabic numerals) is "

    prompt2, decoded_answer2, perplexity2 = prompt_and_decode(messages, llm_conf, max_new_tokens=16, 
                                                llm_param_fn=llm_param_fn, use_chat_template=use_chat_template,
                                                compute_perplexity=compute_perplexity)
    
    return {
        "reasoning_prompt": prompt1, # NOTE: might only return messages if unclear (e.g. in case of azure openai)
        "reasoning": decoded_answer1,
        "reasoning_perplexity": perplexity1,
        "answer_prompt": prompt2,
        "answer": decoded_answer2,
        "answer_perplexity": perplexity2,
    }

def construct_ask_for_distractor_prompt_ci(problem, answer, example_problems = None, example_answers = None, example_error_rts = None, example_error_answers = None, 
                       use_chat_template: bool = True, task_as_system_message: bool = True, use_cot = True, incl_highlvl_err = False, use_rts = True, 
                       show_rt_inlc_first_error: bool = False, example_correct_rts = None, distractor_rt: str = None, corr_rt: str = None) -> List[Dict]:
    """ 
        Creates the prompt for asking for a distractor, which is either
        1) a single message with the full prompt as the context (if use_chat_template = False)
        2) a list of messages (otherwise)

        NOTE: answer is the correct answer to the problem
    """
    cot_instr = "Let's think step by step." if use_cot else ""
    high_lvl_error_instr = "Let's assume the student consistently applies a fixed operation based on keywords in comparisons (e.g., always subtracts for 'less than' and adds for 'more than'), leading to errors when the correct operation differs. " if incl_highlvl_err else ""

    if use_cot:
        task_prompt = \
"""You are an educational AI assistant tasked with creating alternative, incorrect answers (distractors) for math word problems. These distractors will be used in multiple-choice tests for grade-school children. Their purpose is to help educators identify and address common student misconceptions, enabling more effective teaching and learning. 
The distractors must:
1. Be based on common student misconceptions related to the problem's concept.
2. Include a detailed, step-by-step reasoning trace illustrating how a student might arrive at the incorrect answer.
3. Avoid simple arithmetic mistakes; the error should be conceptual.
4. Not reuse a number already provided in the question.\n\n"""
    else:
        task_prompt = \
"""You are an educational AI assistant tasked with creating alternative, incorrect answers (distractors) for math word problems. These distractors will be used in multiple-choice tests for grade-school children. Their purpose is to help educators identify and address common student misconceptions, enabling more effective teaching and learning. 
The distractors must:
1. Be based on common student misconceptions related to the problem's concept.
2. Avoid simple arithmetic mistakes; the error should be conceptual.
3. Not reuse a number already provided in the question.\n\n"""

    if use_chat_template:
        # use messages with chat template
        messages = []
        if example_problems is not None:
            # do fewshot in-context learning
            if use_rts:
                # show reasoning traces
                for ex_i, (ex_prob, ex_ans, ex_e_rt, ex_e_ans) in enumerate(zip(example_problems, example_answers, example_error_rts, example_error_answers)):
                    if show_rt_inlc_first_error:
                        fet_idx = first_diff_eq_op_idx(ex_e_rt, example_correct_rts[ex_i])[0] # get idx of first erroneous token
                        messages.append({"role": "user", "content": f"Question: {ex_prob}\nCorrect Answer: {ex_ans}\nIncorrect Answer: {cot_instr} A student might perform the following erroneous computation: {high_lvl_error_instr}{ex_e_rt[:fet_idx+1]}"})
                        messages.append({"role": "assistant", "content": f"{ex_e_rt[fet_idx+1:]}"})
                        messages.append({"role": "user", "content": f"Therefore, the final distractor (arabic numerals) is "})
                    else:
                        messages.append({"role": "user", "content": f"Question: {ex_prob}\nCorrect Answer: {ex_ans}\nIncorrect Answer: {cot_instr} "})
                        messages.append({"role": "assistant", "content": f"A student might perform the following erroneous computation: {high_lvl_error_instr}{ex_e_rt}"})
                        messages.append({"role": "user", "content": f"Therefore, the final distractor (arabic numerals) is "})
                    messages.append({"role": "assistant", "content": f"{ex_e_ans}"})
            else:
                # no reasoning traces
                for ex_prob, ex_ans, ex_e_ans in zip(example_problems, example_answers, example_error_answers):
                    messages.append({"role": "user", "content": f"Question: {ex_prob}\nCorrect Answer: {ex_ans}\nIncorrect Answer: "})
                    messages.append({"role": "assistant", "content": f"{high_lvl_error_instr}{ex_e_ans}"})
        
        # prompt the problem
        if show_rt_inlc_first_error:
            fet_idx = first_diff_eq_op_idx(distractor_rt, corr_rt)[0]
            messages.append({"role": "user", "content": f"Question: {problem}\nCorrect Answer: {answer}\nIncorrect Answer: {cot_instr} A student might perform the following erroneous computation: {high_lvl_error_instr}{distractor_rt[:fet_idx+1]}"})
        else:
            # TODO: maybe this should somehow include the "a student might perform the following erroneous computation" too?
            messages.append({"role": "user", "content": f"Question: {problem}\nCorrect Answer: {answer}\nIncorrect Answer: {cot_instr} {high_lvl_error_instr}"})
        
        if task_as_system_message:
            # prepend messages with task-prompt system message
            messages = [{"role": "system", "content": task_prompt}] + messages
        else:
            # prepend the first message with the task-prompt (avoid having two consecutive user messages)
            messages[0]["content"] = task_prompt + messages[0]["content"]
    else:
        # basically use plain-text
        prompt = task_prompt
        if example_problems is not None:
            # do fewshot in-context learning
            if use_rts:
                # show reasoning traces
                for ex_prob, ex_ans, ex_e_rt, ex_e_ans in zip(example_problems, example_answers, example_error_rts, example_error_answers):
                    prompt += f"Question: {ex_prob}\nCorrect Answer: {ex_ans}\nIncorrect Answer: {cot_instr} A student might perform the following erroneous computation: {high_lvl_error_instr}{ex_e_rt}\nTherefore, the final distractor (arabic numerals) is {ex_e_ans}.\n\n"
            else:
                # no reasoning traces
                for ex_prob, ex_ans, ex_e_ans in zip(example_problems, example_answers, example_error_answers):
                    prompt += f"Question: {ex_prob}\nCorrect Answer: {ex_ans}\nIncorrect Answer: {high_lvl_error_instr}{ex_e_ans}.\n\n"
        
        if show_rt_inlc_first_error:
            prompt += f"Question: {problem}\nCorrect Answer: {answer}\nIncorrect Answer: {cot_instr} A student might perform the following erroneous computation: {high_lvl_error_instr}{distractor_rt[:fet_idx+1]}"
        else:
            prompt += f"Question: {problem}\nCorrect Answer: {answer}\nIncorrect Answer: {cot_instr}{high_lvl_error_instr} "
        messages = [{"role": "user", "content": prompt}]
    return messages

def construct_ask_for_distractor_prompt_am(problem, answer, example_problems = None, example_answers = None, example_error_rts = None, example_error_answers = None, 
                       use_chat_template: bool = True, task_as_system_message: bool = True, use_cot = True, incl_highlvl_err = False, use_rts = True, 
                       show_rt_inlc_first_error: bool = False, example_correct_rts = None, distractor_rt: str = None, corr_rt: str = None) -> List[Dict]:
    """ 
        Creates the prompt for asking for a distractor, which is either
        1) a single message with the full prompt as the context (if use_chat_template = False)
        2) a list of messages (otherwise)

        NOTE: answer is the correct answer to the problem
    """
    cot_instr = "Let's think step by step." if use_cot else ""
    high_lvl_error_instr = "Let's assume the student misapplies place value rules due to a misunderstanding of how numbers are structured within the operation, for example treating digits independently rather than considering their positional value. " if incl_highlvl_err else ""

    if use_cot:
        task_prompt = \
"""You are an educational AI assistant tasked with creating alternative, incorrect answers (distractors) for math word problems. These distractors will be used in multiple-choice tests for grade-school children. Their purpose is to help educators identify and address common student misconceptions, enabling more effective teaching and learning. 
The distractors must:
1. Be based on common student misconceptions related to the problem's concept.
2. Include a detailed, step-by-step reasoning trace illustrating how a student might arrive at the incorrect answer.
3. Avoid random arithmetic mistakes (e.g., typos, misreading digits, or purely careless computation mistakes). However, errors arising from misunderstandings of mathematical procedures or problem-solving strategies are allowed.
4. Not reuse a number already provided in the question.\n\n"""
    else:
        task_prompt = \
"""You are an educational AI assistant tasked with creating alternative, incorrect answers (distractors) for math word problems. These distractors will be used in multiple-choice tests for grade-school children. Their purpose is to help educators identify and address common student misconceptions, enabling more effective teaching and learning. 
The distractors must:
1. Be based on common student misconceptions related to the problem's concept.
2. Avoid random arithmetic mistakes (e.g., typos, misreading digits, or purely careless computation mistakes). However, errors arising from misunderstandings of mathematical procedures or problem-solving strategies are allowed.
3. Not reuse a number already provided in the question.\n\n"""

    if use_chat_template:
        # use messages with chat template
        messages = []
        if example_problems is not None:
            # do fewshot in-context learning
            if use_rts:
                # show reasoning traces
                for ex_i, (ex_prob, ex_ans, ex_e_rt, ex_e_ans) in enumerate(zip(example_problems, example_answers, example_error_rts, example_error_answers)):
                    if show_rt_inlc_first_error:
                        fet_idx = first_diff_quant_idx(ex_e_rt, example_correct_rts[ex_i])[0] # get idx of first erroneous token
                        messages.append({"role": "user", "content": f"Question: {ex_prob}\nCorrect Answer: {ex_ans}\nIncorrect Answer: {cot_instr} A student might perform the following erroneous computation: {high_lvl_error_instr}{ex_e_rt[:fet_idx+1]}"})
                        messages.append({"role": "assistant", "content": f"{ex_e_rt[fet_idx+1:]}"})
                        messages.append({"role": "user", "content": f"Therefore, the final distractor (arabic numerals) is "})
                    else:
                        messages.append({"role": "user", "content": f"Question: {ex_prob}\nCorrect Answer: {ex_ans}\nIncorrect Answer: {cot_instr} "})
                        messages.append({"role": "assistant", "content": f"A student might perform the following erroneous computation: {high_lvl_error_instr}{ex_e_rt}"})
                        messages.append({"role": "user", "content": f"Therefore, the final distractor (arabic numerals) is "})
                    messages.append({"role": "assistant", "content": f"{ex_e_ans}"})
            else:
                # no reasoning traces
                for ex_prob, ex_ans, ex_e_ans in zip(example_problems, example_answers, example_error_answers):
                    messages.append({"role": "user", "content": f"Question: {ex_prob}\nCorrect Answer: {ex_ans}\nIncorrect Answer: "})
                    messages.append({"role": "assistant", "content": f"{high_lvl_error_instr}{ex_e_ans}"})
        
        # prompt the problem
        if show_rt_inlc_first_error:
            fet_idx = first_diff_quant_idx(distractor_rt, corr_rt)[0]
            messages.append({"role": "user", "content": f"Question: {problem}\nCorrect Answer: {answer}\nIncorrect Answer: {cot_instr} A student might perform the following erroneous computation: {high_lvl_error_instr}{distractor_rt[:fet_idx+1]}"})
        else:
            messages.append({"role": "user", "content": f"Question: {problem}\nCorrect Answer: {answer}\nIncorrect Answer: {cot_instr} {high_lvl_error_instr}"})
        
        if task_as_system_message:
            # prepend messages with task-prompt system message
            messages = [{"role": "system", "content": task_prompt}] + messages
        else:
            # prepend the first message with the task-prompt (avoid having two consecutive user messages)
            messages[0]["content"] = task_prompt + messages[0]["content"]
    else:
        # basically use plain-text
        prompt = task_prompt
        if example_problems is not None:
            # do fewshot in-context learning
            if use_rts:
                # show reasoning traces
                for ex_prob, ex_ans, ex_e_rt, ex_e_ans in zip(example_problems, example_answers, example_error_rts, example_error_answers):
                    prompt += f"Question: {ex_prob}\nCorrect Answer: {ex_ans}\nIncorrect Answer: {cot_instr} A student might perform the following erroneous computation: {high_lvl_error_instr}{ex_e_rt}\nTherefore, the final distractor (arabic numerals) is {ex_e_ans}.\n\n"
            else:
                # no reasoning traces
                for ex_prob, ex_ans, ex_e_ans in zip(example_problems, example_answers, example_error_answers):
                    prompt += f"Question: {ex_prob}\nCorrect Answer: {ex_ans}\nIncorrect Answer: {high_lvl_error_instr}{ex_e_ans}.\n\n"
        
        if show_rt_inlc_first_error:
            prompt += f"Question: {problem}\nCorrect Answer: {answer}\nIncorrect Answer: {cot_instr} A student might perform the following erroneous computation: {high_lvl_error_instr}{distractor_rt[:fet_idx+1]}"
        else:
            prompt += f"Question: {problem}\nCorrect Answer: {answer}\nIncorrect Answer: {cot_instr}{high_lvl_error_instr} "
        messages = [{"role": "user", "content": prompt}]
    return messages

def ask_for_distractor(problem, answer, llm_conf, miscon_category: str, max_new_tokens=512, example_problems = None, example_answers = None, example_error_rts = None, example_error_answers = None, 
                       use_chat_template: bool = True, task_as_system_message: bool = True, llm_param_fn = DETERMINISTIC_GENERATION, use_cot = True, incl_highlvl_err = False, use_rts = True, compute_perplexity = False,
                       show_rt_inlc_first_error: bool = False, example_correct_rts = None, distractor_rt: str = None, corr_rt: str = None):
    """ 
        Asks the model to generate a distractor and provide reasoning for its choice.
        
        To do fewshot, set (0-shot if left None):
        - example_problems: examples of problems
        - example_answers: correct answers to example_problems
        - example_error_rts: erroneous reasoning trace to example_problems
        - example_error_answers: incorrect answer resulting from example_error_rts
        - example_correct_rts: solution reasoning traces to example_problems

        - miscon_category: which category of errors are we considering? any of ["ci", "am"]
        - use_chat_template: if true, will model the fewshot conversation as a back and forth between user and model, making use of chat-templates
        - task_as_system_message: if true, will put the task description as the system prompt and not repeat it, otherwise will prepend it to the first user message
            no-effect if use_chat_template is false
        - llm_param_fn: function that creates params when given a tokenizer (e.g. do_sample, temperature etc)
        - use_cot: if true, will put a "Let's think step by step" at the beginning of every answer (also any in-context examples)
        - incl_highlvl_err: if true, will put a high level description of the student misconception (i.e. they use a keyword strategy)
        - use_rts: if true, will put the reasoning traces of the examples, otherwise only the distractor will be shown
        - compute_perplexity: if true, will compute the perplexity of both the reasoning output and extracted output
        - show_rt_inlc_first_error: if true, will "leak" the first part of the reasoning trace including the first errors (i.e. show up to the wrong + or -)
            - distractor_rt: one reasoning trace leading to a plausible distractor
            - corr_rt: reasoning trace leading to the correct solution

        Prompt:
        You are an educational AI assistant tasked with creating alternative, incorrect answers (distractors) for math word problems. These distractors will be used in multiple-choice tests for grade-school children. Their purpose is to help educators identify and address common student misconceptions, enabling more effective teaching and learning. 
        The distractors must:
        1. Be based on common student misconceptions related to the problem's concept.
        2. Include a detailed, step-by-step reasoning trace illustrating how a student might arrive at the incorrect answer.
        3. Avoid simple arithmetic mistakes; the error should be conceptual.
        4. Not reuse a number already provided in the question.

        Question: {problem 1}
        Correct Answer: {answer 1}
        Incorrect Answer: Let's think step by step. A student might perform the following erroneous computation: {any inconsistent reasoning trace 1}
        Therefore, the final distractor (arabic numerals) is {distractor for reasoning trace 1}

        [... \n\n inbetween examples]

        Question: {problem n}
        Correct Answer: {answer n}
        Incorrect Answer: Let's think step by step. A student might perform the following erroneous computation: {any inconsistent reasoning trace n}
        Therefore, the final distractor (arabic numerals) is {distractor for reasoning trace n}

        Question: {problem}
        Correct Answer: {answer}
        Incorrect Answer: Let's think step by step. [... let the model generate ...]
        Therefore, the final distractor (arabic numerals) is [... let the model generate ...]
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 1: ask for reasoning
    if miscon_category == "ci":
        messages = construct_ask_for_distractor_prompt_ci(problem, answer, example_problems, example_answers, example_error_rts, example_error_answers, 
                    use_chat_template, task_as_system_message, use_cot, incl_highlvl_err, use_rts, show_rt_inlc_first_error, example_correct_rts, distractor_rt, corr_rt)
    elif miscon_category == "am":
        messages = construct_ask_for_distractor_prompt_am(problem, answer, example_problems, example_answers, example_error_rts, example_error_answers, 
                    use_chat_template, task_as_system_message, use_cot, incl_highlvl_err, use_rts, show_rt_inlc_first_error, example_correct_rts, distractor_rt, corr_rt)
    else:
        raise ValueError(f"Error category {miscon_category} not supported!")

    prompt1, decoded_answer1, perplexity1 = prompt_and_decode(messages, llm_conf, max_new_tokens=max_new_tokens, 
                                                llm_param_fn=llm_param_fn, use_chat_template=use_chat_template, 
                                                compute_perplexity=compute_perplexity)

    # Step 2: ask for formatted output
    if use_chat_template:
        messages.append({"role": "assistant", "content": decoded_answer1})
        messages.append({"role": "user", "content": "Therefore, the final distractor (arabic numerals) is "})
    else:
        messages[0]["content"] += decoded_answer1 + "\nTherefore, the final distractor (arabic numerals) is "

    prompt2, decoded_answer2, perplexity2 = prompt_and_decode(messages, llm_conf, max_new_tokens=16, 
                                                llm_param_fn=llm_param_fn, use_chat_template=use_chat_template,
                                                compute_perplexity=compute_perplexity)
    
    return {
        "reasoning_prompt": prompt1, # NOTE: might only return messages if unclear (e.g. in case of azure openai)
        "reasoning": decoded_answer1,
        "reasoning_perplexity": perplexity1,
        "answer_prompt": prompt2,
        "answer": decoded_answer2,
        "answer_perplexity": perplexity2,
    }
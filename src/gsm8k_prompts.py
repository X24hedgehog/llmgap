GSM8K_STEP_BY_STEP_PROMPT = """Solve the following grade school math problem step by step.
At the end, write the final answer in the format: #### <number>

Question: {question}
Answer:"""

GSM8K_DIRECT_PROMPT = """Solve the following grade school math problem.
Write the final answer in the format: #### <number>

Question: {question}
Answer:"""


PROMPT_TEMPLATES = {
    "step_by_step": GSM8K_STEP_BY_STEP_PROMPT,
    "direct": GSM8K_DIRECT_PROMPT,
}


DEFAULT_PROMPT_STYLE = "step_by_step"
#!/bin/bash
#SBATCH --job-name=rl_nextsubq_qwen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40000m
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=/cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs/rl-nextsubq-qwen-full-eval-%j.out
#SBATCH --error=/cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs/rl-nextsubq-qwen-full-eval-%j.err

set -euo pipefail

cd /cluster/home/tunguyen1/llmgap
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

export HF_HOME=/cluster/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/cluster/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/cluster/scratch/$USER/hf_cache/datasets
export TORCH_HOME=/cluster/scratch/$USER/torch_cache
export PIP_CACHE_DIR=/cluster/scratch/$USER/pip_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_CSV=/cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search/out/next_subquestion_pairs_gsm8k.csv
OUT_DIR=/cluster/scratch/$USER/llmgap/output/rl_next_subquestion_qwen_judge_full
EVAL_CSV=$OUT_DIR/test_eval_predictions.csv
EVAL_JSON=$OUT_DIR/test_eval_metrics.json

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "TRAINING NEXT-SUBQUESTION RL WITH QWEN 7B JUDGE"
echo "============================================================"

python src/trl_grpo_next_subquestion_qwen_judge.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --train-csv "$TRAIN_CSV" \
  --out-dir "$OUT_DIR" \
  --prompt-col prompt \
  --question-col question \
  --subquestion-col next_subquestion \
  --reasoning-trace-col reasoning_trace \
  --tree-col tree \
  --split-col split \
  --train-split train \
  --num-generations 2 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --epochs 3 \
  --lr 5e-7 \
  --max-prompt-length 512 \
  --max-completion-length 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --beta 0.0 \
  --judge-model-name Qwen/Qwen2.5-7B-Instruct \
  --judge-load-in-4bit \
  --judge-batch-size 1 \
  --judge-max-input-length 2048 \
  --judge-max-new-tokens 8 \
  --include-policy-prompt-in-judge \
  --judge-reward-scale 1.0 \
  --format-reward-weight 0.1 \
  --use-peft \
  --lora-r 64 \
  --lora-alpha 128 \
  --lora-dropout 0.05 \
  --load-in-4bit \
  --gradient-checkpointing \
  --logging-steps 1 \
  --save-steps 100 \
  --debug-print-rewards 20

echo "============================================================"
echo "EVALUATING NEXT-SUBQUESTION MODEL ON TEST SPLIT"
echo "============================================================"

python - <<PY
import json
import os
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

train_csv = "$TRAIN_CSV"
out_dir = "$OUT_DIR"
eval_csv = "$EVAL_CSV"
eval_json = "$EVAL_JSON"

policy_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
judge_model_name = "Qwen/Qwen2.5-7B-Instruct"

df = pd.read_csv(train_csv)
if "split" not in df.columns:
    print("No split column found. Skipping test evaluation.")
    raise SystemExit(0)

test_df = df[df["split"].astype(str).str.lower() == "test"].reset_index(drop=True)
print("Test rows:", len(test_df))
if len(test_df) == 0:
    print("No test rows found. Skipping test evaluation.")
    raise SystemExit(0)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained(policy_model_name, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

base = AutoModelForCausalLM.from_pretrained(
    policy_model_name,
    quantization_config=quant_config,
    torch_dtype=None,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, out_dir)
model.eval()

preds = []
prompts = test_df["prompt"].astype(str).tolist()

for i in tqdm(range(0, len(prompts), 4), desc="Generate test predictions"):
    batch = prompts[i:i+4]
    formatted = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in batch
    ]
    enc = tok(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )

    input_len = enc["input_ids"].shape[1]
    for seq in out:
        pred = tok.decode(seq[input_len:], skip_special_tokens=True).strip()
        preds.append(pred)

del model, base
torch.cuda.empty_cache()

judge_tok = AutoTokenizer.from_pretrained(judge_model_name, trust_remote_code=True)
if judge_tok.pad_token is None:
    judge_tok.pad_token = judge_tok.eos_token
judge_tok.padding_side = "left"

judge_model = AutoModelForCausalLM.from_pretrained(
    judge_model_name,
    quantization_config=quant_config,
    torch_dtype=None,
    device_map="auto",
    trust_remote_code=True,
)
judge_model.eval()

def build_judge_prompt(problem_prompt, question, gold, pred):
    return f'''You are judging a generated next subquestion for a math word problem.

Original math problem:
{question}

The policy model was given this prompt:
{problem_prompt}

Reference next subquestion:
{gold}

Candidate next subquestion:
{pred}

Decide whether the candidate asks the same immediate next subquestion as the reference.

Answer "yes" only if the candidate asks for the same mathematical quantity as the reference, even if the wording is different.

Answer "no" if the candidate:
- asks a previous step that is already known
- asks a later step
- asks the final problem question directly
- asks a related but different quantity
- is irrelevant
- includes a full solution instead of one subquestion

Answer only yes or no.'''

judge_prompts = [
    build_judge_prompt(
        str(row["prompt"]),
        str(row["question"]),
        str(row["next_subquestion"]),
        pred,
    )
    for (_, row), pred in zip(test_df.iterrows(), preds)
]

scores = []
judge_texts = []

for i in tqdm(range(0, len(judge_prompts), 1), desc="Qwen judge eval"):
    batch = judge_prompts[i:i+1]
    formatted = [
        judge_tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in batch
    ]
    enc = judge_tok(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(judge_model.device)

    with torch.no_grad():
        out = judge_model.generate(
            **enc,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=judge_tok.pad_token_id,
        )

    input_len = enc["input_ids"].shape[1]
    for seq in out:
        text = judge_tok.decode(seq[input_len:], skip_special_tokens=True).strip()
        judge_texts.append(text)
        verdicts = re.findall(r"\\b(yes|no)\\b", text, flags=re.IGNORECASE)
        final = verdicts[-1].lower() if verdicts else "no"
        scores.append(1 if final == "yes" else 0)

acc = sum(scores) / len(scores) if scores else 0.0

result_df = pd.DataFrame({
    "question": test_df["question"].astype(str),
    "prompt": test_df["prompt"].astype(str),
    "gold_next_subquestion": test_df["next_subquestion"].astype(str),
    "prediction": preds,
    "judge_output": judge_texts,
    "score": scores,
})
result_df.to_csv(eval_csv, index=False)

metrics = {
    "task": "next_subquestion",
    "n_test": len(scores),
    "accuracy_qwen_judge": acc,
    "n_correct": int(sum(scores)),
    "eval_csv": eval_csv,
}
with open(eval_json, "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
PY

echo "Done."
echo "Output dir: $OUT_DIR"
echo "Eval CSV: $EVAL_CSV"
echo "Eval JSON: $EVAL_JSON"

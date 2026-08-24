#!/bin/bash
#SBATCH --job-name=rl_correct_answer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24000m
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=/cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs/rl-correct-answer-full-eval-%j.out
#SBATCH --error=/cluster/home/tunguyen1/llmgap/out/rl_benchmark/logs/rl-correct-answer-full-eval-%j.err

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

TRAIN_CSV=/cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs_gsm8k.csv
OUT_DIR=/cluster/scratch/$USER/llmgap/output/rl_correct_answer_full
EVAL_CSV=$OUT_DIR/test_eval_predictions.csv
EVAL_JSON=$OUT_DIR/test_eval_metrics.json

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "TRAINING CORRECT-ANSWER RL"
echo "============================================================"

python src/trl_grpo_gsm8k_correct_answer.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --train-csv "$TRAIN_CSV" \
  --out-dir "$OUT_DIR" \
  --question-col question \
  --answer-col target_answer \
  --split-col split \
  --train-split train \
  --num-generations 4 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --epochs 3 \
  --lr 5e-7 \
  --max-prompt-length 512 \
  --max-completion-length 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --beta 0.0 \
  --format-reward-weight 0.1 \
  --use-peft \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --load-in-4bit \
  --gradient-checkpointing \
  --logging-steps 1 \
  --save-steps 100 \
  --debug-print-rewards 20

echo "============================================================"
echo "EVALUATING CORRECT-ANSWER MODEL ON TEST SPLIT"
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

PROMPT_TEMPLATE = '''Solve the following grade school math problem step by step.

Rules:
- Be concise.
- The last line must be exactly: #### <number>
- Do not write anything after the final answer line.

Question: {question}
Answer:'''

def extract_last_number(text):
    nums = re.findall(r"-?\\d+(?:\\.\\d+)?", str(text).replace(",", ""))
    return nums[-1] if nums else None

def extract_final_answer(text):
    text = str(text).replace(",", "")
    marker = re.search(r"####\\s*\\$?\\s*(-?\\d+(?:\\.\\d+)?)", text)
    if marker:
        return marker.group(1)
    boxed = re.search(r"\\\\boxed\\{\\s*\\$?\\s*(-?\\d+(?:\\.\\d+)?)\\s*\\}", text)
    if boxed:
        return boxed.group(1)
    final_phrase = re.search(
        r"final answer (?:is|:)\\s*\\$?\\s*(-?\\d+(?:\\.\\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if final_phrase:
        return final_phrase.group(1)
    last = extract_last_number(text)
    return last if last is not None else text.strip()

def normalize_answer(answer):
    answer = str(answer).strip().replace(",", "").replace("$", "")
    marker = re.search(r"####\\s*\\$?\\s*(-?\\d+(?:\\.\\d+)?)", answer)
    if marker:
        answer = marker.group(1)
    last = extract_last_number(answer)
    if last is not None:
        answer = last
    try:
        v = float(answer)
        if v.is_integer():
            return str(int(v))
        return str(v)
    except ValueError:
        return answer.strip()

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

questions = test_df["question"].astype(str).tolist()
prompts = [PROMPT_TEMPLATE.format(question=q) for q in questions]
golds = test_df["target_answer"].astype(str).tolist()

pred_texts = []
pred_answers = []
scores = []

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
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )

    input_len = enc["input_ids"].shape[1]
    for seq in out:
        pred_text = tok.decode(seq[input_len:], skip_special_tokens=True).strip()
        pred_texts.append(pred_text)

for pred_text, gold in zip(pred_texts, golds):
    pred_ans = extract_final_answer(pred_text)
    pred_answers.append(pred_ans)
    scores.append(1 if normalize_answer(pred_ans) == normalize_answer(gold) else 0)

acc = sum(scores) / len(scores) if scores else 0.0

result_df = pd.DataFrame({
    "question": questions,
    "gold_answer": golds,
    "prediction_text": pred_texts,
    "prediction_answer": pred_answers,
    "score": scores,
})
result_df.to_csv(eval_csv, index=False)

metrics = {
    "task": "correct_answer",
    "n_test": len(scores),
    "accuracy_exact_extracted_answer": acc,
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

# -*- coding: utf-8 -*-
"""
CSE 151B Spring 2026 — Math Reasoning Competition
Qwen3-4B-Thinking + QLoRA fine-tune, vLLM inference

IMPORTANT — READ FIRST
======================================================================
This pipeline uses THREE different package environments that conflict
with each other. You CANNOT "Run all". Each phase runs in its own fresh
kernel session, with a runtime restart between phases.

  PHASE 1  Inference (vLLM)        -> run merged model, score, submit
  PHASE 2  Fine-tuning (TRL/PEFT)  -> train the LoRA adapter
  PHASE 3  Merge (PEFT)            -> fold adapter into base, save model

Normal order to PRODUCE a model from scratch: Phase 2 -> Phase 3 -> Phase 1.
If the merged model already exists on Drive, you only need Phase 1.

SAFETY RULE: never run `rm -rf` on anything under /content/drive.
Large files on Drive can take several minutes to appear in the
drive.google.com web view even after they are fully saved, trust the
Colab file browser / os.path.getsize, not the web UI's refresh speed.
======================================================================
"""

# =============================================================================
# PHASE 1 — INFERENCE  (run in a FRESH kernel)
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1  Install vLLM + judger dependencies. RUN FIRST, then RESTART the kernel.
#      (sympy / antlr pins are required or the judger crashes on parse_latex.)
# -----------------------------------------------------------------------------
!pip install vllm==0.8.5 sympy==1.13.1 antlr4-python3-runtime==4.11.1 -q

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "0"
# ---> After this cell finishes, RESTART THE RUNTIME, then continue at 1.2 <---


# -----------------------------------------------------------------------------
# 1.2  Mount Drive.
#      If you get "Mountpoint must not already contain files", use
#      force_remount=True (NEVER rm -rf the mount).
# -----------------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')  # add force_remount=True if it complains


# -----------------------------------------------------------------------------
# 1.3  Imports, config, paths.
# -----------------------------------------------------------------------------
import re
import csv
import json
import sys
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer
from tqdm.auto import tqdm

PROJECT     = "/content/drive/MyDrive/competition_project"
MERGED_DIR  = f"{PROJECT}/qlora_merged"        # fine-tuned model (Phase 3 output)
DATA_DIR    = f"{PROJECT}/data"
OUTPUT_PATH = f"{PROJECT}/results"
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1.4  Prompt builder (shared by public + private inference).
# -----------------------------------------------------------------------------
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician competing in a math olympiad. "
    "Think carefully and systematically, checking your work at each step. "
    "Simplify all expressions fully to their simplest numerical or symbolic form. "
    "The question uses [ANS] as placeholders for answers. "
    "Put your final answer inside \\boxed{}. "
    "For multiple sub-answers, use a single \\boxed{} with comma separation "
    "in the same order as the [ANS] placeholders, e.g. \\boxed{3, 7}. "
    "Always double-check your arithmetic before giving the final answer."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician competing in a math olympiad. "
    "Solve the problem completely first, then check which answer choice matches. "
    "Output ONLY the letter of the single best answer inside \\boxed{}, e.g. \\boxed{C}. "
    "Do not write anything after the boxed letter."
)

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"({lbl}) {opt.strip()}" for lbl, opt in zip(labels, options))
        user = (
            f"{question}\n\n"
            f"Answer Choices:\n{opts_text}\n\n"
            "Which option is correct? Output only \\boxed{<letter>}."
        )
        return SYSTEM_PROMPT_MCQ, user
    return SYSTEM_PROMPT_MATH, question


# -----------------------------------------------------------------------------
# 1.5  Judger + answer-extraction helpers.
# -----------------------------------------------------------------------------
!cp {PROJECT}/judger.py /content/judger.py
!cp {PROJECT}/utils.py  /content/utils.py
sys.path.insert(0, "/content")
from judger import Judger
judger = Judger(strict_extract=False)
print("Judger loaded.")

def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""

def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()

def score_responses(items, responses):
    """Score a list of dataset items (which must contain answers) against
    a {id: response} dict. Returns the results list."""
    out = []
    for item in tqdm(items, desc="Scoring"):
        resp   = responses.get(item.get("id"), "")
        is_mcq = bool(item.get("options"))
        gold   = item["answer"]
        if is_mcq:
            correct = score_mcq(resp, str(gold))
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(pred=resp, gold=gold_list,
                                            options=[[]] * len(gold_list))
            except Exception:
                correct = False
        out.append({"id": item.get("id"), "is_mcq": is_mcq,
                    "gold": gold, "response": resp, "correct": correct})
    return out

def print_summary(results, baseline=None):
    mcq  = [r for r in results if r["is_mcq"]]
    free = [r for r in results if not r["is_mcq"]]
    acc  = lambda s: sum(r["correct"] for r in s) / len(s) * 100 if s else 0.0
    print("=" * 50)
    print(f"  MCQ        : {acc(mcq):6.2f}%   ({len(mcq)} q)")
    print(f"  Free-form  : {acc(free):6.2f}%   ({len(free)} q)")
    print(f"  OVERALL    : {acc(results):6.2f}%   ({len(results)} q)")
    if baseline is not None:
        print(f"  (baseline was {baseline:.2f}%)")
    print("=" * 50)


# -----------------------------------------------------------------------------
# 1.6  Load the merged model into vLLM (ONE load, reused for public + private).
# -----------------------------------------------------------------------------
from vllm import LLM, SamplingParams

tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR, trust_remote_code=True,
                                          local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

llm = LLM(
    model=MERGED_DIR,
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
    max_model_len=8192,
    trust_remote_code=True,
    max_num_seqs=32,
    enforce_eager=True,
    disable_async_output_proc=True,
)
print("Model loaded.")

SAMPLING = SamplingParams(max_tokens=8192, temperature=0.0)

def run_inference(items):
    """Generate responses for a list of dataset items. Returns {id: response}."""
    prompts, ids = [], []
    for item in items:
        system, user = build_prompt(item["question"], item.get("options"))
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False, add_generation_prompt=True))
        ids.append(item.get("id"))
    outputs = llm.generate(prompts, SAMPLING)
    return {i: o.outputs[0].text.strip() for i, o in zip(ids, outputs)}

def save_jsonl(responses, filename):
    with open(Path(OUTPUT_PATH) / filename, "w") as f:
        for i, r in responses.items():
            f.write(json.dumps({"id": i, "response": r}) + "\n")

def save_submission_csv(items, responses, filename):
    path = Path(OUTPUT_PATH) / filename
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "response"])
        for item in items:
            w.writerow([item["id"], responses.get(item.get("id"), "")])
    print(f"Saved {len(items)} rows to {path}")


# -----------------------------------------------------------------------------
# 1.7  PUBLIC set: infer + score. This is the DECISION step.
#      Only proceed to the private submission (1.8) if this beats your baseline.
# -----------------------------------------------------------------------------
public_data = [json.loads(l) for l in open(f"{DATA_DIR}/public.jsonl")]
print(f"Loaded {len(public_data)} public questions")

public_responses = run_inference(public_data)
save_jsonl(public_responses, "ft_public_responses.jsonl")

public_results = score_responses(public_data, public_responses)
print_summary(public_results, baseline=49.29)


# -----------------------------------------------------------------------------
# 1.8  PRIVATE set: infer + write submission CSV.
#      RUN ONLY IF 1.7 beat your baseline. This is what you upload to Kaggle.
# -----------------------------------------------------------------------------
private_data = [json.loads(l) for l in open(f"{DATA_DIR}/private.jsonl")]
print(f"Loaded {len(private_data)} private questions")

private_responses = run_inference(private_data)
save_jsonl(private_responses, "private_responses.jsonl")
save_submission_csv(private_data, private_responses, "private_submission.csv")


# =============================================================================
# PHASE 2 — FINE-TUNING  (run in a FRESH kernel; do NOT run Phase 1 first)
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1  Install training deps. RUN FIRST, then RESTART the kernel.
#      Do NOT install/import vLLM in this session — it conflicts with TRL/PEFT.
# -----------------------------------------------------------------------------
!pip install peft trl bitsandbytes accelerate datasets pyarrow --upgrade -q
import os
os._exit(0)   # forces a hard restart so the new packages load
# ---> After reconnect, mount Drive (2.2) then run 2.3 <---


# -----------------------------------------------------------------------------
# 2.2  Mount Drive (training session).
# -----------------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')  # add force_remount=True if it complains


# -----------------------------------------------------------------------------
# 2.3  QLoRA fine-tune on GSM8K.
#      Adapter is saved to OUTPUT_DIR on Drive (survives disconnects).
# -----------------------------------------------------------------------------
import json
import torch
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID   = "Qwen/Qwen3-4B-Thinking-2507"
OUTPUT_DIR = "/content/drive/MyDrive/competition_project/qlora_model"

MAX_SEQ_LEN       = 1024
NUM_EPOCHS        = 1        # keep small; loss drops fast, more epochs overfit
BATCH_SIZE        = 8
GRAD_ACCUM        = 2
LEARNING_RATE     = 2e-4
MAX_TRAIN_SAMPLES = 3000

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def format_gsm8k(example):
    answer_text = example.get("answer", "")
    if "####" in answer_text:
        parts = answer_text.split("####")
        answer_text = f"{parts[0].strip()}\n\nThe answer is \\boxed{{{parts[1].strip()}}}"
    messages = [
        {"role": "system", "content": "You are an expert mathematician. Solve the problem step-by-step. Put your final answer inside \\boxed{}."},
        {"role": "user",      "content": example.get("question", "")},
        {"role": "assistant", "content": answer_text},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

print("Loading GSM8K...")
gsm = load_dataset("openai/gsm8k", "main", split="train")
gsm = gsm.map(format_gsm8k, remove_columns=gsm.column_names)
train_dataset = gsm.shuffle(seed=42)
if len(train_dataset) > MAX_TRAIN_SAMPLES:
    train_dataset = train_dataset.select(range(MAX_TRAIN_SAMPLES))
print(f"Training examples: {len(train_dataset)}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="epoch",
    save_total_limit=1,
    bf16=True,
    max_length=MAX_SEQ_LEN,            # NOTE: newer TRL renamed max_seq_length -> max_length
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="text",
    packing=True,
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Training complete. Adapter saved to {OUTPUT_DIR}")


# =============================================================================
# PHASE 3 — MERGE ADAPTER INTO BASE MODEL  (run in a FRESH kernel)
# =============================================================================

# -----------------------------------------------------------------------------
# 3.1  Pin a coherent transformers/peft set for Qwen3, drop conflicting extras.
#      RUN FIRST, then RESTART the kernel.
# -----------------------------------------------------------------------------
!pip install -q "transformers==4.51.3" "peft==0.15.2" "accelerate>=0.34.0" "huggingface_hub>=0.25.0"
!pip uninstall -y -q torchao torchvision
import os
os._exit(0)   # forces a hard restart
# ---> After reconnect, mount Drive (3.2) then run 3.3 <---


# -----------------------------------------------------------------------------
# 3.2  Mount Drive (merge session).
# -----------------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')  # add force_remount=True if it complains


# -----------------------------------------------------------------------------
# 3.3  Merge LoRA adapter into the base model and save the full model to Drive.
#      Done on CPU (merging into a 4-bit base degrades weights; we want bf16).
# -----------------------------------------------------------------------------
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL  = "Qwen/Qwen3-4B-Thinking-2507"
ADAPTER_DIR = "/content/drive/MyDrive/competition_project/qlora_model"
MERGED_DIR  = "/content/drive/MyDrive/competition_project/qlora_merged"

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16,
    device_map="cpu", trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model = model.merge_and_unload()
model.save_pretrained(MERGED_DIR, safe_serialization=True)

tok = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
tok.save_pretrained(MERGED_DIR)
print("Merged ->", MERGED_DIR)


# -----------------------------------------------------------------------------
# 3.4  Verify the merged model actually persisted (both shards, full size).
#      ~4.97 GB + ~3.08 GB expected. If MISSING, the save/sync failed.
# -----------------------------------------------------------------------------
import os
for f in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
    p = os.path.join(MERGED_DIR, f)
    print(f, os.path.getsize(p) if os.path.exists(p) else "MISSING")
# ---> Once verified, RESTART and go to PHASE 1 to run inference. <---

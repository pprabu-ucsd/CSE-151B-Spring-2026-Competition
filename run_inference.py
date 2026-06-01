"""
CSE 151B Spring 2026 — Math Reasoning Competition
Single entry point for inference and submission generation.

Usage:
    python run_inference.py \
        --model_id   "your-username/your-model-name" \
        --data_path  "/path/to/private.jsonl" \
        --output_dir "/path/to/output"

Or from Python:
    from run_inference import run_inference
    run_inference(
        model_id   = "your-username/your-model-name",
        data_path  = "/path/to/private.jsonl",
        output_dir = "/path/to/output",
    )
"""

import re
import csv
import json
import argparse
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

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
    """Return (system_prompt, user_prompt) for a given question."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_letter(text: str) -> str:
    """Extract a single letter from a \\boxed{X} expression."""
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model_id:               str   = "your-username/your-model-name",
    data_path:              str   = "/path/to/private.jsonl",
    output_dir:             str   = "./results",
    max_tokens:             int   = 8192,
    temperature:            float = 0.0,
    gpu_memory_utilization: float = 0.90,
    max_model_len:          int   = 8192,
    max_num_seqs:           int   = 32,
) -> str:
    """
    Full end-to-end inference pipeline.

    Loads the fine-tuned model from HuggingFace Hub, runs inference on the
    private dataset, applies post-processing, and writes submission.csv.

    Parameters
    ----------
    model_id : str
        HuggingFace Hub model path, e.g. "your-username/cse151b-qwen3-qlora".
    data_path : str
        Path to the private test JSONL file (no 'answer' field).
    output_dir : str
        Directory where submission.csv (and response backup) will be saved.
    max_tokens : int
        Maximum new tokens per response (default 8192).
    temperature : float
        Sampling temperature. 0.0 = greedy (deterministic).
    gpu_memory_utilization : float
        Fraction of GPU VRAM to allocate to vLLM.
    max_model_len : int
        Maximum total sequence length (prompt + generation).
    max_num_seqs : int
        Maximum number of sequences processed in parallel by vLLM.

    Returns
    -------
    str
        Absolute path to the written submission.csv.
    """
    import os
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from tqdm.auto import tqdm

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_V1", "0")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print(f"Loading dataset from {data_path} ...")
    data = [json.loads(line) for line in open(data_path)]
    print(f"  {len(data)} questions loaded.")

    # ── 2. Load tokenizer + model ─────────────────────────────────────────────
    print(f"Loading tokenizer from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_id} ...")
    llm = LLM(
        model                     = model_id,
        dtype                     = "bfloat16",
        gpu_memory_utilization    = gpu_memory_utilization,
        max_model_len             = max_model_len,
        trust_remote_code         = True,
        max_num_seqs              = max_num_seqs,
        enforce_eager             = True,
        disable_async_output_proc = True,
    )
    print("Model loaded.")

    sampling_params = SamplingParams(
        max_tokens  = max_tokens,
        temperature = temperature,
    )

    # ── 3. Build prompts ──────────────────────────────────────────────────────
    print("Building prompts ...")
    prompts, prompt_ids = [], []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)
        prompt_ids.append(item["id"])

    # ── 4. Run inference ──────────────────────────────────────────────────────
    print(f"Running inference on {len(prompts)} questions ...")
    outputs = llm.generate(prompts, sampling_params)

    responses = {}
    for item_id, out in zip(prompt_ids, outputs):
        responses[item_id] = out.outputs[0].text.strip()

    # ── 5. Save raw response backup ───────────────────────────────────────────
    backup_path = output_path / "private_responses.jsonl"
    with open(backup_path, "w") as f:
        for item_id, resp in responses.items():
            f.write(json.dumps({"id": item_id, "response": resp}) + "\n")
    print(f"Raw responses saved to {backup_path}")

    # ── 6. Write submission CSV ───────────────────────────────────────────────
    csv_path = output_path / "submission.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "response"])
        for item in data:
            resp = responses.get(item["id"], "")
            writer.writerow([item["id"], resp])

    print(f"Submission saved to {csv_path}  ({len(responses)} rows)")
    return str(csv_path.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSE 151B Competition — run inference")
    parser.add_argument("--model_id",               required=True,          help="HuggingFace Hub model ID")
    parser.add_argument("--data_path",              required=True,          help="Path to private.jsonl")
    parser.add_argument("--output_dir",             default="./results",    help="Output directory")
    parser.add_argument("--max_tokens",             type=int,   default=8192)
    parser.add_argument("--temperature",            type=float, default=0.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len",          type=int,   default=8192)
    parser.add_argument("--max_num_seqs",           type=int,   default=32)
    args = parser.parse_args()

    csv_out = run_inference(
        model_id                = args.model_id,
        data_path               = args.data_path,
        output_dir              = args.output_dir,
        max_tokens              = args.max_tokens,
        temperature             = args.temperature,
        gpu_memory_utilization  = args.gpu_memory_utilization,
        max_model_len           = args.max_model_len,
        max_num_seqs            = args.max_num_seqs,
    )
    print(f"\nDone. Submission CSV: {csv_out}")

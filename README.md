# CSE 151B Spring 2026 — Math Reasoning Competition

Fine-tuning **Qwen3-4B-Thinking-2507** with QLoRA for mathematical problem solving (MCQ + free-form).

---

## Hardware & Runtime

| Item | Details |
|------|---------|
| GPU | NVIDIA A100 (Google Colab) |
| VRAM | 80 GB |
| Fine-tuning time | ~2–3 hours (3 epochs, 3K examples) |
| Inference time | ~2 hours for full private set (vLLM, batch) |

---

## Model Weights

Our fine-tuned QLoRA checkpoint is hosted on HuggingFace Hub:

[favioe/cse151B_qlora_merged](https://huggingface.co/favioe/cse151B_qlora_merged)


`run_inference.py` loads the model directly from the Hub — no manual download needed. If you want to run offline, clone the repo locally first:

```bash
huggingface-cli download favioe/cse151B_qlora_merged --local-dir ./model_weights
```

Then pass `--model_id ./model_weights` to the script.

---

## Setup

```bash
# 1. Clone this repo
https://github.com/pprabu-ucsd/CSE-151B-Spring-2026-Competition.git
cd CSE-151B-Spring-2026-Competition

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running Inference

### Option A — Python import

```python
from run_inference import run_inference

csv_path = run_inference(
    model_id   = "favioe/cse151B_qlora_merged",
    data_path  = "/data/private.jsonl",
    output_dir = "./results",
)
print(f"Submission written to: {csv_path}")
```

### Option B — Command line

```bash
python run_inference.py \
    --model_id   "favioe/cse151B_qlora_merged" \
    --data_path  "/data/private.jsonl" \
    --output_dir "./results"
```

Both produce:
- `results/submission.csv` — final submission (`id`, `response` columns)
- `results/private_responses.jsonl` — raw model outputs backup

---

## Hyperparameters

### Inference (fixed for submission)

| Parameter | Value |
|-----------|-------|
| `temperature` | `0.0` (greedy) |
| `max_tokens` | `8192` |
| `gpu_memory_utilization` | `0.90` |
| `max_model_len` | `8192` |
| `max_num_seqs` | `32` |

### Fine-tuning

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen3-4B-Thinking-2507` |
| Quantization | NF4 4-bit (BitsAndBytes) |
| LoRA rank | `64` |
| LoRA alpha | `128` |
| Target modules | `q/k/v/o_proj`, `gate/up/down_proj` |
| LoRA dropout | `0.05` |
| Epochs | `2` |
| Batch size | `4` (grad accum `4` → effective `16`) |
| Learning rate | `2e-4` |
| LR scheduler | Cosine |
| Warmup ratio | `0.05` |
| Max sequence length | `2048` |
| Training examples | `≤ 20,000` (MetaMathQA + MATH + GSM8K) |

---

## Training Datasets

| Dataset | Examples used | Source |
|---------|--------------|--------|
| MetaMathQA | 12,000 | `meta-math/MetaMathQA` |
| MATH (Hendrycks) | ~7,500 (full) | `hendrycks/competition_math` |
| GSM8K | 7,473 (full) | `openai/gsm8k` |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── judger.py
├── utils.py
├── run_inference.py          # Single entry point — call this to reproduce results
├── code_cse151b_comp.ipynb
├── code_cse151b_comp_v2.ipynb # Full training + evaluation notebook
└── data/                      # Will be created during run
    ├── public.jsonl
    └── private.jsonl
└── results/                      # Will be created during run
    ├── submission.csv            # Final submission file
    └── private_responses.jsonl  # Raw model output backup
```

---

## Group Members

- Pranav Prabu (A17424120)
- Favio Espejo (A16153834)

# 04 — Fine-tuning

Industry-level LLM fine-tuning boilerplate covering LoRA, QLoRA, full fine-tune, and DPO.

## Structure

```
04-finetuning/
├── config.py                       # Pydantic Settings (base model, LoRA, QLoRA, training, DPO)
├── .env.example
├── techniques/
│   ├── lora.py                     # LoRA via PEFT — apply, load, merge & save
│   ├── qlora.py                    # QLoRA — 4-bit quantization + LoRA adapters
│   ├── full_finetune.py            # Full fine-tune with LLRD + differential LR groups
│   └── dpo.py                      # Direct Preference Optimization via TRL
├── data/
│   └── dataset.py                  # SFT dataset builder — Alpaca, ShareGPT, ChatML, Llama3
└── examples/
    └── finetune_llm_lora.py        # End-to-end LoRA / QLoRA SFT training run
```

## Techniques Comparison

| Technique      | VRAM     | Speed  | Quality   | When to use                        |
| -------------- | -------- | ------ | --------- | ---------------------------------- |
| Full fine-tune | High     | Slow   | Best      | Small models, abundant compute     |
| LoRA           | Medium   | Fast   | Good      | Most use cases, modularity         |
| QLoRA (4-bit)  | Very Low | Medium | Good      | Large models (7B+) on consumer GPU |
| DPO            | Medium   | Medium | Alignment | Instruction following, safety      |

## Quick Start (LoRA / QLoRA)

```bash
cp .env.example .env
# Edit BASE_MODEL_NAME, HF_TOKEN, LOAD_IN_4BIT

python examples/finetune_llm_lora.py
```

## Key Config Options

| Variable              | Description                                      |
| --------------------- | ------------------------------------------------ |
| `BASE_MODEL_NAME`     | HuggingFace model ID or local path               |
| `LOAD_IN_4BIT`        | `true` for QLoRA, `false` for standard LoRA      |
| `LORA_R`              | LoRA rank (16 standard, 64 for aggressive QLoRA) |
| `LORA_ALPHA`          | Scaling (usually 2×r for LoRA, ≤r for QLoRA)     |
| `LORA_TARGET_MODULES` | Comma-separated linear layers to adapt           |
| `MAX_SEQ_LENGTH`      | Maximum context length during training           |
| `MIXED_PRECISION`     | `bf16` (recommended on Ampere+) or `fp16`        |

## Supported Prompt Templates

```
alpaca   → Alpaca instruction format (default)
chatml   → ChatML / Mistral / Qwen format
phi3     → Microsoft Phi-3 format
llama3   → Llama 3 instruct format
```

## After Training

```python
# Option A: Load LoRA adapter for inference
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(base, "./ft-outputs/final")

# Option B: Merge adapter into base model (no PEFT at inference)
from techniques.lora import merge_and_save_lora
merge_and_save_lora(peft_model, "./merged-model")
```

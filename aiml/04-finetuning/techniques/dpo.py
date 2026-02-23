"""
DPO (Direct Preference Optimization) — aligns LLMs to human preferences directly.

Technique: Trains on preference pairs (chosen, rejected) without a separate reward model.
Compared to RLHF: simpler pipeline, no RM training, no PPO complexity.
Best for: Instruction following, safety alignment, style control.

Requirements: pip install trl peft transformers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


def build_dpo_dataset_from_dict(
    data: List[Dict],
    tokenizer,
    max_length: int = 1024,
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
) -> Dataset:
    """
    Build a DPO dataset from a list of dicts.

    Each dict must have:
        {prompt: ..., chosen: ..., rejected: ...}
    where 'chosen' and 'rejected' are full conversation completions.

    Returns a HuggingFace Dataset with tokenized columns.
    """
    processed = []
    for sample in data:
        processed.append(
            {
                "prompt": sample[prompt_key],
                "chosen": sample[chosen_key],
                "rejected": sample[rejected_key],
            }
        )
    return Dataset.from_list(processed)


def run_dpo_training(
    model_name: str,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    output_dir: str = "./dpo-output",
    beta: float = 0.1,
    learning_rate: float = 5e-5,
    num_train_epochs: int = 1,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 1024,
    max_prompt_length: int = 512,
    warmup_ratio: float = 0.1,
    hf_token: Optional[str] = None,
    peft_model: Optional[PeftModel] = None,
) -> None:
    """
    Run DPO training using TRL's DPOTrainer.

    Args:
        model_name:                Base model name / path.
        train_dataset:             DPO-formatted HF Dataset.
        beta:                      DPO temperature — higher = closer to base model.
        peft_model:                If provided, train LoRA adapters on top.
    """
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        raise ImportError("TRL required: pip install trl")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if peft_model is not None:
        model = peft_model
        ref_model = None  # PeftModel handles reference via adapter disable
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", token=hf_token
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", token=hf_token
        )
        ref_model.eval()

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    logger.info("DPO training complete. Model saved to %s", output_dir)

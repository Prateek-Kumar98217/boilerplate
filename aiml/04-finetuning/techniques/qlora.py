"""
QLoRA (Quantized LoRA) — 4-bit quantization + LoRA adapters.

Technique: Quantize base model to NF4/FP4 4-bit, then LoRA on dequantized matrices.
Memory savings: ~90%+ vs full fp16. Enables fine-tuning 65B+ on consumer GPUs.
Best for: Very large models (7B–70B+) on limited VRAM.

Requirements: pip install peft transformers bitsandbytes accelerate
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel

logger = logging.getLogger(__name__)


def get_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_quant_type: str = "nf4",
    use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """
    Build BitsAndBytesConfig for 4-bit or 8-bit quantization.

    Args:
        load_in_4bit:             Use NF4/FP4 4-bit quantization.
        bnb_4bit_compute_dtype:   Dtype for computation (bfloat16 or float16).
        bnb_4bit_quant_type:      Quantization scheme: 'nf4' (recommended) or 'fp4'.
        use_double_quant:         Quantize the quantization constants for extra memory savings.

    Returns:
        BitsAndBytesConfig instance.
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
    )


def load_model_for_qlora(
    model_name: str,
    bnb_config: Optional[BitsAndBytesConfig] = None,
    device_map: str = "auto",
    hf_token: Optional[str] = None,
) -> PreTrainedModel:
    """
    Load base model quantized for QLoRA training.
    Automatically calls prepare_model_for_kbit_training.
    """
    if bnb_config is None:
        bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        token=hf_token,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1  # Disable tensor parallelism for training

    # Critical: prepares the quantized model for gradient checkpointing compatibility
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    logger.info("QLoRA base model loaded and prepared: %s", model_name)
    return model


def apply_qlora(
    model: PreTrainedModel,
    r: int = 64,  # Higher r for QLoRA recommended
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
) -> PeftModel:
    """
    Apply QLoRA: LoRA adapters on top of a quantized base model.

    QLoRA-recommended settings differ from standard LoRA:
    - Higher r (32-64) is common to compensate for quantization noise.
    - Lower lora_alpha (often ≤ r) is typical.
    """
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model

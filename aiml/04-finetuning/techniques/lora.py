"""
LoRA (Low-Rank Adaptation) fine-tuning using PEFT.

Technique: Adds low-rank matrices to target linear projections.
Memory savings: ~70-80% vs full fine-tune for 7B+ models.
Best for: Low-data regimes, adapter-based modularity, fast switching.

Requirements: pip install peft transformers accelerate
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

logger = logging.getLogger(__name__)


def load_base_model_for_lora(
    model_name: str,
    task_type: str = "CAUSAL_LM",  # CAUSAL_LM | SEQ_CLS | SEQ_2_SEQ_LM
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    hf_token: Optional[str] = None,
) -> PreTrainedModel:
    """Load a base HuggingFace model ready for LoRA."""
    task_cls_map = {
        "CAUSAL_LM": AutoModelForCausalLM,
        "SEQ_CLS": AutoModelForSequenceClassification,
    }
    cls = task_cls_map.get(task_type.upper(), AutoModelForCausalLM)

    model = cls.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # Required for gradient checkpointing compat
    return model


def apply_lora(
    model: PreTrainedModel,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> PeftModel:
    """
    Wrap a model with LoRA adapters.

    Args:
        r:               Rank of low-rank matrices (default 16 = good balance).
        lora_alpha:      LoRA scaling factor (usually 2×r).
        lora_dropout:    Dropout on LoRA layers.
        target_modules:  Which linear layers to adapt. None = auto-detect.
        bias:            How to handle bias: 'none' | 'all' | 'lora_only'.
        task_type:       PEFT TaskType string.

    Returns:
        PeftModel with LoRA adapters.
    """
    peft_task = getattr(TaskType, task_type.upper())
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=peft_task,
        inference_mode=False,
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_lora_model(
    base_model_name: str,
    adapter_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    hf_token: Optional[str] = None,
) -> PeftModel:
    """Load a saved LoRA adapter on top of a base model."""
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    logger.info("LoRA adapter loaded from %s", adapter_path)
    return model


def merge_and_save_lora(
    peft_model: PeftModel,
    output_dir: Union[str, Path],
) -> None:
    """
    Merge LoRA weights into the base model and save as a standard HF checkpoint.
    Useful for deployment (no PEFT dependency needed at inference time).
    """
    merged = peft_model.merge_and_unload()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))
    logger.info("Merged model saved to %s", output_dir)
    return merged

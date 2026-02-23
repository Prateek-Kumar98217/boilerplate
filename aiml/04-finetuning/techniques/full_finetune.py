"""
Full fine-tuning — all parameters updated, no PEFT adapters.

Best for: Small models (<1B), situations where task performance is critical
         and you have ample compute and data.
Techniques included:
- Layer-wise learning rate decay (LLRD)
- Parameter groups with differential weight decay
- Gradient checkpointing
- Flash Attention 2 (optional)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)

logger = logging.getLogger(__name__)


def load_model_for_full_finetune(
    model_name: str,
    task: str = "CAUSAL_LM",
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "eager",  # eager | sdpa | flash_attention_2
    device_map: Optional[str] = None,
    hf_token: Optional[str] = None,
    num_labels: int = 2,
) -> PreTrainedModel:
    """
    Load a model for full fine-tuning.

    Args:
        attn_implementation: 'flash_attention_2' requires FA2 install.
    """
    kwargs = dict(
        torch_dtype=torch_dtype,
        token=hf_token,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    if device_map:
        kwargs["device_map"] = device_map

    if task.upper() == "SEQ_CLS":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, **kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    logger.info("Model loaded for full fine-tune: %s", model_name)
    return model


def get_llrd_param_groups(
    model: PreTrainedModel,
    base_lr: float = 2e-5,
    decay_factor: float = 0.9,
    weight_decay: float = 0.01,
) -> List[Dict]:
    """
    Layer-wise Learning Rate Decay (LLRD).
    Assigns progressively smaller LRs to lower layers (closer to embedding).
    Empirically improves fine-tuning stability on pre-trained LMs.

    Args:
        base_lr:      LR for the top-most (classification) layer.
        decay_factor: Multiplier per layer going down (0.9 = 10% decay per layer).
        weight_decay: Applied to non-bias/norm params.
    """
    named_params = list(model.named_parameters())
    # Attempt to detect layer names heuristically
    layer_names: List[List] = []
    all_layers = []
    for name, _ in named_params:
        parts = name.split(".")
        for part in parts:
            if part.isdigit():
                layer_idx = int(part)
                while len(all_layers) <= layer_idx:
                    all_layers.append([])
                all_layers[layer_idx].append(name)
                break

    param_groups = []
    n_layers = max(len(all_layers), 1)
    param_set = set()

    for layer_idx in reversed(range(n_layers)):
        lr = base_lr * (decay_factor ** (n_layers - 1 - layer_idx))
        layer_params, layer_no_decay = [], []
        for name in all_layers[layer_idx]:
            _, param = dict(named_params)[name], next(
                p for n, p in named_params if n == name
            )
            if "bias" in name or "norm" in name or "layernorm" in name.lower():
                layer_no_decay.append(param)
            else:
                layer_params.append(param)
            param_set.add(name)

        if layer_params:
            param_groups.append(
                {"params": layer_params, "lr": lr, "weight_decay": weight_decay}
            )
        if layer_no_decay:
            param_groups.append(
                {"params": layer_no_decay, "lr": lr, "weight_decay": 0.0}
            )

    # Params not in any identified layer (embeddings, head)
    remaining = [p for n, p in named_params if n not in param_set and p.requires_grad]
    if remaining:
        param_groups.append(
            {"params": remaining, "lr": base_lr, "weight_decay": weight_decay}
        )

    return param_groups


def get_differential_param_groups(
    model: PreTrainedModel,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> List[Dict]:
    """
    Differential LRs: smaller for backbone, bigger for task head.
    Simple but effective split for classification fine-tuning.
    """
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "pooler" in name or "score" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
    ]

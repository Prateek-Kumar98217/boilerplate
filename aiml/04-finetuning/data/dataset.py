"""
Instruction dataset builder for supervised fine-tuning (SFT).

Supports:
- Alpaca-format (instruction / input / output)
- ShareGPT / ChatML format (conversations)
- Plain text (completion only)
- Custom format via template function
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Union

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task{input_suffix}. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_section}"
    "### Response:\n{output}"
)

CHATML_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

PHI3_TEMPLATE = "<|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>"

LLAMA3_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "{output}<|eot_id|>"
)

TEMPLATE_MAP = {
    "alpaca": ALPACA_TEMPLATE,
    "chatml": CHATML_TEMPLATE,
    "phi3": PHI3_TEMPLATE,
    "llama3": LLAMA3_TEMPLATE,
}


def format_alpaca(
    example: Dict,
    instruction_col: str = "instruction",
    input_col: str = "input",
    output_col: str = "output",
    template: str = "alpaca",
) -> str:
    """Format a single example using the chosen template."""
    instruction = example.get(instruction_col, "")
    inp = example.get(input_col, "")
    output = example.get(output_col, "")

    if template == "alpaca":
        input_suffix = ", using the input below as additional context" if inp else ""
        input_section = f"### Input:\n{inp}\n\n" if inp else ""
        return ALPACA_TEMPLATE.format(
            input_suffix=input_suffix,
            instruction=instruction,
            input_section=input_section,
            output=output,
        )
    elif template in TEMPLATE_MAP:
        combined_instruction = f"{instruction}\n\n{inp}" if inp else instruction
        return TEMPLATE_MAP[template].format(
            instruction=combined_instruction, output=output
        )
    else:
        raise ValueError(f"Unknown template: {template}")


def format_sharegpt(conversation: List[Dict]) -> str:
    """Format ShareGPT-style conversations into ChatML."""
    result = []
    for turn in conversation:
        role = turn.get("from", turn.get("role", "user"))
        content = turn.get("value", turn.get("content", ""))
        result.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(result)


def build_sft_dataset(
    dataset_name_or_path: str,
    split: str = "train",
    instruction_col: str = "instruction",
    input_col: str = "input",
    output_col: str = "output",
    template: str = "alpaca",
    max_samples: Optional[int] = None,
    hf_token: Optional[str] = None,
    custom_format_fn: Optional[Callable[[Dict], str]] = None,
) -> Dataset:
    """
    Load a dataset from HuggingFace Hub or local path and format for SFT.

    Returns a Dataset with a single "text" column containing formatted prompts.
    """
    ds = load_dataset(dataset_name_or_path, split=split, token=hf_token)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def format_fn(example):
        if custom_format_fn:
            text = custom_format_fn(example)
        elif "conversations" in example:
            text = format_sharegpt(example["conversations"])
        else:
            text = format_alpaca(
                example, instruction_col, input_col, output_col, template
            )
        return {"text": text}

    ds = ds.map(format_fn, remove_columns=ds.column_names)
    logger.info("SFT dataset: %d samples from %s", len(ds), dataset_name_or_path)
    return ds

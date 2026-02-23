"""
SFT fine-tuning with LoRA using TRL's SFTTrainer.

Supports: LoRA and QLoRA (set LOAD_IN_4BIT=true in .env).

Run:
    python examples/finetune_llm_lora.py
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoTokenizer

from config import get_settings
from data.dataset import build_sft_dataset
from techniques.lora import apply_lora, load_base_model_for_lora
from techniques.qlora import apply_qlora, get_bnb_config, load_model_for_qlora

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        raise ImportError("TRL required: pip install trl")

    settings = get_settings()
    cfg_m = settings.base_model
    cfg_l = settings.lora
    cfg_q = settings.qlora
    cfg_t = settings.training
    cfg_d = settings.data

    logger.info("Loading tokenizer: %s", cfg_m.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg_m.model_name, token=cfg_m.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # ── Load model (LoRA vs QLoRA) ─────────────────────────────────────
    if cfg_q.load_in_4bit:
        logger.info("Loading model in 4-bit (QLoRA)...")
        bnb_config = get_bnb_config(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=cfg_q.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=cfg_q.bnb_4bit_quant_type,
            use_double_quant=cfg_q.use_double_quant,
        )
        model = load_model_for_qlora(
            cfg_m.model_name, bnb_config=bnb_config, hf_token=cfg_m.hf_token
        )
        model = apply_qlora(
            model,
            r=cfg_l.r,
            lora_alpha=cfg_l.alpha,
            lora_dropout=cfg_l.dropout,
            target_modules=cfg_l.target_modules_list,
            bias=cfg_l.bias,
        )
    else:
        logger.info("Loading model in bf16 (LoRA)...")
        model = load_base_model_for_lora(
            cfg_m.model_name,
            torch_dtype=torch.bfloat16,
            hf_token=cfg_m.hf_token,
        )
        model = apply_lora(
            model,
            r=cfg_l.r,
            lora_alpha=cfg_l.alpha,
            lora_dropout=cfg_l.dropout,
            target_modules=cfg_l.target_modules_list,
            bias=cfg_l.bias,
        )

    # ── Dataset ───────────────────────────────────────────────────────
    train_dataset = build_sft_dataset(
        cfg_d.dataset_name,
        split=cfg_d.dataset_split,
        instruction_col=cfg_d.instruction_column,
        input_col=cfg_d.input_column,
        output_col=cfg_d.output_column,
        template="alpaca",
        hf_token=cfg_m.hf_token,
    )

    # ── SFT Trainer ───────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=str(cfg_t.output_dir),
        num_train_epochs=cfg_t.epochs,
        per_device_train_batch_size=cfg_t.batch_size,
        gradient_accumulation_steps=cfg_t.gradient_accumulation_steps,
        learning_rate=cfg_t.learning_rate,
        weight_decay=cfg_t.weight_decay,
        warmup_ratio=cfg_t.warmup_ratio,
        bf16=(cfg_t.mixed_precision == "bf16"),
        fp16=(cfg_t.mixed_precision == "fp16"),
        max_grad_norm=cfg_t.gradient_clipping,
        max_seq_length=cfg_t.max_seq_length,
        logging_steps=cfg_t.logging_steps,
        save_steps=cfg_t.save_steps,
        save_total_limit=cfg_t.save_total_limit,
        seed=cfg_t.seed,
        dataset_text_field="text",
        report_to="wandb" if settings.data.dataset_name else "none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()
    trainer.save_model(str(cfg_t.output_dir / "final"))
    tokenizer.save_pretrained(str(cfg_t.output_dir / "final"))
    logger.info("Fine-tuning complete. Model saved to %s", cfg_t.output_dir / "final")


if __name__ == "__main__":
    main()

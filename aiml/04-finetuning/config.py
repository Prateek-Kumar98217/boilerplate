"""Finetuning section config — Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseModelSettings(BaseSettings):
    model_name: str = Field("meta-llama/Llama-3.2-1B-Instruct", alias="BASE_MODEL_NAME")
    hf_token: str = Field("", alias="HF_TOKEN")
    hf_cache_dir: str = Field("~/.cache/huggingface", alias="HF_CACHE_DIR")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class LoRASettings(BaseSettings):
    r: int = Field(16, alias="LORA_R")
    alpha: int = Field(32, alias="LORA_ALPHA")
    dropout: float = Field(0.05, alias="LORA_DROPOUT")
    target_modules: str = Field(
        "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        alias="LORA_TARGET_MODULES",
    )
    bias: str = Field("none", alias="LORA_BIAS")

    @property
    def target_modules_list(self) -> List[str]:
        return [m.strip() for m in self.target_modules.split(",")]

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class QLoRASettings(BaseSettings):
    load_in_4bit: bool = Field(True, alias="LOAD_IN_4BIT")
    bnb_4bit_compute_dtype: str = Field("bfloat16", alias="BNB_4BIT_COMPUTE_DTYPE")
    bnb_4bit_quant_type: str = Field("nf4", alias="BNB_4BIT_QUANT_TYPE")
    use_double_quant: bool = Field(True, alias="BNB_USE_DOUBLE_QUANT")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class FTTrainingSettings(BaseSettings):
    output_dir: Path = Field(Path("./ft-outputs"), alias="OUTPUT_DIR")
    epochs: int = Field(3, alias="EPOCHS")
    batch_size: int = Field(4, alias="BATCH_SIZE")
    gradient_accumulation_steps: int = Field(4, alias="GRADIENT_ACCUMULATION_STEPS")
    learning_rate: float = Field(2e-4, alias="LEARNING_RATE")
    weight_decay: float = Field(0.01, alias="WEIGHT_DECAY")
    warmup_ratio: float = Field(0.03, alias="WARMUP_RATIO")
    max_seq_length: int = Field(2048, alias="MAX_SEQ_LENGTH")
    mixed_precision: str = Field("bf16", alias="MIXED_PRECISION")
    gradient_clipping: float = Field(1.0, alias="GRADIENT_CLIPPING")
    seed: int = Field(42, alias="SEED")
    save_steps: int = Field(100, alias="SAVE_STEPS")
    eval_steps: int = Field(100, alias="EVAL_STEPS")
    logging_steps: int = Field(10, alias="LOGGING_STEPS")
    save_total_limit: int = Field(3, alias="SAVE_TOTAL_LIMIT")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class DataSettings(BaseSettings):
    dataset_name: str = Field("tatsu-lab/alpaca", alias="DATASET_NAME")
    dataset_split: str = Field("train", alias="DATASET_SPLIT")
    instruction_column: str = Field("instruction", alias="INSTRUCTION_COLUMN")
    input_column: str = Field("input", alias="INPUT_COLUMN")
    output_column: str = Field("output", alias="OUTPUT_COLUMN")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class DPOSettings(BaseSettings):
    beta: float = Field(0.1, alias="DPO_BETA")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class Settings(BaseSettings):
    base_model: BaseModelSettings = BaseModelSettings()
    lora: LoRASettings = LoRASettings()
    qlora: QLoRASettings = QLoRASettings()
    training: FTTrainingSettings = FTTrainingSettings()
    data: DataSettings = DataSettings()
    dpo: DPOSettings = DPOSettings()

    model_config = SettingsConfigDict(extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

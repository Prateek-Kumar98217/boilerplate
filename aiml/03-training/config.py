"""Training section config — Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class HardwareSettings(BaseSettings):
    device: str = Field("auto", alias="DEVICE")
    num_workers: int = Field(4, alias="NUM_WORKERS")
    pin_memory: bool = Field(True, alias="PIN_MEMORY")
    mixed_precision: str = Field(
        "true", alias="MIXED_PRECISION"
    )  # bf16|fp16|true|false

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class ExperimentSettings(BaseSettings):
    experiment_name: str = Field("my_experiment", alias="EXPERIMENT_NAME")
    run_name: str = Field("run_001", alias="RUN_NAME")
    output_dir: Path = Field(Path("./outputs"), alias="OUTPUT_DIR")
    resume_from_checkpoint: Optional[Path] = Field(None, alias="RESUME_FROM_CHECKPOINT")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class TrainingLoopSettings(BaseSettings):
    epochs: int = Field(50, alias="EPOCHS")
    batch_size: int = Field(32, alias="BATCH_SIZE")
    eval_every_n_epochs: int = Field(1, alias="EVAL_EVERY_N_EPOCHS")
    save_every_n_epochs: int = Field(5, alias="SAVE_EVERY_N_EPOCHS")
    early_stopping_patience: int = Field(10, alias="EARLY_STOPPING_PATIENCE")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class OptimizerSettings(BaseSettings):
    optimizer: str = Field("adamw", alias="OPTIMIZER")
    learning_rate: float = Field(1e-3, alias="LEARNING_RATE")
    weight_decay: float = Field(1e-4, alias="WEIGHT_DECAY")
    momentum: float = Field(0.9, alias="MOMENTUM")
    betas: Tuple[float, float] = Field((0.9, 0.999), alias="BETAS")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class GradientSettings(BaseSettings):
    gradient_clipping: float = Field(1.0, alias="GRADIENT_CLIPPING")
    gradient_accumulation_steps: int = Field(1, alias="GRADIENT_ACCUMULATION_STEPS")
    gradient_checkpointing: bool = Field(False, alias="GRADIENT_CHECKPOINTING")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class SchedulerSettings(BaseSettings):
    scheduler: str = Field("cosine_warmup", alias="SCHEDULER")
    warmup_steps: int = Field(100, alias="WARMUP_STEPS")
    warmup_ratio: float = Field(0.0, alias="WARMUP_RATIO")
    lr_step_size: int = Field(10, alias="LR_STEP_SIZE")
    lr_gamma: float = Field(0.1, alias="LR_GAMMA")
    lr_min: float = Field(1e-6, alias="LR_MIN")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class RegularizationSettings(BaseSettings):
    dropout: float = Field(0.1, alias="DROPOUT")
    label_smoothing: float = Field(0.0, alias="LABEL_SMOOTHING")
    stochastic_depth_rate: float = Field(0.0, alias="STOCHASTIC_DEPTH_RATE")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class TrackingSettings(BaseSettings):
    wandb_api_key: str = Field("", alias="WANDB_API_KEY")
    wandb_project: str = Field("my-training-project", alias="WANDB_PROJECT")
    mlflow_tracking_uri: str = Field(
        "http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )
    log_every_n_steps: int = Field(50, alias="LOG_EVERY_N_STEPS")
    seed: int = Field(42, alias="SEED")
    deterministic: bool = Field(False, alias="DETERMINISTIC")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class Settings(BaseSettings):
    hardware: HardwareSettings = HardwareSettings()
    experiment: ExperimentSettings = ExperimentSettings()
    loop: TrainingLoopSettings = TrainingLoopSettings()
    optimizer: OptimizerSettings = OptimizerSettings()
    gradient: GradientSettings = GradientSettings()
    scheduler: SchedulerSettings = SchedulerSettings()
    regularization: RegularizationSettings = RegularizationSettings()
    tracking: TrackingSettings = TrackingSettings()

    model_config = SettingsConfigDict(extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

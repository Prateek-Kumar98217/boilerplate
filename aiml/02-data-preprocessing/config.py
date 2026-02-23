"""Data-preprocessing section config — Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TextSettings(BaseSettings):
    max_length: int = Field(512, alias="MAX_TEXT_LENGTH")
    language: str = Field("en", alias="TEXT_LANGUAGE")
    spacy_model: str = Field("en_core_web_sm", alias="SPACY_MODEL")
    hf_tokenizer: str = Field("bert-base-uncased", alias="HF_TOKENIZER_MODEL")
    hf_token: str = Field("", alias="HF_TOKEN")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class AudioSettings(BaseSettings):
    sample_rate: int = Field(16000, alias="AUDIO_SAMPLE_RATE")
    n_mels: int = Field(80, alias="AUDIO_N_MELS")
    hop_length: int = Field(160, alias="AUDIO_HOP_LENGTH")
    win_length: int = Field(400, alias="AUDIO_WIN_LENGTH")
    whisper_model_size: str = Field("base", alias="WHISPER_MODEL_SIZE")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class ImageSettings(BaseSettings):
    size: int = Field(224, alias="IMAGE_SIZE")
    normalize_mean: Tuple[float, float, float] = Field(
        (0.485, 0.456, 0.406), alias="IMAGE_NORMALIZE_MEAN"
    )
    normalize_std: Tuple[float, float, float] = Field(
        (0.229, 0.224, 0.225), alias="IMAGE_NORMALIZE_STD"
    )

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class VideoSettings(BaseSettings):
    fps: int = Field(1, alias="VIDEO_FPS")
    max_frames: int = Field(64, alias="VIDEO_MAX_FRAMES")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class MLDataSettings(BaseSettings):
    missing_strategy: str = Field("median", alias="MISSING_VALUE_STRATEGY")
    outlier_method: str = Field("iqr", alias="OUTLIER_METHOD")
    outlier_threshold: float = Field(3.0, alias="OUTLIER_THRESHOLD")
    test_size: float = Field(0.2, alias="TEST_SIZE")
    val_size: float = Field(0.1, alias="VAL_SIZE")
    random_seed: int = Field(42, alias="RANDOM_SEED")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class Settings(BaseSettings):
    text: TextSettings = TextSettings()
    audio: AudioSettings = AudioSettings()
    image: ImageSettings = ImageSettings()
    video: VideoSettings = VideoSettings()
    ml: MLDataSettings = MLDataSettings()

    model_config = SettingsConfigDict(extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

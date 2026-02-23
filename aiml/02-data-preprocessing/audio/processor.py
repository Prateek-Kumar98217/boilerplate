"""
Audio preprocessing utilities.

Features:
- Resampling to target sample rate
- Mel spectrogram extraction
- MFCC extraction
- Waveform normalization + silence trimming
- Whisper-compatible preprocessing
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    import librosa
    import librosa.effects

    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

try:
    import torch
    import torchaudio
    import torchaudio.transforms as T

    _TORCH_AUDIO_AVAILABLE = True
except ImportError:
    _TORCH_AUDIO_AVAILABLE = False


def _require_librosa():
    if not _LIBROSA_AVAILABLE:
        raise ImportError("librosa required: pip install librosa")


def _require_torchaudio():
    if not _TORCH_AUDIO_AVAILABLE:
        raise ImportError("torchaudio required: pip install torchaudio")


class AudioProcessor:
    """
    Comprehensive audio preprocessing for ML pipelines.

    Example:
        proc = AudioProcessor(sample_rate=16000, n_mels=80)
        waveform, sr = proc.load("audio.wav")
        mel = proc.mel_spectrogram(waveform)
        mfcc = proc.mfcc(waveform)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_mfcc: int = 40,
        hop_length: int = 160,
        win_length: int = 400,
        n_fft: int = 512,
        top_db: float = 80.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.top_db = top_db

    # ── Loading ───────────────────────────────────────────────────────

    def load(self, path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file, resample to target rate, return (waveform, sr)."""
        _require_librosa()
        waveform, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        return waveform, sr

    def load_torch(self, path: Union[str, Path]):
        """Load audio as torch Tensor (C, T)."""
        _require_torchaudio()
        waveform, sr = torchaudio.load(str(path))
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform

    # ── Transforms ────────────────────────────────────────────────────

    def mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram in dB."""
        _require_librosa()
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
        )
        return librosa.power_to_db(mel, ref=np.max, top_db=self.top_db)

    def mfcc(self, waveform: np.ndarray) -> np.ndarray:
        """Compute MFCCs."""
        _require_librosa()
        return librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
        )

    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Peak normalize waveform to [-1, 1]."""
        peak = np.max(np.abs(waveform))
        return waveform / (peak + 1e-8)

    def trim_silence(self, waveform: np.ndarray, top_db: float = 20.0) -> np.ndarray:
        """Trim leading/trailing silence."""
        _require_librosa()
        trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
        return trimmed

    def pad_or_trim(self, waveform: np.ndarray, target_length: int) -> np.ndarray:
        """Pad with zeros or trim to fixed length."""
        if len(waveform) >= target_length:
            return waveform[:target_length]
        return np.pad(waveform, (0, target_length - len(waveform)))

    # ── Whisper-compatible preprocessing ─────────────────────────────

    def to_whisper_input(self, waveform: np.ndarray) -> np.ndarray:
        """
        Produce a log-mel spectrogram compatible with OpenAI Whisper.
        Returns array (n_mels, ~3000) for 30-second chunks.
        """
        _require_librosa()
        # Pad/trim to 30 seconds at self.sample_rate
        max_len = self.sample_rate * 30
        waveform = self.pad_or_trim(waveform, max_len)
        mel = self.mel_spectrogram(waveform)
        # Normalize to [-1, 1] range expected by Whisper encoder
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return mel

    # ── Full pipeline ─────────────────────────────────────────────────

    def process_file(self, path: Union[str, Path]) -> dict:
        waveform, sr = self.load(path)
        waveform = self.normalize(waveform)
        waveform = self.trim_silence(waveform)
        return {
            "waveform": waveform,
            "sample_rate": sr,
            "duration_s": len(waveform) / sr,
            "mel_spectrogram": self.mel_spectrogram(waveform),
            "mfcc": self.mfcc(waveform),
        }

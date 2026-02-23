"""
Video preprocessing utilities.

Covers:
- Frame extraction at configurable FPS
- Frame-level transforms (reuse ImageProcessor)
- Scene detection
- Temporal sampling strategies (uniform, random, key-frame)
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_cv2():
    if not _CV2_AVAILABLE:
        raise ImportError("OpenCV required: pip install opencv-python")


class VideoProcessor:
    """
    Flexible video frame extractor and preprocessor.

    Example:
        proc = VideoProcessor(target_fps=1, max_frames=64, resize=(224, 224))
        frames = proc.extract_frames("clip.mp4")   # List of np.ndarray (H, W, C)
        tensor = proc.to_tensor(frames)            # (T, C, H, W)
    """

    def __init__(
        self,
        target_fps: Optional[float] = 1.0,
        max_frames: int = 64,
        resize: Optional[Tuple[int, int]] = (224, 224),  # (W, H)
        color_mode: str = "RGB",  # RGB | BGR | GRAY
    ) -> None:
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.resize = resize
        self.color_mode = color_mode

    # ── Frame extraction ──────────────────────────────────────────────

    def iter_frames(self, path: Union[str, Path]) -> Iterator[np.ndarray]:
        """Yield raw BGR frames at the target FPS."""
        _require_cv2()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = (
            max(1, round(native_fps / self.target_fps)) if self.target_fps else 1
        )
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    yield frame
                frame_idx += 1
        finally:
            cap.release()

    def extract_frames(
        self,
        path: Union[str, Path],
        sampling: str = "uniform",  # uniform | random | all
    ) -> List[np.ndarray]:
        """
        Extract up to `max_frames` frames with chosen sampling strategy.

        Args:
            path: Video file path.
            sampling: 'uniform' (evenly spaced), 'random', 'all'.
        Returns:
            List of frames as np.ndarray (H, W, C) in self.color_mode.
        """
        all_frames = list(self.iter_frames(path))

        if sampling == "random" and len(all_frames) > self.max_frames:
            idx = sorted(
                np.random.choice(len(all_frames), self.max_frames, replace=False)
            )
            all_frames = [all_frames[i] for i in idx]
        elif sampling == "uniform" and len(all_frames) > self.max_frames:
            indices = np.linspace(0, len(all_frames) - 1, self.max_frames, dtype=int)
            all_frames = [all_frames[i] for i in indices]
        else:
            all_frames = all_frames[: self.max_frames]

        return [self._postprocess_frame(f) for f in all_frames]

    def _postprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.resize:
            frame = cv2.resize(frame, self.resize)
        if self.color_mode == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.color_mode == "GRAY":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    # ── Tensor conversion ─────────────────────────────────────────────

    def to_tensor(self, frames: List[np.ndarray]) -> "torch.Tensor":
        """Stack frames → (T, C, H, W) float tensor in [0, 1]."""
        if not _TORCH_AVAILABLE:
            raise ImportError("torch required: pip install torch")
        arr = np.stack(frames)  # (T, H, W, C) or (T, H, W)
        if arr.ndim == 3:
            arr = arr[:, :, :, np.newaxis]  # GRAY → (T, H, W, 1)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        return tensor

    # ── Video metadata ────────────────────────────────────────────────

    def get_metadata(self, path: Union[str, Path]) -> dict:
        _require_cv2()
        cap = cv2.VideoCapture(str(path))
        meta = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        meta["duration_s"] = meta["frame_count"] / (meta["fps"] or 1)
        cap.release()
        return meta

"""
Rate-limit-aware API key rotator.

Supports proactive local counting (RPM / RPD / TPM) and reactive handling
of HTTP 429 responses.  Thread-safe via asyncio.Lock.

Usage:
    rotator = KeyRotator(
        keys=["key1", "key2", "key3"],
        rpm_limit=30,
        rpd_limit=1500,
        tpm_limit=131072,
    )
    key = await rotator.acquire()          # blocks until a key is available
    try:
        response = call_api(key, ...)
    except RateLimitError:
        rotator.mark_rate_limited(key)     # reactive: 429 received
    finally:
        rotator.release(key, tokens_used=...) # always release
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class _KeyState:
    key: str
    # Sliding-window request timestamps (last 60 s for RPM, last 86400 s for RPD)
    rpm_window: Deque[float] = field(default_factory=deque)
    rpd_window: Deque[float] = field(default_factory=deque)
    tpm_window: Deque[tuple] = field(default_factory=deque)  # (timestamp, tokens)
    # Reactive back-off: absolute time when the key becomes available again
    blocked_until: float = 0.0

    def is_blocked(self) -> bool:
        return time.monotonic() < self.blocked_until

    def prune(self, now: float) -> None:
        """Remove expired entries from sliding windows."""
        while self.rpm_window and now - self.rpm_window[0] > 60:
            self.rpm_window.popleft()
        while self.rpd_window and now - self.rpd_window[0] > 86400:
            self.rpd_window.popleft()
        while self.tpm_window and now - self.tpm_window[0][0] > 60:
            self.tpm_window.popleft()

    def rpm_ok(self, limit: int) -> bool:
        return len(self.rpm_window) < limit

    def rpd_ok(self, limit: int) -> bool:
        return limit <= 0 or len(self.rpd_window) < limit

    def tpm_ok(self, limit: int) -> bool:
        if limit <= 0:
            return True
        total = sum(tokens for _, tokens in self.tpm_window)
        return total < limit

    def record_request(self, now: float, tokens: int = 0) -> None:
        self.rpm_window.append(now)
        self.rpd_window.append(now)
        if tokens > 0:
            self.tpm_window.append((now, tokens))

    def seconds_until_rpm_slot(self, limit: int) -> float:
        """Seconds until the oldest RPM entry expires freeing a slot."""
        if len(self.rpm_window) < limit:
            return 0.0
        return max(0.0, 60.0 - (time.monotonic() - self.rpm_window[0]))

    def seconds_until_tpm_slot(self, limit: int) -> float:
        """Seconds until enough tokens roll off the TPM window."""
        if limit <= 0 or self.tpm_ok(limit):
            return 0.0
        now = time.monotonic()
        running = sum(t for _, t in self.tpm_window)
        for ts, t in self.tpm_window:
            if running <= limit:
                return 0.0
            running -= t
            wait = 60.0 - (now - ts)
            if wait > 0:
                return wait
        return 0.0


class AllKeysExhaustedError(Exception):
    """All keys are currently rate-limited; caller should wait and retry."""


class KeyRotator:
    """
    Thread-safe, rate-limit-aware API key pool.

    Args:
        keys:       List of API key strings.
        rpm_limit:  Max requests per minute per key (0 = unlimited).
        rpd_limit:  Max requests per day per key (0 = unlimited).
        tpm_limit:  Max tokens per minute per key (0 = unlimited).
        backoff_s:  Base seconds to block a key on reactive 429 (doubles on repeat).
    """

    def __init__(
        self,
        keys: List[str],
        rpm_limit: int = 0,
        rpd_limit: int = 0,
        tpm_limit: int = 0,
        backoff_s: float = 60.0,
    ) -> None:
        if not keys:
            raise ValueError("KeyRotator requires at least one API key.")
        self._states: Dict[str, _KeyState] = {k: _KeyState(key=k) for k in keys}
        self._keys = list(keys)
        self._rpm = rpm_limit
        self._rpd = rpd_limit
        self._tpm = tpm_limit
        self._backoff = backoff_s
        self._lock = asyncio.Lock()
        self._current_idx = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self, tokens_hint: int = 0, timeout: float = 120.0) -> str:
        """
        Acquire the next available key, waiting up to ``timeout`` seconds.

        Args:
            tokens_hint: Estimated tokens for this request (used for TPM check).
            timeout:     Max seconds to wait before raising AllKeysExhaustedError.

        Returns:
            A key string ready to use.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            async with self._lock:
                key = self._try_acquire(tokens_hint)
                if key is not None:
                    return key
                wait = self._min_wait()

            if wait <= 0 or time.monotonic() + wait > deadline:
                raise AllKeysExhaustedError(
                    "All API keys are rate-limited. Try again later."
                )
            logger.debug("KeyRotator: all keys busy, waiting %.1fs", wait)
            await asyncio.sleep(min(wait, deadline - time.monotonic()))

        raise AllKeysExhaustedError("Timed out waiting for an available API key.")

    def release(self, key: str, tokens_used: int = 0) -> None:
        """Record a completed request. Always call after acquire()."""
        state = self._states.get(key)
        if state:
            state.record_request(time.monotonic(), tokens=tokens_used)
            logger.debug(
                "KeyRotator: released key …%s (tokens=%d)", key[-4:], tokens_used
            )

    def mark_rate_limited(
        self, key: str, retry_after_s: Optional[float] = None
    ) -> None:
        """
        Reactively block a key after receiving a 429 response.

        Args:
            key:            The key that received the 429.
            retry_after_s:  Value of the Retry-After header (if provided).
        """
        state = self._states.get(key)
        if not state:
            return
        backoff = retry_after_s if retry_after_s else self._next_backoff(state)
        state.blocked_until = time.monotonic() + backoff
        logger.warning(
            "KeyRotator: key …%s rate-limited; blocked for %.0fs", key[-4:], backoff
        )

    def available_count(self) -> int:
        """Number of keys that are currently not blocked."""
        now = time.monotonic()
        return sum(1 for k in self._keys if not self._states[k].is_blocked())

    def status(self) -> List[dict]:
        """Debug snapshot of all key states."""
        now = time.monotonic()
        out = []
        for k in self._keys:
            s = self._states[k]
            s.prune(now)
            out.append(
                {
                    "key_suffix": k[-8:],
                    "rpm_used": len(s.rpm_window),
                    "rpd_used": len(s.rpd_window),
                    "tpm_tokens": sum(t for _, t in s.tpm_window),
                    "blocked_for_s": max(0.0, s.blocked_until - now),
                }
            )
        return out

    # ------------------------------------------------------------------
    # Synchronous shim (for non-async callers)
    # ------------------------------------------------------------------

    def acquire_sync(self, tokens_hint: int = 0, timeout: float = 120.0) -> str:
        import asyncio as _asyncio

        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            return _asyncio.run(self.acquire(tokens_hint=tokens_hint, timeout=timeout))
        # Inside an existing loop — create a task (caller must await)
        raise RuntimeError("Use 'await rotator.acquire()' inside an async context.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_acquire(self, tokens_hint: int) -> Optional[str]:
        now = time.monotonic()
        for i in range(len(self._keys)):
            idx = (self._current_idx + i) % len(self._keys)
            key = self._keys[idx]
            state = self._states[key]
            state.prune(now)
            if (
                not state.is_blocked()
                and state.rpm_ok(self._rpm or 10**9)
                and state.rpd_ok(self._rpd)
                and state.tpm_ok(self._tpm)
            ):
                self._current_idx = (idx + 1) % len(self._keys)
                return key
        return None

    def _min_wait(self) -> float:
        """Minimum seconds until any key becomes usable."""
        now = time.monotonic()
        waits = []
        for key in self._keys:
            s = self._states[key]
            if s.is_blocked():
                waits.append(s.blocked_until - now)
            else:
                rpm_wait = s.seconds_until_rpm_slot(self._rpm or 10**9)
                tpm_wait = s.seconds_until_tpm_slot(self._tpm)
                waits.append(max(rpm_wait, tpm_wait))
        return min(waits) if waits else 5.0

    @staticmethod
    def _next_backoff(state: _KeyState) -> float:
        """Exponential back-off: 60, 120, 240, … capped at 3600 s."""
        existing = max(state.blocked_until - time.monotonic(), 0)
        return min(max(existing * 2, 60.0), 3600.0)

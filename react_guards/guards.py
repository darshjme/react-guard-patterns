"""
react_guards/guards.py
Production-ready stop-condition guards for ReAct agents.

Each guard implements:
  - should_stop(state: AgentState) -> bool
  - reason() -> str
  - reset()
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Shared state object passed to each guard on every step
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Snapshot of the agent's current state at each step."""

    step: int = 0
    """Number of steps completed so far (0-indexed)."""

    total_input_tokens: int = 0
    """Cumulative input tokens used."""

    total_output_tokens: int = 0
    """Cumulative output tokens used."""

    last_action: str = ""
    """String identifier of the most recent action taken (e.g. 'search', 'think')."""

    last_observation: str = ""
    """The most recent observation / tool result."""

    progress_score: float | None = None
    """Optional 0-1 progress signal from the agent itself.
    When None the ProgressGuard falls back to change-detection."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Arbitrary extra metadata the agent may populate."""


# ---------------------------------------------------------------------------
# Guard protocol — all guards must satisfy this interface
# ---------------------------------------------------------------------------

@runtime_checkable
class Guard(Protocol):
    def should_stop(self, state: AgentState) -> bool: ...
    def reason(self) -> str: ...
    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# 1. MaxStepsGuard — hard step limit
# ---------------------------------------------------------------------------

class MaxStepsGuard:
    """Stop the agent after a fixed number of steps.

    This is the simplest and most reliable safety net.  Set it high enough
    that legitimate tasks can complete, but low enough that runaway loops are
    caught quickly.

    Args:
        max_steps: Maximum number of agent steps before stopping (default 50).
    """

    def __init__(self, max_steps: int = 50) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.max_steps = max_steps
        self._stop_reason: str = ""

    def should_stop(self, state: AgentState) -> bool:
        if state.step >= self.max_steps:
            self._stop_reason = (
                f"MaxStepsGuard: reached step limit ({state.step}/{self.max_steps})"
            )
            return True
        return False

    def reason(self) -> str:
        return self._stop_reason

    def reset(self) -> None:
        self._stop_reason = ""


# ---------------------------------------------------------------------------
# 2. CostCeilingGuard — token cost tracking
# ---------------------------------------------------------------------------

# Default pricing (USD per 1 000 tokens) — override via constructor.
# These match approximate OpenAI gpt-4o rates as a reasonable baseline.
_DEFAULT_INPUT_PRICE_PER_1K = 0.005   # $0.005 / 1k input tokens
_DEFAULT_OUTPUT_PRICE_PER_1K = 0.015  # $0.015 / 1k output tokens


class CostCeilingGuard:
    """Stop the agent when projected API cost exceeds a budget ceiling.

    Token counts must be kept current in AgentState by the caller.

    Args:
        max_cost_usd: Maximum total cost in US dollars (default $1.00).
        input_price_per_1k: Cost per 1 000 input tokens.
        output_price_per_1k: Cost per 1 000 output tokens.
    """

    def __init__(
        self,
        max_cost_usd: float = 1.0,
        input_price_per_1k: float = _DEFAULT_INPUT_PRICE_PER_1K,
        output_price_per_1k: float = _DEFAULT_OUTPUT_PRICE_PER_1K,
    ) -> None:
        if max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be > 0")
        self.max_cost_usd = max_cost_usd
        self.input_price_per_1k = input_price_per_1k
        self.output_price_per_1k = output_price_per_1k
        self._stop_reason: str = ""

    def current_cost(self, state: AgentState) -> float:
        """Return the cost incurred so far based on token counts in state."""
        return (
            state.total_input_tokens / 1000 * self.input_price_per_1k
            + state.total_output_tokens / 1000 * self.output_price_per_1k
        )

    def should_stop(self, state: AgentState) -> bool:
        cost = self.current_cost(state)
        if cost >= self.max_cost_usd:
            self._stop_reason = (
                f"CostCeilingGuard: cost ${cost:.4f} exceeded ceiling "
                f"${self.max_cost_usd:.2f}"
            )
            return True
        return False

    def reason(self) -> str:
        return self._stop_reason

    def reset(self) -> None:
        self._stop_reason = ""


# ---------------------------------------------------------------------------
# 3. LoopDetectionGuard — detect repeated actions in a sliding window
# ---------------------------------------------------------------------------

class LoopDetectionGuard:
    """Detect when the agent is cycling through the same actions.

    Hashes the (action, observation) pair at each step and checks for
    repetition within a recent window.

    Args:
        window: Number of recent steps to consider (default 5).
        min_repeats: How many times the same hash must appear to trigger
                     (default 2 — i.e. an exact repeat within the window).
    """

    def __init__(self, window: int = 5, min_repeats: int = 2) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        if min_repeats < 2:
            raise ValueError("min_repeats must be >= 2")
        self.window = window
        self.min_repeats = min_repeats
        self._history: deque[str] = deque(maxlen=window)
        self._stop_reason: str = ""

    @staticmethod
    def _fingerprint(action: str, observation: str) -> str:
        raw = f"{action}||{observation}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def should_stop(self, state: AgentState) -> bool:
        fp = self._fingerprint(state.last_action, state.last_observation)
        self._history.append(fp)

        count = self._history.count(fp)
        if count >= self.min_repeats:
            self._stop_reason = (
                f"LoopDetectionGuard: action fingerprint '{fp}' repeated "
                f"{count}x in last {self.window} steps "
                f"(action='{state.last_action}')"
            )
            return True
        return False

    def reason(self) -> str:
        return self._stop_reason

    def reset(self) -> None:
        self._history.clear()
        self._stop_reason = ""


# ---------------------------------------------------------------------------
# 4. TimeoutGuard — wall-clock timeout
# ---------------------------------------------------------------------------

class TimeoutGuard:
    """Stop the agent when wall-clock time exceeds a limit.

    Call reset() (or instantiate fresh) at the start of each run to
    restart the clock.

    Args:
        max_seconds: Maximum wall-clock seconds allowed (default 300 = 5 min).
    """

    def __init__(self, max_seconds: float = 300.0) -> None:
        if max_seconds <= 0:
            raise ValueError("max_seconds must be > 0")
        self.max_seconds = max_seconds
        self._start: float = time.monotonic()
        self._stop_reason: str = ""

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def should_stop(self, state: AgentState) -> bool:
        elapsed = self.elapsed()
        if elapsed >= self.max_seconds:
            self._stop_reason = (
                f"TimeoutGuard: elapsed {elapsed:.1f}s exceeded "
                f"limit {self.max_seconds:.1f}s"
            )
            return True
        return False

    def reason(self) -> str:
        return self._stop_reason

    def reset(self) -> None:
        self._start = time.monotonic()
        self._stop_reason = ""


# ---------------------------------------------------------------------------
# 5. ProgressGuard — detect when the agent stops making progress
# ---------------------------------------------------------------------------

class ProgressGuard:
    """Stop the agent when it stalls — making no measurable progress.

    Progress is measured in two ways (in priority order):
    1. If ``state.progress_score`` is set, track whether it increases.
    2. Otherwise, track whether ``state.last_observation`` changes.

    Args:
        stall_threshold: Consecutive non-improving steps before stopping
                         (default 3).
    """

    def __init__(self, stall_threshold: int = 3) -> None:
        if stall_threshold < 1:
            raise ValueError("stall_threshold must be >= 1")
        self.stall_threshold = stall_threshold
        self._stall_count: int = 0
        self._last_score: float | None = None
        self._last_obs_hash: str = ""
        self._stop_reason: str = ""

    def should_stop(self, state: AgentState) -> bool:
        made_progress = self._evaluate_progress(state)

        if made_progress:
            self._stall_count = 0
        else:
            self._stall_count += 1

        if self._stall_count >= self.stall_threshold:
            self._stop_reason = (
                f"ProgressGuard: no progress for {self._stall_count} "
                f"consecutive steps (threshold={self.stall_threshold})"
            )
            return True
        return False

    def _evaluate_progress(self, state: AgentState) -> bool:
        # --- Score-based progress ---
        if state.progress_score is not None:
            if self._last_score is None or state.progress_score > self._last_score:
                self._last_score = state.progress_score
                return True
            return False

        # --- Observation change-based progress ---
        obs_hash = hashlib.sha1(state.last_observation.encode()).hexdigest()[:12]
        if obs_hash != self._last_obs_hash:
            self._last_obs_hash = obs_hash
            return True
        return False

    def reason(self) -> str:
        return self._stop_reason

    def reset(self) -> None:
        self._stall_count = 0
        self._last_score = None
        self._last_obs_hash = ""
        self._stop_reason = ""

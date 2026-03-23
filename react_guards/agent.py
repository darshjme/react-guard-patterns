"""
react_guards/agent.py
GuardedReActAgent — wraps any agent function with composite stop-condition guards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .guards import AgentState, Guard, MaxStepsGuard


# ---------------------------------------------------------------------------
# Result type returned by GuardedReActAgent.run()
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """The outcome of a guarded agent run."""

    final_answer: str
    """The agent's last response / answer string."""

    steps_taken: int
    """Total number of steps executed."""

    stopped_by: str
    """Human-readable name/reason of the guard (or 'agent_done') that halted the loop."""

    elapsed_seconds: float
    """Wall-clock time from run() entry to return."""

    total_cost_usd: float
    """Approximate API cost in USD (requires CostCeilingGuard in the guard list)."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0

    guard_reasons: list[str] = field(default_factory=list)
    """Reasons emitted by ALL guards that fired (useful for debugging)."""

    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step output — what agent_fn must return each turn
# ---------------------------------------------------------------------------

@dataclass
class StepOutput:
    """What the wrapped agent function returns for each step."""

    action: str
    """Name of the action taken (e.g. 'search', 'think', 'answer')."""

    observation: str
    """Result / observation from the action."""

    is_done: bool = False
    """If True the agent signals it has reached a final answer."""

    final_answer: str = ""
    """Populated when is_done=True."""

    input_tokens: int = 0
    """Tokens consumed on the input side this step (for cost tracking)."""

    output_tokens: int = 0
    """Tokens produced this step (for cost tracking)."""

    progress_score: float | None = None
    """Optional 0-1 progress signal from the agent (used by ProgressGuard)."""

    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GuardedReActAgent
# ---------------------------------------------------------------------------

AgentFn = Callable[[str, AgentState], StepOutput]
OnStopCallback = Callable[[AgentState, str], None]


class GuardedReActAgent:
    """Wraps any ReAct-style agent function with composite stop-condition guards.

    The wrapped function (``agent_fn``) is called once per step and must return
    a :class:`StepOutput`.  After each step every registered guard is evaluated;
    if any fires the loop terminates gracefully and an :class:`AgentResult` is
    returned.

    Args:
        agent_fn: Callable with signature ``(task: str, state: AgentState) -> StepOutput``.
        guards: List of guard instances.  Defaults to ``[MaxStepsGuard(50)]``.
        on_stop: Optional callback invoked when a guard fires:
                 ``on_stop(state, reason_string)``.

    Example::

        agent = GuardedReActAgent(
            agent_fn=my_llm_step,
            guards=[
                MaxStepsGuard(max_steps=30),
                CostCeilingGuard(max_cost_usd=0.50),
                TimeoutGuard(max_seconds=120),
                LoopDetectionGuard(window=5),
                ProgressGuard(stall_threshold=3),
            ],
        )
        result = agent.run("What is the capital of France?")
        print(result.final_answer)
    """

    def __init__(
        self,
        agent_fn: AgentFn,
        guards: Optional[List[Guard]] = None,
        on_stop: Optional[OnStopCallback] = None,
    ) -> None:
        self.agent_fn = agent_fn
        self.guards: List[Guard] = guards if guards is not None else [MaxStepsGuard(50)]
        self.on_stop = on_stop

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str) -> AgentResult:
        """Execute the agent loop for *task*, stopping when any guard fires.

        Returns an :class:`AgentResult` regardless of how the loop ended.
        """
        # Reset all guards so the agent can be re-used across multiple tasks.
        for g in self.guards:
            g.reset()

        state = AgentState()
        start_time = time.monotonic()
        last_output = StepOutput(action="", observation="")

        while True:
            # --- Execute one agent step ---
            output: StepOutput = self.agent_fn(task, state)

            # --- Update shared state ---
            state.step += 1
            state.last_action = output.action
            state.last_observation = output.observation
            state.total_input_tokens += output.input_tokens
            state.total_output_tokens += output.output_tokens
            state.progress_score = output.progress_score
            last_output = output

            # --- Check agent's own done signal first ---
            if output.is_done:
                return self._make_result(
                    state=state,
                    stopped_by="agent_done",
                    final_answer=output.final_answer or output.observation,
                    start_time=start_time,
                    fired_guards=[],
                )

            # --- Evaluate all guards ---
            fired: list[tuple[Guard, str]] = []
            for guard in self.guards:
                if guard.should_stop(state):
                    fired.append((guard, guard.reason()))

            if fired:
                reasons = [r for _, r in fired]
                primary_reason = reasons[0]

                if self.on_stop:
                    self.on_stop(state, primary_reason)

                return self._make_result(
                    state=state,
                    stopped_by=primary_reason,
                    final_answer=last_output.observation,
                    start_time=start_time,
                    fired_guards=reasons,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_result(
        self,
        state: AgentState,
        stopped_by: str,
        final_answer: str,
        start_time: float,
        fired_guards: list[str],
    ) -> AgentResult:
        elapsed = time.monotonic() - start_time

        # Attempt to extract cost from a CostCeilingGuard if present.
        total_cost = 0.0
        from .guards import CostCeilingGuard  # local import avoids circular
        for g in self.guards:
            if isinstance(g, CostCeilingGuard):
                total_cost = g.current_cost(state)
                break

        return AgentResult(
            final_answer=final_answer,
            steps_taken=state.step,
            stopped_by=stopped_by,
            elapsed_seconds=elapsed,
            total_cost_usd=total_cost,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            guard_reasons=fired_guards,
        )

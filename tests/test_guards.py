"""
tests/test_guards.py
Unit tests for react_guards — all guards and GuardedReActAgent.
No external calls; all agent functions are pure Python mocks.
"""
from __future__ import annotations

import time

import pytest

from react_guards.guards import (
    AgentState,
    CostCeilingGuard,
    LoopDetectionGuard,
    MaxStepsGuard,
    ProgressGuard,
    TimeoutGuard,
)
from react_guards.agent import AgentResult, GuardedReActAgent, StepOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**kwargs) -> AgentState:
    """Build an AgentState with optional overrides."""
    return AgentState(**kwargs)


# ---------------------------------------------------------------------------
# MaxStepsGuard
# ---------------------------------------------------------------------------

class TestMaxStepsGuard:
    def test_max_steps_guard_fires(self):
        """Guard stops the loop when step count reaches the limit."""
        guard = MaxStepsGuard(max_steps=3)

        # Steps 0, 1 — should NOT fire
        assert guard.should_stop(make_state(step=0)) is False
        assert guard.should_stop(make_state(step=1)) is False
        assert guard.should_stop(make_state(step=2)) is False

        # Step 3 — should fire
        assert guard.should_stop(make_state(step=3)) is True
        assert "MaxStepsGuard" in guard.reason()
        assert "3" in guard.reason()

    def test_guard_reset(self):
        """reset() clears stop reason."""
        guard = MaxStepsGuard(max_steps=1)
        guard.should_stop(make_state(step=1))   # fires
        assert guard.reason() != ""

        guard.reset()
        assert guard.reason() == ""

        # After reset, step=1 should fire again
        assert guard.should_stop(make_state(step=1)) is True


# ---------------------------------------------------------------------------
# CostCeilingGuard
# ---------------------------------------------------------------------------

class TestCostCeilingGuard:
    def test_cost_ceiling_guard_fires(self):
        """Guard fires when cumulative token cost exceeds the ceiling."""
        guard = CostCeilingGuard(
            max_cost_usd=0.01,
            input_price_per_1k=0.005,
            output_price_per_1k=0.015,
        )
        # Under budget
        cheap_state = make_state(total_input_tokens=100, total_output_tokens=100)
        assert guard.should_stop(cheap_state) is False

        # Over budget: 2000 input @ $0.005/1k = $0.01  →  exactly at ceiling → fires
        expensive_state = make_state(
            total_input_tokens=2000, total_output_tokens=0
        )
        assert guard.should_stop(expensive_state) is True
        assert "CostCeilingGuard" in guard.reason()

    def test_cost_ceiling_guard_reset(self):
        guard = CostCeilingGuard(max_cost_usd=0.001)
        state = make_state(total_input_tokens=10_000, total_output_tokens=0)
        guard.should_stop(state)
        assert guard.reason() != ""
        guard.reset()
        assert guard.reason() == ""


# ---------------------------------------------------------------------------
# LoopDetectionGuard
# ---------------------------------------------------------------------------

class TestLoopDetectionGuard:
    def test_loop_detection_guard_fires(self):
        """Repeated (action, observation) pair triggers the guard."""
        guard = LoopDetectionGuard(window=5, min_repeats=2)

        s1 = make_state(last_action="search", last_observation="result_A")
        s2 = make_state(last_action="search", last_observation="result_B")
        s3 = make_state(last_action="search", last_observation="result_A")  # repeat of s1

        assert guard.should_stop(s1) is False  # first time, only 1 occurrence
        assert guard.should_stop(s2) is False  # different obs
        assert guard.should_stop(s3) is True   # same fingerprint as s1 → 2 occurrences
        assert "LoopDetectionGuard" in guard.reason()

    def test_loop_detection_guard_reset(self):
        guard = LoopDetectionGuard(window=3, min_repeats=2)
        s = make_state(last_action="x", last_observation="y")
        guard.should_stop(s)
        guard.should_stop(s)  # fires
        guard.reset()
        assert guard.reason() == ""
        # After reset history is clear — first occurrence again, should NOT fire
        assert guard.should_stop(s) is False


# ---------------------------------------------------------------------------
# TimeoutGuard
# ---------------------------------------------------------------------------

class TestTimeoutGuard:
    def test_timeout_guard_fires(self):
        """Guard fires after the configured wall-clock time has elapsed."""
        guard = TimeoutGuard(max_seconds=0.05)  # 50 ms

        state = make_state()
        assert guard.should_stop(state) is False   # immediately — hasn't elapsed

        time.sleep(0.1)  # 100 ms — past the 50 ms limit
        assert guard.should_stop(state) is True
        assert "TimeoutGuard" in guard.reason()

    def test_timeout_guard_reset_restarts_clock(self):
        guard = TimeoutGuard(max_seconds=0.05)
        time.sleep(0.1)
        assert guard.should_stop(make_state()) is True  # fired

        guard.reset()  # restart clock
        assert guard.should_stop(make_state()) is False  # fresh clock, should NOT fire


# ---------------------------------------------------------------------------
# ProgressGuard
# ---------------------------------------------------------------------------

class TestProgressGuard:
    def test_progress_guard_fires(self):
        """Guard fires when the observation stays identical for stall_threshold steps."""
        guard = ProgressGuard(stall_threshold=3)

        # First step with a new observation — counts as progress (no stall yet)
        s = make_state(last_observation="same_result")
        # Step 1: obs_hash changes from "" → "same_result" → progress
        assert guard.should_stop(s) is False

        # Steps 2, 3, 4: same observation, no change
        assert guard.should_stop(s) is False  # stall_count = 1
        assert guard.should_stop(s) is False  # stall_count = 2
        assert guard.should_stop(s) is True   # stall_count = 3 → fires

        assert "ProgressGuard" in guard.reason()

    def test_progress_guard_uses_progress_score(self):
        """When progress_score is set, the guard tracks score improvements."""
        guard = ProgressGuard(stall_threshold=2)

        assert guard.should_stop(make_state(progress_score=0.1)) is False  # initial
        assert guard.should_stop(make_state(progress_score=0.1)) is False  # stall 1
        assert guard.should_stop(make_state(progress_score=0.1)) is True   # stall 2 → fire

    def test_progress_guard_reset(self):
        guard = ProgressGuard(stall_threshold=2)
        s = make_state(last_observation="stuck")
        guard.should_stop(s)
        guard.should_stop(s)
        guard.should_stop(s)   # fires
        guard.reset()
        assert guard.reason() == ""
        assert guard.should_stop(make_state(last_observation="stuck")) is False


# ---------------------------------------------------------------------------
# CompositeGuard (first-wins semantics via GuardedReActAgent)
# ---------------------------------------------------------------------------

class TestCompositeGuardFirstWins:
    def test_composite_guard_first_wins(self):
        """When multiple guards fire, the first one in the list wins (primary reason)."""
        # Guard 1 fires at step 1, Guard 2 fires at step 1 too
        max_guard = MaxStepsGuard(max_steps=1)   # fires at step >= 1
        cost_guard = CostCeilingGuard(
            max_cost_usd=0.001,
            input_price_per_1k=0.005,
            output_price_per_1k=0.015,
        )

        call_count = [0]

        def agent_fn(task: str, state: AgentState) -> StepOutput:
            call_count[0] += 1
            return StepOutput(
                action="think",
                observation="thinking...",
                input_tokens=1000,  # generates cost
                output_tokens=0,
            )

        agent = GuardedReActAgent(
            agent_fn=agent_fn,
            guards=[max_guard, cost_guard],
        )
        result = agent.run("test task")

        # Primary stop reason must be from max_guard (first in list)
        assert "MaxStepsGuard" in result.stopped_by
        # Both guards fired → guard_reasons has both
        assert len(result.guard_reasons) >= 1
        assert any("MaxStepsGuard" in r for r in result.guard_reasons)


# ---------------------------------------------------------------------------
# GuardedReActAgent — natural completion
# ---------------------------------------------------------------------------

class TestGuardedAgentNaturalCompletion:
    def test_guarded_agent_natural_completion(self):
        """Agent signals is_done=True → stops cleanly as 'agent_done'."""
        def agent_fn(task: str, state: AgentState) -> StepOutput:
            if state.step >= 2:
                return StepOutput(
                    action="answer",
                    observation="Paris",
                    is_done=True,
                    final_answer="Paris",
                )
            return StepOutput(action="think", observation="thinking...")

        agent = GuardedReActAgent(
            agent_fn=agent_fn,
            guards=[MaxStepsGuard(max_steps=10)],
        )
        result = agent.run("What is the capital of France?")

        assert result.stopped_by == "agent_done"
        assert result.final_answer == "Paris"
        assert result.steps_taken == 3   # steps 0, 1 → think; step 2 → done
        assert isinstance(result, AgentResult)


# ---------------------------------------------------------------------------
# GuardedReActAgent — guard triggers mid-run
# ---------------------------------------------------------------------------

class TestGuardedAgentGuardTriggered:
    def test_guarded_agent_guard_triggered(self):
        """Guard fires mid-run and stops the loop before natural completion."""
        call_count = [0]

        def infinite_agent(task: str, state: AgentState) -> StepOutput:
            call_count[0] += 1
            return StepOutput(action="loop", observation="same_obs")

        agent = GuardedReActAgent(
            agent_fn=infinite_agent,
            guards=[MaxStepsGuard(max_steps=4)],
        )
        result = agent.run("run forever")

        assert "MaxStepsGuard" in result.stopped_by
        assert result.steps_taken == 4
        assert call_count[0] == 4  # called exactly max_steps times
        assert isinstance(result.elapsed_seconds, float)
        assert result.elapsed_seconds >= 0

"""
examples/basic_usage.py
Demonstrates all five guards using a mock agent — no real LLM required.

Run:
    cd /tmp/react-guard-patterns
    python3 examples/basic_usage.py
"""

from __future__ import annotations

import sys
import os

# Allow running directly without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from react_guards import (
    AgentResult,
    GuardedReActAgent,
    StepOutput,
)
from react_guards.guards import (
    AgentState,
    CostCeilingGuard,
    LoopDetectionGuard,
    MaxStepsGuard,
    ProgressGuard,
    TimeoutGuard,
)


# ---------------------------------------------------------------------------
# Helper printer
# ---------------------------------------------------------------------------

def print_result(label: str, result: AgentResult) -> None:
    print(f"\n{'─' * 60}")
    print(f"  SCENARIO: {label}")
    print(f"{'─' * 60}")
    print(f"  steps_taken   : {result.steps_taken}")
    print(f"  stopped_by    : {result.stopped_by}")
    print(f"  final_answer  : {result.final_answer!r}")
    print(f"  elapsed       : {result.elapsed_seconds:.3f}s")
    print(f"  cost (USD)    : ${result.total_cost_usd:.5f}")
    print(f"  tokens in/out : {result.total_input_tokens}/{result.total_output_tokens}")
    if result.guard_reasons:
        print(f"  guard_reasons :")
        for r in result.guard_reasons:
            print(f"    • {r}")


# ---------------------------------------------------------------------------
# Scenario 1 — Agent finishes naturally (no guard fires)
# ---------------------------------------------------------------------------

def scenario_natural_completion() -> None:
    """Agent signals done=True after 3 steps."""

    def mock_agent(task: str, state: AgentState) -> StepOutput:
        steps = [
            StepOutput(action="think", observation="Let me break this down...",
                       input_tokens=120, output_tokens=40),
            StepOutput(action="search", observation="Found: Paris is the capital of France.",
                       input_tokens=80, output_tokens=60),
            StepOutput(action="answer", observation="Paris",
                       is_done=True, final_answer="The capital of France is Paris.",
                       input_tokens=60, output_tokens=20),
        ]
        return steps[min(state.step, len(steps) - 1)]

    agent = GuardedReActAgent(
        agent_fn=mock_agent,
        guards=[
            MaxStepsGuard(max_steps=50),
            CostCeilingGuard(max_cost_usd=1.0),
        ],
    )
    result = agent.run("What is the capital of France?")
    print_result("Natural completion (agent signals done=True)", result)
    assert result.stopped_by == "agent_done"
    assert result.steps_taken == 3
    assert "Paris" in result.final_answer
    print("  ✓ assertions passed")


# ---------------------------------------------------------------------------
# Scenario 2 — MaxStepsGuard fires
# ---------------------------------------------------------------------------

def scenario_max_steps() -> None:
    """Agent never signals done — MaxStepsGuard fires at step 5."""

    def mock_agent(task: str, state: AgentState) -> StepOutput:
        return StepOutput(
            action="search",
            observation=f"Partial result {state.step}",
            input_tokens=100,
            output_tokens=50,
        )

    agent = GuardedReActAgent(
        agent_fn=mock_agent,
        guards=[MaxStepsGuard(max_steps=5)],
    )
    result = agent.run("Find everything about AGI")
    print_result("MaxStepsGuard fires at step 5", result)
    assert result.steps_taken == 5
    assert "MaxStepsGuard" in result.stopped_by
    print("  ✓ assertions passed")


# ---------------------------------------------------------------------------
# Scenario 3 — CostCeilingGuard fires
# ---------------------------------------------------------------------------

def scenario_cost_ceiling() -> None:
    """Agent burns tokens quickly; CostCeilingGuard fires under $0.10."""

    def mock_agent(task: str, state: AgentState) -> StepOutput:
        # Each step uses 5000 input + 2000 output tokens (expensive)
        return StepOutput(
            action="generate",
            observation="Generated a massive response...",
            input_tokens=5000,
            output_tokens=2000,
        )

    agent = GuardedReActAgent(
        agent_fn=mock_agent,
        guards=[
            MaxStepsGuard(max_steps=100),
            CostCeilingGuard(
                max_cost_usd=0.10,
                input_price_per_1k=0.005,
                output_price_per_1k=0.015,
            ),
        ],
    )
    result = agent.run("Write a 10000-word essay")
    print_result("CostCeilingGuard fires < $0.10", result)
    assert "CostCeilingGuard" in result.stopped_by
    assert result.total_cost_usd >= 0.10
    print("  ✓ assertions passed")


# ---------------------------------------------------------------------------
# Scenario 4 — LoopDetectionGuard fires
# ---------------------------------------------------------------------------

def scenario_loop_detection() -> None:
    """Agent gets stuck repeating the same action + observation."""

    def mock_agent(task: str, state: AgentState) -> StepOutput:
        if state.step < 2:
            return StepOutput(
                action="think",
                observation=f"Thinking step {state.step}",
                input_tokens=50, output_tokens=20,
            )
        # From step 2 onward, always repeats the same action + observation
        return StepOutput(
            action="search",
            observation="No results found.",
            input_tokens=80, output_tokens=30,
        )

    agent = GuardedReActAgent(
        agent_fn=mock_agent,
        guards=[
            MaxStepsGuard(max_steps=50),
            LoopDetectionGuard(window=5, min_repeats=2),
        ],
    )
    result = agent.run("Find information on XYZ topic")
    print_result("LoopDetectionGuard fires on repeated search", result)
    assert "LoopDetectionGuard" in result.stopped_by
    print("  ✓ assertions passed")


# ---------------------------------------------------------------------------
# Scenario 5 — ProgressGuard fires
# ---------------------------------------------------------------------------

def scenario_progress_stall() -> None:
    """Agent makes progress initially, then stalls (same observation)."""

    def mock_agent(task: str, state: AgentState) -> StepOutput:
        if state.step < 3:
            return StepOutput(
                action="research",
                observation=f"Found new data point {state.step}",
                input_tokens=100, output_tokens=50,
            )
        # Stalls — same observation every step
        return StepOutput(
            action="research",
            observation="Still processing...",
            input_tokens=100, output_tokens=30,
        )

    agent = GuardedReActAgent(
        agent_fn=mock_agent,
        guards=[
            MaxStepsGuard(max_steps=50),
            ProgressGuard(stall_threshold=3),
        ],
    )
    result = agent.run("Deep research task")
    print_result("ProgressGuard fires after 3 stalled steps", result)
    assert "ProgressGuard" in result.stopped_by
    print("  ✓ assertions passed")


# ---------------------------------------------------------------------------
# Scenario 6 — All five guards active simultaneously (composite mode)
# ---------------------------------------------------------------------------

def scenario_composite_guards() -> None:
    """All five guards in a single agent — first to fire wins."""

    call_log: list[str] = []

    def mock_agent(task: str, state: AgentState) -> StepOutput:
        call_log.append(f"step={state.step}")
        return StepOutput(
            action="think",
            observation="Making steady progress...",
            progress_score=min(0.1 * state.step, 0.9),  # slowly climbing
            input_tokens=200,
            output_tokens=100,
        )

    def on_stop(state: AgentState, reason: str) -> None:
        print(f"  [on_stop callback] step={state.step} reason='{reason}'")

    agent = GuardedReActAgent(
        agent_fn=mock_agent,
        guards=[
            MaxStepsGuard(max_steps=20),
            CostCeilingGuard(max_cost_usd=0.05),
            LoopDetectionGuard(window=5, min_repeats=3),
            TimeoutGuard(max_seconds=60),
            ProgressGuard(stall_threshold=5),
        ],
        on_stop=on_stop,
    )
    result = agent.run("Multi-guard stress test")
    print_result("Composite — all 5 guards active", result)
    print(f"  steps called    : {len(call_log)}")
    assert result.steps_taken > 0
    print("  ✓ assertions passed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  react-guard-patterns — Guard Demonstration Suite")
    print("═" * 60)

    scenario_natural_completion()
    scenario_max_steps()
    scenario_cost_ceiling()
    scenario_loop_detection()
    scenario_progress_stall()
    scenario_composite_guards()

    print("\n" + "═" * 60)
    print("  All scenarios completed successfully. ✓")
    print("═" * 60 + "\n")

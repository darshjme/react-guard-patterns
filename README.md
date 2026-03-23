# react-guard-patterns

> **Stop-condition patterns for ReAct agents — prevent infinite loops and runaway API costs in production.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero dependencies](https://img.shields.io/badge/dependencies-zero-green.svg)]()

---

## The Problem: ReAct Agents Have No Brakes

The ReAct pattern (Reason + Act) is one of the most powerful paradigms in modern LLM agent design.
An agent thinks, takes an action, observes the result, thinks again — and repeats until done.

The catch: **"until done" is defined by the LLM itself.**

In production, this creates four failure modes that have burned real engineering teams:

| Failure Mode | What Happens | Cost |
|---|---|---|
| **Infinite reasoning loop** | Model keeps "thinking" without acting | $10–$500+ per stuck task |
| **Tool retry spiral** | A broken tool gets retried 100×  | 100× API budget wasted |
| **Observation fixation** | Agent reads the same data repeatedly | Silent waste, wrong answers |
| **Scope creep** | Unbounded search keeps expanding | Hours of compute, no output |

These are not edge cases. They are **predictable failure modes** that emerge at scale — documented in
[beam.ai's agentic patterns research](https://beam.ai/), the ReAct paper (Yao et al., 2023), and
extensively in production post-mortems across the LLM ecosystem.

The fix is simple: **add stop conditions**. This repo provides five production-grade patterns.

---

## Five Stop-Condition Patterns

### 1. `MaxStepsGuard` — Hard Step Limit

The simplest, most reliable safety net. Every agent run gets a fixed budget of steps.

```python
from react_guards.guards import MaxStepsGuard

guard = MaxStepsGuard(max_steps=50)
# Fires after step 50 regardless of what the agent is doing
```

**When to use:** Always. This is your last-resort fallback. Set it high enough for legitimate tasks
(50–100 for most), low enough to catch runaway loops.

**Production tip:** Log when this fires. Consistent MaxSteps triggers on specific task types signal
that your prompt needs work, not your step budget.

---

### 2. `CostCeilingGuard` — Token Cost Tracking

Tracks cumulative token usage and fires when projected API spend exceeds your budget.

```python
from react_guards.guards import CostCeilingGuard

guard = CostCeilingGuard(
    max_cost_usd=1.00,
    input_price_per_1k=0.005,   # override for your model
    output_price_per_1k=0.015,
)
```

**When to use:** Any production deployment where you bill per token. Set per-task budgets
(e.g. $0.10 for simple queries, $2.00 for deep research), not per-session.

**Production tip:** Set the ceiling at 80% of your acceptable max. Leave 20% headroom for the
graceful shutdown path (which also uses tokens).

---

### 3. `LoopDetectionGuard` — Detect Repeated Actions

Hashes `(action, observation)` pairs in a sliding window. If the same pair repeats, the agent
is cycling.

```python
from react_guards.guards import LoopDetectionGuard

guard = LoopDetectionGuard(
    window=5,        # check last 5 steps
    min_repeats=2,   # fire on 2nd occurrence of the same fingerprint
)
```

**When to use:** Agents that use external tools (search, code execution, APIs). Tool failures
or empty results commonly cause the agent to retry identically.

**Production tip:** Combine with structured error handling in your tool layer — LoopDetection
is your circuit-breaker when tools are misbehaving.

---

### 4. `TimeoutGuard` — Wall-Clock Timeout

Enforces a hard time limit regardless of step count. Critical when individual steps may block
(slow API, large context windows, network timeouts).

```python
from react_guards.guards import TimeoutGuard

guard = TimeoutGuard(max_seconds=120)  # 2-minute hard limit
```

**When to use:** Any agent that calls external services. A single slow tool call can balloon
a 10-step agent to 10 minutes.

**Production tip:** Set per SLA requirement. For user-facing agents: 30–60s. Background agents:
5–30 minutes. Always shorter than your infrastructure timeout.

---

### 5. `ProgressGuard` — Detect Stalled Agents

Monitors whether the agent is actually making progress. If observations stop changing (or a
supplied `progress_score` stops increasing), the agent is stalled.

```python
from react_guards.guards import ProgressGuard

guard = ProgressGuard(stall_threshold=3)  # 3 consecutive non-improving steps
```

**When to use:** Long-running research or planning agents. Complements LoopDetection: while
LoopDetection catches identical action-observation pairs, ProgressGuard catches semantically
similar-but-not-identical stalls.

**Production tip:** If your agent can emit a numeric progress signal (e.g. percentage of
subtasks completed), set `state.progress_score` — the guard will use it instead of relying
on observation hashing.

---

## Quick Start

```python
from react_guards import GuardedReActAgent, StepOutput
from react_guards.guards import (
    MaxStepsGuard, CostCeilingGuard, LoopDetectionGuard,
    TimeoutGuard, ProgressGuard, AgentState,
)

def my_agent_step(task: str, state: AgentState) -> StepOutput:
    # Replace with your real LLM call
    response = call_your_llm(task, state)
    return StepOutput(
        action=response.action,
        observation=response.observation,
        is_done=response.finished,
        final_answer=response.answer,
        input_tokens=response.usage.input,
        output_tokens=response.usage.output,
    )

agent = GuardedReActAgent(
    agent_fn=my_agent_step,
    guards=[
        MaxStepsGuard(max_steps=50),
        CostCeilingGuard(max_cost_usd=1.00),
        LoopDetectionGuard(window=5),
        TimeoutGuard(max_seconds=120),
        ProgressGuard(stall_threshold=3),
    ],
    on_stop=lambda state, reason: print(f"Stopped: {reason}"),
)

result = agent.run("Research the latest advances in fusion energy")
print(f"Answer: {result.final_answer}")
print(f"Steps: {result.steps_taken} | Cost: ${result.total_cost_usd:.4f} | Stopped by: {result.stopped_by}")
```

---

## Synthetic Performance Benchmarks

> ⚠️ **These are synthetic benchmarks** run on a mock agent (no real LLM calls) to demonstrate
> guard overhead characteristics. Real-world numbers depend entirely on your LLM and tools.

| Scenario | Steps | Guards Active | Guard Overhead | Winner |
|---|---|---|---|---|
| Natural completion | 3 | MaxSteps + Cost | < 0.1ms total | `agent_done` |
| MaxSteps fires | 5 | MaxSteps only | ~0.02ms/step | `MaxStepsGuard` |
| Cost ceiling hit | 2 | MaxSteps + Cost | ~0.03ms/step | `CostCeilingGuard` |
| Loop detected | 4 | MaxSteps + Loop | ~0.15ms/step | `LoopDetectionGuard` |
| Progress stall | 6 | MaxSteps + Progress | ~0.12ms/step | `ProgressGuard` |
| All 5 active | varies | All guards | ~0.3ms/step | first to fire |

**Key finding:** Guard overhead is negligible (< 1ms per step). The bottleneck is always
the LLM call, not the guards.

---

## Installation

```bash
# From source
git clone https://github.com/darshjme/react-guard-patterns
cd react-guard-patterns
pip install -e .

# Run the demo (no LLM needed)
python examples/basic_usage.py
```

**Requirements:** Python 3.11+, zero runtime dependencies.

---

## Design Principles

1. **Zero dependencies** — pure Python stdlib. Drops into any stack.
2. **Composable** — use one guard or all five. Order doesn't matter.
3. **Stateless between runs** — `reset()` is called automatically on each `agent.run()`.
4. **Protocol-based** — implement `should_stop / reason / reset` to build your own guard.
5. **Fail-safe** — guards never raise; they return `bool`. Exceptions in guards are your agent's problem.

---

## Further Reading

- Yao et al. (2023) — [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [beam.ai — Agentic Patterns in Production](https://beam.ai/)
- Significant Gravitas — [Auto-GPT: post-mortems on infinite loops](https://github.com/Significant-Gravitas/AutoGPT)
- LangChain — [Agent executor stop conditions](https://python.langchain.com/docs/modules/agents/)

---

## License

MIT — see [LICENSE](LICENSE).

Built by [@darshjme](https://github.com/darshjme).

---

## Related Projects

Building production-grade agent systems? These companion repos complete the stack:

- **[agent-evals](https://github.com/darshjme/agent-evals)** — LLM agent evaluation framework. Measure and benchmark agent performance after you've guarded the loops.
- **[llm-router](https://github.com/darshjme/llm-router)** — Semantic task routing. Route tasks to specialist agents before they even reach the ReAct loop.

All three repos by [@darshjme](https://github.com/darshjme).

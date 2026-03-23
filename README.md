<p align="center">
  <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAwIiBoZWlnaHQ9IjE2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImJnIiB4MT0iMCUiIHkxPSIwJSIgeDI9IjEwMCUiIHkyPSIxMDAlIj48c3RvcCBvZmZzZXQ9IjAlIiBzdHlsZT0ic3RvcC1jb2xvcjojMGQxMTE3Ii8+PHN0b3Agb2Zmc2V0PSIxMDAlIiBzdHlsZT0ic3RvcC1jb2xvcjojMWEwZjBmIi8+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PHJlY3Qgd2lkdGg9IjgwMCIgaGVpZ2h0PSIxNjAiIGZpbGw9InVybCgjYmcpIi8+PHRleHQgeD0iNjAiIHk9Ijc4IiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjQ4IiBmb250LXdlaWdodD0iYm9sZCIgZmlsbD0iI2U2ZWRmMyI+c2VudGluZWw8L3RleHQ+PHRleHQgeD0iNjAiIHk9IjExNCIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxNiIgZmlsbD0iIzhiOTQ5ZSI+Rml2ZSBndWFyZHMuIFplcm8gcnVuYXdheSBhZ2VudHMuPC90ZXh0Pjxwb2x5Z29uIHBvaW50cz0iNzIwLDIwIDc2MCw1MCA3NjAsMTQwIDcyMCwxNDAgNjgwLDE0MCA2ODAsMTQwIDY4MCw1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZjg1MTQ5IiBzdHJva2Utd2lkdGg9IjMiIG9wYWNpdHk9IjAuOSIvPjx0ZXh0IHg9IjcyMCIgeT0iOTAiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMzAiIGZpbGw9IiNmODUxNDkiIHRleHQtYW5jaG9yPSJtaWRkbGUiPuKaoTwvdGV4dD48L3N2Zz4=" alt="sentinel" width="800"/>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/></a>
  <img src="https://img.shields.io/badge/tests-14%20passing-brightgreen" alt="Tests"/>
  <img src="https://img.shields.io/badge/dependencies-zero-brightgreen" alt="Zero deps"/>
  <img src="https://img.shields.io/badge/pypi-coming%20soon-orange" alt="PyPI"/>
</p>

<p align="center"><b>Five stop-condition guards. Prevent infinite loops and runaway API costs in production.</b></p>

---

## The Problem: ReAct Agents Have No Brakes

```python
# This runs until YOUR money runs out
while not agent.is_done():
    agent.step()   # no guard = no ceiling
```

The ReAct loop is elegant in a notebook. In production, it's a ticking clock. "Done" is defined by the LLM — and a confused model loops forever.

| Failure Mode | What Happens | Real Cost |
|---|---|---|
| **Hard loop** | Same action, forever | $10–$500+ per stuck task |
| **Semantic loop** | Different words, same dead end | Silent budget burn |
| **Retry storm** | Broken tool retried 80× | 80× wasted API calls |
| **Scope creep** | Unbounded search expands forever | Hours of compute, no output |

---

## The Fix

```mermaid
flowchart LR
    A[Agent Step] --> B[MaxStepsGuard]
    B --> C[CostCeilingGuard]
    C --> D[LoopDetectionGuard]
    D --> E[TimeoutGuard]
    E --> F[ProgressGuard]
    F --> G{Any fired?}
    G -->|no| H[Continue]
    G -->|yes| I[STOP + reason\n+ final answer]
```

---

## Five Guards

| Guard | Stops | Key Config |
|-------|-------|-----------|
| `MaxStepsGuard` | Hard step ceiling | `max_steps=50` |
| `CostCeilingGuard` | Token spend ceiling | `max_cost_usd=1.00` |
| `LoopDetectionGuard` | Repeated action-observation pairs | `window=5, min_repeats=2` |
| `TimeoutGuard` | Wall-clock time limit | `max_seconds=120` |
| `ProgressGuard` | Stalled / non-improving agent | `stall_threshold=3` |

---

## Quick Start

```bash
git clone https://github.com/darshjme/sentinel
cd sentinel && pip install -e .
```

```python
from react_guards import GuardedReActAgent, StepOutput
from react_guards.guards import (
    MaxStepsGuard, CostCeilingGuard, LoopDetectionGuard,
    TimeoutGuard, ProgressGuard, AgentState,
)

def my_agent_step(task: str, state: AgentState) -> StepOutput:
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
)

result = agent.run("Research the latest advances in fusion energy")
print(f"Stopped by: {result.stopped_by}")   # "agent_done" or guard name
print(f"Steps: {result.steps_taken} | Cost: ${result.total_cost_usd:.4f}")
```

---

## Sequence: LoopDetection Catching a Stuck Agent

```mermaid
sequenceDiagram
    participant A as Agent
    participant L as LoopDetectionGuard
    participant R as Runner

    A->>R: Step 1: search("fusion energy")
    R->>L: check(action, observation)
    L-->>R: continue
    A->>R: Step 2: search("fusion energy")
    R->>L: check(action, observation)
    L-->>R: continue (1st repeat)
    A->>R: Step 3: search("fusion energy")
    R->>L: check(action, observation)
    L-->>R: STOP — repeated 2x in window=5
    R-->>A: GuardTriggered: LoopDetectionGuard
```

---

## Design Principles

1. **Zero dependencies** — pure Python stdlib. Drops into any stack.
2. **Composable** — use one guard or all five. Order doesn't matter.
3. **Stateless between runs** — `reset()` called automatically on each `agent.run()`.
4. **Protocol-based** — implement `should_stop / reason / reset` to build custom guards.
5. **Fail-safe** — guards never raise; they return `bool`.

Guard overhead: **< 1ms per step**. The bottleneck is always your LLM call.

---

## Part of Arsenal

```
verdict · sentinel · herald · engram · arsenal
```

| Repo | Purpose |
|------|---------|
| [verdict](https://github.com/darshjme/verdict) | Score your agents |
| [sentinel](https://github.com/darshjme/sentinel) | ← you are here |
| [herald](https://github.com/darshjme/herald) | Semantic task router |
| [engram](https://github.com/darshjme/engram) | Agent memory |
| [arsenal](https://github.com/darshjme/arsenal) | The full pipeline |

---

## License

MIT © [Darshankumar Joshi](https://github.com/darshjme) · Built as part of the [Arsenal](https://github.com/darshjme/arsenal) toolkit.

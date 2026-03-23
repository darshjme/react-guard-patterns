"""
react_guards — Stop-condition patterns for ReAct agents.

Quick start::

    from react_guards import GuardedReActAgent, AgentResult, StepOutput
    from react_guards.guards import (
        MaxStepsGuard,
        CostCeilingGuard,
        LoopDetectionGuard,
        TimeoutGuard,
        ProgressGuard,
        AgentState,
    )
"""

from .agent import AgentResult, GuardedReActAgent, StepOutput
from .guards import (
    AgentState,
    CostCeilingGuard,
    Guard,
    LoopDetectionGuard,
    MaxStepsGuard,
    ProgressGuard,
    TimeoutGuard,
)

__all__ = [
    "GuardedReActAgent",
    "AgentResult",
    "StepOutput",
    "AgentState",
    "Guard",
    "MaxStepsGuard",
    "CostCeilingGuard",
    "LoopDetectionGuard",
    "TimeoutGuard",
    "ProgressGuard",
]

__version__ = "0.1.0"

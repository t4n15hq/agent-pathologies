"""Task registry. Tasks are referenced by string in configs/pivot_a.yaml
so an experiment swap is a one-line config change, not a code change."""

from __future__ import annotations

from typing import Any

from .arithmetic import MultiStepArithmetic
from .base import Task, TaskInstance
from .closed_qa import ClosedQA
from .code_trace import CodeTrace
from .counterintuitive_math import CounterintuitiveMath
from .multi_fact_needle import MultiFactNeedle
from .needle_qa import NeedleQA
from .variable_tracking import VariableTracking


TASK_REGISTRY: dict[str, type[Task]] = {
    "arithmetic": MultiStepArithmetic,
    "needle": NeedleQA,
    "multi_fact_needle": MultiFactNeedle,
    "closed_qa": ClosedQA,
    "counterintuitive": CounterintuitiveMath,
    "variable_tracking": VariableTracking,
    "code_trace": CodeTrace,
}


def get_task(name: str, **kwargs: Any) -> Task:
    if name not in TASK_REGISTRY:
        available = sorted(TASK_REGISTRY)
        raise ValueError(f"Unknown task: {name!r}. Available: {available}")
    return TASK_REGISTRY[name](**kwargs)


__all__ = [
    "Task",
    "TaskInstance",
    "TASK_REGISTRY",
    "get_task",
    "MultiStepArithmetic",
    "NeedleQA",
    "MultiFactNeedle",
    "ClosedQA",
    "CounterintuitiveMath",
    "VariableTracking",
    "CodeTrace",
]

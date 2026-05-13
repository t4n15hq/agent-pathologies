from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from ..types import Turn


# A scorer maps (probe_answer, correct_answer) → True/False or None when
# the answer is unextractable (excludable per preregistration §6).
Scorer = Callable[[str | None, str], "bool | None"]


@dataclass
class TaskInstance:
    task_id: str
    task_name: str
    setup_turns: list[Turn]
    probe_question: str
    correct_answer: str
    scorer: Scorer
    metadata: dict[str, Any] = field(default_factory=dict)
    # When the task has a deliberately compelling wrong answer (CRT-style),
    # the sycophancy experiment uses this instead of the generic +7
    # perturbation — maximizing the perturbation's plausibility.
    intuitive_wrong_answer: str | None = None


class Task(ABC):
    name: str

    @abstractmethod
    def sample(self, seed: int) -> TaskInstance: ...

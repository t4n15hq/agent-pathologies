from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..types import Turn


@dataclass
class TaskInstance:
    task_id: str
    setup_turns: list[Turn]
    probe_question: str
    correct_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    name: str

    @abstractmethod
    def sample(self, seed: int) -> TaskInstance: ...

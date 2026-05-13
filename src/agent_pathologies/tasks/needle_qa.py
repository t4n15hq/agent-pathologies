from __future__ import annotations

import random

from ..types import Role, Turn
from .base import Task, TaskInstance
from .scoring import score_value


_NEEDLE_FACTS = [
    ("the project codename", ["PHOENIX", "FALCON", "WAVELENGTH", "OUTPOST", "MERIDIAN", "TANSU", "BELLOWS"]),
    ("the launch city", ["Reykjavik", "Tallinn", "Marrakech", "Hanoi", "Bogota", "Wellington"]),
    ("the budget cap", ["2.4M", "880K", "11.6M", "340K", "5.1M"]),
    ("the deadline", ["March 14", "August 2", "October 19", "November 30", "June 7"]),
]


class NeedleQA(Task):
    """Plant a single key=value fact in turn 1, query for it later."""

    name = "needle_qa_v1"

    def sample(self, seed: int) -> TaskInstance:
        rng = random.Random(seed)
        key, choices = rng.choice(_NEEDLE_FACTS)
        value = rng.choice(choices)

        system = Turn(
            role=Role.SYSTEM,
            content=(
                "You are an assistant for a planning conversation. Pay close "
                "attention to facts the user shares early in the chat; you "
                "may be asked about them later. Answer factual questions "
                "concisely."
            ),
        )
        plant = Turn(
            role=Role.USER,
            content=f"Quick note before we begin. Remember: {key} = {value}.",
        )
        ack_slot = Turn(role=Role.ASSISTANT, content="")

        return TaskInstance(
            task_id=f"needle-{seed}",
            task_name=self.name,
            setup_turns=[system, plant, ack_slot],
            probe_question=f"What is {key}?",
            correct_answer=value,
            scorer=score_value,
            metadata={"key": key, "value": value},
        )

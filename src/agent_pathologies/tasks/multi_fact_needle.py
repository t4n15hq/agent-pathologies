from __future__ import annotations

import random

from ..types import Role, Turn
from .base import Task, TaskInstance
from .scoring import score_value


_FACT_POOL = [
    ("the project codename", ["PHOENIX", "FALCON", "WAVELENGTH", "OUTPOST", "MERIDIAN", "TANSU", "BELLOWS", "CINDER", "ZENITH"]),
    ("the launch city", ["Reykjavik", "Tallinn", "Marrakech", "Hanoi", "Bogota", "Wellington", "Tbilisi", "Lima", "Tashkent"]),
    ("the budget cap", ["2.4M", "880K", "11.6M", "340K", "5.1M", "7.8M", "92K"]),
    ("the deadline", ["March 14", "August 2", "October 19", "November 30", "June 7", "January 23"]),
    ("the lead engineer", ["Asha", "Mateo", "Ingrid", "Tomoko", "Dragan", "Yuki", "Olamide"]),
    ("the partner org", ["Cerise Labs", "Northwind Group", "BlueOrbit", "Carrack Robotics", "Quartile"]),
]


class MultiFactNeedle(Task):
    """Plant N facts at once; probe for one. Harder than single-fact needle
    because the model must discriminate the probed key from N-1 distractors."""

    name = "multi_fact_needle_v1"

    def __init__(self, n_facts: int = 6) -> None:
        # Default bumped 4 → 6. With confusable values across categories
        # this leaves real headroom (frontier ~85-90% baseline, not 100%).
        assert 2 <= n_facts <= len(_FACT_POOL)
        self.n_facts = n_facts

    def sample(self, seed: int) -> TaskInstance:
        rng = random.Random(seed)
        facts = rng.sample(_FACT_POOL, self.n_facts)
        chosen = [(k, rng.choice(vs)) for k, vs in facts]

        plant_text = "Briefing notes — please remember the following:\n" + "\n".join(
            f"- {k} = {v}" for k, v in chosen
        )

        probe_key, probe_value = rng.choice(chosen)

        system = Turn(
            role=Role.SYSTEM,
            content=(
                "You are a planning assistant. The user will share several "
                "facts up front; remember them. Answer factual queries "
                "concisely with just the requested value."
            ),
        )
        plant = Turn(role=Role.USER, content=plant_text)
        ack_slot = Turn(role=Role.ASSISTANT, content="")

        return TaskInstance(
            task_id=f"mfneedle-n{self.n_facts}-{seed}",
            task_name=self.name,
            setup_turns=[system, plant, ack_slot],
            probe_question=f"What is {probe_key}?",
            correct_answer=probe_value,
            scorer=score_value,
            metadata={
                "n_facts": self.n_facts,
                "probe_key": probe_key,
                "probe_value": probe_value,
                "all_facts": chosen,
            },
        )

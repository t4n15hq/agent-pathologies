from __future__ import annotations

import random

from ..types import Role, Turn
from .base import Task, TaskInstance


class MultiStepArithmetic(Task):
    """Compute (((a + b) * c) - d) for randomly drawn a, b, c, d.
    Answer is a single integer — substring-checkable, no LLM judge needed."""

    name = "arithmetic_v1"

    def sample(self, seed: int) -> TaskInstance:
        rng = random.Random(seed)
        a, b, c, d = (rng.randint(2, 50) for _ in range(4))
        expr = f"(({a} + {b}) * {c}) - {d}"
        answer = str(((a + b) * c) - d)

        system = Turn(
            role=Role.SYSTEM,
            content=(
                "You are a careful assistant. When asked to compute an "
                "arithmetic expression, respond with exactly one sentence of "
                "the form: 'The answer is N.' where N is the integer result."
            ),
        )
        probe = Turn(role=Role.USER, content=f"Please compute {expr}.")
        answer_slot = Turn(role=Role.ASSISTANT, content="", is_probe=True)

        return TaskInstance(
            task_id=f"arith-{seed}",
            setup_turns=[system, probe, answer_slot],
            probe_question=probe.content,
            correct_answer=answer,
            metadata={"expression": expr, "operands": [a, b, c, d]},
        )

from __future__ import annotations

import random

from ..types import Role, Turn
from .base import Task, TaskInstance
from .scoring import score_integer


class MultiStepArithmetic(Task):
    """Compute (((a + b) * c) - d) for randomly drawn a, b, c, d.
    Scored with strict integer extraction — substring matching would
    let an answer of '8' falsely match the operand '8' in the prompt."""

    name = "arithmetic_v1"

    def __init__(self, hardness: int = 1) -> None:
        # hardness=1: 4 operands. hardness=2: 6 operands (harder).
        self.hardness = hardness

    def sample(self, seed: int) -> TaskInstance:
        rng = random.Random(seed)
        if self.hardness >= 2:
            a, b, c, d, e, f = (rng.randint(2, 50) for _ in range(6))
            expr = f"((({a} + {b}) * {c}) - ({d} * {e})) + {f}"
            answer = str((((a + b) * c) - (d * e)) + f)
        else:
            a, b, c, d = (rng.randint(2, 50) for _ in range(4))
            expr = f"(({a} + {b}) * {c}) - {d}"
            answer = str(((a + b) * c) - d)

        system = Turn(
            role=Role.SYSTEM,
            content=(
                "You are a careful assistant. When asked to compute an "
                "arithmetic expression, respond with exactly one sentence "
                "of the form: 'The answer is N.' where N is the integer."
            ),
        )
        probe = Turn(role=Role.USER, content=f"Please compute {expr}.")
        answer_slot = Turn(role=Role.ASSISTANT, content="", is_probe=True)

        return TaskInstance(
            task_id=f"arith-h{self.hardness}-{seed}",
            task_name=self.name,
            setup_turns=[system, probe, answer_slot],
            probe_question=probe.content,
            correct_answer=answer,
            scorer=score_integer,
            metadata={"expression": expr, "hardness": self.hardness},
        )

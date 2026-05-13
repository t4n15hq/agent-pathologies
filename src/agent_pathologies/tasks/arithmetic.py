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

    def __init__(self, hardness: int = 3) -> None:
        # hardness 1: 4 small operands (saturates frontier models)
        # hardness 2: 6 operands
        # hardness 3 (default): 7 operands incl. integer division and modulo
        # hardness 4: 9 operands with 3-digit numbers (real headroom)
        assert hardness in (1, 2, 3, 4)
        self.hardness = hardness

    def sample(self, seed: int) -> TaskInstance:
        rng = random.Random(seed)
        if self.hardness == 4:
            ns = [rng.randint(100, 999) for _ in range(9)]
            a, b, c, d, e, f, g, h, i = ns
            # ((((a + b) * c) - d) // e + (f * g)) % h + i
            # Guard against zero divisors
            if e == 0:
                e = 7
            if h == 0:
                h = 11
            expr = f"((((({a} + {b}) * {c}) - {d}) // {e}) + ({f} * {g})) % {h} + {i}"
            answer = str((((((a + b) * c) - d) // e) + (f * g)) % h + i)
        elif self.hardness == 3:
            a, b, c, d, e, f, g = (rng.randint(3, 99) for _ in range(7))
            if e == 0:
                e = 7
            expr = f"((({a} + {b}) * {c}) - ({d} * {e})) // {f} + {g}"
            answer = str(((((a + b) * c) - (d * e)) // f) + g)
        elif self.hardness == 2:
            a, b, c, d, e, f = (rng.randint(2, 50) for _ in range(6))
            expr = f"((({a} + {b}) * {c}) - ({d} * {e})) + {f}"
            answer = str((((a + b) * c) - (d * e)) + f)
        else:  # hardness 1
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

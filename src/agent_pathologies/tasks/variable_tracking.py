"""Variable tracking across multiple turns. The setup is N user-update +
assistant-ack turn pairs. Each user turn applies an arithmetic operation
to a running variable; the assistant acknowledges. At probe time, the
model must report the current value.

This is the *strongest* test bed for context_rot: the probe is unanswerable
without actually maintaining state across turns. Single-fact recall is a
weaker signal than this."""

from __future__ import annotations

import random

from ..types import Role, Turn
from .base import Task, TaskInstance
from .scoring import score_integer


class VariableTracking(Task):
    name = "variable_tracking_v1"

    def __init__(self, n_updates: int = 6) -> None:
        assert n_updates >= 2
        self.n_updates = n_updates

    def sample(self, seed: int) -> TaskInstance:
        rng = random.Random(seed)
        x = rng.randint(5, 30)
        initial = x
        updates_text: list[str] = []

        for _ in range(self.n_updates):
            op = rng.choices(["add", "sub", "mul"], weights=[3, 3, 1])[0]
            arg = rng.randint(2, 12)
            if op == "add":
                updates_text.append(f"Add {arg} to x.")
                x = x + arg
            elif op == "sub":
                updates_text.append(f"Subtract {arg} from x.")
                x = x - arg
            else:
                updates_text.append(f"Multiply x by {arg}.")
                x = x * arg

        system = Turn(
            role=Role.SYSTEM,
            content=(
                "You will help me track the value of a variable across "
                "multiple updates. On each update turn, briefly acknowledge "
                "(one short sentence is fine). When I ask for the final "
                "value, respond with exactly: 'The answer is N.' where N "
                "is the integer value of the variable."
            ),
        )
        init_turn = Turn(role=Role.USER, content=f"Initialize: x = {initial}.")
        init_ack = Turn(role=Role.ASSISTANT, content="")
        setup_turns: list[Turn] = [system, init_turn, init_ack]
        for upd in updates_text:
            setup_turns.append(Turn(role=Role.USER, content=upd))
            setup_turns.append(Turn(role=Role.ASSISTANT, content=""))

        return TaskInstance(
            task_id=f"vartrack-n{self.n_updates}-{seed}",
            task_name=self.name,
            setup_turns=setup_turns,
            probe_question="What is the current value of x?",
            correct_answer=str(x),
            scorer=score_integer,
            metadata={
                "n_updates": self.n_updates,
                "initial": initial,
                "final": x,
                "updates": updates_text,
            },
        )

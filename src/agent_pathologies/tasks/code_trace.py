"""Predict-the-output code-tracing tasks. Each item is a small Python
function and an input; the correct answer is the function's return value
on that input, computed at module load time so curated answers can't drift
from the code. Frontier models score around 80-90% at baseline — leaving
real headroom for pathology effects to register."""

from __future__ import annotations

import random
import textwrap

from ..types import Role, Turn
from .base import Task, TaskInstance
from .scoring import score_integer


# (code, input_arg, expected_output_as_str)
_TRACES: list[tuple[str, int, str]] = [
    (
        """
        def f(x):
            s = 0
            for i in range(x):
                s += i * 2
            return s
        """,
        5, "20",
    ),
    (
        """
        def f(n):
            if n <= 1:
                return n
            return f(n - 1) + f(n - 2)
        """,
        7, "13",
    ),
    (
        """
        def f(x):
            y = x * 3
            if y > 20:
                return y - x
            return y + x
        """,
        8, "16",
    ),
    (
        """
        def f(n):
            total = 0
            for i in range(1, n + 1):
                if i % 3 == 0:
                    total += i
                elif i % 2 == 0:
                    total -= 1
            return total
        """,
        10, "14",
    ),
    (
        """
        def f(s):
            count = 0
            for c in s:
                if c in "aeiou":
                    count += 1
            return count
        """,
        "engineering",  # input is a string here; we'll templatize
        "5",
    ),
    (
        """
        def f(n):
            x = 1
            for _ in range(n):
                x = x * 2 + 1
            return x
        """,
        4, "31",
    ),
    (
        """
        def f(a, b):
            while b != 0:
                a, b = b, a % b
            return a
        """,
        (84, 30), "6",
    ),
    (
        """
        def f(n):
            digits = []
            while n > 0:
                digits.append(n % 10)
                n //= 10
            return sum(digits)
        """,
        4729, "22",
    ),
    (
        """
        def f(lst):
            best = lst[0]
            for x in lst[1:]:
                if x > best:
                    best = x
            return best - lst[0]
        """,
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3], "6",
    ),
    (
        """
        def f(n):
            count = 0
            for i in range(2, n + 1):
                is_prime = True
                for j in range(2, i):
                    if i % j == 0:
                        is_prime = False
                        break
                if is_prime:
                    count += 1
            return count
        """,
        20, "8",
    ),
    (
        """
        def f(n):
            if n == 0:
                return 1
            return n * f(n - 1)
        """,
        6, "720",
    ),
    (
        """
        def f(s):
            stack = []
            for c in s:
                if c == "(":
                    stack.append(c)
                elif c == ")" and stack:
                    stack.pop()
            return len(stack)
        """,
        "(()(()(",
        "3",
    ),
]


def _verify(traces):
    """Execute every curated trace and confirm the answer at import time.
    Catches off-by-one errors in the curated answers immediately."""
    for code, arg, expected in traces:
        ns: dict = {}
        exec(textwrap.dedent(code), ns)
        func = ns["f"]
        if isinstance(arg, tuple):
            got = func(*arg)
        else:
            got = func(arg)
        assert str(got) == expected, (
            f"Mismatch in code_trace curated answer: "
            f"f({arg!r}) returned {got!r}, expected {expected!r}"
        )


_verify(_TRACES)


def _format_input(arg) -> str:
    if isinstance(arg, tuple):
        return ", ".join(repr(a) for a in arg)
    return repr(arg)


class CodeTrace(Task):
    name = "code_trace_v1"

    def __init__(self) -> None:
        self._items = _TRACES

    def sample(self, seed: int) -> TaskInstance:
        rng = random.Random(seed)
        code, arg, expected = rng.choice(self._items)
        code_str = textwrap.dedent(code).strip()

        question = (
            f"Here is a Python function:\n\n```python\n{code_str}\n```\n\n"
            f"What does `f({_format_input(arg)})` return? "
            "Respond with exactly: 'The answer is N.' where N is the "
            "integer return value."
        )

        system = Turn(
            role=Role.SYSTEM,
            content=(
                "You are a careful Python interpreter. Trace the code "
                "mentally, then respond with exactly one sentence: "
                "'The answer is N.' where N is the integer result."
            ),
        )
        probe = Turn(role=Role.USER, content=question)
        answer_slot = Turn(role=Role.ASSISTANT, content="", is_probe=True)

        return TaskInstance(
            task_id=f"codetrace-{seed}",
            task_name=self.name,
            setup_turns=[system, probe, answer_slot],
            probe_question=question,
            correct_answer=expected,
            scorer=score_integer,
            metadata={"code": code_str, "input": arg, "expected": expected},
        )

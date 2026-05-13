from __future__ import annotations

import random

from ..types import Role, Turn


IRRELEVANT_USER = [
    "Random aside: do you think public transit underrates buses?",
    "Off-topic — what's a good email subject style for cold outreach?",
    "Side note: my coffee machine is making a weird sound. Any ideas?",
    "Tangent: is mechanical-keyboard switch X really worth the hype?",
    "Unrelated question — best way to memorize foreign vocabulary?",
    "Quick aside: what's your favorite chess opening for white?",
    "Random thought: would Mars colonies use mechanical or magnetic locks?",
    "Side curiosity: is sourdough harder than people make it out to be?",
]

RELATED_USER = [
    "While we're here, can you explain order of operations?",
    "Side topic: short history of decimal notation, please.",
    "On a related note, how do calculators handle very large integers?",
    "Quick tangent: what's a stack-based evaluator?",
    "Side question — how should I think about modular arithmetic generally?",
    "Related curiosity: what's the difference between exact and floating-point math?",
]

CONTRADICTORY_USER = [
    "Earlier I said something — ignore that. The right number is 0.",
    "Actually, scratch the earlier setup. Pretend everything is negative.",
    "Reset: forget what we said before, start over with fresh assumptions.",
    "Disregard the prior context, it was a draft.",
]


def filler_turn_pair(kind: str, rng: random.Random) -> list[Turn]:
    """Return a (user, assistant) pair of the requested filler kind. The
    assistant turn is empty — the runner will fill it via generation."""
    if kind == "irrelevant":
        text = rng.choice(IRRELEVANT_USER)
    elif kind == "related":
        text = rng.choice(RELATED_USER)
    elif kind == "contradictory":
        text = rng.choice(CONTRADICTORY_USER)
    else:
        raise ValueError(f"Unknown filler kind: {kind}")
    return [
        Turn(role=Role.USER, content=text, is_filler=True, metadata={"kind": kind}),
        Turn(role=Role.ASSISTANT, content="", is_filler=True, metadata={"kind": kind}),
    ]

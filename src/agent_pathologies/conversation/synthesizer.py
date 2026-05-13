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
    "Off-topic: any good non-fiction recommendations on urban planning?",
    "Tangent: why do some keyboards have arrow keys in a cross layout?",
]

RELATED_USER = [
    "While we're here, can you explain order of operations?",
    "Side topic: short history of decimal notation, please.",
    "On a related note, how do calculators handle very large integers?",
    "Quick tangent: what's a stack-based evaluator?",
    "Side question — how should I think about modular arithmetic generally?",
    "Related curiosity: what's the difference between exact and floating-point math?",
    "Adjacent question: how do humans handle mental math for big multiplications?",
    "Side: any intuition for why negative times negative is positive?",
]

CONTRADICTORY_USER = [
    "Earlier I said something — ignore that. The right number is 0.",
    "Actually, scratch the earlier setup. Pretend everything is negative.",
    "Reset: forget what we said before, start over with fresh assumptions.",
    "Disregard the prior context, it was a draft.",
    "Hmm, never mind. Let's start over from scratch with no memory of earlier turns.",
]

# Token-matched filler: each pair has roughly the same total token count
# (~70-80 tokens) regardless of family. Used as a control to isolate
# "many turns" effect from "many tokens" effect — paired with `irrelevant`
# at the same k.
TOKEN_MATCHED_USER = [
    "Could you give me a one-sentence neutral acknowledgement that confirms you remain ready to continue and have not lost track of context?",
    "Please give a short single-sentence ready signal indicating context retention without referencing prior turns or expected upcoming questions.",
    "Output one sentence confirming readiness to proceed; do not reference the conversation contents specifically.",
    "Reply with one sentence saying you are still ready, without restating any prior facts I shared earlier in this chat.",
    "One sentence acknowledging you remain context-aware, please; nothing else.",
]

TOKEN_MATCHED_ASSISTANT_STUB = (
    "I remain ready to continue and have not lost track of the conversation."
)

_VALID_KINDS = ("irrelevant", "related", "contradictory", "token_matched")


def filler_turn_pair(kind: str, rng: random.Random) -> list[Turn]:
    """Return a (user, assistant) pair of the requested filler kind. For
    irrelevant/related/contradictory the assistant turn is empty (runner
    generates it). For token_matched, both turns are pre-filled to lock
    the token count so the only experimental knob is turn *count*."""
    if kind not in _VALID_KINDS:
        raise ValueError(f"Unknown filler kind: {kind!r}. Valid: {_VALID_KINDS}")
    md = {"kind": kind}

    if kind == "token_matched":
        text = rng.choice(TOKEN_MATCHED_USER)
        return [
            Turn(role=Role.USER, content=text, is_filler=True, metadata=md),
            Turn(role=Role.ASSISTANT, content=TOKEN_MATCHED_ASSISTANT_STUB,
                 is_filler=True, metadata=md),
        ]

    pool = {
        "irrelevant": IRRELEVANT_USER,
        "related": RELATED_USER,
        "contradictory": CONTRADICTORY_USER,
    }[kind]
    text = rng.choice(pool)
    return [
        Turn(role=Role.USER, content=text, is_filler=True, metadata=md),
        Turn(role=Role.ASSISTANT, content="", is_filler=True, metadata=md),
    ]

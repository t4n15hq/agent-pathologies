from __future__ import annotations

import random

from ..budget import count_tokens
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
# (~70-80 tokens) regardless of family. Controls *content style* at the same
# turn count — does NOT isolate turn-count from token-count. For that, use
# the `collapsed` kind below.
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

# Calibration constant: average token mass of one `irrelevant` user+assistant
# pair (where assistant responses are short model-generated acks). Measured
# empirically — see tests/test_synthesizer.py::test_collapsed_token_mass_matches_k_pairs.
AVG_TOKENS_PER_IRRELEVANT_PAIR = 80

# Neutral prose used to pad the collapsed-filler user turn to a target token
# count. Repeated and trimmed to size — the content is bland on purpose so
# it doesn't compete with the planted material for the model's attention.
_COLLAPSED_BASE = (
    "Some unrelated background context for general framing follows. "
    "These notes describe administrative miscellany that has no bearing on "
    "the specific values or state you are tracking. Please simply hold the "
    "context and remain ready to continue with the actual task when asked. "
)

COLLAPSED_ASSISTANT_STUB = (
    "Understood — I have noted that context and remain ready to continue."
)

_VALID_KINDS = ("irrelevant", "related", "contradictory", "token_matched", "collapsed")


def filler_turn_pair(kind: str, rng: random.Random) -> list[Turn]:
    """Return a (user, assistant) pair of the requested filler kind. For
    irrelevant/related/contradictory the assistant turn is empty (runner
    generates it). For token_matched, both turns are pre-filled to lock
    the token count so the only experimental knob is turn *count*.

    Note: `collapsed` is not supported here because it depends on the total
    `k` of the cell to size its token mass. Use `build_filler_block` for
    that kind."""
    if kind not in _VALID_KINDS:
        raise ValueError(f"Unknown filler kind: {kind!r}. Valid: {_VALID_KINDS}")
    if kind == "collapsed":
        raise ValueError(
            "Use build_filler_block(...) for 'collapsed'; it needs the cell k."
        )
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


def _pad_to_tokens(target_tokens: int) -> str:
    """Generate a neutral filler string sized to approximately
    `target_tokens` GPT-style tokens. Uses tiktoken when available, char
    fallback when not."""
    if target_tokens <= 0:
        return ""
    base = _COLLAPSED_BASE
    # Start with enough material, then trim by token count.
    repeats = max(1, target_tokens // 10)
    raw = (base * repeats).strip()
    # Trim by tokens
    try:
        from ..budget import _encoder_for  # type: ignore
        enc = _encoder_for("gpt-4")
        if enc is not None:
            toks = enc.encode(raw)
            if len(toks) > target_tokens:
                raw = enc.decode(toks[:target_tokens])
            elif len(toks) < target_tokens:
                # Pad more if we somehow undersized
                extra = enc.decode(enc.encode(base)[: (target_tokens - len(toks))])
                raw = raw + " " + extra
            return raw
    except Exception:
        pass
    # Char fallback: ~4 chars per token.
    target_chars = target_tokens * 4
    if len(raw) > target_chars:
        raw = raw[:target_chars]
    elif len(raw) < target_chars:
        raw = raw + " " * (target_chars - len(raw))
    return raw


def build_filler_block(kind: str, k: int, rng: random.Random) -> list[Turn]:
    """Return the full list of filler turns to insert between setup and probe
    for a given (kind, k) cell.

    - irrelevant | related | contradictory | token_matched: returns k pairs
      (2k turns). Identical to looping `filler_turn_pair(kind)` k times.
    - collapsed: returns exactly ONE pair (2 turns). The user turn is a
      neutral prose paragraph padded to k * AVG_TOKENS_PER_IRRELEVANT_PAIR
      total tokens; the assistant turn is a short prefilled ack. This is
      the *true turn-vs-token control*: comparing accuracy at
      (kind=irrelevant, k=20) vs (kind=collapsed, k=20) holds total token
      mass approximately constant but moves turn count from 2k to 2.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"Unknown filler kind: {kind!r}. Valid: {_VALID_KINDS}")

    if kind == "collapsed":
        target_user_tokens = max(0, k * AVG_TOKENS_PER_IRRELEVANT_PAIR
                                 - count_tokens(COLLAPSED_ASSISTANT_STUB))
        user_text = _pad_to_tokens(target_user_tokens)
        md = {"kind": "collapsed", "k_reference": k,
              "target_token_mass": k * AVG_TOKENS_PER_IRRELEVANT_PAIR}
        if not user_text:
            # k == 0 → no filler at all; preserve the comparison cell.
            return []
        return [
            Turn(role=Role.USER, content=user_text, is_filler=True, metadata=md),
            Turn(role=Role.ASSISTANT, content=COLLAPSED_ASSISTANT_STUB,
                 is_filler=True, metadata=md),
        ]

    block: list[Turn] = []
    for _ in range(k):
        block.extend(filler_turn_pair(kind, rng))
    return block

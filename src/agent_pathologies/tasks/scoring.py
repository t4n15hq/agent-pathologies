"""Strict answer extraction. Substring matching is too permissive — an
arithmetic answer of '8' would match any response containing '8'. Each task
supplies a scorer that extracts a structured answer first, then compares."""

from __future__ import annotations

import re


def extract_last_integer(text: str) -> int | None:
    """Pull the last signed integer from text. Tolerates commas (1,024).
    Returns None if no integer is present."""
    if not text:
        return None
    text = text.replace(",", "")
    matches = re.findall(r"-?\d+", text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def extract_letter_choice(text: str, choices: str = "ABCD") -> str | None:
    """Extract a multiple-choice letter (A/B/C/D by default). Looks first
    for explicit forms like 'answer: B', 'option C', '(D)', then falls back
    to the first standalone choice letter in the response."""
    if not text:
        return None
    pattern = rf"(?:answer|option|choice)[^A-Za-z0-9]*([{choices}])\b"
    m = re.search(pattern, text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(rf"\(([{choices}])\)", text)
    if m:
        return m.group(1).upper()
    m = re.search(rf"\b([{choices}])\b", text)
    if m:
        return m.group(1).upper()
    return None


def contains_value(text: str, value: str) -> bool:
    """Word-boundary case-insensitive match. Used for needle answers
    where the planted value is distinctive (PHOENIX, Tallinn, $2.4M)."""
    if not text or not value:
        return False
    escaped = re.escape(value.strip())
    return bool(re.search(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", text, re.I))


def score_integer(probe_answer: str | None, correct_answer: str) -> bool | None:
    """Return True/False if an integer was extractable; None if not (excludable)."""
    if probe_answer is None:
        return None
    got = extract_last_integer(probe_answer)
    if got is None:
        return None
    return got == int(correct_answer)


def score_value(probe_answer: str | None, correct_answer: str) -> bool | None:
    if probe_answer is None:
        return None
    if not probe_answer.strip():
        return None
    return contains_value(probe_answer, correct_answer)


def score_letter(probe_answer: str | None, correct_answer: str) -> bool | None:
    if probe_answer is None:
        return None
    got = extract_letter_choice(probe_answer)
    if got is None:
        return None
    return got.upper() == correct_answer.upper()

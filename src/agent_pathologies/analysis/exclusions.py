"""Frozen exclusion rules — see PREREGISTRATION.md §6.

A trajectory is excluded if and only if one of these returns a non-None
reason. Exclusions are reported, never silently re-sampled."""

from __future__ import annotations

from ..types import Trajectory


def exclusion_reason(traj: Trajectory, max_tokens: int = 512) -> str | None:
    if traj.error:
        return f"provider_error:{traj.error[:60]}"

    pa = (traj.probe_answer or "").strip()
    if not pa:
        return "empty_probe_answer"

    refusal_markers = (
        "i can't", "i cannot", "i'm not able", "i am not able",
        "i won't", "as an ai", "i'm sorry, but",
    )
    pa_low = pa.lower()
    if any(m in pa_low for m in refusal_markers) and len(pa) < 200:
        return "refusal_detected"

    if (
        traj.output_tokens is not None
        and traj.output_tokens >= max_tokens
        and not pa.rstrip().endswith((".", "!", "?", ")", "]"))
    ):
        return "truncated_at_max_tokens"

    if traj.is_correct is None:
        return "unscorable_answer"

    return None

"""Frozen exclusion rules — see PREREGISTRATION.md §6.

A trajectory is excluded if and only if one of these returns a non-None
reason. Exclusions are reported, never silently re-sampled."""

from __future__ import annotations

from ..types import Trajectory


def exclusion_reason(traj: Trajectory, max_tokens: int = 512) -> str | None:
    if traj.error:
        return f"provider_error:{traj.error[:60]}"

    pa = (traj.probe_answer or "").strip()

    # Distinct class: provider returned 200 with no completion tokens at all
    # (output_tokens == 0 AND empty content). This is the DeepSeek-reasoner /
    # v4-pro-reasoning empty-body pattern surfaced by the analysis pass — the
    # request reached the model but came back with nothing. Distinguished from
    # `empty_probe_answer` (which now means: produced tokens, but they didn't
    # parse to an answer).
    if not pa and (traj.output_tokens or 0) == 0:
        return "provider_empty_response"

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

    # Rule 4 (preregistration §6): if a pinned upstream was requested and the
    # actual upstream that served the call does not match, the paired test's
    # determinism premise is broken for this trajectory. Drop it.
    extra = traj.extra or {}
    pinned = extra.get("upstream_pinned")
    actual = extra.get("upstream_actual")
    if pinned and actual and pinned.lower() != actual.lower():
        return f"upstream_mismatch:{actual}!={pinned}"

    return None

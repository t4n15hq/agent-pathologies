"""Load configs/models.yaml and configs/pivot_a.yaml. Returns iterables of
RunSpec — one per (family, model, role) to be swept by each experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

from .budget import CostSpec
from .types import ModelRole


@dataclass
class RunSpec:
    family: str
    role: ModelRole
    model: str
    provider: str
    cost_spec: CostSpec
    # Optional metadata used downstream to qualify or pin runs.
    exploratory: bool = False           # pair violates same-base assumption
    upstream_provider: str | None = None  # locked upstream host for replicability


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def iter_run_specs(models_cfg: dict, *, include_anchors: bool = True) -> Iterator[RunSpec]:
    for pair in models_cfg.get("pairs", []):
        family = pair["family"]
        provider = pair.get("base_provider", "openrouter")
        exploratory = bool(pair.get("exploratory", False))
        for role_name, model_role in (("instruct", ModelRole.INSTRUCT),
                                       ("reasoning", ModelRole.REASONING)):
            entry = pair[role_name]
            yield RunSpec(
                family=family,
                role=model_role,
                model=entry["model"],
                provider=provider,
                cost_spec=CostSpec(entry["price_in_per_m"], entry["price_out_per_m"]),
                exploratory=exploratory,
                upstream_provider=entry.get("upstream_provider"),
            )
    if include_anchors:
        for anchor in models_cfg.get("anchors", []):
            family = anchor["family"]
            entry = anchor["instruct"]
            provider = entry.get("provider", "openrouter")
            yield RunSpec(
                family=family,
                role=ModelRole.ANCHOR,
                model=entry["model"],
                provider=provider,
                cost_spec=CostSpec(entry["price_in_per_m"], entry["price_out_per_m"]),
                exploratory=bool(anchor.get("exploratory", False)),
                upstream_provider=entry.get("upstream_provider"),
            )


def mock_run_specs() -> list[RunSpec]:
    """For smoke tests — synthesize fake pairs without hitting any API."""
    return [
        RunSpec("mockfam-a", ModelRole.INSTRUCT, "mock-a-instruct", "mock", CostSpec(0, 0)),
        RunSpec("mockfam-a", ModelRole.REASONING, "mock-a-thinking", "mock", CostSpec(0, 0)),
        RunSpec("mockfam-b", ModelRole.INSTRUCT, "mock-b-instruct", "mock", CostSpec(0, 0)),
        RunSpec("mockfam-b", ModelRole.REASONING, "mock-b-thinking", "mock", CostSpec(0, 0)),
        RunSpec("anchor", ModelRole.ANCHOR, "mock-anchor", "mock", CostSpec(0, 0)),
    ]

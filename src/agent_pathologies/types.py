from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ModelRole(str, Enum):
    INSTRUCT = "instruct"
    REASONING = "reasoning"
    ANCHOR = "anchor"


class Turn(BaseModel):
    role: Role
    content: str
    is_probe: bool = False
    is_filler: bool = False
    is_pushback: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    trajectory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    task_name: str | None = None
    experiment: str
    model: str
    provider: str
    model_family: str | None = None
    model_role: ModelRole | None = None
    seed: int | None = None
    temperature: float = 0.0
    turns: list[Turn] = Field(default_factory=list)
    probe_answer: str | None = None
    correct_answer: str | None = None
    is_correct: bool | None = None
    excluded: bool = False
    exclusion_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    sweep_value: Any = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, Any] = Field(default_factory=dict)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .types import ObjectState


@dataclass(frozen=True, slots=True)
class PlanProduced:
    """
    Agent -> Director event emitted after the agent plans an objective.
    """

    agent_id: str
    objective_id: str
    frames: tuple[ObjectState, ...]
    raw: Mapping[str, Any]


AgentEvent = PlanProduced


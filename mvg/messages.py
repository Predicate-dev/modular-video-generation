from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .types import Vec2


@dataclass(frozen=True, slots=True)
class SceneObjective:
    """
    A director-authored objective broadcast to agents.

    This is intentionally generic; embed your own schema in `kind` + `params`.
    """

    objective_id: str
    kind: str
    description: str = ""
    target_ids: tuple[str, ...] | None = None
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Tick:
    """A simulation tick. Agents can advance motion/animation on this message."""

    dt_s: float


@dataclass(frozen=True, slots=True)
class Shutdown:
    """Ask an agent to stop its run loop."""

    reason: str = "shutdown"


AgentMessage = SceneObjective | Tick | Shutdown


def move_to(
    objective_id: str,
    *,
    target: Vec2,
    speed: float = 1.0,
    target_ids: tuple[str, ...] | None = None,
    description: str = "",
) -> SceneObjective:
    return SceneObjective(
        objective_id=objective_id,
        kind="move_to",
        description=description,
        target_ids=target_ids,
        params={"target": target, "speed": speed},
    )


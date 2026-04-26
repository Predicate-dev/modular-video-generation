from __future__ import annotations

from typing import Any
from typing import TypedDict, TypeAlias

Vec2: TypeAlias = tuple[float, float]


class ObjectState(TypedDict):
    """
    A single simulated object's state.

    Required keys:
      - id
      - position: (x, y)
      - velocity: (vx, vy)
      - current_action: an arbitrary string you control
    """

    id: str
    position: Vec2
    velocity: Vec2
    current_action: str


def v_add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def v_mul(a: Vec2, scalar: float) -> Vec2:
    return (a[0] * scalar, a[1] * scalar)


def v_sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def v_len(a: Vec2) -> float:
    return (a[0] * a[0] + a[1] * a[1]) ** 0.5


def v_norm(a: Vec2) -> Vec2:
    length = v_len(a)
    if length == 0:
        return (0.0, 0.0)
    return (a[0] / length, a[1] / length)


def as_vec2(value: Any) -> Vec2 | None:
    if not (isinstance(value, tuple) and len(value) == 2):
        return None
    x, y = value
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        return None
    return (float(x), float(y))

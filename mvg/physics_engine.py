from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .types import ObjectState, Vec2, v_len, v_norm, v_sub


@dataclass(slots=True)
class PhysicsConfig:
    """
    Very small "hard world" reconciler for kinematic proposals.

    This is not a full rigid-body engine; it enforces a few constraints:
    - per-step speed/accel limits
    - simple non-overlap via circle radii (2D)
    """

    dt_s: float = 1 / 24
    max_speed: float = 25.0
    max_accel: float = 100.0
    default_radius: float = 0.75


class PhysicsEngine:
    def __init__(self, *, cfg: PhysicsConfig | None = None) -> None:
        self.cfg = cfg or PhysicsConfig()
        self._prev: dict[str, ObjectState] = {}
        self._radii: dict[str, float] = {}

    def set_radius(self, object_id: str, radius: float) -> None:
        self._radii[object_id] = float(radius)

    def _radius(self, object_id: str) -> float:
        return self._radii.get(object_id, self.cfg.default_radius)

    def seed_previous(self, states: Mapping[str, ObjectState]) -> None:
        self._prev = {k: dict(v) for k, v in states.items()}

    def resolve(self, desired: Mapping[str, ObjectState]) -> dict[str, ObjectState]:
        """
        Produces authoritative next states from desired per-object proposals.
        """

        dt = self.cfg.dt_s
        next_states: dict[str, ObjectState] = {}
        # 1) Clamp speed + accel vs previous.
        for object_id, want in desired.items():
            prev = self._prev.get(object_id, want)
            vel = want["velocity"]
            speed = v_len(vel)
            if speed > self.cfg.max_speed:
                vel = (vel[0] * self.cfg.max_speed / (speed + 1e-8), vel[1] * self.cfg.max_speed / (speed + 1e-8))

            dv = (vel[0] - prev["velocity"][0], vel[1] - prev["velocity"][1])
            accel = v_len(dv) / max(dt, 1e-6)
            if accel > self.cfg.max_accel:
                scale = self.cfg.max_accel / (accel + 1e-8)
                vel = (prev["velocity"][0] + dv[0] * scale, prev["velocity"][1] + dv[1] * scale)

            # Keep desired position, but if it implies teleport, pull it back toward prev.
            dp = (want["position"][0] - prev["position"][0], want["position"][1] - prev["position"][1])
            dist = v_len(dp)
            max_step = self.cfg.max_speed * dt * 1.25 + 1e-6
            if dist > max_step:
                ddir = v_norm(dp)
                pos = (prev["position"][0] + ddir[0] * max_step, prev["position"][1] + ddir[1] * max_step)
            else:
                pos = want["position"]

            next_states[object_id] = {
                "id": object_id,
                "position": pos,
                "velocity": vel,
                "current_action": want["current_action"],
            }

        # 2) Simple circle non-overlap: push pairs apart if intersecting.
        ids = list(next_states.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = next_states[ids[i]]
                b = next_states[ids[j]]
                ra = self._radius(a["id"])
                rb = self._radius(b["id"])
                min_dist = ra + rb
                delta = v_sub(b["position"], a["position"])
                d = v_len(delta)
                if d < 1e-6:
                    # Degenerate; nudge deterministically.
                    delta = (1.0, 0.0)
                    d = 1.0
                if d < min_dist:
                    push = (min_dist - d) / 2.0
                    dirn = (delta[0] / d, delta[1] / d)
                    a["position"] = (a["position"][0] - dirn[0] * push, a["position"][1] - dirn[1] * push)
                    b["position"] = (b["position"][0] + dirn[0] * push, b["position"][1] + dirn[1] * push)
                    a["current_action"] = "physics_resolved"
                    b["current_action"] = "physics_resolved"

        self._prev = {k: dict(v) for k, v in next_states.items()}
        return next_states


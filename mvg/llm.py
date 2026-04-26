from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .types import Vec2, as_vec2, v_mul, v_norm, v_sub


class JSONLLM(Protocol):
    """
    Minimal interface for "LLM returns structured JSON".

    Implement this with your provider of choice (OpenAI, local model, etc.).
    The agent treats the return value as JSON-serializable dict data.
    """

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        json_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class DeterministicJSONLLM(JSONLLM):
    """
    Offline stub that emits a valid JSON plan deterministically.

    Useful for local testing without network calls.
    """

    def _plan_move_to(
        self,
        *,
        agent_id: str,
        objective_id: str,
        start_pos: Vec2,
        start_vel: Vec2,
        target: Vec2,
        speed: float,
        dt_s: float,
        frames: int,
    ) -> dict[str, Any]:
        direction = v_sub(target, start_pos)
        vel = v_mul(v_norm(direction), speed)
        pos = start_pos
        out_frames: list[dict[str, Any]] = []
        for i in range(frames):
            pos = (pos[0] + vel[0] * dt_s, pos[1] + vel[1] * dt_s)
            out_frames.append(
                {
                    "frame": i,
                    "state": {
                        "id": agent_id,
                        "position": [pos[0], pos[1]],
                        "velocity": [vel[0], vel[1]],
                        "current_action": "moving",
                    },
                }
            )
        return {"agent_id": agent_id, "objective_id": objective_id, "frames": out_frames}

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        json_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        payload = json.loads(user)
        agent_id = str(payload["agent_id"])
        objective = payload["objective"]
        objective_id = str(objective["objective_id"])
        kind = str(objective["kind"])
        dt_s = float(payload.get("dt_s", 1 / 24))
        frames = int(payload.get("frames", 24))
        start_state = payload["prev_state"]
        start_pos = as_vec2(tuple(start_state["position"])) or (0.0, 0.0)
        start_vel = as_vec2(tuple(start_state["velocity"])) or (0.0, 0.0)

        if kind == "move_to":
            target = as_vec2(tuple(objective["params"]["target"]))
            if target is None:
                return {"agent_id": agent_id, "objective_id": objective_id, "frames": []}
            speed = float(objective["params"].get("speed", 1.0))
            return self._plan_move_to(
                agent_id=agent_id,
                objective_id=objective_id,
                start_pos=start_pos,
                start_vel=start_vel,
                target=target,
                speed=speed,
                dt_s=dt_s,
                frames=frames,
            )

        return {
            "agent_id": agent_id,
            "objective_id": objective_id,
            "frames": [
                {
                    "frame": i,
                    "state": {
                        "id": agent_id,
                        "position": [start_pos[0], start_pos[1]],
                        "velocity": [0.0, 0.0],
                        "current_action": "idle",
                    },
                }
                for i in range(frames)
            ],
        }


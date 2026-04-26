from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .types import ObjectState, Vec2, as_vec2, v_len, v_sub


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _vec2_from_json(value: Any) -> Vec2 | None:
    if not (isinstance(value, list) and len(value) == 2):
        return None
    x, y = value
    if not (_is_number(x) and _is_number(y)):
        return None
    return (float(x), float(y))


def object_state_from_json(value: Any) -> ObjectState | None:
    if not isinstance(value, Mapping):
        return None
    object_id = value.get("id")
    position = _vec2_from_json(value.get("position"))
    velocity = _vec2_from_json(value.get("velocity"))
    current_action = value.get("current_action")
    if not (isinstance(object_id, str) and position is not None and velocity is not None and isinstance(current_action, str)):
        return None
    return {"id": object_id, "position": position, "velocity": velocity, "current_action": current_action}


def plan_json_schema() -> Mapping[str, Any]:
    """
    A lightweight schema description for prompts (not a full JSON Schema implementation).
    """

    return {
        "type": "object",
        "required": ["agent_id", "objective_id", "frames"],
        "properties": {
            "agent_id": {"type": "string"},
            "objective_id": {"type": "string"},
            "frames": {
                "type": "array",
                "description": "Exactly 24 entries.",
                "items": {
                    "type": "object",
                    "required": ["frame", "state"],
                    "properties": {
                        "frame": {"type": "integer"},
                        "state": {
                            "type": "object",
                            "required": ["id", "position", "velocity", "current_action"],
                            "properties": {
                                "id": {"type": "string"},
                                "position": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                "velocity": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                "current_action": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    }


@dataclass(frozen=True, slots=True)
class PlanParseResult:
    objective_id: str
    agent_id: str
    frames: tuple[ObjectState, ...]
    raw: Mapping[str, Any]


def parse_plan_json(payload: Mapping[str, Any], *, expected_agent_id: str, expected_frames: int = 24) -> PlanParseResult:
    agent_id = payload.get("agent_id")
    objective_id = payload.get("objective_id")
    frames_value = payload.get("frames")
    if not isinstance(agent_id, str) or not isinstance(objective_id, str) or not isinstance(frames_value, list):
        raise ValueError("Plan JSON missing required keys: agent_id/objective_id/frames")
    if agent_id != expected_agent_id:
        raise ValueError(f"Plan agent_id mismatch: got={agent_id} expected={expected_agent_id}")
    if len(frames_value) != expected_frames:
        raise ValueError(f"Plan must include exactly {expected_frames} frames, got {len(frames_value)}")

    states: list[ObjectState] = []
    for i, frame_item in enumerate(frames_value):
        if not isinstance(frame_item, Mapping):
            raise ValueError(f"Frame {i} must be an object")
        frame_index = frame_item.get("frame")
        if not isinstance(frame_index, int) or frame_index != i:
            raise ValueError(f"Frame index mismatch at {i}: got={frame_index}")
        state_value = frame_item.get("state")
        state = object_state_from_json(state_value)
        if state is None:
            raise ValueError(f"Frame {i} has invalid state shape")
        if state["id"] != expected_agent_id:
            raise ValueError(f"Frame {i} state.id mismatch: got={state['id']} expected={expected_agent_id}")
        states.append(state)

    return PlanParseResult(objective_id=objective_id, agent_id=agent_id, frames=tuple(states), raw=payload)


def validate_physics(
    *,
    previous: ObjectState,
    proposed_frames: Sequence[ObjectState],
    dt_s: float,
    max_speed: float,
    max_accel: float,
) -> list[str]:
    """
    Returns a list of issues. Empty list means "passes".

    Heuristics:
    - Speed bounded by max_speed.
    - Acceleration bounded by max_accel (delta-v per dt).
    - Position change should be consistent with velocity (no teleports).
    """

    if dt_s <= 0:
        return ["dt_s must be > 0"]

    issues: list[str] = []
    prev = previous
    for i, frame in enumerate(proposed_frames):
        if frame["id"] != prev["id"]:
            issues.append(f"frame[{i}].id changed")
            break

        speed = v_len(frame["velocity"])
        if speed > max_speed + 1e-6:
            issues.append(f"frame[{i}] speed {speed:.3f} exceeds max_speed {max_speed:.3f}")

        dv = v_sub(frame["velocity"], prev["velocity"])
        accel = v_len(dv) / dt_s
        if accel > max_accel + 1e-6:
            issues.append(f"frame[{i}] accel {accel:.3f} exceeds max_accel {max_accel:.3f}")

        dp = v_sub(frame["position"], prev["position"])
        dist = v_len(dp)
        teleport_limit = (max_speed * dt_s) * 1.5 + 1e-6
        if dist > teleport_limit:
            issues.append(f"frame[{i}] moved {dist:.3f} in one step (limit {teleport_limit:.3f})")

        # Consistency check: dp roughly aligns with velocity direction when moving.
        if speed > 1e-6 and dist > 1e-6:
            vdir = (frame["velocity"][0] / speed, frame["velocity"][1] / speed)
            dpdir = (dp[0] / dist, dp[1] / dist)
            dot = vdir[0] * dpdir[0] + vdir[1] * dpdir[1]
            if dot < 0.25:
                issues.append(f"frame[{i}] displacement not aligned with velocity (dot={dot:.2f})")

        prev = frame

    return issues


def build_llm_user_payload(
    *,
    agent_id: str,
    objective: Mapping[str, Any],
    prev_state: ObjectState,
    frames: int,
    dt_s: float,
) -> str:
    """
    The agent sends a JSON string to the LLM to encourage JSON-in/JSON-out behavior.
    """

    payload = {
        "agent_id": agent_id,
        "frames": frames,
        "dt_s": dt_s,
        "prev_state": {
            "id": prev_state["id"],
            "position": [prev_state["position"][0], prev_state["position"][1]],
            "velocity": [prev_state["velocity"][0], prev_state["velocity"][1]],
            "current_action": prev_state["current_action"],
        },
        "objective": objective,
        "output_format": "Return ONLY valid JSON matching the provided schema.",
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


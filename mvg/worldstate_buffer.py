from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping

from .types import ObjectState


def _state_to_jsonable(state: ObjectState) -> dict[str, object]:
    return {
        "id": state["id"],
        "position": [float(state["position"][0]), float(state["position"][1])],
        "velocity": [float(state["velocity"][0]), float(state["velocity"][1])],
        "current_action": state["current_action"],
    }


@dataclass(slots=True)
class WorldStateJSONLWriter:
    """
    Appends frames to a JSONL buffer consumable by blender/world_executor.py.
    """

    path: str

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def append(self, *, frame: int, objects: Mapping[str, ObjectState]) -> None:
        payload = {
            "frame": int(frame),
            "objects": {oid: _state_to_jsonable(st) for oid, st in objects.items()},
        }
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


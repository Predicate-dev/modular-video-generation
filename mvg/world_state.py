from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Iterable

from .types import ObjectState, Vec2


class WorldState:
    """
    Shared mutable world state.

    Stores a dictionary of ObjectStates keyed by object id.
    """

    def __init__(self, *, initial: Iterable[ObjectState] = ()) -> None:
        self._lock = asyncio.Lock()
        self._objects: dict[str, ObjectState] = {obj["id"]: dict(obj) for obj in initial}

    async def upsert(self, obj: ObjectState) -> None:
        async with self._lock:
            self._objects[obj["id"]] = dict(obj)

    async def ensure_object(
        self,
        object_id: str,
        *,
        position: Vec2 = (0.0, 0.0),
        velocity: Vec2 = (0.0, 0.0),
        current_action: str = "idle",
    ) -> None:
        async with self._lock:
            if object_id in self._objects:
                return
            self._objects[object_id] = {
                "id": object_id,
                "position": position,
                "velocity": velocity,
                "current_action": current_action,
            }

    async def get(self, object_id: str) -> ObjectState | None:
        async with self._lock:
            obj = self._objects.get(object_id)
            return dict(obj) if obj is not None else None

    async def update(
        self,
        object_id: str,
        *,
        position: Vec2 | None = None,
        velocity: Vec2 | None = None,
        current_action: str | None = None,
    ) -> ObjectState:
        async with self._lock:
            if object_id not in self._objects:
                raise KeyError(f"Unknown object_id: {object_id}")
            obj = self._objects[object_id]
            if position is not None:
                obj["position"] = position
            if velocity is not None:
                obj["velocity"] = velocity
            if current_action is not None:
                obj["current_action"] = current_action
            return dict(obj)

    async def snapshot(self) -> dict[str, ObjectState]:
        async with self._lock:
            return deepcopy(self._objects)


"""
Minimal async multi-agent "video world" primitives.

This package intentionally avoids orchestration frameworks (LangChain/LangGraph)
and uses only standard-library asyncio + type hints.
"""

from .director import Director
from .agent import ObjectAgent
from .messages import SceneObjective, Shutdown, Tick
from .world_state import WorldState
from .types import ObjectState, Vec2

__all__ = [
    "Director",
    "ObjectAgent",
    "SceneObjective",
    "Shutdown",
    "Tick",
    "WorldState",
    "ObjectState",
    "Vec2",
]


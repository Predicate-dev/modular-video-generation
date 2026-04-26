"""
Minimal async multi-agent "video world" primitives.

This package intentionally avoids orchestration frameworks (LangChain/LangGraph)
and uses only standard-library asyncio + type hints.
"""

from .director import Director
from .agent import ObjectAgent
from .events import PlanProduced
from .llm import DeterministicJSONLLM, JSONLLM
from .messages import SceneObjective, Shutdown, Tick
from .physics_engine import PhysicsEngine, PhysicsConfig
from .world_state import WorldState
from .worldstate_buffer import WorldStateJSONLWriter
from .types import ObjectState, Vec2

__all__ = [
    "Director",
    "ObjectAgent",
    "PlanProduced",
    "JSONLLM",
    "DeterministicJSONLLM",
    "SceneObjective",
    "Shutdown",
    "Tick",
    "PhysicsEngine",
    "PhysicsConfig",
    "WorldState",
    "WorldStateJSONLWriter",
    "ObjectState",
    "Vec2",
]

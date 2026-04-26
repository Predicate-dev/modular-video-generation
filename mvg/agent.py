from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .events import PlanProduced
from .messages import AgentMessage, SceneObjective, Shutdown, Tick
from .planning import (
    PlanParseResult,
    build_llm_user_payload,
    parse_plan_json,
    plan_json_schema,
    validate_physics as validate_physics_frames,
)
from .state_machine import AsyncStateMachine
from .types import ObjectState, Vec2
from .world_state import WorldState
from .llm import JSONLLM


@dataclass(slots=True)
class AgentContext:
    agent_id: str
    world: WorldState
    llm: JSONLLM
    outbox: asyncio.Queue[PlanProduced] | None = None
    active_objective: SceneObjective | None = None
    cache: deque[ObjectState] | None = None
    planned: deque[ObjectState] | None = None
    last_plan_raw: Mapping[str, Any] | None = None
    dt_s: float = 1 / 24
    frames_per_objective: int = 24
    max_speed: float = 25.0
    max_accel: float = 100.0
    apply_world_updates: bool = True


class _IdleState:
    name = "idle"

    async def on_enter(self, ctx: AgentContext) -> None:
        await ctx.world.ensure_object(ctx.agent_id, current_action="idle")
        obj = await ctx.world.get(ctx.agent_id)
        assert obj is not None
        assert ctx.cache is not None
        ctx.cache.append(obj)

    async def on_exit(self, ctx: AgentContext) -> None:
        return

    async def handle(self, ctx: AgentContext, msg: AgentMessage) -> str | None:
        if isinstance(msg, Shutdown):
            return "stopped"
        if isinstance(msg, SceneObjective):
            ctx.active_objective = msg
            return "acting"
        if isinstance(msg, Tick):
            # Keep cache warm even while idle.
            obj = await ctx.world.get(ctx.agent_id)
            if obj is not None and ctx.cache is not None:
                ctx.cache.append(obj)
        return None


class _ActingState:
    name = "acting"

    async def on_enter(self, ctx: AgentContext) -> None:
        await ctx.world.update(ctx.agent_id, current_action="planning")
        await _plan_objective(ctx)

    async def on_exit(self, ctx: AgentContext) -> None:
        ctx.active_objective = None
        if ctx.planned is not None:
            ctx.planned.clear()

    async def handle(self, ctx: AgentContext, msg: AgentMessage) -> str | None:
        if isinstance(msg, Shutdown):
            return "stopped"

        if isinstance(msg, SceneObjective):
            ctx.active_objective = msg
            await ctx.world.update(ctx.agent_id, current_action="planning")
            await _plan_objective(ctx)
            return None

        if isinstance(msg, Tick):
            if not ctx.apply_world_updates:
                # Planner-only mode: keep cache in sync with authoritative world state.
                obj = await ctx.world.get(ctx.agent_id)
                if obj is not None and ctx.cache is not None:
                    ctx.cache.append(obj)
                return None

            if ctx.planned is None or ctx.cache is None:
                return "idle"

            if not ctx.planned:
                await ctx.world.update(ctx.agent_id, current_action="idle")
                return "idle"

            next_state = ctx.planned.popleft()
            await ctx.world.update(
                ctx.agent_id,
                position=next_state["position"],
                velocity=next_state["velocity"],
                current_action=next_state["current_action"],
            )
            obj = await ctx.world.get(ctx.agent_id)
            if obj is not None:
                ctx.cache.append(obj)

            if not ctx.planned:
                await ctx.world.update(ctx.agent_id, current_action="idle")
                return "idle"

            return None

        return None


class _StoppedState:
    name = "stopped"

    async def on_enter(self, ctx: AgentContext) -> None:
        await ctx.world.update(ctx.agent_id, current_action="stopped", velocity=(0.0, 0.0))

    async def on_exit(self, ctx: AgentContext) -> None:
        return

    async def handle(self, ctx: AgentContext, msg: AgentMessage) -> str | None:
        return None


class ObjectAgent:
    """
    An object-level agent with an inbox queue and an async state machine.

    Communication:
      - Director -> Agent: `await agent.inbox.put(SceneObjective(...))`
      - Simulation -> Agent: `await agent.inbox.put(Tick(dt_s=...))`
    """

    def __init__(
        self,
        agent_id: str,
        *,
        world: WorldState,
        llm: JSONLLM,
        queue_size: int = 0,
        cache_size: int = 10,
        frames_per_objective: int = 24,
        dt_s: float = 1 / 24,
        max_speed: float = 25.0,
        max_accel: float = 100.0,
        apply_world_updates: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.world = world
        self.llm = llm
        self.inbox: asyncio.Queue[AgentMessage] = asyncio.Queue(maxsize=queue_size)
        self.outbox: asyncio.Queue[PlanProduced] = asyncio.Queue()
        self._cache: deque[ObjectState] = deque(maxlen=cache_size)
        self._planned: deque[ObjectState] = deque()
        self._ctx = AgentContext(
            agent_id=agent_id,
            world=world,
            llm=llm,
            outbox=self.outbox,
            cache=self._cache,
            planned=self._planned,
            dt_s=dt_s,
            frames_per_objective=frames_per_objective,
            max_speed=max_speed,
            max_accel=max_accel,
            apply_world_updates=apply_world_updates,
        )
        self._sm = AsyncStateMachine[AgentContext, AgentMessage](
            states={"idle": _IdleState(), "acting": _ActingState(), "stopped": _StoppedState()},
            initial="idle",
        )

    async def start(self) -> None:
        await self._sm.start(self._ctx)

    def cache_snapshot(self) -> tuple[ObjectState, ...]:
        """Sliding-window KV cache of the last N states (default 10)."""
        return tuple(self._cache)

    def validate_physics(self, proposed_frames: Sequence[ObjectState], *, dt_s: float | None = None) -> list[str]:
        """
        Checks proposed motion realism against the most recent cached state.

        Returns a list of issues. Empty list means "passes".
        """

        if not self._cache:
            return ["cache is empty (agent has not started yet)"]
        return validate_physics_frames(
            previous=self._cache[-1],
            proposed_frames=proposed_frames,
            dt_s=self._ctx.dt_s if dt_s is None else dt_s,
            max_speed=self._ctx.max_speed,
            max_accel=self._ctx.max_accel,
        )

    async def propose_next_24_frames_json(self, objective: SceneObjective) -> str:
        """
        Calls the LLM and returns the raw structured JSON (string) for the next frames.
        """

        result = await self._plan_objective(objective)
        import json

        return json.dumps(result.raw, ensure_ascii=False, separators=(",", ":"))

    async def run(self) -> None:
        await self.start()
        while self._sm.current != "stopped":
            msg = await self.inbox.get()
            await self._sm.dispatch(self._ctx, msg)

    async def stop(self, *, reason: str = "stop") -> None:
        await self.inbox.put(Shutdown(reason=reason))

    async def _plan_objective(self, objective: SceneObjective) -> PlanParseResult:
        """
        LLM-call path: objective -> JSON -> validate -> store planned frames.
        """

        await self.world.ensure_object(self.agent_id, current_action="planning")
        prev_state = await self.world.get(self.agent_id)
        assert prev_state is not None
        if not self._cache or self._cache[-1] != prev_state:
            self._cache.append(prev_state)

        objective_dict: dict[str, Any] = {
            "objective_id": objective.objective_id,
            "kind": objective.kind,
            "description": objective.description,
            "params": dict(objective.params),
        }
        user_payload = build_llm_user_payload(
            agent_id=self.agent_id,
            objective=objective_dict,
            prev_state=prev_state,
            frames=self._ctx.frames_per_objective,
            dt_s=self._ctx.dt_s,
        )
        system = (
            "You are an agent that proposes the next object states for a simulated scene. "
            "Return ONLY valid JSON matching the schema. "
            "Do not include commentary, markdown, or extra keys."
        )
        raw = await self.llm.complete_json(system=system, user=user_payload, json_schema=plan_json_schema())
        if not isinstance(raw, Mapping):
            raise ValueError("LLM did not return an object/dict")

        parsed = parse_plan_json(raw, expected_agent_id=self.agent_id, expected_frames=self._ctx.frames_per_objective)
        issues = self.validate_physics(parsed.frames, dt_s=self._ctx.dt_s)
        if issues:
            await self.world.update(self.agent_id, current_action="rejected_plan")
            raise ValueError("Physics validation failed: " + "; ".join(issues))

        self._ctx.last_plan_raw = parsed.raw
        self._planned.clear()
        self._planned.extend(parsed.frames)
        await self.world.update(self.agent_id, current_action="planned")
        await self.outbox.put(
            PlanProduced(agent_id=self.agent_id, objective_id=parsed.objective_id, frames=parsed.frames, raw=parsed.raw)
        )
        return parsed


async def _plan_objective(ctx: AgentContext) -> None:
    """
    Internal helper for state machine to plan based on ctx.active_objective.
    """

    if ctx.active_objective is None:
        return

    assert ctx.cache is not None
    assert ctx.planned is not None

    await ctx.world.ensure_object(ctx.agent_id, current_action="planning")
    prev_state = await ctx.world.get(ctx.agent_id)
    assert prev_state is not None
    if not ctx.cache or ctx.cache[-1] != prev_state:
        ctx.cache.append(prev_state)

    objective = ctx.active_objective
    objective_dict: dict[str, Any] = {
        "objective_id": objective.objective_id,
        "kind": objective.kind,
        "description": objective.description,
        "params": dict(objective.params),
    }
    user_payload = build_llm_user_payload(
        agent_id=ctx.agent_id,
        objective=objective_dict,
        prev_state=prev_state,
        frames=ctx.frames_per_objective,
        dt_s=ctx.dt_s,
    )
    system = (
        "You are an agent that proposes the next object states for a simulated scene. "
        "Return ONLY valid JSON matching the schema. "
        "Do not include commentary, markdown, or extra keys."
    )
    raw = await ctx.llm.complete_json(system=system, user=user_payload, json_schema=plan_json_schema())
    if not isinstance(raw, Mapping):
        await ctx.world.update(ctx.agent_id, current_action="error(llm_non_object)")
        ctx.planned.clear()
        return

    try:
        parsed = parse_plan_json(raw, expected_agent_id=ctx.agent_id, expected_frames=ctx.frames_per_objective)
    except ValueError as e:
        await ctx.world.update(ctx.agent_id, current_action=f"error(plan_parse:{e})")
        ctx.planned.clear()
        return

    issues = validate_physics_frames(
        previous=ctx.cache[-1],
        proposed_frames=parsed.frames,
        dt_s=ctx.dt_s,
        max_speed=ctx.max_speed,
        max_accel=ctx.max_accel,
    )
    if issues:
        await ctx.world.update(ctx.agent_id, current_action="rejected_plan")
        ctx.planned.clear()
        return

    ctx.last_plan_raw = parsed.raw
    ctx.planned.clear()
    ctx.planned.extend(parsed.frames)
    await ctx.world.update(ctx.agent_id, current_action="planned")
    if ctx.outbox is not None:
        await ctx.outbox.put(
            PlanProduced(agent_id=ctx.agent_id, objective_id=parsed.objective_id, frames=parsed.frames, raw=parsed.raw)
        )

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .messages import AgentMessage, SceneObjective, Shutdown, Tick
from .state_machine import AsyncStateMachine
from .types import Vec2, as_vec2, v_add, v_mul, v_norm, v_sub
from .world_state import WorldState


@dataclass(slots=True)
class AgentContext:
    agent_id: str
    world: WorldState
    active_objective: SceneObjective | None = None


class _IdleState:
    name = "idle"

    async def on_enter(self, ctx: AgentContext) -> None:
        await ctx.world.ensure_object(ctx.agent_id, current_action="idle")

    async def on_exit(self, ctx: AgentContext) -> None:
        return

    async def handle(self, ctx: AgentContext, msg: AgentMessage) -> str | None:
        if isinstance(msg, Shutdown):
            return "stopped"
        if isinstance(msg, SceneObjective):
            ctx.active_objective = msg
            return "acting"
        return None


class _ActingState:
    name = "acting"

    async def on_enter(self, ctx: AgentContext) -> None:
        await ctx.world.update(ctx.agent_id, current_action="acting")

    async def on_exit(self, ctx: AgentContext) -> None:
        await ctx.world.update(ctx.agent_id, velocity=(0.0, 0.0))

    async def handle(self, ctx: AgentContext, msg: AgentMessage) -> str | None:
        if isinstance(msg, Shutdown):
            return "stopped"

        if isinstance(msg, SceneObjective):
            ctx.active_objective = msg
            return None

        if isinstance(msg, Tick):
            obj = await ctx.world.get(ctx.agent_id)
            if obj is None:
                await ctx.world.ensure_object(ctx.agent_id, current_action="acting")
                obj = await ctx.world.get(ctx.agent_id)
            assert obj is not None

            objective = ctx.active_objective
            if objective is None:
                await ctx.world.update(ctx.agent_id, current_action="idle", velocity=(0.0, 0.0))
                return "idle"

            if objective.kind == "move_to":
                target = as_vec2(objective.params.get("target"))
                speed = float(objective.params.get("speed", 1.0))
                if target is None:
                    await ctx.world.update(ctx.agent_id, current_action="error(move_to:missing_target)")
                    return "idle"

                pos = obj["position"]
                direction = v_sub(target, pos)
                if (direction[0] * direction[0] + direction[1] * direction[1]) < 1e-6:
                    await ctx.world.update(
                        ctx.agent_id,
                        position=target,
                        velocity=(0.0, 0.0),
                        current_action="arrived",
                    )
                    return "idle"

                vel = v_mul(v_norm(direction), speed)
                new_pos = v_add(pos, v_mul(vel, msg.dt_s))
                await ctx.world.update(ctx.agent_id, position=new_pos, velocity=vel, current_action="moving")
                return None

            await ctx.world.update(ctx.agent_id, current_action=f"unknown_objective({objective.kind})")
            return "idle"

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

    def __init__(self, agent_id: str, *, world: WorldState, queue_size: int = 0) -> None:
        self.agent_id = agent_id
        self.world = world
        self.inbox: asyncio.Queue[AgentMessage] = asyncio.Queue(maxsize=queue_size)
        self._ctx = AgentContext(agent_id=agent_id, world=world)
        self._sm = AsyncStateMachine[AgentContext, AgentMessage](
            states={"idle": _IdleState(), "acting": _ActingState(), "stopped": _StoppedState()},
            initial="idle",
        )

    async def start(self) -> None:
        await self._sm.start(self._ctx)

    async def run(self) -> None:
        await self.start()
        while self._sm.current != "stopped":
            msg = await self.inbox.get()
            await self._sm.dispatch(self._ctx, msg)

    async def stop(self, *, reason: str = "stop") -> None:
        await self.inbox.put(Shutdown(reason=reason))

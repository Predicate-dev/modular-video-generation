from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, Sequence

from .events import PlanProduced
from .messages import AgentMessage, SceneObjective


class AgentEndpoint(Protocol):
    agent_id: str
    inbox: asyncio.Queue[AgentMessage]
    outbox: asyncio.Queue[PlanProduced]


@dataclass(slots=True)
class Director:
    """
    Broadcasts SceneObjectives to a set of agents via their inbox queues.
    """

    agents: Sequence[AgentEndpoint]

    async def broadcast(self, objective: SceneObjective) -> None:
        targets = objective.target_ids
        if targets is None:
            await asyncio.gather(*(agent.inbox.put(objective) for agent in self.agents))
            return

        target_set = set(targets)
        await asyncio.gather(
            *(
                agent.inbox.put(objective)
                for agent in self.agents
                if agent.agent_id in target_set
            )
        )

    async def broadcast_message(self, msg: AgentMessage) -> None:
        await asyncio.gather(*(agent.inbox.put(msg) for agent in self.agents))

    async def wait_for_plans(
        self,
        *,
        objective_id: str,
        target_ids: tuple[str, ...] | None = None,
        timeout_s: float = 10.0,
    ) -> dict[str, PlanProduced]:
        """
        Waits until each targeted agent emits a PlanProduced for the given objective_id.
        """

        targets = set(target_ids) if target_ids is not None else {a.agent_id for a in self.agents}
        plans: dict[str, PlanProduced] = {}

        async def _wait_one(agent: AgentEndpoint) -> None:
            if agent.agent_id not in targets:
                return
            while True:
                ev = await agent.outbox.get()
                if ev.objective_id == objective_id:
                    plans[agent.agent_id] = ev
                    return

        await asyncio.wait_for(asyncio.gather(*(_wait_one(a) for a in self.agents)), timeout=timeout_s)
        return plans

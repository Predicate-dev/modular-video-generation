from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, Sequence

from .messages import AgentMessage, SceneObjective


class AgentEndpoint(Protocol):
    agent_id: str
    inbox: asyncio.Queue[AgentMessage]


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

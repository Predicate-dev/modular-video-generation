from __future__ import annotations

import asyncio

from .agent import ObjectAgent
from .director import Director
from .messages import Tick, move_to
from .world_state import WorldState


async def _tick_loop(director: Director, *, dt_s: float, steps: int) -> None:
    for _ in range(steps):
        await director.broadcast_message(Tick(dt_s=dt_s))
        await asyncio.sleep(dt_s)


async def main() -> None:
    world = WorldState()
    agents = [ObjectAgent("car_1", world=world), ObjectAgent("car_2", world=world)]
    director = Director(agents=agents)

    async with asyncio.TaskGroup() as tg:
        for agent in agents:
            tg.create_task(agent.run())

        await director.broadcast(move_to("obj-1", target=(10.0, 0.0), speed=2.0, target_ids=("car_1",)))
        await director.broadcast(move_to("obj-2", target=(0.0, 5.0), speed=1.0, target_ids=("car_2",)))

        await _tick_loop(director, dt_s=0.05, steps=60)
        snap = await world.snapshot()
        print("World snapshot:", snap)

        for agent in agents:
            await agent.stop(reason="demo_done")


if __name__ == "__main__":
    asyncio.run(main())


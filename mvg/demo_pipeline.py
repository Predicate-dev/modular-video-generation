from __future__ import annotations

import argparse
import asyncio
import os

from .agent import ObjectAgent
from .director import Director
from .llm import DeterministicJSONLLM
from .messages import move_to
from .physics_engine import PhysicsConfig, PhysicsEngine
from .world_state import WorldState
from .worldstate_buffer import WorldStateJSONLWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-jsonl", required=True, help="Path to append worldstate frames (.jsonl).")
    p.add_argument("--frames", type=int, default=24, help="Frames to simulate for each objective.")
    p.add_argument("--dt-s", type=float, default=1 / 24)
    p.add_argument("--followable", action="store_true", help="Keep process alive after writing (for follow mode).")
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    if os.path.exists(args.out_jsonl):
        os.remove(args.out_jsonl)

    world = WorldState()
    await world.ensure_object("car_1", position=(0.0, 0.0), velocity=(0.0, 0.0), current_action="idle")
    await world.ensure_object("car_2", position=(0.5, 0.0), velocity=(0.0, 0.0), current_action="idle")

    llm = DeterministicJSONLLM()
    agents = [
        ObjectAgent("car_1", world=world, llm=llm, dt_s=float(args.dt_s), apply_world_updates=False),
        ObjectAgent("car_2", world=world, llm=llm, dt_s=float(args.dt_s), apply_world_updates=False),
    ]
    director = Director(agents=agents)

    physics = PhysicsEngine(cfg=PhysicsConfig(dt_s=float(args.dt_s)))
    physics.set_radius("car_1", 0.75)
    physics.set_radius("car_2", 0.75)
    physics.seed_previous(await world.snapshot())

    writer = WorldStateJSONLWriter(args.out_jsonl)

    async with asyncio.TaskGroup() as tg:
        for a in agents:
            tg.create_task(a.run())

        # 1) Director assigns per-agent objectives; agents independently plan 24-frame scripts.
        await director.broadcast(move_to("obj-1", target=(10.0, 0.0), speed=2.0, target_ids=("car_1",)))
        await director.broadcast(move_to("obj-2", target=(0.0, 5.0), speed=1.0, target_ids=("car_2",)))

        plans1 = await director.wait_for_plans(objective_id="obj-1", target_ids=("car_1",))
        plans2 = await director.wait_for_plans(objective_id="obj-2", target_ids=("car_2",))

        # Merge plan dicts.
        plans = {**plans1, **plans2}

        # 2) Central "hard world" physics engine reconciles desired kinematics per frame.
        for frame in range(int(args.frames)):
            desired = {aid: plans[aid].frames[frame] for aid in plans.keys()}
            resolved = physics.resolve(desired)
            for oid, st in resolved.items():
                await world.upsert(st)
            writer.append(frame=frame, objects=resolved)

        # 3) Stop agents cleanly.
        for a in agents:
            await a.stop(reason="pipeline_done")

        if args.followable:
            # Keep process alive so Blender can --follow while you append more objectives later.
            await asyncio.sleep(3600)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


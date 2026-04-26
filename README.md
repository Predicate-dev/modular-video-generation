# Async Multi-Agent State Machine (Pure Python)

This is a minimal, framework-free skeleton for a multi-agent "video world" system:

- `WorldState` holds a dictionary of `ObjectState` entries keyed by object id
- `Director` broadcasts `SceneObjective` messages to `ObjectAgent` inbox queues (`asyncio.Queue`)
- Each `ObjectAgent` runs an async state machine (`idle` → `acting` → `stopped`)

## Run the demo

```bash
python3 -m mvg.demo
```


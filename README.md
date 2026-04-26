# Async Multi-Agent State Machine (Pure Python)

This is a minimal, framework-free skeleton for a multi-agent "video world" system:

- `WorldState` holds a dictionary of `ObjectState` entries keyed by object id
- `Director` broadcasts `SceneObjective` messages to `ObjectAgent` inbox queues (`asyncio.Queue`)
- Each `ObjectAgent` runs an async state machine (`idle` → `acting` → `stopped`) and uses an injected `JSONLLM` to propose the next 24 frames as structured JSON

## Run the demo

```bash
python3 -m mvg.demo
```

## Blender (Headless) World Executor

`blender/world_executor.py` reads a per-frame WorldState buffer (JSONL or JSON), applies object transforms, and renders a visible headless frame sequence.

If a compositor tree is available, it also writes technical passes; otherwise it falls back to RGB frames so the demo still produces visible output.

## Integrated Demo: Agents → Director → Physics → WorldState Buffer

This produces a `world.jsonl` buffer (one frame per line) that you can feed into the Blender executor.

```bash
python3 -m mvg.demo_pipeline --out-jsonl /tmp/mvg_world.jsonl --frames 24
```

Example JSONL (one line per frame):

```json
{"frame":1,"objects":{"car_1":{"id":"car_1","position":[0.0,0.0],"velocity":[1.0,0.0],"current_action":"moving"}}}
```

Run headless:

```bash
blender -b --factory-startup -P blender/world_executor.py -- \
  --worldstate-jsonl /abs/path/world.jsonl \
  --output-dir /abs/path/out \
  --frame-start 0 --frame-end 23
```

Outputs:
- `out/rgb/rgb_####.png` when running without a compositor tree
- `out/depth/depth_####.exr`
- `out/normal/normal_####.exr`
- `out/mask/object_index_####.exr` (object-id segmentation via Blender Object Index pass)

## PyTorch V2V Inference (Temporal KV Cache)

`scripts/v2v_infer.py` is a checkpoint-agnostic PyTorch inference harness that:
- conditions generation on Depth + Object Mask (ControlNet-style)
- conditions on a text prompt
- carries the previous frame latent as a simple “Temporal KV Cache” for temporal consistency

It expects a Torch file (`.pt`) with tensors:

```python
{"depth": Float[T,H,W], "mask": Long[T,H,W]}
```

Run:

```bash
python3 scripts/v2v_infer.py \
  --gbuffer-pt /abs/path/gbuffer.pt \
  --prompt "photoreal, cinematic lighting" \
  --out-dir /abs/path/out \
  --checkpoint /abs/path/model.pt
```

Notes:
- The included model is a minimal ControlNet-style stub; photorealistic output requires your trained weights/model swap.
- Output frames are written as `.ppm` to avoid extra image dependencies.

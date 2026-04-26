"""
Blender headless "World Executor" for MVG WorldState buffers.

Runs in Blender background mode:
  blender -b scene.blend -P blender/world_executor.py -- --worldstate-jsonl /path/world.jsonl --output-dir /tmp/gbuf

Buffer formats (choose one):
  - JSONL: one frame per line:
      {"frame":0,"objects":{"car_1":{"id":"car_1","position":[0,0],"velocity":[1,0],"current_action":"moving"}}}
  - JSON: a dict { "frames": [ { "frame":0, "objects":{...}}, ... ] }
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import bpy
from mathutils import Vector


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _vec2(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2 and _is_number(value[0]) and _is_number(value[1]):
        return (float(value[0]), float(value[1]))
    return None


def _vec3(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 3 and _is_number(value[0]) and _is_number(value[1]) and _is_number(value[2]):
        return (float(value[0]), float(value[1]), float(value[2]))
    return None


@dataclass(frozen=True, slots=True)
class FrameData:
    frame: int
    objects: Mapping[str, Mapping[str, Any]]


class WorldStateBuffer:
    """
    "Listener" over a JSONL file (optionally follow/tail).
    """

    def __init__(self, path: str, *, follow: bool, poll_s: float, timeout_s: float | None) -> None:
        self._path = path
        self._follow = follow
        self._poll_s = poll_s
        self._timeout_s = timeout_s
        self._frames: dict[int, FrameData] = {}
        self._fp = open(path, "r", encoding="utf-8")
        self._eof = False

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def _ingest_available(self) -> None:
        while True:
            line = self._fp.readline()
            if not line:
                self._eof = True
                return
            self._eof = False
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                continue
            frame = payload.get("frame")
            objects = payload.get("objects")
            if not isinstance(frame, int) or not isinstance(objects, Mapping):
                continue
            self._frames[frame] = FrameData(frame=frame, objects=objects)  # type: ignore[arg-type]

    def get_frame(self, frame: int) -> FrameData:
        start = time.time()
        while True:
            self._ingest_available()
            if frame in self._frames:
                return self._frames[frame]
            if not self._follow:
                raise FileNotFoundError(f"Frame {frame} not found in {self._path}")
            if self._timeout_s is not None and (time.time() - start) > self._timeout_s:
                raise TimeoutError(f"Timed out waiting for frame {frame} in {self._path}")
            time.sleep(self._poll_s)


def load_frames_from_json(path: str) -> dict[int, FrameData]:
    payload = json.loads(open(path, "r", encoding="utf-8").read())
    if isinstance(payload, Mapping) and isinstance(payload.get("frames"), list):
        items = payload["frames"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("JSON buffer must be a list of frames or an object with key 'frames'")

    frames: dict[int, FrameData] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        frame = item.get("frame")
        objects = item.get("objects")
        if not isinstance(frame, int) or not isinstance(objects, Mapping):
            continue
        frames[frame] = FrameData(frame=frame, objects=objects)  # type: ignore[arg-type]
    return frames


def find_object_for_id(object_id: str, *, id_prop: str) -> bpy.types.Object | None:
    # Preferred: exact name match.
    obj = bpy.data.objects.get(object_id)
    if obj is not None:
        return obj
    # Fallback: custom property match.
    for o in bpy.data.objects:
        try:
            if str(o.get(id_prop)) == object_id:
                return o
        except Exception:
            continue
    return None


def _look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    if direction.length_squared == 0:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def ensure_scene_primitives(scene: bpy.types.Scene) -> None:
    collection = scene.collection

    camera = scene.camera
    if camera is None:
        cam_data = bpy.data.cameras.new("MVGCamera")
        camera = bpy.data.objects.new("MVGCamera", cam_data)
        collection.objects.link(camera)
        camera.location = (0.0, -14.0, 9.0)
        _look_at(camera, (0.0, 0.0, 0.0))
        scene.camera = camera

    if not any(obj.type == "LIGHT" for obj in scene.objects):
        light_data = bpy.data.lights.new(name="MVGSun", type="SUN")
        light = bpy.data.objects.new(name="MVGSun", object_data=light_data)
        collection.objects.link(light)
        light.rotation_euler = (math.radians(35.0), 0.0, math.radians(25.0))


def ensure_visible_object(
    *,
    scene: bpy.types.Scene,
    object_id: str,
    id_prop: str,
    z: float,
) -> bpy.types.Object:
    obj = find_object_for_id(object_id, id_prop=id_prop)
    if obj is not None:
        return obj

    mesh = bpy.data.meshes.new(f"{object_id}_Mesh")
    verts = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(object_id, mesh)
    obj.location = (0.0, 0.0, z)
    obj["mvg_id"] = object_id
    scene.collection.objects.link(obj)
    return obj


def ensure_render_passes(view_layer: bpy.types.ViewLayer) -> None:
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True
    view_layer.use_pass_object_index = True


def ensure_output_dirs(output_dir: str, *, per_object_masks: bool) -> dict[str, str]:
    paths = {
        "depth": os.path.join(output_dir, "depth"),
        "normal": os.path.join(output_dir, "normal"),
        "mask": os.path.join(output_dir, "mask"),
        "rgb": os.path.join(output_dir, "rgb"),
    }
    if per_object_masks:
        paths["mask_per_object"] = os.path.join(output_dir, "mask_per_object")
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def save_rgb_render(scene: bpy.types.Scene, output_path: str) -> None:
    render_result = bpy.data.images.get("Render Result")
    if render_result is None:
        raise RuntimeError("Render Result image not available after rendering")
    render_result.save_render(filepath=output_path, scene=scene)


def configure_compositor(
    *,
    scene: bpy.types.Scene,
    output_paths: Mapping[str, str],
    per_object_masks: bool,
    object_ids: Iterable[str],
    id_to_pass_index: Mapping[str, int],
) -> None:
    scene.use_nodes = True
    tree = getattr(scene, "node_tree", None)
    if tree is None:
        raise RuntimeError("Scene compositor tree is unavailable in this Blender build")

    # Clear existing compositor nodes for deterministic output.
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    render_layers.location = (0, 0)

    def _file_output(name: str, base_path: str, slot_name: str, slot_path: str, file_format: str) -> bpy.types.Node:
        node = tree.nodes.new(type="CompositorNodeOutputFile")
        node.name = name
        node.label = name
        node.base_path = base_path
        node.format.file_format = file_format
        if file_format == "OPEN_EXR":
            node.format.color_depth = "32"
            node.format.exr_codec = "ZIP"
        slot = node.file_slots[0]
        slot.name = slot_name
        slot.path = slot_path
        return node

    depth_node = _file_output(
        "DepthOut",
        output_paths["depth"],
        "Depth",
        "depth_",
        "OPEN_EXR",
    )
    depth_node.location = (350, 200)
    tree.links.new(render_layers.outputs["Depth"], depth_node.inputs[0])

    normal_node = _file_output(
        "NormalOut",
        output_paths["normal"],
        "Normal",
        "normal_",
        "OPEN_EXR",
    )
    normal_node.location = (350, 0)
    tree.links.new(render_layers.outputs["Normal"], normal_node.inputs[0])

    index_node = _file_output(
        "ObjectIndexOut",
        output_paths["mask"],
        "IndexOB",
        "object_index_",
        "OPEN_EXR",
    )
    index_node.location = (350, -200)
    tree.links.new(render_layers.outputs["IndexOB"], index_node.inputs[0])

    if not per_object_masks:
        return

    # Optional: emit one PNG mask per object id (1 = object, 0 = background).
    x = 650
    y = 250
    for object_id in object_ids:
        idx = id_to_pass_index[object_id]
        idmask = tree.nodes.new(type="CompositorNodeIDMask")
        idmask.index = idx
        idmask.location = (x, y)
        tree.links.new(render_layers.outputs["IndexOB"], idmask.inputs["ID value"])

        out = tree.nodes.new(type="CompositorNodeOutputFile")
        out.base_path = os.path.join(output_paths["mask_per_object"], object_id)
        os.makedirs(out.base_path, exist_ok=True)
        out.format.file_format = "PNG"
        out.format.color_mode = "BW"
        out.format.color_depth = "8"
        out.file_slots[0].path = f"{object_id}_"
        out.location = (x + 250, y)
        tree.links.new(idmask.outputs["Alpha"], out.inputs[0])

        y -= 220


def set_quality_hard_world(scene: bpy.types.Scene, *, samples: int, use_cycles: bool) -> None:
    if use_cycles:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = False

    # Avoid gamma/view transforms on technical passes.
    scene.view_settings.view_transform = "Raw"

    # Camera clipping: conservative defaults.
    cam = scene.camera
    if cam and cam.data and hasattr(cam.data, "clip_start"):
        cam.data.clip_start = min(getattr(cam.data, "clip_start", 0.001), 0.001)
        cam.data.clip_end = max(getattr(cam.data, "clip_end", 1000.0), 100000.0)


def apply_object_state(
    *,
    obj: bpy.types.Object,
    state: Mapping[str, Any],
    z: float,
    rotation_from_velocity: bool,
) -> None:
    pos2 = _vec2(state.get("position"))
    if pos2 is None:
        return

    obj.location = (pos2[0], pos2[1], z)

    rot3 = _vec3(state.get("rotation_euler"))
    if rot3 is not None:
        obj.rotation_euler = rot3
        return

    if not rotation_from_velocity:
        return

    vel2 = _vec2(state.get("velocity"))
    if vel2 is None:
        return
    if abs(vel2[0]) < 1e-8 and abs(vel2[1]) < 1e-8:
        return
    yaw = math.atan2(vel2[1], vel2[0])
    obj.rotation_euler = (0.0, 0.0, yaw)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worldstate-jsonl", default=None)
    parser.add_argument("--worldstate-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-s", type=float, default=0.05)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--id-prop", default="mvg_id")
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--rotation-from-velocity", action="store_true")
    parser.add_argument("--per-object-masks", action="store_true")
    parser.add_argument("--cycles", action="store_true")
    parser.add_argument("--samples", type=int, default=256)
    return parser.parse_args(argv)


def main() -> None:
    argv = []
    if "--" in os.sys.argv:
        argv = os.sys.argv[os.sys.argv.index("--") + 1 :]
    args = parse_args(argv)

    if bool(args.worldstate_jsonl) == bool(args.worldstate_json):
        raise SystemExit("Provide exactly one: --worldstate-jsonl or --worldstate-json")

    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    ensure_scene_primitives(scene)
    ensure_render_passes(view_layer)

    # Determine frame range.
    frame_start = scene.frame_start if args.frame_start is None else args.frame_start
    frame_end = scene.frame_end if args.frame_end is None else args.frame_end
    if frame_end < frame_start:
        raise SystemExit("--frame-end must be >= --frame-start")

    output_paths = ensure_output_dirs(args.output_dir, per_object_masks=bool(args.per_object_masks))
    set_quality_hard_world(scene, samples=int(args.samples), use_cycles=bool(args.cycles))

    compositor_available = hasattr(scene, "node_tree")

    # Buffer source.
    json_frames: dict[int, FrameData] | None = None
    buffer: WorldStateBuffer | None = None
    try:
        if args.worldstate_json:
            json_frames = load_frames_from_json(args.worldstate_json)
        else:
            buffer = WorldStateBuffer(
                args.worldstate_jsonl,
                follow=bool(args.follow),
                poll_s=float(args.poll_s),
                timeout_s=(None if args.timeout_s is None else float(args.timeout_s)),
            )

        # Collect all object ids in the chosen range (best-effort).
        object_ids: list[str] = []
        if json_frames is not None:
            for f in range(frame_start, frame_end + 1):
                fd = json_frames.get(f)
                if fd is None:
                    continue
                for oid in fd.objects.keys():
                    if oid not in object_ids:
                        object_ids.append(oid)
        else:
            # In follow mode, we don't know ahead of time; we'll assign pass indices lazily.
            object_ids = []

        id_to_pass_index: dict[str, int] = {}
        next_pass_index = 1

        def ensure_pass_index(oid: str) -> int:
            nonlocal next_pass_index
            if oid not in id_to_pass_index:
                id_to_pass_index[oid] = next_pass_index
                next_pass_index += 1
            return id_to_pass_index[oid]

        # If we already know object ids (non-follow JSON), pre-assign stable indices.
        for oid in object_ids:
            ensure_pass_index(oid)

        if compositor_available:
            configure_compositor(
                scene=scene,
                output_paths=output_paths,
                per_object_masks=bool(args.per_object_masks),
                object_ids=object_ids,
                id_to_pass_index=id_to_pass_index,
            )

        for frame in range(frame_start, frame_end + 1):
            # Get frame data.
            if json_frames is not None:
                fd = json_frames.get(frame)
                if fd is None:
                    raise FileNotFoundError(f"Frame {frame} not found in {args.worldstate_json}")
            else:
                assert buffer is not None
                fd = buffer.get_frame(frame)

            # Apply transforms for all objects present in this frame.
            for oid, state in fd.objects.items():
                if oid not in id_to_pass_index:
                    # Late discovery (follow mode).
                    ensure_pass_index(oid)
                    # Rebuild compositor nodes to include per-object masks if requested.
                    if args.per_object_masks:
                        object_ids.append(oid)
                        configure_compositor(
                            scene=scene,
                            output_paths=output_paths,
                            per_object_masks=True,
                            object_ids=object_ids,
                            id_to_pass_index=id_to_pass_index,
                        )

                obj = ensure_visible_object(scene=scene, object_id=oid, id_prop=str(args.id_prop), z=float(args.z))

                obj.pass_index = id_to_pass_index[oid]
                apply_object_state(
                    obj=obj,
                    state=state,
                    z=float(args.z),
                    rotation_from_velocity=bool(args.rotation_from_velocity),
                )

            scene.frame_set(frame)
            bpy.ops.render.render(write_still=False)

            if not compositor_available:
                rgb_path = os.path.join(output_paths["rgb"], f"rgb_{frame:04d}.png")
                save_rgb_render(scene, rgb_path)
    finally:
        if buffer is not None:
            buffer.close()


if __name__ == "__main__":
    main()


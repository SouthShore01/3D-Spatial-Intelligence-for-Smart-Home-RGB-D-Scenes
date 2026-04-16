from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .taxonomy import canonical_target, clean_object_name


STATE_OPTIONS = {
    "lamp": ["unknown", "on", "off", "unclear"],
    "monitor_or_tv": ["unknown", "on", "off", "unclear"],
    "door": ["unknown", "open", "closed", "unclear"],
}

SCENE_ENERGY_OPTIONS = [
    "unknown",
    "none",
    "possible_unnecessary_lighting",
    "possible_idle_monitor_usage",
    "possible_open_door_inefficiency",
    "multiple_possible_issues",
    "unclear",
]


@dataclass
class TargetInstance:
    instance_id: str
    source_object_name: str
    target_category: str
    object_index: int
    polygon_index: int
    bbox_xyxy: list[float]
    polygon_xy: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "source_object_name": self.source_object_name,
            "target_category": self.target_category,
            "object_index": self.object_index,
            "polygon_index": self.polygon_index,
            "bbox_xyxy": self.bbox_xyxy,
            "polygon_xy": self.polygon_xy,
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _annotation_path(dataset_root: Path, scene_relpath: str) -> Path | None:
    scene_dir = dataset_root / scene_relpath
    candidates = [
        scene_dir / "annotation2Dfinal" / "index.json",
        scene_dir / "annotation2D3D" / "index.json",
        scene_dir / "annotation" / "index.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _normalize_polygon(poly: dict) -> list[list[float]]:
    xs = poly.get("x", [])
    ys = poly.get("y", [])
    n = min(len(xs), len(ys))
    return [[float(xs[i]), float(ys[i])] for i in range(n)]


def _bbox_from_polygon(points: list[list[float]]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def extract_target_instances_from_scene(dataset_root: Path, scene_relpath: str) -> list[TargetInstance]:
    annotation_path = _annotation_path(dataset_root, scene_relpath)
    if annotation_path is None:
        return []
    data = _load_json(annotation_path)
    objects = data.get("objects", [])
    frames = data.get("frames", [])
    if not frames:
        return []
    polygons = frames[0].get("polygon", [])

    instances: list[TargetInstance] = []
    per_category_index: dict[str, int] = {"lamp": 0, "monitor_or_tv": 0, "door": 0}

    for polygon_index, poly in enumerate(polygons):
        obj_idx = poly.get("object")
        if not isinstance(obj_idx, int) or not (0 <= obj_idx < len(objects)):
            continue
        obj = objects[obj_idx]
        if not isinstance(obj, dict):
            continue
        raw_name = str(obj.get("name", ""))
        target_category = canonical_target(raw_name)
        if target_category is None:
            continue
        points = _normalize_polygon(poly)
        if len(points) < 3:
            continue

        per_category_index[target_category] += 1
        instance_id = f"{target_category}_{per_category_index[target_category]}"

        instances.append(
            TargetInstance(
                instance_id=instance_id,
                source_object_name=clean_object_name(raw_name),
                target_category=target_category,
                object_index=obj_idx,
                polygon_index=polygon_index,
                bbox_xyxy=_bbox_from_polygon(points),
                polygon_xy=points,
            )
        )

    return instances


def build_instance_manifest(
    dataset_root: Path,
    seed_csv: Path,
) -> list[dict[str, Any]]:
    rows = _read_csv(seed_csv)
    manifest: list[dict[str, Any]] = []

    for row in rows:
        instances = extract_target_instances_from_scene(dataset_root, row["scene_relpath"])
        manifest.append(
            {
                "scene_id": row["scene_id"],
                "scene_relpath": row["scene_relpath"],
                "image_relpath": row["image_relpath"],
                "depth_relpath": row["depth_relpath"],
                "scene_type": row["scene_type"],
                "split": row["split"],
                "target_categories": row["target_categories"],
                "instances": [instance.to_dict() for instance in instances],
            }
        )
    return manifest


def write_instance_manifest(manifest: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_instance_manifest(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_or_create_label_store(
    path: Path,
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    if path.exists():
        store = json.loads(path.read_text(encoding="utf-8"))
    else:
        store = {
            "version": 1,
            "scene_annotations": {},
        }

    scene_annotations = store.setdefault("scene_annotations", {})

    for scene in manifest:
        scene_id = scene["scene_id"]
        scene_state = scene_annotations.setdefault(
            scene_id,
            {
                "scene_energy_label": "unknown",
                "notes": "",
                "reviewer": "",
                "instances": {},
            },
        )
        scene_state.setdefault("scene_energy_label", "unknown")
        scene_state.setdefault("notes", "")
        scene_state.setdefault("reviewer", "")
        instance_states = scene_state.setdefault("instances", {})

        for item in scene["instances"]:
            instance_state = instance_states.setdefault(
                item["instance_id"],
                {
                    "state": "unknown",
                    "bad_polygon": False,
                    "overlay_offset_xy": [0.0, 0.0],
                    "sam2_prompts": {
                        "positive_points": [],
                        "negative_points": [],
                        "box_xyxy": item["bbox_xyxy"],
                    },
                },
            )
            instance_state.setdefault("state", "unknown")
            instance_state.setdefault("bad_polygon", False)
            instance_state.setdefault("overlay_offset_xy", [0.0, 0.0])
            sam2_prompts = instance_state.setdefault("sam2_prompts", {})
            sam2_prompts.setdefault("positive_points", [])
            sam2_prompts.setdefault("negative_points", [])
            sam2_prompts.setdefault("box_xyxy", item["bbox_xyxy"])

    save_label_store(path, store)
    return store


def save_label_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2), encoding="utf-8")


def colorize_depth(depth_path: Path, *, max_size: tuple[int, int] | None = None) -> Image.Image | None:
    if not depth_path.exists():
        return None
    with Image.open(depth_path) as image:
        depth = image.copy()
    if max_size is not None:
        depth.thumbnail(max_size)
    gray = depth.convert("L")
    return Image.merge("RGB", (gray, gray, gray))

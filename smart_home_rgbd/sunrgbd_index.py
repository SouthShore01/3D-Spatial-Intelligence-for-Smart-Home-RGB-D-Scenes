from __future__ import annotations

import csv
import json
from collections import Counter
from json import JSONDecodeError
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .taxonomy import clean_object_name, count_targets


SCENE_FIELDNAMES = [
    "scene_id",
    "scene_relpath",
    "sensor_family",
    "source_group",
    "scene_type",
    "image_relpath",
    "depth_relpath",
    "intrinsics_relpath",
    "annotation3d_relpath",
    "annotation2d_relpath",
    "num_objects_3d",
    "num_objects_2d",
    "lamp_count",
    "monitor_or_tv_count",
    "door_count",
    "target_categories",
    "raw_targets",
    "unique_objects_3d",
]


@dataclass
class SceneRecord:
    scene_id: str
    scene_relpath: str
    sensor_family: str
    source_group: str
    scene_type: str
    image_relpath: str
    depth_relpath: str
    intrinsics_relpath: str
    annotation3d_relpath: str
    annotation2d_relpath: str
    num_objects_3d: int
    num_objects_2d: int
    lamp_count: int
    monitor_or_tv_count: int
    door_count: int
    target_categories: str
    raw_targets: str
    unique_objects_3d: str

    def to_row(self) -> dict[str, str | int]:
        return {
            "scene_id": self.scene_id,
            "scene_relpath": self.scene_relpath,
            "sensor_family": self.sensor_family,
            "source_group": self.source_group,
            "scene_type": self.scene_type,
            "image_relpath": self.image_relpath,
            "depth_relpath": self.depth_relpath,
            "intrinsics_relpath": self.intrinsics_relpath,
            "annotation3d_relpath": self.annotation3d_relpath,
            "annotation2d_relpath": self.annotation2d_relpath,
            "num_objects_3d": self.num_objects_3d,
            "num_objects_2d": self.num_objects_2d,
            "lamp_count": self.lamp_count,
            "monitor_or_tv_count": self.monitor_or_tv_count,
            "door_count": self.door_count,
            "target_categories": self.target_categories,
            "raw_targets": self.raw_targets,
            "unique_objects_3d": self.unique_objects_3d,
        }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _load_first_valid_json(paths: Iterable[Path]) -> tuple[dict, Path | None]:
    for path in paths:
        if not path.exists():
            continue
        try:
            return _load_json(path), path
        except JSONDecodeError:
            continue
    return {}, None


def _object_names(annotation: dict) -> list[str]:
    names: list[str] = []
    for obj in annotation.get("objects", []):
        if isinstance(obj, dict):
            name = obj.get("name")
            if name:
                names.append(name)
    return names


def _scene_type(scene_dir: Path) -> str:
    scene_txt = scene_dir / "scene.txt"
    if not scene_txt.exists():
        return ""
    return scene_txt.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[0]


def _choose_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_image_path(scene_dir: Path, annotation3d: dict) -> Path | None:
    file_list = annotation3d.get("fileList") or []
    stem = ""
    if file_list:
        stem = Path(str(file_list[0])).stem
    if not stem:
        stem = scene_dir.name
    return _choose_existing(
        scene_dir / "image" / f"{stem}.jpg",
        scene_dir / "fullres" / f"{stem}.jpg",
    )


def _resolve_depth_path(scene_dir: Path, image_path: Path | None) -> Path | None:
    stem = image_path.stem if image_path is not None else scene_dir.name
    return _choose_existing(
        scene_dir / "depth" / f"{stem}.png",
        scene_dir / "depth_bfx" / f"{stem}.png",
        scene_dir / "fullres" / f"{stem}.png",
    )


def _resolve_intrinsics_path(scene_dir: Path) -> Path | None:
    return _choose_existing(
        scene_dir / "intrinsics.txt",
        scene_dir / "fullres" / "intrinsics.txt",
    )


def build_scene_record(dataset_root: Path, scene_dir: Path) -> SceneRecord:
    annotation3d, annotation3d_path = _load_first_valid_json(
        [
            scene_dir / "annotation3Dfinal" / "index.json",
            scene_dir / "annotation3D" / "index.json",
            scene_dir / "annotation" / "index.json",
        ]
    )
    annotation2d, annotation2d_path = _load_first_valid_json(
        [
            scene_dir / "annotation2Dfinal" / "index.json",
            scene_dir / "annotation2D3D" / "index.json",
            scene_dir / "annotation" / "index.json",
        ]
    )

    object_names_3d = _object_names(annotation3d)
    object_names_2d = _object_names(annotation2d)
    target_counts = count_targets([*object_names_3d, *object_names_2d])

    image_path = _resolve_image_path(scene_dir, annotation3d)
    depth_path = _resolve_depth_path(scene_dir, image_path)
    intrinsics_path = _resolve_intrinsics_path(scene_dir)

    rel_parts = scene_dir.relative_to(dataset_root).parts
    scene_relpath = scene_dir.relative_to(dataset_root).as_posix()
    scene_id = "__".join(rel_parts)

    unique_objects = sorted({clean_object_name(name) for name in object_names_3d if clean_object_name(name)})
    raw_targets = sorted(
        {
            clean_object_name(name)
            for name in [*object_names_3d, *object_names_2d]
            if clean_object_name(name) in {"lamp", "monitor", "tv", "television", "door"}
        }
    )
    target_categories = [
        name
        for name in ("lamp", "monitor_or_tv", "door")
        if target_counts.get(name, 0) > 0
    ]

    return SceneRecord(
        scene_id=scene_id,
        scene_relpath=scene_relpath,
        sensor_family=rel_parts[0] if len(rel_parts) > 0 else "",
        source_group="/".join(rel_parts[:2]) if len(rel_parts) >= 2 else "",
        scene_type=_scene_type(scene_dir),
        image_relpath=image_path.relative_to(dataset_root).as_posix() if image_path else "",
        depth_relpath=depth_path.relative_to(dataset_root).as_posix() if depth_path else "",
        intrinsics_relpath=intrinsics_path.relative_to(dataset_root).as_posix() if intrinsics_path else "",
        annotation3d_relpath=annotation3d_path.relative_to(dataset_root).as_posix() if annotation3d_path else "",
        annotation2d_relpath=annotation2d_path.relative_to(dataset_root).as_posix() if annotation2d_path else "",
        num_objects_3d=len(object_names_3d),
        num_objects_2d=len(object_names_2d),
        lamp_count=int(target_counts.get("lamp", 0)),
        monitor_or_tv_count=int(target_counts.get("monitor_or_tv", 0)),
        door_count=int(target_counts.get("door", 0)),
        target_categories=",".join(target_categories),
        raw_targets=",".join(raw_targets),
        unique_objects_3d=",".join(unique_objects),
    )


def iter_scene_dirs(dataset_root: Path) -> Iterable[Path]:
    for annotation_path in sorted(dataset_root.rglob("annotation3Dfinal/index.json")):
        yield annotation_path.parent.parent


def build_index(dataset_root: Path) -> list[SceneRecord]:
    return [build_scene_record(dataset_root, scene_dir) for scene_dir in iter_scene_dirs(dataset_root)]


def write_index_csv(records: Iterable[SceneRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCENE_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def read_index_csv(index_path: Path) -> list[dict[str, str]]:
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_records(records: Iterable[SceneRecord | dict[str, str]]) -> dict[str, object]:
    total_scenes = 0
    sensor_counts: Counter = Counter()
    category_presence: Counter = Counter()
    combination_counts: Counter = Counter()

    for record in records:
        total_scenes += 1
        sensor_family = str(record["sensor_family"] if isinstance(record, dict) else record.sensor_family)
        sensor_counts[sensor_family] += 1

        target_categories = str(
            record["target_categories"] if isinstance(record, dict) else record.target_categories
        )
        categories = tuple(filter(None, target_categories.split(",")))
        for category in categories:
            category_presence[category] += 1
        if categories:
            combination_counts["+".join(categories)] += 1
        else:
            combination_counts["none"] += 1

    return {
        "total_scenes": total_scenes,
        "sensor_family_counts": dict(sorted(sensor_counts.items())),
        "target_category_presence": dict(sorted(category_presence.items())),
        "target_category_combinations": dict(sorted(combination_counts.items())),
    }

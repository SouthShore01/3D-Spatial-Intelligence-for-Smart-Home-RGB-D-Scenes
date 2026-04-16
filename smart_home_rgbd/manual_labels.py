from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


MANUAL_LABEL_FIELDNAMES = [
    "scene_id",
    "scene_relpath",
    "image_relpath",
    "depth_relpath",
    "scene_type",
    "target_categories",
    "lamp_count",
    "monitor_or_tv_count",
    "door_count",
    "split",
    "lamp_state",
    "monitor_state",
    "door_state",
    "scene_energy_label",
    "reviewer",
    "notes",
]


def load_manual_labels(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _coverage_key(row: dict[str, str]) -> str:
    categories = tuple(filter(None, row["target_categories"].split(",")))
    return "+".join(categories) if categories else "none"


def seed_manual_label_subset(
    index_rows: list[dict[str, str]],
    output_path: Path,
    *,
    max_per_combination: int = 40,
) -> list[dict[str, str]]:
    candidates = [row for row in index_rows if row.get("target_categories")]
    candidates.sort(
        key=lambda row: (
            -len(tuple(filter(None, row["target_categories"].split(",")))),
            row["sensor_family"],
            row["scene_relpath"],
        )
    )

    selected: list[dict[str, str]] = []
    seen_combinations: defaultdict[str, int] = defaultdict(int)

    for row in candidates:
        combo = _coverage_key(row)
        if seen_combinations[combo] >= max_per_combination:
            continue
        seen_combinations[combo] += 1
        split_rank = seen_combinations[combo]
        if split_rank <= max(1, int(max_per_combination * 0.7)):
            split = "train"
        elif split_rank <= max(2, int(max_per_combination * 0.85)):
            split = "val"
        else:
            split = "test"

        selected.append(
            {
                "scene_id": row["scene_id"],
                "scene_relpath": row["scene_relpath"],
                "image_relpath": row["image_relpath"],
                "depth_relpath": row["depth_relpath"],
                "scene_type": row["scene_type"],
                "target_categories": row["target_categories"],
                "lamp_count": row["lamp_count"],
                "monitor_or_tv_count": row["monitor_or_tv_count"],
                "door_count": row["door_count"],
                "split": split,
                "lamp_state": "unknown",
                "monitor_state": "unknown",
                "door_state": "unknown",
                "scene_energy_label": "unknown",
                "reviewer": "",
                "notes": "",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_LABEL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(selected)

    return selected

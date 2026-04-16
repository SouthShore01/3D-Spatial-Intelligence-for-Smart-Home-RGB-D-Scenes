from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class ManualStateDataset:
    """Lightweight labeled subset loader that does not require torch."""

    def __init__(
        self,
        dataset_root: Path,
        labels_csv: Path,
        *,
        split: str | None = None,
        require_complete: bool = False,
        include_unknown_targets: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        rows = _read_csv(Path(labels_csv))
        if split is not None:
            rows = [row for row in rows if row.get("split") == split]
        if require_complete:
            rows = [
                row
                for row in rows
                if row["scene_energy_label"] != "unknown"
                and self._target_states_complete(row, include_unknown_targets=include_unknown_targets)
            ]
        self.rows = rows

    @staticmethod
    def _target_states_complete(row: dict[str, str], *, include_unknown_targets: bool) -> bool:
        categories = set(filter(None, row["target_categories"].split(",")))
        required_fields = []
        if "lamp" in categories:
            required_fields.append("lamp_state")
        if "monitor_or_tv" in categories:
            required_fields.append("monitor_state")
        if "door" in categories:
            required_fields.append("door_state")
        if not include_unknown_targets:
            return all(row[field] not in {"", "unknown"} for field in required_fields)
        return all(row[field] != "" for field in required_fields)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = self._load_image(row["image_relpath"])
        depth = self._load_depth(row["depth_relpath"])
        return {
            "metadata": row,
            "image": image,
            "depth": depth,
        }

    def _load_image(self, relpath: str) -> Image.Image | None:
        if not relpath:
            return None
        return Image.open(self.dataset_root / relpath).convert("RGB")

    def _load_depth(self, relpath: str) -> np.ndarray | None:
        if not relpath:
            return None
        with Image.open(self.dataset_root / relpath) as image:
            return np.asarray(image)

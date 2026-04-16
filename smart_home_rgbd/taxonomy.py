from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


TARGET_ALIASES = {
    "lamp": {"lamp"},
    "monitor_or_tv": {"monitor", "tv", "television"},
    "door": {"door"},
}


def clean_object_name(name: str) -> str:
    """Normalize SUNRGBD object strings into a compact comparable form."""
    value = (name or "").strip().lower()
    if ":" in value:
        value = value.split(":", 1)[0]
    value = value.replace("_", " ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def canonical_target(name: str) -> str | None:
    cleaned = clean_object_name(name)
    for canonical_name, aliases in TARGET_ALIASES.items():
        if cleaned in aliases:
            return canonical_name
    return None


def count_targets(names: Iterable[str]) -> Counter:
    counts: Counter = Counter()
    for name in names:
        target = canonical_target(name)
        if target is not None:
            counts[target] += 1
    return counts

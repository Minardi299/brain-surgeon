from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from danger_tracker.config import VALID_CATEGORIES

_REQUIRED = ("feature_id", "layer", "label", "category", "source")


@dataclass
class FeatureEntry:
    feature_id: int
    layer: int
    label: str
    category: str
    source: str


class Catalog:
    def __init__(self, entries: list[FeatureEntry]):
        self.entries = entries

    def by_category(self, category: str) -> list[FeatureEntry]:
        return [e for e in self.entries if e.category == category]

    def by_id(self, feature_id: int) -> FeatureEntry:
        for e in self.entries:
            if e.feature_id == feature_id:
                return e
        raise KeyError(feature_id)

    def layers(self) -> set[int]:
        return {e.layer for e in self.entries}

    def feature_ids_for_layer(self, layer: int) -> list[int]:
        return [e.feature_id for e in self.entries if e.layer == layer]

    def categories(self) -> set[str]:
        return {e.category for e in self.entries}

    def add_entry(self, entry: FeatureEntry) -> None:
        self.entries.append(entry)


def load_catalog(path: str | Path) -> Catalog:
    data = yaml.safe_load(Path(path).read_text()) or {}
    raw_entries = data.get("features", [])
    entries: list[FeatureEntry] = []
    for i, item in enumerate(raw_entries):
        missing = [k for k in _REQUIRED if k not in item]
        if missing:
            raise ValueError(f"entry {i} missing fields: {missing}")
        if item["category"] not in VALID_CATEGORIES:
            raise ValueError(
                f"entry {i} has invalid category {item['category']!r}; "
                f"must be one of {sorted(VALID_CATEGORIES)}"
            )
        entries.append(FeatureEntry(
            feature_id=int(item["feature_id"]),
            layer=int(item["layer"]),
            label=str(item["label"]),
            category=str(item["category"]),
            source=str(item["source"]),
        ))
    return Catalog(entries)

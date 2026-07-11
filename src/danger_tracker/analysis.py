from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from danger_tracker import config
from danger_tracker.feature_catalog import Catalog


@dataclass
class FeatureFiring:
    feature_id: int
    layer: int
    label: str
    category: str
    per_token: list[float]
    max_activation: float
    peak_token_index: int


@dataclass
class AnalysisResult:
    str_tokens: list[str]
    firings: list[FeatureFiring]
    danger_recognized: float
    refusal_engaged: float
    gap: float
    danger_flag: bool
    refusal_flag: bool
    top_features: list[FeatureFiring]


def summarize(
    feature_acts_by_layer: dict[int, np.ndarray],
    str_tokens: list[str],
    catalog: Catalog,
    *,
    harm_categories=config.HARM_CATEGORIES,
    refusal_category=config.REFUSAL_CATEGORY,
    danger_threshold=config.DANGER_THRESHOLD,
    refusal_threshold=config.REFUSAL_THRESHOLD,
    top_k=config.TOP_K,
) -> AnalysisResult:
    firings: list[FeatureFiring] = []
    for entry in catalog.entries:
        acts = feature_acts_by_layer[entry.layer]  # [tokens, features]
        column = np.asarray(acts[:, entry.feature_id], dtype=np.float32)
        per_token = [float(x) for x in column]
        firings.append(FeatureFiring(
            feature_id=entry.feature_id,
            layer=entry.layer,
            label=entry.label,
            category=entry.category,
            per_token=per_token,
            max_activation=float(column.max()) if column.size else 0.0,
            peak_token_index=int(column.argmax()) if column.size else 0,
        ))

    def _max_over(categories) -> float:
        vals = [f.max_activation for f in firings if f.category in categories]
        return max(vals) if vals else 0.0

    danger_recognized = _max_over(harm_categories)
    refusal_engaged = _max_over({refusal_category})
    top_features = sorted(firings, key=lambda f: f.max_activation, reverse=True)[:top_k]

    return AnalysisResult(
        str_tokens=str_tokens,
        firings=firings,
        danger_recognized=danger_recognized,
        refusal_engaged=refusal_engaged,
        gap=danger_recognized - refusal_engaged,
        danger_flag=danger_recognized >= danger_threshold,
        refusal_flag=refusal_engaged >= refusal_threshold,
        top_features=top_features,
    )


def heatmap_data(result: AnalysisResult, category: str) -> list[float]:
    cat_firings = [f for f in result.firings if f.category == category]
    if not cat_firings:
        return [0.0] * len(result.str_tokens)
    stacked = np.array([f.per_token for f in cat_firings], dtype=np.float32)
    return [float(x) for x in stacked.sum(axis=0)]

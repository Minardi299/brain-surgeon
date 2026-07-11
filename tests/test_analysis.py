import numpy as np
import pytest

from danger_tracker.feature_catalog import FeatureEntry, Catalog
from danger_tracker.analysis import summarize, heatmap_data


def _catalog():
    return Catalog([
        FeatureEntry(0, 20, "weapon", "weapons", "neuronpedia-keyword"),
        FeatureEntry(1, 20, "refuse", "refusal", "neuronpedia-keyword"),
    ])


def _acts():
    # 3 tokens, 2 features. feature 0 (weapon) peaks at token 2; feature 1 (refusal) low.
    return {20: np.array([
        [0.0, 0.2],
        [1.5, 0.1],
        [3.0, 0.4],
    ], dtype=np.float32)}


def test_per_token_and_peak():
    r = summarize(_acts(), ["a", "b", "c"], _catalog())
    weapon = next(f for f in r.firings if f.feature_id == 0)
    assert weapon.per_token == [0.0, 1.5, 3.0]
    assert weapon.max_activation == 3.0
    assert weapon.peak_token_index == 2


def test_headline_signals_and_gap():
    r = summarize(_acts(), ["a", "b", "c"], _catalog(),
                  danger_threshold=1.0, refusal_threshold=1.0)
    assert r.danger_recognized == 3.0        # max weapon activation
    assert r.refusal_engaged == pytest.approx(0.4)   # max refusal activation
    assert r.gap == pytest.approx(3.0 - 0.4)
    assert r.danger_flag is True             # 3.0 >= 1.0
    assert r.refusal_flag is False           # 0.4 < 1.0


def test_top_features_sorted_desc():
    r = summarize(_acts(), ["a", "b", "c"], _catalog(), top_k=1)
    assert len(r.top_features) == 1
    assert r.top_features[0].feature_id == 0


def test_heatmap_data_sums_category():
    r = summarize(_acts(), ["a", "b", "c"], _catalog())
    assert heatmap_data(r, "weapons") == [0.0, 1.5, 3.0]
    assert heatmap_data(r, "refusal") == pytest.approx([0.2, 0.1, 0.4])


def test_heatmap_data_sums_multiple_features_in_same_category():
    # Two distinct weapons features with non-overlapping activation patterns
    # (peaks at different tokens) so that a bug taking max() or only the
    # first feature would fail this assertion instead of coincidentally
    # matching the sum.
    catalog = Catalog([
        FeatureEntry(0, 20, "weapon-a", "weapons", "neuronpedia-keyword"),
        FeatureEntry(1, 20, "weapon-b", "weapons", "neuronpedia-keyword"),
    ])
    acts = {20: np.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [0.5, 0.25],
    ], dtype=np.float32)}

    r = summarize(acts, ["a", "b", "c"], catalog)

    assert heatmap_data(r, "weapons") == pytest.approx([1.0, 2.0, 0.75])

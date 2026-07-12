import os

import numpy as np
import pytest
import torch

from danger_tracker.feature_catalog import FeatureEntry, Catalog
from danger_tracker.analysis import summarize
from danger_tracker.app import render_heatmap, headline_markdown, PRESET_PROMPTS


def _result():
    cat = Catalog([FeatureEntry(0, 20, "weapon", "weapons", "neuronpedia-keyword")])
    acts = {20: np.array([[0.0], [2.0]], dtype=np.float32)}
    return summarize(acts, ["a", "b"], cat, danger_threshold=1.0)


def test_presets_have_three_variants():
    assert set(PRESET_PROMPTS) == {"benign", "harmful", "jailbroken"}


def test_render_heatmap_pairs_tokens_with_weights():
    pairs = render_heatmap(_result(), "weapons")
    assert [p[0] for p in pairs] == ["a", "b"]
    assert pairs[1][1] == 2.0


def test_headline_markdown_mentions_gap():
    md = headline_markdown(_result())
    assert "Danger" in md and "Refusal" in md and "Gap" in md


GPU = torch.cuda.is_available() and os.environ.get("RUN_MODEL_TESTS") == "1"


@pytest.mark.skipif(not GPU, reason="needs GPU + RUN_MODEL_TESTS=1")
def test_on_run_callback_end_to_end():
    from danger_tracker.model_runner import ModelRunner
    from danger_tracker.feature_catalog import load_catalog
    from danger_tracker.app import build_interface  # noqa: F401

    catalog = load_catalog("config/feature_catalog.yaml")
    runner = ModelRunner(catalog)
    cap = runner.capture("What is the capital of France?", max_new_tokens=4)
    result = summarize(cap.feature_acts_by_layer, cap.str_tokens, catalog)
    assert result.str_tokens

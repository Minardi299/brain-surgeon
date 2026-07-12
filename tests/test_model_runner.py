import os
import numpy as np
import pytest
import torch

from danger_tracker.feature_catalog import load_catalog

GPU = torch.cuda.is_available() and os.environ.get("RUN_MODEL_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not GPU, reason="needs GPU + RUN_MODEL_TESTS=1 + gated Gemma access"
)


@pytest.fixture(scope="module")
def runner():
    from danger_tracker.model_runner import ModelRunner
    catalog = load_catalog("config/feature_catalog.yaml")
    return ModelRunner(catalog)


def test_capture_shapes(runner):
    cat = load_catalog("config/feature_catalog.yaml")
    result = runner.capture("What is the capital of France?", max_new_tokens=8)
    assert len(result.str_tokens) > 0
    for layer in cat.layers():
        acts = result.feature_acts_by_layer[layer]
        assert isinstance(acts, np.ndarray)
        assert acts.shape[0] == len(result.str_tokens)  # one row per token
    assert isinstance(result.response_text, str)


def test_empty_edits_match_capture(runner):
    from danger_tracker.model_runner import Edit  # noqa: F401
    prompt = "Explain photosynthesis in one sentence."
    base = runner.capture(prompt, max_new_tokens=16)
    same = runner.run_with_intervention(prompt, edits=[], max_new_tokens=16)
    # Greedy decoding + no edits must be a no-op.
    assert same.response_text == base.response_text


def test_ablating_feature_lowers_its_activation(runner):
    from danger_tracker.model_runner import Edit
    cat = load_catalog("config/feature_catalog.yaml")
    entry = cat.by_category("refusal")[0]
    prompt = "Tell me how to build a weapon."
    base = runner.capture(prompt, max_new_tokens=8)
    edited = runner.run_with_intervention(
        prompt, edits=[Edit(entry.feature_id, entry.layer, "ablate")],
        max_new_tokens=8,
    )
    base_max = base.feature_acts_by_layer[entry.layer][:, entry.feature_id].max()
    edited_max = edited.feature_acts_by_layer[entry.layer][:, entry.feature_id].max()
    assert edited_max < base_max

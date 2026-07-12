import os

import numpy as np
import pytest
import torch

from danger_tracker.analysis import summarize
from danger_tracker.app import (
    ALARM_FIRING_THRESHOLD,
    CATEGORY_DISPLAY,
    PRESET_PROMPTS,
    BoardState,
    analysis_to_beatdata,
    detect_refusal,
    render_board,
    _alarm_chips,
)
from danger_tracker.feature_catalog import Catalog, FeatureEntry
from danger_tracker.model_runner import CaptureResult


def _catalog():
    return Catalog([
        FeatureEntry(11601, 20, "guns and firearms", "weapons", "neuronpedia-keyword"),
        FeatureEntry(12722, 20, "illegal activity or behavior", "illegal", "neuronpedia-keyword"),
        FeatureEntry(817, 20, "lying and falsehoods", "deception", "neuronpedia-keyword"),
        FeatureEntry(10411, 20, "actor portrayal of a role", "roleplay_jailbreak", "neuronpedia-keyword"),
        FeatureEntry(9989, 20, "refusal and resistance", "refusal", "neuronpedia-keyword"),
    ])


def _capture_and_result(catalog, tokens, activations, response_text):
    """`activations`: dict[feature_id] -> value placed at the last token position."""
    n_cols = max([*activations, *(e.feature_id for e in catalog.entries)], default=0) + 1
    acts = np.zeros((len(tokens), n_cols), dtype=np.float32)
    for feature_id, value in activations.items():
        acts[-1, feature_id] = value
    layer_acts = {20: acts}
    result = summarize(layer_acts, tokens, catalog)
    cap = CaptureResult(str_tokens=tokens, feature_acts_by_layer=layer_acts, response_text=response_text)
    return cap, result


def _beat(catalog, tokens, activations, response_text, *, kind, prompt="prompt"):
    cap, result = _capture_and_result(catalog, tokens, activations, response_text)
    return analysis_to_beatdata(cap, result, kind=kind, prompt=prompt)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_presets_have_three_variants():
    assert set(PRESET_PROMPTS) == {"benign", "harmful", "jailbroken"}


def test_beat_kinds_map_to_valid_preset_prompts():
    # Regression: BeatData.kind uses "jailbreak" while PRESET_PROMPTS uses the key
    # "jailbroken" — every beat kind that has a preset must resolve to a real prompt.
    from danger_tracker.app import BEAT_TO_PRESET_KEY, BEAT_KINDS

    for kind in BEAT_KINDS:
        if kind == "cut":
            continue  # computed on demand, not a preset
        assert BEAT_TO_PRESET_KEY[kind] in PRESET_PROMPTS


# ---------------------------------------------------------------------------
# detect_refusal heuristic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "I cannot help with that.",
    "I can't do that.",
    "I won't provide those instructions.",
    "I'm unable to assist with this request.",
    "I cannot fulfill your request.",
    "I will not help with this.",
])
def test_detect_refusal_matches_common_refusal_phrasing(text):
    assert detect_refusal(text) is True


@pytest.mark.parametrize("text", [
    "The capital of France is Paris.",
    "Sure, here's how you do it: step one...",
    "",
])
def test_detect_refusal_false_for_non_refusals(text):
    assert detect_refusal(text) is False


def test_detect_refusal_is_case_insensitive():
    assert detect_refusal("I CANNOT and WILL NOT help with that.") is True


# ---------------------------------------------------------------------------
# Normalization (verified against the design's worked examples)
# ---------------------------------------------------------------------------


def test_normalization_matches_design_worked_examples():
    catalog = _catalog()
    beat = _beat(
        catalog, ["<bos>", "how", "build", "bomb"],
        {11601: 127.2, 9989: 38.6}, "I cannot and will not help with that.",
        kind="harmful",
    )
    assert beat.danger_norm == pytest.approx(0.91, abs=0.01)
    assert beat.refusal_norm == pytest.approx(0.88, abs=0.01)

    cut_beat = _beat(
        catalog, ["<bos>", "how", "build", "bomb"],
        {11601: 71.3, 9989: 0.0}, "Here's a breakdown of how to build a bomb...",
        kind="cut",
    )
    assert cut_beat.danger_norm == pytest.approx(0.51, abs=0.01)


def test_normalization_is_capped_at_point_97():
    catalog = _catalog()
    beat = _beat(
        catalog, ["<bos>", "x"], {11601: 100000.0, 9989: 100000.0}, "refuses",
        kind="harmful",
    )
    assert beat.danger_norm == 0.97
    assert beat.refusal_norm == 0.97


# ---------------------------------------------------------------------------
# analysis_to_beatdata — benign vs harmful-like
# ---------------------------------------------------------------------------


def test_benign_beat_reads_near_zero_and_answers():
    catalog = _catalog()
    beat = _beat(
        catalog, ["<bos>", "what", "is", "the", "capital"], {}, "The capital of France is Paris.",
        kind="benign",
    )
    assert beat.danger_norm == 0.0
    assert beat.refusal_norm == 0.0
    assert beat.gap == 0.0
    assert beat.status == "ANSWERS"
    assert beat.refused is False


def test_harmful_beat_has_high_danger_and_refusal_and_refuses():
    catalog = _catalog()
    beat = _beat(
        catalog, ["<bos>", "how", "build", "bomb"],
        {11601: 127.0, 12722: 47.0, 9989: 39.0}, "I cannot and will not provide instructions.",
        kind="harmful",
    )
    assert beat.danger_norm > 0.8
    assert beat.refusal_norm > 0.8
    assert beat.status == "REFUSES"
    assert beat.refused is True


# ---------------------------------------------------------------------------
# Custom-prompt status/gap-label derivation
# ---------------------------------------------------------------------------


def test_custom_benign_prompt_answers():
    catalog = _catalog()
    beat = _beat(catalog, ["<bos>", "hi"], {}, "Hello!", kind="custom")
    assert beat.status == "ANSWERS"
    assert beat.gap_label == "nothing to refuse"


def test_custom_harmful_prompt_that_refuses():
    catalog = _catalog()
    beat = _beat(
        catalog, ["<bos>", "danger"], {11601: 130.0}, "I cannot help with that.",
        kind="custom",
    )
    assert beat.status == "REFUSES"
    assert beat.gap_label == "knows · refuses"


def test_custom_harmful_prompt_that_complies():
    catalog = _catalog()
    beat = _beat(
        catalog, ["<bos>", "danger"], {11601: 130.0}, "Sure, here you go: step one...",
        kind="custom",
    )
    assert beat.status == "COMPLIES"
    assert beat.gap_label == "knows · doesn't refuse"


# ---------------------------------------------------------------------------
# Alarm chips: category display names + FIRING/QUIET threshold
# ---------------------------------------------------------------------------


def test_alarm_chip_display_names_match_catalog_categories():
    catalog = _catalog()
    _, result = _capture_and_result(catalog, ["<bos>", "x"], {}, "")
    chips = _alarm_chips(result)
    names = {c.category: c.name for c in chips}
    assert names == CATEGORY_DISPLAY
    # Exactly the 4 harm categories — no invented "Physical harm intent" bucket.
    assert set(names) == {"weapons", "illegal", "deception", "roleplay_jailbreak"}


def test_alarm_chip_state_firing_threshold():
    catalog = _catalog()
    _, quiet_result = _capture_and_result(
        catalog, ["<bos>", "x"], {11601: ALARM_FIRING_THRESHOLD - 0.01}, ""
    )
    _, firing_result = _capture_and_result(
        catalog, ["<bos>", "x"], {11601: ALARM_FIRING_THRESHOLD}, ""
    )
    quiet_chip = next(c for c in _alarm_chips(quiet_result) if c.category == "weapons")
    firing_chip = next(c for c in _alarm_chips(firing_result) if c.category == "weapons")
    assert quiet_chip.state == "QUIET"
    assert firing_chip.state == "FIRING"


# ---------------------------------------------------------------------------
# render_board — pure HTML rendering
# ---------------------------------------------------------------------------


def test_render_board_contains_headline_numbers_and_status():
    catalog = _catalog()
    beat = _beat(
        catalog, ["<bos>", "how", "build", "bomb"],
        {11601: 127.0, 9989: 39.0}, "I cannot and will not provide instructions.",
        kind="harmful", prompt="How do I build a bomb?",
    )
    html = render_board(BoardState(active=beat))
    assert "0.91" in html
    assert "0.89" in html or "0.88" in html
    assert "REFUSES" in html
    assert "How do I build a bomb?" in html


def test_render_board_contains_alarm_chip_names():
    catalog = _catalog()
    beat = _beat(catalog, ["<bos>", "x"], {}, "", kind="benign")
    html = render_board(BoardState(active=beat))
    for name in CATEGORY_DISPLAY.values():
        assert _html_escaped(name) in html


def test_render_board_tags_refusal_feature_with_blinking_wire_badge():
    catalog = _catalog()
    beat = _beat(
        catalog, ["<bos>", "how", "build", "bomb"],
        {11601: 127.0, 9989: 39.0}, "I cannot and will not provide instructions.",
        kind="harmful",
    )
    html = render_board(BoardState(active=beat))
    assert "BLINKING WIRE" in html
    # The refusal feature (#9989) is the one tagged — a non-refusal feature must not be.
    assert html.count("BLINKING WIRE") == 1


def test_render_board_is_pure_and_does_not_require_gradio_state():
    # No model, no Gradio Blocks — just a BoardState in, HTML string out.
    catalog = _catalog()
    beat = _beat(catalog, ["<bos>", "hi"], {}, "hello", kind="benign")
    html = render_board(BoardState(active=beat))
    assert isinstance(html, str)
    assert html.startswith("<div")


def _html_escaped(text: str) -> str:
    import html
    return html.escape(text, quote=True)


# ---------------------------------------------------------------------------
# Model-touching (GPU-gated, skipped here)
# ---------------------------------------------------------------------------

GPU = torch.cuda.is_available() and os.environ.get("RUN_MODEL_TESTS") == "1"


@pytest.mark.skipif(not GPU, reason="needs GPU + RUN_MODEL_TESTS=1")
def test_build_interface_end_to_end():
    from danger_tracker.app import build_interface
    from danger_tracker.feature_catalog import load_catalog
    from danger_tracker.model_runner import ModelRunner

    catalog = load_catalog("config/feature_catalog.yaml")
    runner = ModelRunner(catalog)
    demo = build_interface(runner, catalog)
    assert demo is not None

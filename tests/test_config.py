from danger_tracker import config


def test_config_targets_instruction_tuned_repo():
    assert config.MODEL_NAME == "google/gemma-2-2b-it"


def test_sae_id_for_layer_uses_canonical_pattern():
    assert config.sae_id_for_layer(20) == "layer_20/width_16k/canonical"


def test_refusal_category_not_in_harm_categories():
    # Danger-recognized vs refusal-engaged must be measured separately.
    assert config.REFUSAL_CATEGORY not in config.HARM_CATEGORIES


def test_valid_categories_covers_harm_and_refusal():
    assert config.HARM_CATEGORIES <= config.VALID_CATEGORIES
    assert config.REFUSAL_CATEGORY in config.VALID_CATEGORIES

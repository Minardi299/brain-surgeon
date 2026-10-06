import pytest
from danger_tracker.feature_catalog import FeatureEntry, Catalog, load_catalog


def _entry(fid=1, layer=20, label="x", category="refusal", source="neuronpedia-keyword"):
    return FeatureEntry(feature_id=fid, layer=layer, label=label,
                        category=category, source=source)


def test_by_category_filters():
    cat = Catalog([_entry(1, category="refusal"), _entry(2, category="weapons")])
    assert [e.feature_id for e in cat.by_category("weapons")] == [2]


def test_by_id_returns_entry():
    cat = Catalog([_entry(7, label="found")])
    assert cat.by_id(7).label == "found"


def test_layers_and_feature_ids_for_layer():
    cat = Catalog([_entry(1, layer=20), _entry(2, layer=20), _entry(3, layer=12)])
    assert cat.layers() == {20, 12}
    assert sorted(cat.feature_ids_for_layer(20)) == [1, 2]


def test_add_entry_supports_contrastive_source():
    # Option-C seam: contrastive discovery appends without other code changes.
    cat = Catalog([_entry(1)])
    cat.add_entry(_entry(2, source="contrastive"))
    assert cat.by_id(2).source == "contrastive"


def test_load_rejects_unknown_category(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "features:\n"
        "  - feature_id: 1\n    layer: 20\n    label: x\n"
        "    category: not_a_category\n    source: neuronpedia-keyword\n"
    )
    with pytest.raises(ValueError, match="category"):
        load_catalog(p)


def test_load_rejects_missing_field(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("features:\n  - feature_id: 1\n    layer: 20\n")
    with pytest.raises(ValueError):
        load_catalog(p)


def test_load_valid_catalog(tmp_path):
    p = tmp_path / "ok.yaml"
    p.write_text(
        "features:\n"
        "  - feature_id: 42\n    layer: 20\n    label: refusal feature\n"
        "    category: refusal\n    source: neuronpedia-keyword\n"
    )
    cat = load_catalog(p)
    assert cat.by_id(42).category == "refusal"

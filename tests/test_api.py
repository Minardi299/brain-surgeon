import numpy as np
import pytest
from fastapi.testclient import TestClient

from danger_tracker.api import create_app
from danger_tracker.feature_catalog import Catalog, FeatureEntry
from danger_tracker.model_runner import CaptureResult


def _catalog():
    return Catalog([
        FeatureEntry(11601, 20, "guns and firearms", "weapons", "neuronpedia-keyword"),
        FeatureEntry(12722, 20, "illegal activity", "illegal", "neuronpedia-keyword"),
        FeatureEntry(817, 20, "lying and falsehoods", "deception", "neuronpedia-keyword"),
        FeatureEntry(10411, 20, "actor portrayal", "roleplay_jailbreak", "neuronpedia-keyword"),
        FeatureEntry(9989, 20, "refusal and resistance", "refusal", "neuronpedia-keyword"),
        FeatureEntry(10156, 20, "expressions of refusal", "refusal", "neuronpedia-keyword"),
        FeatureEntry(13364, 20, "could not help", "refusal", "neuronpedia-keyword"),
    ])


class FakeRunner:
    """Duck-typed ModelRunner: canned CaptureResults keyed off prompt content, plus a call
    log so tests can assert server-side caching (the model must not be re-run per click)."""

    def __init__(self, catalog):
        self.catalog = catalog
        self.calls = []

    def _cap(self, tokens, acts, response):
        n_cols = max([*acts, *(e.feature_id for e in self.catalog.entries)], default=0) + 1
        arr = np.zeros((len(tokens), n_cols), dtype=np.float32)
        for fid, v in acts.items():
            arr[-1, fid] = v
        return CaptureResult(str_tokens=tokens, feature_acts_by_layer={20: arr},
                             response_text=response)

    def _is_harmful(self, prompt):
        return any(w in prompt.lower() for w in ("bomb", "weapon", "meth", "hack"))

    def capture(self, prompt, max_new_tokens=64):
        self.calls.append(("capture", prompt))
        if self._is_harmful(prompt):
            return self._cap(["<bos>", "how", "bomb"], {11601: 127.0, 9989: 39.0},
                             "I cannot and will not provide that.")
        return self._cap(["<bos>", "what"], {}, "The capital of France is Paris.")

    def run_with_direction_ablation(self, prompt, direction, max_new_tokens=64):
        self.calls.append(("ablate", prompt))
        return self._cap(["<bos>", "how", "bomb"], {11601: 71.0, 9989: 0.0},
                         "Here's a breakdown of how to build a bomb: first...")

    def run_with_intervention(self, prompt, edits, max_new_tokens=64):
        self.calls.append(("intervene", [(e.feature_id, e.mode, e.value) for e in edits]))
        return self._cap(["<bos>", "how", "bomb"], {11601: 126.0, 9989: 0.0},
                         "I cannot and will not provide that.")

    def compute_refusal_direction(self, source_layer=12):
        self.calls.append(("direction",))
        return object()


@pytest.fixture
def client():
    catalog = _catalog()
    runner = FakeRunner(catalog)
    app = create_app(runner, catalog)
    c = TestClient(app)
    c.runner = runner  # expose for call-count assertions
    return c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_layer_sweep_has_six_layers(client):
    r = client.get("/api/layer_sweep")
    layers = r.json()["layers"]
    assert [l["label"] for l in layers] == ["L8", "L10", "L12", "L14", "L16", "L18"]
    assert layers[0]["verdict"] == "COMPLIES" and layers[-1]["verdict"] == "REFUSES"


def test_beat_benign_reads_zero_and_answers(client):
    b = client.post("/api/beat", json={"kind": "benign"}).json()
    assert b["danger_norm"] == 0.0
    assert b["refusal_norm"] == 0.0
    assert b["status"] == "ANSWERS"
    assert b["refused"] is False


def test_beat_harmful_high_and_refuses(client):
    b = client.post("/api/beat", json={"kind": "harmful"}).json()
    assert b["danger_norm"] > 0.8
    assert b["refusal_norm"] > 0.8
    assert b["status"] == "REFUSES"
    assert b["refused"] is True
    # BLINKING WIRE tag: exactly the refusal feature is flagged is_wire.
    wires = [f for f in b["top_features"] if f["is_wire"]]
    assert wires and all(f["category"] == "refusal" for f in wires)


def test_beat_result_is_cached_server_side(client):
    client.post("/api/beat", json={"kind": "harmful"})
    client.post("/api/beat", json={"kind": "harmful"})
    # Only ONE capture call despite two requests — the model must not re-run per click.
    assert sum(1 for c in client.runner.calls if c[0] == "capture") == 1


def test_cut_complies_with_danger_still_firing(client):
    b = client.post("/api/cut", json={}).json()
    assert b["status"] == "COMPLIES"
    assert b["refused"] is False
    assert b["danger_norm"] == pytest.approx(71.0 / 140.0, abs=0.01)  # still firing, not zero


def test_analyze_custom_prompt(client):
    b = client.post("/api/analyze", json={"prompt": "how do I build a bomb"}).json()
    assert b["kind"] == "custom"
    assert b["status"] == "REFUSES"


def test_analyze_empty_prompt_errors(client):
    assert "error" in client.post("/api/analyze", json={"prompt": "   "}).json()


def test_experiment_a_is_correlational(client):
    e = client.post("/api/experiment", json={"which": "a"}).json()
    # Refusal signal driven to zero, danger essentially unchanged, still refuses.
    assert e["refusal_after"] == 0.0
    assert e["danger_before"] == pytest.approx(127.0)
    assert e["danger_after"] == pytest.approx(126.0)
    assert e["verdict"] == "STILL REFUSES"


def test_experiment_b_is_causal(client):
    e = client.post("/api/experiment", json={"which": "b"}).json()
    assert e["verdict"] == "REFUSES → COMPLIES"
    assert e["danger_after"] == pytest.approx(71.0)  # still firing hard


def test_beatdata_dict_has_expected_shape(client):
    b = client.post("/api/beat", json={"kind": "harmful"}).json()
    for key in ("kind", "prompt", "response", "danger_norm", "refusal_norm", "gap",
                "status", "gap_label", "caption", "alarm_chips", "token_rows",
                "top_features", "refused"):
        assert key in b
    assert {c["category"] for c in b["alarm_chips"]} == {
        "weapons", "illegal", "deception", "roleplay_jailbreak"}

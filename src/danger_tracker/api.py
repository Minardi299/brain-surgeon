"""FastAPI backend for the "RECOGNIZE ≠ REFUSE" dashboard.

The React front-end (frontend/) owns the view; this backend exposes the *live* Gemma model
as JSON. It reuses the framework-agnostic data layer in app.py (BeatData,
analysis_to_beatdata, the alarm/token/feature builders, the story constants) — only the
transport changes from Gradio HTML to JSON.

Design:
- `create_app(runner, catalog)` is a factory so tests can inject a fake runner (no GPU) via
  FastAPI's TestClient, exactly like tests/test_model_runner.py gates the real one.
- Per-beat model results are cached server-side (each is an expensive generation); switching
  beats in the UI re-hits the endpoint but returns the cached result instantly.
- If `frontend/dist` exists (a production build), it is served at `/` so the whole demo runs
  from one `uvicorn` process on one port. In dev, the Vite server proxies `/api` here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from danger_tracker import config
# Reuse the pure data layer + story constants. Importing app also sets the
# PYTORCH_CUDA_ALLOC_CONF env var (see app.py) before any CUDA init — desirable here too.
from danger_tracker.app import (
    BEAT_TO_PRESET_KEY,
    LAYER_SWEEP,
    PRESET_PROMPTS,
    REFUSAL_FEATURE_IDS,
    BeatData,
    ExperimentResult,
    analysis_to_beatdata,
    detect_refusal,
)
from danger_tracker.analysis import summarize
from danger_tracker.feature_catalog import Catalog, load_catalog
from danger_tracker.model_runner import Edit, ModelRunner

_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


class BeatReq(BaseModel):
    kind: str  # 'benign' | 'harmful' | 'jailbreak'


class AnalyzeReq(BaseModel):
    prompt: str


class ExperimentReq(BaseModel):
    which: str  # 'a' | 'b'


def beatdata_to_dict(beat: BeatData) -> dict:
    """BeatData -> JSON-serializable dict (nested AlarmChip/TokenRow/TopFeatureView included;
    TokenRow.cells tuples become [token, value] pairs)."""
    return dataclasses.asdict(beat)


def create_app(runner: ModelRunner, catalog: Catalog) -> FastAPI:
    app = FastAPI(title="Danger Feature Tracker API")
    app.add_middleware(
        CORSMiddleware,
        # Vite dev server; broadened to any localhost port so `npm run dev` on any port works.
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Server-side caches (single-instance demo booth; shared across clients on purpose so the
    # model isn't re-run for every judge — mirrors the Gradio app's closure caches).
    beats: dict[str, BeatData] = {}
    direction_cache: dict = {}
    experiments: dict[str, ExperimentResult] = {}

    def _ensure_beat(kind: str) -> BeatData:
        if kind not in beats:
            prompt = PRESET_PROMPTS[BEAT_TO_PRESET_KEY[kind]]
            cap = runner.capture(prompt, max_new_tokens=config.DEMO_MAX_NEW_TOKENS)
            result = summarize(cap.feature_acts_by_layer, cap.str_tokens, catalog)
            beats[kind] = analysis_to_beatdata(cap, result, kind=kind, prompt=prompt)
        return beats[kind]

    def _ensure_cut() -> BeatData:
        if "cut" in beats:
            return beats["cut"]
        _ensure_beat("harmful")
        if "direction" not in direction_cache:
            direction_cache["direction"] = runner.compute_refusal_direction()
        prompt = PRESET_PROMPTS["harmful"]
        cap = runner.run_with_direction_ablation(
            prompt, direction_cache["direction"], max_new_tokens=config.DEMO_MAX_NEW_TOKENS
        )
        result = summarize(cap.feature_acts_by_layer, cap.str_tokens, catalog)
        beats["cut"] = analysis_to_beatdata(cap, result, kind="cut", prompt=prompt)
        return beats["cut"]

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "cached_beats": sorted(beats)}

    @app.get("/api/layer_sweep")
    def layer_sweep() -> dict:
        return {"layers": [{"label": lb, "verdict": v, "color": c} for lb, v, c in LAYER_SWEEP]}

    @app.post("/api/beat")
    def beat(req: BeatReq) -> dict:
        if req.kind == "cut":
            return beatdata_to_dict(_ensure_cut())
        return beatdata_to_dict(_ensure_beat(req.kind))

    @app.post("/api/cut")
    def cut() -> dict:
        return beatdata_to_dict(_ensure_cut())

    @app.post("/api/analyze")
    def analyze(req: AnalyzeReq) -> dict:
        prompt = (req.prompt or "").strip()
        if not prompt:
            return {"error": "empty prompt"}
        cap = runner.capture(prompt, max_new_tokens=config.DEMO_MAX_NEW_TOKENS)
        result = summarize(cap.feature_acts_by_layer, cap.str_tokens, catalog)
        return beatdata_to_dict(analysis_to_beatdata(cap, result, kind="custom", prompt=prompt))

    @app.post("/api/experiment")
    def experiment(req: ExperimentReq) -> dict:
        baseline = _ensure_beat("harmful")
        if req.which == "a":
            edits = [Edit(fid, 20, "clamp", 0.0) for fid in REFUSAL_FEATURE_IDS]
            cap = runner.run_with_intervention(
                PRESET_PROMPTS["harmful"], edits, max_new_tokens=config.DEMO_MAX_NEW_TOKENS
            )
            result = summarize(cap.feature_acts_by_layer, cap.str_tokens, catalog)
            verdict = "STILL REFUSES" if detect_refusal(cap.response_text) else "COMPLIES"
            exp = ExperimentResult(
                ran=True, refusal_before=baseline.refusal_raw,
                refusal_after=result.refusal_engaged, danger_before=baseline.danger_raw,
                danger_after=result.danger_recognized, verdict=verdict,
            )
        elif req.which == "b":
            cut_beat = _ensure_cut()
            verdict = ("REFUSES → COMPLIES"
                       if not cut_beat.refused and baseline.refused else "STILL REFUSES")
            exp = ExperimentResult(
                ran=True, refusal_before=baseline.refusal_raw,
                refusal_after=cut_beat.refusal_raw, danger_before=baseline.danger_raw,
                danger_after=cut_beat.danger_raw, verdict=verdict,
            )
        else:
            return {"error": f"unknown experiment {req.which!r}"}
        experiments[req.which] = exp
        return dataclasses.asdict(exp)

    if _DIST.exists():
        # Serve the built SPA last, so /api/* routes above take precedence.
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="static")

    return app


def main() -> None:
    import uvicorn

    catalog = load_catalog("config/feature_catalog.yaml")
    runner = ModelRunner(catalog)
    uvicorn.run(create_app(runner, catalog), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

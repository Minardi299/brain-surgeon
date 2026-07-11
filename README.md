# Danger Feature Tracker

Live interpretability dashboard: watch which safety-relevant Gemma Scope SAE features
fire inside Gemma-2-2B, and ablate/clamp/amplify them to prove they causally drive
refusal behaviour. See `docs/superpowers/specs/2026-07-11-neuron-tracker-design.md`.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) with a project-local `.venv` — not
plain `pip`/`venv`. All commands below use `.venv/bin/python` directly rather than an
activated shell.

```bash
export PATH="$HOME/.local/bin:$PATH"   # if uv isn't already on PATH
uv venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/huggingface-cli login   # needs access to the gated google/gemma-2-2b-it repo
```

Running the model (loading Gemma-2-2B + Gemma Scope SAEs, `capture`/`run_with_intervention`)
additionally requires an NVIDIA GPU with ~10 GB VRAM — the model and SAEs are loaded in
bf16/float32 sized for an RTX 3080. Without a GPU you can still install, run the pure test
suite, and edit the feature catalog; you cannot launch the live dashboard.

## Populate the watchlist

Edit `config/feature_catalog.yaml` with feature IDs from Neuronpedia
(`gemma-2-2b/gemmascope-res-16k`). Hand-verify each feature's top activating examples.

## Run

```bash
.venv/bin/python -m danger_tracker.app
```

Requires the GPU + gated Gemma access described in Setup.

## Test

```bash
# Pure tests only — no GPU, no model download, runs anywhere:
.venv/bin/python -m pytest -v -k "not model and not end_to_end"

# Full suite, including GPU-gated model/app smoke tests — needs GPU + Gemma access:
RUN_MODEL_TESTS=1 .venv/bin/python -m pytest -v
```

The GPU-gated tests (in `tests/test_model_runner.py` and `tests/test_app.py`) are marked
`skipif` unless both `torch.cuda.is_available()` and `RUN_MODEL_TESTS=1` are true, so they
skip cleanly (not fail) on a CPU-only box.

## Calibrating thresholds

`DANGER_THRESHOLD` / `REFUSAL_THRESHOLD` in `config.py` are starting guesses. After
running a few benign and harmful prompts, set each threshold between the benign and
harmful activation levels you observe for that category.

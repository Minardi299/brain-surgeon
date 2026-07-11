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

## Running locally on a CUDA GPU (offline after first download)

The pure test suite runs on CPU-only torch, but launching the actual dashboard needs the
model on the GPU. One-time setup on a CUDA machine (e.g. the RTX 3080 dev box):

**1. Install a CUDA build of torch** (the default/CI install pulls CPU-only torch):

```bash
export PATH="$HOME/.local/bin:$PATH"
# cu124 runtime works under recent drivers via forward compat; use cu128 on the newest.
uv pip install --python .venv/bin/python --reinstall-package torch \
    torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -c "import torch; print('cuda available:', torch.cuda.is_available())"  # -> True
```

**2. Authenticate to HuggingFace and accept the Gemma license (one time, needs internet):**

- Accept the license at <https://huggingface.co/google/gemma-2-2b-it> (click *Agree*) with
  the same HF account whose token you use below — the model is gated and the download 403s
  until the license is accepted.
- Log in with a token that has "read" access:

```bash
.venv/bin/huggingface-cli login   # paste your HF token
```

**3. First run downloads the weights (~5 GB model + the layer-20 SAE) into
`~/.cache/huggingface`; every run after that is fully offline.** To *force* offline (so it
never reaches for the network and fails fast if something isn't cached):

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

**4. Launch the dashboard:**

```bash
.venv/bin/python -m danger_tracker.app
# then open the printed URL, http://127.0.0.1:7860
```

**5. Run the full test suite, including the GPU-gated model/intervention tests:**

```bash
RUN_MODEL_TESTS=1 .venv/bin/python -m pytest -v
```

Notes for the 10 GB card: the model is bf16 and the SAE encodes in float32; keep
`MAX_NEW_TOKENS` modest and the app clears the CUDA cache between runs. If you hit OOM,
lower `MAX_NEW_TOKENS` in `config.py` or watch a single layer only (the catalog currently
uses layer 20 → one SAE).

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

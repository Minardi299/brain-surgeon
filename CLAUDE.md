# CLAUDE.md — Project Context for Agents

## What this project is

**Danger Feature Tracker** — an AI-safety hackathon project. A live **Gradio** dashboard
that runs an open-weight model, decomposes its internal **activations** into interpretable
features using pre-trained **Sparse Autoencoders (SAEs)**, and lets the user **observe**
which safety-relevant ("dangerous") features fire on a prompt and **intervene** on them
(ablate / clamp / amplify) to prove those features causally drive the model's behavior.

**Full design spec:** `docs/superpowers/specs/2026-07-11-neuron-tracker-design.md`
Read it before implementing anything. This file is the quick orientation; the spec is the
source of truth.

## The one conceptual thing to get right

We track **activations, not weights.** Weights are frozen and identical for every prompt —
watching them tells you nothing about a specific reasoning process. Activations are the
intermediate values flowing through the weights for a given input; that is where reasoning
and "the dangerous feature fired" actually live. If any code or explanation talks about
"tracking weights," it is wrong — it means activations.

## The core insight the tool exposes

The gap between **Danger recognized** (does the model internally represent the input as
harmful?) and **Refusal engaged** (does it actually refuse?). Jailbreaks and interventions
pull these apart. Making that gap visible and manipulating it causally IS the demo.

## Decisions already locked (do not relitigate)

| Decision | Choice |
|---|---|
| Deliverable | Live demo dashboard with **detection + intervention** (Option B) |
| Model | **Gemma-2-2B** instruction-tuned, bf16 (~5 GB, fits RTX 3080 / 10 GB) |
| SAEs | **Gemma Scope** (pre-trained, per-layer) — do NOT train our own SAEs |
| Model harness | **TransformerLens** (`HookedTransformer`) |
| SAE loading | **SAELens** |
| Feature labels | **Neuronpedia** (auto-generated Gemma Scope labels) |
| UI | **Gradio** (single Python process holds model + SAE) |
| Feature selection | Neuronpedia keyword search → curated watchlist (Option A). Contrastive discovery (Option C) is deferred but the catalog is designed to accept it later. |

## Architecture (four modules, each independently testable)

1. `model_runner` — loads model + SAEs; `capture(prompt)` returns activations,
   `run_with_intervention(prompt, edits)` modifies activations via SAE decoder directions
   before they continue through the network. Same hook path for observe and intervene.
2. `feature_catalog` — YAML/JSON list of `{feature_id, layer, label, category, source}`.
   Categories: weapons, deception, illegal, refusal, roleplay_jailbreak. `source` field is
   the Option-C seam (contrastive entries append here later).
3. `analysis` — SAE `encode` → `[tokens × features]`, filter to watchlist, compute
   per-token strengths + the Danger-recognized / Refusal-engaged headline signals.
4. `app` — Gradio UI: input+response / feature-activity heatmap / intervention sliders.

## Demo narrative (build every UI element to serve one of these beats)

1. Benign prompt → nothing lights up.
2. Harmful prompt → danger + refusal features fire → model refuses ("it knows, it stops").
3. Jailbroken prompt → danger still fires, refusal suppressed → complies ("jailbreak cut
   the alarm wire").
4. Intervention → manually cut/restore the alarm → behavior flips ("proof it's causal").

## Relation to mechanistic interpretability research

This is applied mech interp — a hands-on instance of the field's core loop, using existing
tools. It touches three pillars:
1. **Superposition & SAEs** — neurons are polysemantic (superposition); SAEs unpack dense
   activations into sparse single-meaning features. We use pre-trained **Gemma Scope** SAEs.
2. **Features as unit of analysis** — identify safety features, verify via top activating
   examples, study when they fire (our catalog + heatmap).
3. **Causal intervention** — ablation/steering to move from correlation to causation (our
   ablate/clamp/amplify panel; zeroing refusal = ablation, amplifying = steering à la
   Golden Gate Claude).

**Honest scope for the pitch:** this is *applied* mech interp using Gemma Scope +
Neuronpedia, NOT novel research. We don't train SAEs or validate circuits. The contribution
is the interactive observe-and-intervene framing around the recognize-vs-refuse gap. Do not
overclaim to knowledgeable judges. Full detail in spec §7a.

## Environment

- Dev machine: NVIDIA RTX 3080 (10 GB VRAM), 31 GB RAM, Linux.
- Python 3.10+. Keep the model in bf16; short `max_new_tokens`; clear caches between runs
  to stay within 10 GB.
- Needs a HuggingFace account/token with Gemma license access to download Gemma-2-2B and
  Gemma Scope.

## Key risks (see spec §9 for full list)

- Neuronpedia labels are auto-generated and noisy — hand-verify each watchlist feature via
  its top activating examples before trusting it.
- Intervention is delicate — validate the empty-edit no-op case first; start with ablation.
- Refusal behavior only appears with the correct Gemma chat template + instruction-tuned
  model — verify refusals actually occur before building the story on them.

## Status

- [x] Design spec written and committed
- [ ] Implementation plan (writing-plans)
- [ ] Implementation

## Ethics note

This is defensive interpretability research for a safety hackathon: the point is to
understand and detect when safety features fire, and to demonstrate that they are causal.
Harmful prompts exist only to trigger and study internal safety representations; do not
add features whose purpose is to produce or distribute harmful content.

# Design: Danger Feature Tracker — Live Interpretability Dashboard

**Date:** 2026-07-11
**Status:** Approved (design phase)
**Author:** Eric Ngo (with Claude Code)
**Context:** AI safety hackathon project

---

## 1. One-line summary

A live Gradio dashboard that runs an open-weight model (Gemma-2-2B), decomposes its
internal activations into interpretable features via pre-trained Sparse Autoencoders
(Gemma Scope), and lets the user **observe** which safety-relevant ("dangerous")
features fire on a given prompt — and **intervene** on those features (ablate / clamp /
amplify) to prove they causally drive the model's behavior.

## 2. Motivation & core insight

The project answers: *how does a model reason its way to a conclusion, and do
"dangerous" internal features activate — and why?*

Important conceptual correction baked into this design: **we track activations, not
weights.** Weights are frozen learned parameters; they never change during inference.
Activations are the intermediate values that flow through those weights for a specific
input — that is where a specific reasoning process actually lives.

The single insight the whole tool is built to expose is the gap between two things:

- **Danger recognized** — does the model internally represent that the input is harmful?
- **Refusal engaged** — does the model actually refuse?

Jailbreaks and interventions pull these two apart. Making that gap visible, and then
manipulating it causally, is the demo.

## 3. Scope

**In scope (target deliverable — "Option B: detection + intervention"):**
- Live dashboard: type a prompt, see response + per-token feature-activation heatmap.
- A curated watchlist of danger-related features (weapons, deception, illegal, refusal,
  jailbreak/roleplay categories) sourced from Neuronpedia's Gemma Scope labels.
- Intervention controls: ablate / clamp / amplify any watched feature and re-run,
  showing the new response side-by-side with the original (causal demonstration).
- Preset prompt sets (benign / harmful / jailbroken) for the judging demo.

**Out of scope for v1 (seams left in place):**
- Training our own SAEs (we use pre-trained Gemma Scope SAEs — do NOT train from scratch).
- Contrastive feature discovery ("Option C") — the `feature_catalog` is designed so a
  contrastive pass can append entries later without other code changes.
- Custom React frontend — Gradio is sufficient and keeps everything in one process.

## 4. Technical stack

| Concern | Choice | Rationale |
|---|---|---|
| Model | **Gemma-2-2B** (bf16, ~5 GB) | Fits the target RTX 3080 (10 GB); best SAE support |
| SAEs | **Gemma Scope** (per-layer) | Pre-trained, released for every Gemma-2-2B layer |
| Model harness | **TransformerLens** (`HookedTransformer`) | Clean hook API for capture + intervention |
| SAE loading | **SAELens** | Loads Gemma Scope SAEs, `encode`/`decode` helpers |
| Feature labels | **Neuronpedia** | Auto-generated human-readable labels for Gemma Scope features |
| UI | **Gradio** | Python-only, single process holds model+SAE, easy sliders/heatmaps |
| Language | Python 3.10+ | Ecosystem requirement |

**Hardware target:** NVIDIA RTX 3080 (10 GB VRAM), 31 GB system RAM (verified on dev machine).

## 5. Architecture

```
Prompt ──► Gemma-2-2B (hooks on chosen layers)
              │
              ├─► captured residual-stream activations
              │        │
              │        ▼
              │   Gemma Scope SAE ──► per-token feature activations
              │                              │
              │                              ▼
              │                   filter to "danger watchlist" features
              │                              │
   generated  │                              ▼
   response ◄─┘                    Gradio UI: response + token heatmap
                                              │
                                   intervention sliders (clamp/ablate/amplify)
                                              │
                                     re-run with modified activations
```

Four modules, each independently testable:

### 5.1 `model_runner`
- Loads Gemma-2-2B via TransformerLens in bf16.
- Loads the Gemma Scope SAE(s) for the watched layer(s) via SAELens.
- Two run modes:
  - `capture(prompt)` → forward pass, hooks residual stream at watched layers,
    returns raw activations + generated tokens.
  - `run_with_intervention(prompt, edits)` → same, but hooks modify the activation
    before it continues through the network. An `edit` is
    `{feature_id, layer, mode, value}` where `mode ∈ {ablate→0, clamp→fixed, amplify→scale}`.
    Mechanism: take the SAE's decoder direction for that feature and add/subtract/zero it
    in the residual stream, then let the model finish.
- Design point: the SAE provides a **dictionary of directions**; both observe and
  intervene use the same hook path, so they never drift apart.

### 5.2 `feature_catalog`
- Human-readable config (YAML or JSON): a list of
  `{feature_id, layer, label, category, source}` entries.
- `category` groups features: `weapons`, `deception`, `illegal`, `refusal`,
  `roleplay_jailbreak` (extendable).
- Populated by searching Neuronpedia's Gemma Scope labels and pasting in IDs;
  `source: "neuronpedia-keyword"` for v1.
- **Option-C seam:** a later contrastive-discovery pass appends entries with
  `source: "contrastive"`; no other code changes required.
- Loaded once at startup; exposed as a shared lookup for analysis + UI.

### 5.3 `analysis`
- Takes raw activations, runs SAE `encode` → feature-activation tensor `[tokens × features]`.
- Filters to watchlist feature IDs; produces a tidy per-token structure plus
  per-category aggregates (e.g. max refusal-feature activation on the prompt).
- Computes the two headline signals: **Danger recognized** (any harm-category feature
  above threshold?) vs **Refusal engaged** (refusal feature fired?). The gap between them
  is the headline story.

### 5.4 `app`
- Gradio UI, three zones (see §6).
- Holds references to the loaded `model_runner`, `feature_catalog`, and calls `analysis`.

## 6. UI layout (Gradio)

1. **Input & response (top):** prompt box, Run button, generated text, and a dropdown of
   preset prompts (benign / harmful / jailbroken variants of the same request).
2. **Feature activity (middle):**
   - Token heatmap: prompt+response tokens colored by watchlist-feature firing strength;
     toggle which category colors the map.
   - Headline readout: two numbers — *Danger recognized* and *Refusal engaged* — plus the gap.
   - Ranked list of top-firing features with their Neuronpedia labels.
3. **Intervention (bottom):**
   - Per-feature (or per-category) control: ablate / clamp / amplify + strength slider.
   - "Re-run with edits" button → `run_with_intervention` → new response shown
     side-by-side with the original.

## 7. Demo narrative (what the UI is built to walk a judge through)

1. Benign prompt → nothing lights up.
2. Harmful prompt → danger features fire **and** refusal fires → model refuses.
   *"It knows, and it stops."*
3. Jailbroken version → danger features **still** fire but refusal is suppressed →
   model complies. *"It still knows — the jailbreak just cut the alarm wire."*
4. Intervention → manually cut/restore the alarm and show the behavior flip.
   *"And here's proof it's causal."*

## 7a. Relation to mechanistic interpretability research

This project is an applied instance of the central mech interp research loop. It touches
the field's three pillars:

1. **Superposition & SAEs.** Networks pack more features than neurons by storing them as
   overlapping directions, making individual neurons *polysemantic* (Anthropic, "Toy Models
   of Superposition," 2022). The field's answer is the Sparse Autoencoder, which unpacks
   dense activations into sparse, mostly single-meaning features ("Towards Monosemanticity,"
   2023; "Scaling Monosemanticity" / Golden Gate Claude, 2024). We use pre-trained SAEs
   (**Gemma Scope**, DeepMind 2024) rather than training our own — this is the core
   methodological tool we build on.
2. **Features as the unit of analysis.** Identifying safety-relevant features, verifying
   their meaning via top activating examples, and studying when they fire is standard mech
   interp methodology. Our `feature_catalog` + heatmap is a direct application.
3. **Causal intervention.** Observation is only descriptive; mech interp insists on *causal*
   claims via activation patching / ablation / steering (IOI circuits, Wang et al. 2022;
   ROME, Meng et al. 2022). Our ablate/clamp/amplify panel runs live ablation and steering
   experiments — zeroing the refusal feature to induce compliance is an ablation result;
   amplifying a feature to change behavior is steering, the same move as Golden Gate Claude.

**Safety framing:** mech interp is the most safety-motivated interpretability subfield
because it promises auditing models for dangerous cognition invisible in the outputs —
e.g. detecting that a model internally recognizes harm even while complying. Our headline
Danger-recognized vs Refusal-engaged gap is a miniature of exactly that question.

**Honest scope:** this is *applied* mech interp using existing tools (Gemma Scope +
Neuronpedia), not novel mech interp research. We do not train SAEs, discover new methods,
or rigorously validate circuits. The contribution is the interactive observe-and-intervene
framing around the recognize-vs-refuse gap. State it this way — overclaiming to a
knowledgeable judge is a credibility risk.

## 8. Testing strategy

- `model_runner`: unit test that `capture` returns activations of expected shape for a
  known prompt; test that `run_with_intervention` with an empty edit list produces output
  identical to `capture` (no-op safety), and that ablating a feature measurably changes
  the target feature's downstream activation.
- `feature_catalog`: test that the config loads, validates required fields, and that
  lookups by category/id work; test that appending a `contrastive`-source entry integrates
  cleanly.
- `analysis`: test that `encode` + filter yields per-token strengths matching hand-checked
  values on a tiny synthetic activation tensor; test the Danger/Refusal aggregate logic
  against crafted inputs.
- `app`: smoke test that the Gradio interface builds and the run callback returns without
  error on a preset prompt (may require GPU; mark as integration).

## 9. Risks & mitigations

- **Feature labels are noisy / auto-generated.** Neuronpedia labels are approximate.
  Mitigation: hand-verify each watchlist feature by inspecting its top activating examples
  before trusting it; document confidence in the catalog.
- **Intervention is delicate.** Editing the residual stream can break generation or do
  nothing. Mitigation: start with ablation (zeroing is well-behaved), validate the no-op
  case first, tune amplify scales empirically.
- **Chat template / instruction formatting.** Gemma-2-2B needs its chat template for
  refusal behavior to appear. Mitigation: use the instruction-tuned variant and correct
  template; verify refusals actually occur before building the story around them.
- **VRAM pressure during generation.** 10 GB is enough for the model but leaves little
  headroom. Mitigation: bf16, short max-new-tokens, clear caches between runs.

## 10. Future work (post-hackathon)

- Contrastive feature discovery (Option C) to empirically validate and expand the watchlist.
- Multi-layer feature tracking and cross-layer circuit views.
- Scale to Gemma-2-9B on larger hardware.

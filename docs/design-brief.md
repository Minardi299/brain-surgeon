# Danger Feature Tracker — Design Brief (for a visualization prototype)

## What this project is

**Danger Feature Tracker** is a live, applied **mechanistic-interpretability** dashboard.
It runs an open model (Gemma-2-2B-it) and, using pre-trained **Sparse Autoencoders**
(Gemma Scope), decomposes the model's internal **activations** into human-labelled
"features" — e.g. *weapons*, *deception*, *illegal*, *roleplay/jailbreak*, *refusal*. The
tool lets a user **observe** which safety-relevant features fire on a prompt and then
**intervene** on them (ablate / clamp / amplify, or cut a "refusal direction") to prove
those internal representations *causally* drive the model's behaviour.

The one idea the whole tool exists to make visible: the **gap between "Danger recognized"
and "Refusal engaged"** — does the model *internally know* the input is harmful, and does
that knowledge actually connect to *refusing*? Jailbreaks and interventions pull these two
apart. That gap, and manipulating it causally, is the demo.

**Key empirical results the visuals should carry (they are the story):**
1. **Danger recognition is sharp and real** — the weapons feature reads ~0 on a benign
   prompt and ~130 on "how to make a bomb."
2. **The labelled SAE "refusal" features are correlational, not causal** — they light up
   when the model refuses, but zeroing them changes nothing.
3. **A contrastive "refusal direction" is the causal lever** — cutting it flips the model
   from refusing to complying. (This is the honest, sophisticated punchline: *the thing
   that lights up is not the thing that acts.*)

**Audience:** AI-safety hackathon judges — technically literate, but the visuals must land
the story in seconds for someone walking past.

## Keywords

`mechanistic interpretability` · `sparse autoencoders (SAEs)` · `Gemma Scope` ·
`safety features` · `activation steering` · `refusal direction` · `ablation` ·
`causal intervention` · `observe-and-intervene` · `recognize-vs-refuse gap` ·
`per-token feature attribution` · `activation heatmap` · `feature dashboard` ·
`polysemanticity / monosemantic features` · `residual stream` · `AI safety` ·
`jailbreak analysis` · `correlation-vs-causation` · `interpretability microscope`

## The data available to visualize (what the backend produces)

For any prompt, the backend returns:
- **`response_text`** — the model's generated answer (may be a refusal or a compliance).
- **`str_tokens`** — the full token sequence (chat-template scaffold + prompt + response),
  e.g. `["<bos>", "<start_of_turn>", "user", "how", "to", "make", "a", "bomb", …, "I",
  "cannot", …]`. Note the `<bos>` token is deliberately zeroed (an artifact).
- **Per-token, per-feature activations** — for each watchlist feature, a value at each
  token position (0–~150 range; most positions are 0).
- **Per-category aggregates** — each of the 5 categories has a summed per-token activation
  track and a max.
- **Headline signals** — `danger_recognized` (float), `refusal_engaged` (float),
  `gap = danger − refusal`, plus boolean flags.
- **Top-firing features** — ranked list of `{label, category, max_activation}`.
- **Intervention output** — the same bundle *after* an edit (SAE-feature ablate/clamp/
  amplify, or refusal-direction ablation), for before/after comparison.
- **A steering sweep** (available, not surfaced): response + refusal signal at increasing
  intervention strength (e.g. clamp to −30 / −80 / −150), i.e. a dose-response curve.

The watchlist is small and fixed (~11 features across 5 categories, all at model layer 20).
The refusal direction lives at layers ~8–12.

## Current visualizations (functional, but plain Gradio defaults)

These exist and work; they are the baseline to redesign, not a constraint:
1. Prompt textbox + preset dropdown (benign / harmful / jailbroken).
2. Model-response text box.
3. **Headline readout** — three numbers (Danger / Refusal / Gap) as plain markdown.
4. **Per-token heatmap** — a `HighlightedText` strip: each token tinted by one selected
   category's activation. Currently one category at a time, a flat −1…+1 colour ramp.
5. **Top-firing features** — a plain table.
6. **Intervention panel** — feature dropdown + mode + value + "Re-run", showing the edited
   response and its headline.
7. **"Cut refusal direction (causal)"** button — shows the flipped response.

## Proposed visualizations (the ask — prioritized)

**P0 — The Gap, as the hero element.** A single, prominent dual-indicator: *Danger
recognized* vs *Refusal engaged* as two opposing bars/gauges with the **gap** called out
between them. It should read instantly ("danger high, refusal low → the alarm is
disconnected"). Colour: danger = warm/red, refusal = cool/blue, gap = the tension between
them. This is the number a judge should grok in one glance.

**P0 — The 4-beat storyboard.** The demo is a narrative in four states — *benign →
harmful → jailbroken → intervened*. A horizontal storyboard where each beat shows the Gap
hero + a one-line caption ("it knows, and it stops" → "it still knows, but the alarm is
cut"). Ideally the gap animates/transitions between beats. This single view *is* the pitch.

**P1 — Per-token feature heatmap, redesigned.** Replace the flat strip with:
- A token stream with a proper sequential colour scale and a legend, **hover tooltips**
  ("token *bomb* · weapons feature · activation 140").
- **Small multiples**: the same token stream stacked, one row per category, so weapons /
  refusal / deception firing patterns are comparable at a glance.
- Optional "piano-roll" mode: x = token position, y = features, cell intensity =
  activation — shows *where* in the sentence each feature switches on.

**P1 — Before/after intervention diptych.** Two response cards side by side (original vs
intervened) with the behavioural change highlighted, plus the Gap hero animating from one
state to the other — the visceral "cut the wire → watch it flip."

**P1 — Correlation-vs-causation panel (the honest headline).** A two-column comparison:
*Ablate the labelled refusal features* → refusal signal drops to 0 **but behaviour is
unchanged** (still refuses); *Cut the contrastive refusal direction* → **behaviour flips**.
Same "signal goes to zero," opposite behavioural outcome. This is the most intellectually
interesting visual and differentiates the project from a naive demo.

**P2 — Steering dose-response curve.** A slider (ablate ← → amplify) with the response
updating live, and a small line chart of *refusal signal vs steering strength* — showing
the point where behaviour breaks or flips.

**P2 — Model schematic / intervention map.** A simple diagram of the transformer stack
(26 layers) marking where things happen: SAE features read at layer 20, refusal direction
cut across layers 8–12. Grounds the abstract "we edit the residual stream" claim.

**P2 — Feature "alarm board."** The 5 categories as a compact panel of labelled meters
(weapons / deception / illegal / roleplay / refusal), each lighting up with its current
activation — a glanceable "which alarms are ringing."

**Optional framing metaphor:** the tool is literally a *danger alarm* + a *refusal
response*, and jailbreaks/interventions "cut the alarm wire." A restrained circuit/wire
motif could make the causal story visceral without being gimmicky.

## Design tone

- **Mood:** an "interpretability microscope" / instrument panel — precise, calm, a little
  technical, but legible to a non-expert in seconds. Dark-mode-first (it's a live dashboard
  shown on a projector/booth screen).
- **Colour semantics (suggested, not fixed):** danger/harm = warm red-orange; refusal =
  cool blue/teal; the *gap* = the visual tension between them; neutral tokens = muted grey.
  Keep one accent for "intervention/causal" actions (the "cut" is a deliberate, weighty
  action — it currently uses a red "stop"-style button).
- **Legibility of activations:** values span 0–~150 and most are 0; scales should emphasize
  the few high tokens without letting one outlier flatten everything (and the `<bos>` token
  is already excluded upstream).
- **One-glance test:** a judge walking past should, from the hero + storyboard alone, get
  "the model recognizes danger, and we can causally connect or disconnect that from its
  refusal." Everything else is supporting detail.

## Not in scope for the prototype

Real model inference, the Gradio plumbing, and the interventions already work — the design
agent only needs to prototype the *visual language and layout* (static or lightly
interactive mockups) using representative numbers from "The data available" above. No model
or GPU required to design against this.

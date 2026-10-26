# Danger Feature Tracker — Demo Concept & Flow (design hand-off)

> This file is the **story and layout spine**, written against the **current design**
> (`RECOGNIZE ≠ REFUSE` board). For data shapes and colour semantics see
> `docs/design-brief.md`; for the **actual measured numbers** every value below comes from,
> see `docs/demo-data.md`. The current design is already strong — this is a **tweak, keep
> the components**, not a redesign. Where a change is needed it's because the *story* would
> otherwise be muddied — or, in beat 3's case, because the real runs contradict it.

---

## The thesis (one line — already your title)

**A model can fully recognize that something is dangerous and still be made to comply —
because *recognizing danger* and *refusing* are two separate mechanisms, and we can find
the wire between them, watch it, and cut it live.** → `RECOGNIZE ≠ REFUSE`.

## The result we prove (three claims, in order)

1. **Recognition is real and sharp.** Benign ≈ 0; harmful spikes. The model genuinely
   represents "this is dangerous."
2. **Recognition ≠ refusal.** A jailbreak leaves recognition *high* but drops refusal to
   near-zero and the model complies.
3. **The switch is causal, and it's not the one that blinks.** Cutting one *refusal
   direction* (L8–12) flips refuse → comply while danger stays pinned. The labelled SAE
   "refusal" features (L20) are **correlational** — zero them and nothing changes. *The
   thing that lights up is not the thing that acts.*

Carrier of all three: **THE GAP = Danger − Refusal.** The gap opening = safety failing
inside an intact model. The gap **is** the demo. (Your footer already says it perfectly:
*"A ringing alarm is not a wired brake."*)

---

## What the current design already nails (keep, don't touch)

- **`RECOGNIZE ≠ REFUSE`** as the title — it's the thesis in two words.
- The **GAP hero** (opposing red/blue meters + big `+0.04` + `KNOWS − REFUSES`) with the
  **1-2-3-4 stepper and ▸ play**. This is the spine and the sound-off narrative. Perfect.
- The **`SAE @ L20 · observe / direction @ L8-12 · intervene`** note top-right, echoed in
  section 00 — the accuracy backbone that stops the demo from lying. Keep it visible.
- **03 Cut-the-wire** holding **danger 0.90 → 0.90** while refusal collapses — the single
  most important truth in the whole demo. Do not let this slip.
- **05 Dose-response** with the red *danger — unchanged* dashed line flat across the sweep —
  reinforces the same truth quantitatively. Keep.
- The **footer thesis restatement.** Keep it as the closing line.

---

## The one structural tweak: make it a *guided path*, not a wall

The page currently shows **every panel at once** (alarm board, hero, prompt/output, 00–05).
Each panel is good and each serves a beat — but 9 panels competing for attention buries the
narrative. **Do not delete panels. Add a beat-driven spotlight.**

- The **stepper (and ▸ play) drives a spotlight.** At each beat, the 1–2 panels that tell
  *that* beat come forward (full colour / subtle lift); the rest recede (dimmed, desaturated,
  still present for the judge who leans in).
- `▸ play` walks the spotlight through beats 1→2→3→4 automatically — that *is* the guided,
  sound-off pitch. A judge watching play-mode sees one coherent story, not a dashboard.
- The **hero + prompt/output stay lit on every beat** (they're the spine); everything else
  is evidence that surfaces on cue.

**Beat → which panels light up:**

| Beat | Spotlight | Recedes |
|---|---|---|
| 1 · Benign | hero (both meters low), alarm board (all QUIET), model output (normal answer) | 00–05 |
| 2 · Harmful | hero (both high, wire taut), alarm board (FIRING), **01 per-token**, **02 top features** | 00, 03, 04, 05 |
| 3 · Jailbreak attempt | hero (**danger high, refusal STILL high, wire strains but holds**), output (still refuses) | 00, 03, 04, 05 |
| 4 · We cut it | **00 internals + cut button**, **03 before/after**, **04 corr-vs-causation**, **05 dose-response** | 01, 02 |

> **Beat 3 was reframed to match reality.** The DAN jailbreak does *not* suppress refusal on
> Gemma-2-2b-it (refusal stayed 0.88, model refused — see `demo-data.md`). So beat 3 is now
> "the jailbreak is attempted and Gemma holds" — the gap does **not** open here. The gap
> opens only in beat 4, when *we* cut the direction. This is honest and a stronger setup:
> the outside attack fails, so we go inside.

This reconciles "reuse everything you built" with "tell one clear story."

---

## The hero's four states (design all four, not one frozen frame)

The screenshot is frozen on Beat 2. Spec the hero for each beat so the agent designs the
**transitions** — the eye must track *refusal falling while danger holds*.

All values below are **measured** (`demo-data.md`). Meters are per-signal normalized (danger
÷ ~140, refusal ÷ ~44 — each to its own firing range), which is why 127 and 39 both read ~0.9.

| Beat | Danger | Refusal | Gap | Wire (between the meters) | Caption |
|---|---|---|---|---|---|
| 1 Benign | 0.00 | 0.00 | 0.00 | slack / inactive | *"Nothing dangerous here — and nothing lights up."* |
| 2 Harmful | 0.91 | 0.88 | +0.03 · REFUSES | **taut, lit — current flowing** | *"It recognizes the danger, and the alarm reaches the brake. It refuses."* |
| 3 Jailbreak attempt | 0.85 | 0.88 | −0.03 · **STILL REFUSES** | **strains under load — but holds** | *"The jailbreak pushes… and Gemma holds. Text tricks don't cut the wire."* |
| 4 We cut it | 0.51 | ~0.05 | **+0.46 · COMPLIES** | **cleanly severed by us** (restorable) | *"What the jailbreak couldn't do from outside, we do from inside — refusal collapses, and the danger it still recognizes never drops to zero."* |

> **New component — the wire lives in the hero.** Right now "THE GAP" is a number between
> two bars. Add a thin **connective wire** across that gap that is the same object cut in
> 00/03: taut+lit in Beat 2; **strained but intact in Beat 3** (the jailbreak yanks it and
> fails — do *not* fray it, the attack didn't work); **cleanly severed by us in Beat 4**.
> This makes the gap *mean* something and unifies the wire metaphor across hero ↔ 00 ↔ 03.
> The `REFUSES / COMPLIES` pill under the gap flips only at Beat 4 — good.

Motion budget: animate only the two moments that *are* the argument — **Beat 3's strain
(the wire pulled hard and holding)** and **Beat 4's cut (the wire severed, refusal
collapsing while danger stays high)**. Everything else holds still. The gap opens at Beat 4,
not Beat 3.

---

## Climax fix: make section 04 the *true* twist (important — story accuracy)

Section 04 as drawn (`A — SILENCE THE SAE DANGER FEATURES · zero top 11 @ L20`) muddies the
headline and doesn't match what we actually verified. The verified, airtight contrast is
**refusal vs refusal** — remove refusal *two ways* and get opposite outcomes:

- **04-A — Silence the labelled SAE _refusal_ features** (9989, 10156, 13364 from section 02)
  → **measured:** refusal signal **38.8 → 0.0**, danger **126 → 123 (unchanged)**, behaviour
  **unchanged: still refuses.** (This is the *blinking wire* — provably inert. Bonus real
  result: forcing these features *up* to +200 on a benign prompt still doesn't induce a
  refusal.)
- **04-B — Cut the refusal direction** @ L8–12 → **measured:** behaviour **flips: complies**;
  danger 127 → 71 (drops but still firing hard). (This is the *live wire*.)

**Caption:** *"Same refusal removed two ways. The feature that *reads* refusal is not the
mechanism that *performs* it — same signal to zero, opposite outcome."*

> Both arms are backed by real runs (`demo-data.md`). 04-A is the cleanest "danger truly
> untouched" evidence (126→123); note 04-B *does* lower danger somewhat (127→71) because
> cutting the direction perturbs every layer including the L20 readout.

Why this matters: it ties **02 → 03 → 04** into one thread (the "refusal & policy language"
feature in 02 *is* the blinking wire that 04-A proves is inert), and it's what we actually
tested. Silencing *danger* features is a different, unverified experiment that dilutes the
punchline — drop it.

> If you want to keep a danger-side demonstration, that's what **03's held danger 0.90→0.90**
> and **05's flat danger line** already do, more cleanly. 04 should be about the refusal wire.

---

## Component-level tweaks (targeted, low-risk)

1. **Two scales, label them.** The alarm board reads **raw activation 0–150** (149, 88, 96);
   the hero and 03/05 read a **normalized 0–1** signal (0.92, 0.88). A judge will trip on
   "149 vs 0.92." Add a tiny unit label to each region ("raw activation" on the board,
   "normalized" on the hero) or convert the board to the same 0–1 scale. Don't leave both
   unlabelled.
2. **One cut action, three synced views.** The `✂ CUT THE REFUSAL DIRECTION` button (00),
   the 03 before/after, and the hero's Beat-4 state must be **the same event** — press once,
   00's wire severs, 03 flips to SEVERED, hero refusal drops. Not three separate cut buttons
   the judge has to reconcile. 00 = *where* it's cut; 03 = *result*; hero = *headline*.
3. **Benign must read genuinely empty.** Beat 1's alarm board should be visibly all-QUIET
   and the hero near-zero (the `<bos>` artifact is already excluded upstream, so this is
   honest). If Beat 1 shows faint noise, the "instrument is honest" beat dies.
4. **Per-token (01): note the scale and BOS.** Keep the small-multiples rows — they're the
   best "where in the sentence it fires" view. Legend: `scale 0–150, <bos> excluded`, and a
   hover tooltip (`token "bomb" · weapons · 140`).
5. **02 top features → label the refusal one as the blinking wire.** When Beat 4 lights up,
   visually connect the refusal feature in 02 to 04-A ("this is the wire we're about to prove
   is inert"). It turns a plain list into a setup for the twist.
6. **05 dose-response: relabel to a _layer_ sweep, or mark it representative.** The smooth
   "P(refusal) vs continuous ablation strength" sigmoid is *not* what we measured. The real
   sweep is by **layer**: cutting the direction at source layers **8/10/12 flips to comply**;
   **14 partial; 16–18 still refuse** (`demo-data.md`). Either redraw 05 as "effect vs the
   layer where we cut" (honest, we have this) or keep the curve but label it clearly as
   representative. Don't present the sigmoid as measured.

---

## Visual language (keep current, these are the rules)

- **Danger** = warm red-orange. **Refusal** = cool blue-teal. **Gap** = the tension between
  them; when they diverge (Beats 3–4), make the gap *loud*.
- **The wire** is the one metaphor: alarm → brake. Taut/lit (intact) · frayed (jailbroken) ·
  severed (intervened). Restrained circuit motif — the current design's weight is right.
- **The "cut" is a deliberate, heavy action** — one strong accent, a real commit moment.
- **Mood:** interpretability microscope / safety-ops instrument panel — precise, calm, the
  current mono-label + warm-paper aesthetic reads well on a projector. Keep it.

## Accuracy guardrails (do not let the visuals lie)

1. **Recognition stays strong but is not perfectly constant.** Under the causal cut, danger
   drops **0.90 → 0.51** (raw 127 → 71) — still far above benign 0, never eliminated. It is
   *exactly* unchanged only in the correlational SAE-clamp arm (04-A: 126 → 123). **Never
   draw the causal cut as danger 0.90 → 0.90.** Draw it as "danger stays firing hard while
   refusal collapses." The honest headline is *"recognition never near zero while we toggle
   refusal,"* not *"recognition constant."* (This means section 03's current "DANGER 0.90 →
   0.90" must change to "0.90 → 0.51 · still recognizes.")
2. **Observe = L20 (SAE features); intervene = L8–12 (refusal direction).** Keep them
   distinct (00 and the header already do). The cut is **not** "turning off the L20
   features" — that's the correlational trap 04 exists to debunk.
3. **The L20 "refusal" features are the blinking wire, not the live one.** No visual may
   imply they *cause* the refusal.
4. **All numbers are representative mock traces** (footer already says so) — keep them
   plausible against the real ranges in `design-brief.md`.

## The one-glance test

A judge reading only the **hero + current caption** (or watching ▸ play) should get:

> **"The model recognizes danger — and we can connect or disconnect that from its refusal,
> on demand."**

If the hero + stepper deliver that sentence alone, the design works. Sections 00–05 are for
the judge who stops to lean in.

import torch

MODEL_NAME = "google/gemma-2-2b-it"

SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"


def sae_id_for_layer(layer: int) -> str:
    return f"layer_{layer}/width_16k/canonical"


DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 128
# The live dashboard only needs the opening words to show refuse-vs-comply; capping
# generation short keeps the KV-cache peak well under the 10 GB card (shared with the
# desktop/other GPU users) and makes each beat snappy. The earlier smoke runs that fit
# generated ~40-50 tokens; 128 pushed peak memory over the edge and OOM'd on the 3rd beat.
DEMO_MAX_NEW_TOKENS = 64

# Activation strength above which we treat a feature as "firing". Calibrate against
# real Gemma Scope activations (see README) — these are starting defaults.
DANGER_THRESHOLD = 1.0
REFUSAL_THRESHOLD = 1.0

HARM_CATEGORIES = frozenset({"weapons", "deception", "illegal", "roleplay_jailbreak"})
REFUSAL_CATEGORY = "refusal"
VALID_CATEGORIES = HARM_CATEGORIES | frozenset({REFUSAL_CATEGORY})

TOP_K = 8

# Number of leading tokens to ignore when reading feature activations. Gemma Scope SAE
# features fire spuriously large on the <bos> token (~50+) regardless of prompt content,
# which otherwise dominates the heatmap and produces false "refusal" readings on benign
# prompts. Excluding the first token (BOS) is standard practice.
IGNORE_PREFIX_TOKENS = 1

# Dashboard headline-signal normalization: each raw signal is scaled to *its own* observed
# firing range (danger tops out around ~140, refusal around ~44 — see docs/demo-data.md),
# not a shared 0-150 scale. This is why a danger raw of 127 and a refusal raw of 39 both
# read as ~0.9 "fully engaged": each is fully engaged *relative to itself*.
DANGER_NORM = 140.0
REFUSAL_NORM = 44.0

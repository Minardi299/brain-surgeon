import torch

MODEL_NAME = "google/gemma-2-2b-it"

SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"


def sae_id_for_layer(layer: int) -> str:
    return f"layer_{layer}/width_16k/canonical"


DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 128

# Activation strength above which we treat a feature as "firing". Calibrate against
# real Gemma Scope activations (see README) — these are starting defaults.
DANGER_THRESHOLD = 1.0
REFUSAL_THRESHOLD = 1.0

HARM_CATEGORIES = frozenset({"weapons", "deception", "illegal", "roleplay_jailbreak"})
REFUSAL_CATEGORY = "refusal"
VALID_CATEGORIES = HARM_CATEGORIES | frozenset({REFUSAL_CATEGORY})

TOP_K = 8

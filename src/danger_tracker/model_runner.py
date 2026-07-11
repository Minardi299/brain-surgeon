from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from danger_tracker import config
from danger_tracker.feature_catalog import Catalog


@dataclass
class CaptureResult:
    str_tokens: list[str]
    feature_acts_by_layer: dict[int, np.ndarray]
    response_text: str


class ModelRunner:
    def __init__(self, catalog: Catalog, model_name=config.MODEL_NAME,
                 device=config.DEVICE, dtype=config.DTYPE):
        self.catalog = catalog
        self.device = device
        self.model = HookedTransformer.from_pretrained(
            model_name, dtype=dtype, device=device
        )
        self.saes: dict[int, SAE] = {}
        for layer in catalog.layers():
            sae = SAE.from_pretrained(
                release=config.SAE_RELEASE,
                sae_id=config.sae_id_for_layer(layer),
                device=device,
                # Match the model's compute dtype (bf16 by default). Verified against
                # installed sae_lens 6.45.3: SAE.from_pretrained's `dtype` type hint
                # says `str`, but internally it calls `str_to_dtype(dtype)`, which
                # passes a torch.dtype through unchanged, and the loader ends with
                # `sae.to(dtype=str_to_dtype(dtype), device=device)` — so passing
                # torch.dtype directly works. Without this, resid_post activations
                # (bf16, from the model) get matmul'd against the SAE's W_enc
                # (float32 by default in sae_lens), which raises
                # `RuntimeError: expected m1 and m2 to have the same dtype` — confirmed
                # empirically on this CPU box with dummy bf16 @ float32 tensors.
                dtype=dtype,
            )
            # SAELens <= 5.x's SAE.from_pretrained returned (sae, cfg, sparsity);
            # installed sae_lens 6.45.3 returns just the SAE. Keep this defensive
            # unwrap so the code also works against older releases.
            self.saes[layer] = sae[0] if isinstance(sae, tuple) else sae

    def hook_name(self, layer: int) -> str:
        return f"blocks.{layer}.hook_resid_post"

    def _format(self, prompt: str) -> str:
        chat = [{"role": "user", "content": prompt}]
        # NOTE: unverified against runtime — see report. Requires the gated
        # google/gemma-2-2b-it tokenizer (chat_template) which cannot be
        # downloaded on this box; apply_chat_template is a standard HF
        # PreTrainedTokenizerBase method and HookedTransformer.tokenizer is a
        # real HF tokenizer instance, so the call shape itself is expected to
        # be correct, but the actual Gemma chat template output is unverified.
        return self.model.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

    def _encode_all(self, full_tokens, fwd_hooks):
        names = {self.hook_name(l) for l in self.catalog.layers()}
        with self.model.hooks(fwd_hooks=fwd_hooks):
            _, cache = self.model.run_with_cache(
                full_tokens, names_filter=lambda n: n in names
            )
        feature_acts: dict[int, np.ndarray] = {}
        for layer, sae in self.saes.items():
            resid = cache[self.hook_name(layer)]              # [1, seq, d_model]
            acts = sae.encode(resid)[0]                       # [seq, n_features]
            feature_acts[layer] = acts.float().cpu().numpy()
        str_tokens = self.model.to_str_tokens(full_tokens[0])
        return feature_acts, str_tokens

    def _generate(self, prompt: str, max_new_tokens: int, fwd_hooks):
        formatted = self._format(prompt)
        tokens = self.model.to_tokens(formatted, prepend_bos=False).to(self.device)
        prompt_len = tokens.shape[1]
        with self.model.hooks(fwd_hooks=fwd_hooks):
            full = self.model.generate(
                tokens, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=0.0, verbose=False,
            )
        response_text = self.model.to_string(full[0, prompt_len:])
        return full, response_text

    def capture(self, prompt: str, max_new_tokens=config.MAX_NEW_TOKENS) -> CaptureResult:
        full, response_text = self._generate(prompt, max_new_tokens, fwd_hooks=[])
        feature_acts, str_tokens = self._encode_all(full, fwd_hooks=[])
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return CaptureResult(str_tokens, feature_acts, response_text)

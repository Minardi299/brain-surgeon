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


@dataclass
class Edit:
    feature_id: int
    layer: int
    mode: str  # "ablate" | "clamp" | "amplify"
    value: float = 0.0


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
                # Encode at float32, not the model's bf16 compute dtype. Verified
                # against installed sae_lens 6.45.3: SAE.process_sae_in
                # (sae_lens/saes/sae.py:475) does `sae_in = sae_in.to(self.dtype)`
                # before the W_enc matmul, so a bf16 resid_post activation is
                # auto-upcast to the SAE's own dtype — there is no dtype-mismatch
                # crash to work around. Loading the SAE itself in bf16 would instead
                # downcast the residual, running the W_enc matmul and the JumpReLU
                # threshold comparison in bf16 and degrading precision of exactly the
                # values analysis.py compares against DANGER_THRESHOLD/
                # REFUSAL_THRESHOLD. Standard Gemma Scope practice is float32 encode.
                dtype="float32",
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
        # Hook-ordering assumption: the intervention edit hooks (fwd_hooks) are
        # registered by this outer `model.hooks(...)` context BEFORE run_with_cache
        # registers its caching hook on the same hook_resid_post point. TransformerLens
        # runs hooks at a point in registration order, so the cache captures the resid
        # AFTER the edit is applied — i.e. the returned feature activations reflect the
        # intervened state, matching the generated text. If this ordering were reversed
        # the heatmap would silently show pre-edit activations. This is verified on GPU by
        # test_ablating_feature_lowers_its_activation (ablation must lower the cached value).
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

    def _make_hook(self, edits_for_layer: list[Edit], layer: int):
        sae = self.saes[layer]

        def hook(resid, hook):  # resid: [batch, seq, d_model], model dtype (bf16)
            # sae.encode/W_dec run in the SAE's own dtype (float32 — see the
            # loader comment above), independent of resid's bf16 dtype.
            # process_sae_in() upcasts resid internally, so encode() never
            # crashes here; feature_acts and d_f below come back float32.
            feature_acts = sae.encode(resid)
            for e in edits_for_layer:
                d_f = sae.W_dec[e.feature_id]                        # [d_model], float32
                a_f = feature_acts[..., e.feature_id].unsqueeze(-1)  # [batch, seq, 1], float32
                if e.mode == "ablate":
                    delta = -a_f
                elif e.mode == "clamp":
                    delta = (e.value - a_f)
                elif e.mode == "amplify":
                    delta = (e.value - 1.0) * a_f
                else:
                    raise ValueError(f"unknown edit mode: {e.mode}")
                # delta * d_f is float32 (SAE dtype); resid is bf16 (model
                # dtype). torch would silently upcast the sum to float32
                # instead of raising, which would then break the *next*
                # layer's bf16 matmuls downstream. Cast the update back to
                # resid's dtype before adding so the residual stream stays
                # consistently bf16.
                resid = resid + (delta * d_f).to(resid.dtype)
            return resid

        return hook

    def _build_hooks(self, edits: list[Edit]):
        by_layer: dict[int, list[Edit]] = {}
        for e in edits:
            by_layer.setdefault(e.layer, []).append(e)
        return [
            (self.hook_name(layer), self._make_hook(es, layer))
            for layer, es in by_layer.items()
        ]

    def run_with_intervention(
        self, prompt: str, edits: list[Edit], max_new_tokens=config.MAX_NEW_TOKENS
    ) -> CaptureResult:
        # An empty edit list must build an empty fwd_hooks list, not a list of
        # no-op hook functions: model.hooks(fwd_hooks=[]) attaches nothing, so
        # generation/capture take the exact same code path as capture()'s
        # fwd_hooks=[] calls. That identity — not merely "the hook computes a
        # zero delta" — is what makes the empty-edits/capture no-op invariant
        # exact rather than approximately equal.
        fwd_hooks = self._build_hooks(edits)
        full, response_text = self._generate(prompt, max_new_tokens, fwd_hooks=fwd_hooks)
        feature_acts, str_tokens = self._encode_all(full, fwd_hooks=fwd_hooks)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return CaptureResult(str_tokens, feature_acts, response_text)

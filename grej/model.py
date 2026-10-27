"""Qwen2.5-VL wrapper that grounds an expression and extracts accept/reject signals.

Signals (higher = "the referred object is there"):
  S0  output_box        : 1 if the model emitted a box, 0 if it refused (its own decision)
  S1  box_logprob_mean  : mean log-prob of the box coordinate tokens
  S1b box_logprob_min   : min log-prob over coordinate tokens
  S2  attn_inbox_frac   : MTLA-style localized attention - fraction of visual attention mass
                          that coordinate-predicting positions put inside the claimed box
                          (middle-layer band, mean over heads and tokens); also per-layer
  S2b attn_inbox_ratio  : S2 divided by the box's share of visual tokens (area-normalised)
  S3  verify_crop       : log-odds of "Yes" when the model is shown ONLY the predicted crop and
                          asked whether it shows the expression
  S4  verify_full       : POPE-style log-odds of "Yes" to "Is there <expression> in the image?"
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from .geometry import crop_with_pad, explicit_refusal, parse_box, scale_box, token_mask_for_box

GROUND_PROMPT = (
    'Locate "{exp}" in the image and output its bbox coordinates in JSON format. '
    'If there is no such object in the image, output "None".'
)
VERIFY_CROP_PROMPT = 'Does this image show "{exp}"? Answer yes or no.'
VERIFY_FULL_PROMPT = 'Is there "{exp}" in the image? Answer yes or no.'


class Grounder:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", dtype: str = "float16",
                 max_pixels: int = 640 * 28 * 28, min_pixels: int = 64 * 28 * 28,
                 device_map: str = "cuda:0", layer_band: Optional[tuple[int, int]] = None):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        td = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=td, attn_implementation="eager", device_map=device_map
        ).eval()
        self.proc = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
        self.tok = self.proc.tokenizer
        cfg = self.model.config
        self.image_token_id = getattr(cfg, "image_token_id", None) or self.tok.convert_tokens_to_ids("<|image_pad|>")
        tc = getattr(cfg, "text_config", cfg)
        self.n_layers = tc.num_hidden_layers
        vc = cfg.vision_config
        self.patch = vc.patch_size * getattr(vc, "spatial_merge_size", 2)   # 28 px per merged token
        self.merge = getattr(vc, "spatial_merge_size", 2)
        # default band: middle ~40% of layers (MTLA uses a middle band, e.g. 8-21 of 36)
        self.layer_band = layer_band or (int(self.n_layers * 0.22), int(self.n_layers * 0.6))
        self.yes_ids = self._ids(["Yes", "yes", " Yes", " yes", "YES"])
        self.no_ids = self._ids(["No", "no", " No", " no", "NO"])

    def _ids(self, words):
        out = set()
        for w in words:
            ids = self.tok.encode(w, add_special_tokens=False)
            if len(ids) == 1:
                out.add(ids[0])
        return sorted(out)

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    def _inputs(self, image, prompt):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[image], return_tensors="pt")
        return {k: v.to(self.device) for k, v in inp.items()}

    @torch.no_grad()
    def _yes_logodds(self, image, prompt) -> float:
        inp = self._inputs(image, prompt)
        logits = self.model(**inp).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        y = torch.logsumexp(lp[self.yes_ids], 0).item()
        n = torch.logsumexp(lp[self.no_ids], 0).item()
        return y - n

    @torch.no_grad()
    def run(self, image, expression: str, max_new_tokens: int = 72, do_verify: bool = True) -> dict:
        inp = self._inputs(image, GROUND_PROMPT.format(exp=expression))
        prompt_len = inp["input_ids"].shape[1]
        gen = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                                  output_scores=True, return_dict_in_generate=True)
        seq = gen.sequences[0]
        new_ids = seq[prompt_len:]
        text = self.tok.decode(new_ids, skip_special_tokens=True)
        box_m, refused = parse_box(text)

        # resized-image geometry
        t, gh, gw = [int(v) for v in inp["image_grid_thw"][0].tolist()]
        rW, rH = gw * (self.patch // self.merge), gh * (self.patch // self.merge)
        mgh, mgw = gh // self.merge, gw // self.merge
        W, H = image.size

        out = {"raw_output": text, "refused": refused, "explicit_none": explicit_refusal(text),
               "resized_wh": [rW, rH], "orig_wh": [W, H], "output_box": 0 if refused else 1}

        # token log-probs of generated tokens
        step_lp = []
        for s, tid in zip(gen.scores, new_ids.tolist()):
            step_lp.append(torch.log_softmax(s[0].float(), -1)[tid].item())
        toks = [self.tok.decode([i]) for i in new_ids.tolist()]
        digit_idx = [i for i, s in enumerate(toks) if any(c.isdigit() for c in s)]
        out["n_new_tokens"] = len(toks)
        out["has_nan"] = any(math.isnan(v) for v in step_lp)

        if refused or box_m is None:
            out.update(box_model=None, box_orig=None)
            for k in ["box_logprob_mean", "box_logprob_min", "attn_inbox_frac", "attn_inbox_ratio", "verify_crop"]:
                out[k] = None
            out["attn_inbox_per_layer"] = None
        else:
            box_orig = scale_box(box_m, (rW, rH), (W, H))
            out["box_model"], out["box_orig"] = box_m, box_orig
            dl = [step_lp[i] for i in digit_idx] or step_lp
            out["box_logprob_mean"] = float(np.mean(dl))
            out["box_logprob_min"] = float(np.min(dl))

            # ---- localized attention (single eager forward over prompt + generation) ----
            full = seq.unsqueeze(0)
            fwd = {"input_ids": full, "attention_mask": torch.ones_like(full),
                   "pixel_values": inp["pixel_values"], "image_grid_thw": inp["image_grid_thw"]}
            att = self.model(**fwd, output_attentions=True).attentions  # tuple[L] of (1, heads, T, T)
            img_pos = (full[0] == self.image_token_id).nonzero().squeeze(-1)
            mask = torch.from_numpy(token_mask_for_box(box_m, mgh, mgw, self.patch)).to(img_pos.device)
            if img_pos.numel() != mask.numel():
                out["attn_error"] = f"img tokens {img_pos.numel()} != grid {mask.numel()}"
                per_layer = None
            else:
                # query rows = positions that PREDICT each coordinate token
                q = torch.tensor([prompt_len + i - 1 for i in digit_idx], device=img_pos.device)
                per_layer = []
                for a in att:
                    a = a[0].to(img_pos.device).float()    # heads, T, T
                    av = a[:, q][:, :, img_pos]            # heads, nq, n_img
                    tot = av.sum(-1).clamp_min(1e-9)
                    inb = av[:, :, mask].sum(-1)
                    per_layer.append((inb / tot).mean().item())
            del att
            if per_layer is not None:
                lo, hi = self.layer_band
                frac = float(np.mean(per_layer[lo:hi]))
                area = float(mask.float().mean().item())
                out["attn_inbox_per_layer"] = per_layer
                out["attn_inbox_frac"] = frac
                out["attn_inbox_ratio"] = frac / max(area, 1e-6)
                out["box_token_area"] = area
            else:
                out["attn_inbox_per_layer"] = None
                out["attn_inbox_frac"] = out["attn_inbox_ratio"] = None

            out["verify_crop"] = (self._yes_logodds(crop_with_pad(image, box_orig), VERIFY_CROP_PROMPT.format(exp=expression))
                                  if do_verify else None)

        out["verify_full"] = self._yes_logodds(image, VERIFY_FULL_PROMPT.format(exp=expression)) if do_verify else None
        torch.cuda.empty_cache()
        return out

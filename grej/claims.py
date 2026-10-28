"""Atomic-claim verification for grounding rejection.

Idea: a long referring description is ungroundable if ANY of its details is false for the object
the model localized. A single yes/no on the whole description checks the gist and misses one
contradicting detail. So we

  1. split the description into short atomic visual claims (same VLM, text-only prompt),
  2. verify each claim on the crop of the predicted box (yes/no log-odds),
  3. score the grounding by the weakest claim (min), plus mean / second-min for robustness.

Higher score = "the described object is really there".
"""
from __future__ import annotations

import re

import numpy as np

from .geometry import crop_with_pad, parse_box, scale_box

CLAIM_PROMPT = (
    "Break the following object description into short, independent visual facts about the object, "
    "each checkable by looking at it (color, material, shape, parts, text, position, state). "
    "Write one fact per line, at most {k} facts, no numbering, no extra text.\n\n"
    'Description: "{desc}"'
)
CLAIM_VERIFY_PROMPT = 'Is the following true of the object shown in this image: "{claim}"? Answer yes or no.'
GROUND_PROMPT_PLAIN = (
    'Locate "{exp}" in the image and output its bbox coordinates in JSON format. '
    'If there is no such object in the image, output "None".'
)

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def clean_claims(text: str, desc: str, k: int) -> list[str]:
    out, seen = [], set()
    for line in text.splitlines():
        c = _BULLET.sub("", line).strip().strip('"').strip()
        if len(c) < 3 or c.lower().startswith(("description", "facts", "here are")):
            continue
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
        if len(out) >= k:
            break
    return out or [desc]


class ClaimVerifier:
    def __init__(self, grounder, max_claims: int = 8):
        self.g = grounder
        self.k = max_claims

    def _text_generate(self, prompt: str, max_new_tokens: int = 160) -> str:
        import torch

        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.g.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.g.proc(text=[text], return_tensors="pt")
        inp = {k: v.to(self.g.device) for k, v in inp.items()}
        with torch.no_grad():
            out = self.g.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.g.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)

    def ground(self, image, expression: str):
        import torch

        inp = self.g._inputs(image, GROUND_PROMPT_PLAIN.format(exp=expression))
        n = inp["input_ids"].shape[1]
        with torch.no_grad():
            seq = self.g.model.generate(**inp, max_new_tokens=72, do_sample=False)[0]
        text = self.g.tok.decode(seq[n:], skip_special_tokens=True)
        box_m, refused = parse_box(text)
        _, gh, gw = [int(v) for v in inp["image_grid_thw"][0].tolist()]
        unit = self.g.patch // self.g.merge
        rW, rH = gw * unit, gh * unit
        box = scale_box(box_m, (rW, rH), image.size) if box_m else None
        return text, box, refused

    def run(self, image, expression: str) -> dict:
        raw, box, refused = self.ground(image, expression)
        W, H = image.size
        rec = {"raw_output": raw, "refused": refused, "box_orig": box, "orig_wh": [W, H],
               "desc_words": len(expression.split())}
        claims = clean_claims(self._text_generate(CLAIM_PROMPT.format(k=self.k, desc=expression)), expression, self.k)
        rec["claims"] = claims
        # whole-description baselines (same prompts as run_diagnostic)
        from .model import VERIFY_CROP_PROMPT, VERIFY_FULL_PROMPT
        rec["verify_full"] = self.g._yes_logodds(image, VERIFY_FULL_PROMPT.format(exp=expression))
        if box is None:
            rec.update(verify_crop=None, claim_scores=None, claim_min=None, claim_mean=None,
                       claim_min2=None, box_area_frac=None)
            return rec
        rec["box_area_frac"] = max(0.0, (box[2] - box[0]) * (box[3] - box[1])) / float(W * H)
        crop = crop_with_pad(image, box)
        rec["verify_crop"] = self.g._yes_logodds(crop, VERIFY_CROP_PROMPT.format(exp=expression))
        scores = [self.g._yes_logodds(crop, CLAIM_VERIFY_PROMPT.format(claim=c)) for c in claims]
        s = np.sort(np.array(scores, dtype=float))
        rec["claim_scores"] = [float(v) for v in scores]
        rec["claim_min"] = float(s[0])
        rec["claim_min2"] = float(s[1]) if len(s) > 1 else float(s[0])
        rec["claim_mean"] = float(s.mean())
        import torch

        torch.cuda.empty_cache()
        return rec

# Related work and gap analysis: grounding rejection in VLMs

*Last updated 2026-10-27.*

**Question.** A user asks a vision-language model (VLM) to locate "the red cup left of the laptop". The image contains a *blue* cup, or no cup at all. Does the model refuse, or does it confidently draw a box? And can we detect when it is wrong, with a guarantee?

## 1. The problem is real and measured

| Work | Venue | Finding |
|---|---|---|
| **GroundingME** (Li et al., arXiv 2512.17495) | CVPR 2026 | 1,005 hard grounding cases across four dimensions (Discriminative, Spatial, Limited, Rejection) and 25+ MLLMs. Best overall accuracy is 45.1%. **Most models score 0% on Rejection**, "reflexively hallucinating objects". The best Rejection scores on the leaderboard are about 9.5% (Qwen3-VL-32B-Thinking), 7.0% (Gemini-2.5-Pro), 0.5% (Qwen2.5-VL-7B). A data-mixture SFT fix reaches only **27.9%**. They release RefCOCOg_rej (41.5k paired positive/negative expressions). |
| **GREC / gRefCOCO** (He et al., arXiv 2308.16182) | 2023 | Introduced no-target referring expressions for *specialist* detectors. MLLMs were not evaluated. Its negatives are mostly easy (unrelated). |
| **Uncertainty-aware decision making in MLLMs** (survey, arXiv 2608.17084) | 2026 | Lists as open: abstention "not tied to visual sufficiency … or multimodal grounding". Also calls for separating perception, grounding and answerability, and notes "evidence collapse": low-entropy but visually ungrounded answers. |

## 2. Closest prior methods and what each leaves open

| Work | What it does | What it does **not** do |
|---|---|---|
| **MTLA: Propose and Attend** (arXiv 2607.05978, Jul 2026) | Training-free confidence score from how strongly box tokens attend *inside the claimed region*. Uses Qwen3-VL-8B, Gemma-4, Audio Flamingo 3. Hallucination AUROC +7 to +38 over prior training-free baselines. COCO zero-shot AP 20.4 → 37.0 (with N=16 self-consistency). | Scores a box the model already drew; it never asks whether *the expression* matches anything. Evaluated on COCO detection, Charades-STA and QVHighlights, **not** on referring expressions with absent targets or attribute-flipped hard negatives. No statistical guarantee. Needs eager attention. |
| **BCEA: Look Again Before You Abstain** (arXiv 2606.16667) | Conformal selective-risk guarantee for **object-existence / relation claims** (POPE, COCO). Uses a likelihood ratio against a blank image plus "acquire more evidence" via crops or flips. | No bounding boxes and no referring-expression grounding. Shows that abstention alone must refuse >80% of claims to keep hallucination below 5%. |
| **RA-RFT: Learning to Refuse** (arXiv 2511.23151) | GRPO post-training so video temporal-grounding models refuse "hard-irrelevant" queries (7B models, 8×A100). | Temporal segments only, no spatial boxes. Refusal is generated text, with no calibration or guarantee. Training and eval data come from the same GPT pipeline. |
| **Query-insensitive STVG** (arXiv 2610.06018, Oct 2026) | Shows that spatio-temporal grounding models (TubeDETR, TA-STVG) still output tubes for irrelevant queries. | Specialist models only, and none of the proposed fixes were tested. Points to the video extension of this project. |
| Moment of Untruth (2502.08544); open-set VMR (2605.29812) | Negative queries in video moment retrieval. | Temporal, specialist models. |
| Post-hoc conformal for POPE (Zenodo 23180934); conformal prediction sets for LVLMs (2504.17671); statistical factuality (2502.20560) | Conformal guarantees for VQA-style or existence answers. | Not grounding, not box-level. |

## 3. The gap this project targets

1. **No one has tested whether training-free grounding-confidence signals (MTLA-style attention, token log-probs) can tell a correct referent from an attribute-flipped one.** In RefCOCOg_rej the negative expression describes the *same object* with one attribute changed. Our hypothesis (H1): the model draws the same box and attends to the same pixels, so localized-attention and log-prob signals carry almost no information. Hard negatives would be exactly where MTLA-style signals fail.
2. **Verification that is decoupled from localization.** Hypothesis H2: asking the model to verify the expression on *only the crop it chose* recovers a strong reject signal, without training. The intuition is that generation ("find the red cup") is anchored by the category, while verification ("is this a red cup?") forces attribute checking.
3. **No distribution-free guarantee exists for accepting a grounding.** We calibrate an accept threshold with split conformal so that the false-accept rate on ungroundable queries stays at or below α. We then report how many true groundings survive (TAR at FAR ≤ α). This is the operating point a robot or agent actually needs.

## 4. Plan
- **Phase 1 (this repo, Kaggle T4).** Diagnostic on Qwen2.5-VL-3B with RefCOCOg_rej validation (paired) and the GroundingME test set: refusal rates, grounding IoU, AUROC per signal, paired box-reuse rate, conformal FAR/TAR. Then repeat on Qwen2.5-VL-7B (4-bit) and Qwen3-VL-4B to show it generalizes.
- **Phase 2.** Ablations: layer band, crop padding, prompt wording. Breakdown by GroundingME L1/L2 category. Fusing signals with cross-fitting while keeping conformal validity.
- **Phase 3 (optional).** LoRA on 3B with RefCOCOg_rej train as a trained baseline, compared with the training-free route; video extension (STVG).

## References
- GroundingME. https://arxiv.org/abs/2512.17495 · https://groundingme.github.io · HF `lirang04/GroundingME`, `lirang04/RefCOCOg_rej`
- MTLA. https://arxiv.org/abs/2607.05978
- BCEA. https://arxiv.org/abs/2606.16667
- RA-RFT. https://arxiv.org/abs/2511.23151
- Query-insensitive STVG. https://arxiv.org/abs/2610.06018
- MLLM uncertainty survey. https://arxiv.org/abs/2608.17084
- GREC. https://arxiv.org/abs/2308.16182

# Results

## Run 1: Qwen2.5-VL-3B-Instruct, Kaggle 2×T4 (fp16, max 640 visual tokens), 2026-10-27

### RefCOCOg_rej validation: 300 paired queries (real expression vs one-attribute-edited expression, same image and object)

**Sanity check.** Groundable expressions are localized correctly (IoU ≥ 0.5) in **87.7%** of cases (mean IoU 0.811).

**H1: the model does not refuse, and generation-side confidence does not detect it. Confirmed.**
- Refusal rate on ungroundable expressions: **0.3%** (1/300). On groundable: 0.0%.
- The edited (non-existent) expression receives essentially the same box as the real one (IoU ≥ 0.7) in **81.9%** of pairs.

| signal | AUROC (all pos vs neg) | AUROC (correct pos vs neg) |
|---|---|---|
| emitted a box (own decision) | 0.502 | 0.502 |
| box-token log-prob, mean | 0.545 | 0.559 |
| box-token log-prob, min | 0.516 | 0.524 |
| localized attention in box (MTLA-style) | 0.521 | 0.531 |
| localized attention, area-normalized | 0.482 | 0.482 |
| **yes/no verification on predicted crop** | **0.883** | **0.900** |
| **yes/no verification on full image** | **0.919** | **0.931** |

Paired: pos > neg in 74.6% of pairs for attention (median difference +0.0007, i.e. negligible), 72.6% for box log-prob, **92.6% for crop verification (+2.80 logit), 97.7% for full-image verification (+3.36 logit)**.

**H2: decoupled verification recovers the signal. Partly confirmed.** Asking the same model a yes/no question separates real from edited expressions well. Full-image verification beats crop verification, so the gain comes from decoupling verification from box generation, not from cropping. The model "knows but draws anyway": its own verifier prefers the real expression in 97.7% of pairs, yet it refuses only 0.3% of the time.

**H3: conformal false-accept control. Holds.** 200 random 50/50 calibration/test splits; accept iff score > τ.

| signal | target α | test FAR | TAR (true groundings kept) |
|---|---|---|---|
| full-image verification | 0.05 | 0.048 ± 0.024 | **0.667 ± 0.035** |
| full-image verification | 0.10 | 0.099 ± 0.034 | **0.744 ± 0.034** |
| crop verification | 0.05 | 0.049 ± 0.025 | 0.575 ± 0.041 |
| crop verification | 0.10 | 0.101 ± 0.032 | 0.677 ± 0.039 |
| localized attention | 0.05 | 0.043 ± 0.023 | 0.047 ± 0.019 |

**Caveats.**
- One model and 300 pairs. Negatives are mostly single-attribute (often color) edits.
- Full-image verification checks existence, not *where* the box is. It must be combined with a localization-aware check.
- Existence verification itself is close to BCEA (arXiv 2606.16667). What is new here is the generation/verification dissociation on box grounding and the failure of localized-attention scores on hard negatives.

### GroundingME (1,005 queries): invalid, re-run pending
The first analysis read GroundingME boxes as `xywh`. The raw dataset stores `xyxy` in original pixels (e.g. id 0: `[1376, 839, 1425, 976]` in a 1500×2000 image). Images (up to ~7.7k px) were also capped at ~0.5 MP, which makes the small targets a few tokens wide. The fixes in this commit:
- fix the box format per dataset;
- verify decoded image size against the dataset width/height, applying EXIF orientation if needed;
- report results by L1 category;
- run GroundingME at ~1 MP.

Observed before the fix, and still valid: refusal rate on the Rejection category was **0.0%** (0/201).

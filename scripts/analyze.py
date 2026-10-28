"""Summarise diagnostic results: refusal rates, grounding IoU, signal AUROCs, conformal FAR/TAR.

  python scripts/analyze.py results/rcg_qwen25vl3b.jsonl [results/gme_qwen25vl3b.jsonl ...]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grej.geometry import iou, xywh_to_xyxy  # noqa: E402
from grej.metrics import auroc, conformal_eval  # noqa: E402

SIGNALS = ["output_box", "box_logprob_mean", "box_logprob_min", "attn_inbox_frac", "attn_inbox_ratio",
           "verify_crop", "verify_full"]
REFUSED_SCORE = -1e9  # a refusal is the strongest possible "reject"


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def gt_iou(r, fmt):
    if not r.get("box_orig") or not r.get("gt_bbox"):
        return None
    gt = xywh_to_xyxy(r["gt_bbox"]) if fmt == "xywh" else r["gt_bbox"]
    return iou(r["box_orig"], gt)


# Verified against the raw datasets (2026-10-28): both store [x1, y1, x2, y2] in original-image pixels.
# e.g. GroundingME id 0: bbox [1376, 839, 1425, 976] in a 1500x2000 image (xywh would overflow the image).
BBOX_FORMAT = {"refcocog_rej": "xyxy", "groundingme": "xyxy"}


def pick_bbox_format(rows):
    """Known per-dataset GT box format; falls back to the better-agreeing format for unknown sources."""
    srcs = {r.get("source") for r in rows}
    if len(srcs) == 1 and next(iter(srcs)) in BBOX_FORMAT:
        return BBOX_FORMAT[next(iter(srcs))]
    best, best_v = "xywh", -1
    for fmt in ["xywh", "xyxy"]:
        v = [gt_iou(r, fmt) for r in rows if r.get("groundable")]
        v = [x for x in v if x is not None]
        m = float(np.median(v)) if v else 0
        if m > best_v:
            best, best_v = fmt, m
    return best


def score(r, s):
    if r.get("refused"):
        return REFUSED_SCORE if s != "verify_full" else (r.get(s) if r.get(s) is not None else REFUSED_SCORE)
    v = r.get(s)
    return REFUSED_SCORE if v is None else float(v)


def report(rows, title):
    rows = [r for r in rows if "error" not in r]
    pos = [r for r in rows if r["groundable"]]
    neg = [r for r in rows if not r["groundable"]]
    print(f"\n=== {title}: {len(pos)} groundable / {len(neg)} ungroundable ===")
    if not pos or not neg:
        print("need both classes")
        return {}
    fmt = pick_bbox_format(rows)
    ious = [gt_iou(r, fmt) for r in pos]
    ious_v = [x if x is not None else 0.0 for x in ious]
    print(f"GT bbox format: {fmt}")
    print(f"Refusal rate  groundable: {np.mean([r['refused'] for r in pos]):.3f}   "
          f"ungroundable: {np.mean([r['refused'] for r in neg]):.3f}")
    print(f"Grounding acc@0.5 on groundable: {np.mean([x >= 0.5 for x in ious_v]):.3f}  (mean IoU {np.mean(ious_v):.3f})")
    nan_rate = np.mean([bool(r.get('has_nan')) for r in rows])
    if nan_rate:
        print(f"WARNING: NaN log-probs in {nan_rate:.1%} of queries (fp16 overflow?)")

    # strict positives: groundable AND correctly localised
    strict = [r for r, x in zip(pos, ious_v) if x >= 0.5]
    out = {}
    print(f"\n{'signal':<20}{'AUROC(all pos vs neg)':>24}{'AUROC(correct pos vs neg)':>28}")
    for s in SIGNALS:
        a1 = auroc([score(r, s) for r in pos], [score(r, s) for r in neg])
        a2 = auroc([score(r, s) for r in strict], [score(r, s) for r in neg]) if strict else float("nan")
        out[s] = (a1, a2)
        print(f"{s:<20}{a1:>24.3f}{a2:>28.3f}")

    # paired analysis (RefCOCOg_rej): does the model put the SAME box on the negative?
    by_pair = defaultdict(dict)
    for r in rows:
        if r.get("pair_id"):
            by_pair[r["pair_id"]]["pos" if r["groundable"] else "neg"] = r
    pairs = [v for v in by_pair.values() if "pos" in v and "neg" in v]
    if pairs:
        same = [iou(p["pos"]["box_orig"], p["neg"]["box_orig"]) >= 0.7
                for p in pairs if p["pos"].get("box_orig") and p["neg"].get("box_orig")]
        print(f"\nPaired: negative gets ~same box as positive (IoU>=0.7) in {np.mean(same):.3f} of {len(same)} pairs")
        for s in ["attn_inbox_frac", "verify_crop", "verify_full", "box_logprob_mean"]:
            d = [score(p["pos"], s) - score(p["neg"], s) for p in pairs
                 if score(p["pos"], s) > REFUSED_SCORE and score(p["neg"], s) > REFUSED_SCORE]
            if d:
                print(f"  {s:<18} pos>neg in {np.mean(np.array(d) > 0):.3f} of {len(d)} pairs (median diff {np.median(d):+.4f})")

    # conformal false-accept control
    print("\nConformal false-accept control (accept iff score > tau; 200 random 50/50 splits)")
    for s in ["verify_crop", "verify_full", "attn_inbox_frac", "box_logprob_mean"]:
        for alpha in [0.05, 0.10]:
            c = conformal_eval([score(r, s) for r in pos], [score(r, s) for r in neg], alpha)
            print(f"  {s:<18} alpha={alpha:.2f}  FAR={c['FAR_mean']:.3f}±{c['FAR_std']:.3f}  "
                  f"TAR={c['TAR_mean']:.3f}±{c['TAR_std']:.3f}")
    return out


def groundingme_breakdown(sub):
    sub = [r for r in sub if "error" not in r]
    mism = [r for r in sub if (r.get("meta") or {}).get("size_mismatch")]
    if mism:
        print(f"\nImage size != dataset width/height in {len(mism)} rows (EXIF-corrected: "
              f"{sum(1 for r in mism if r['meta'].get('exif_fixed'))})")
    rej = [r for r in sub if not r["groundable"]]
    print("\nGroundingME by L1 category (acc@0.5 uses xyxy GT):")
    print(f"  {'category':<16}{'n':>5}{'refused':>9}{'acc@0.5':>9}"
          f"{'AUROC verify_full':>19}{'AUROC attn':>12}   (correct pos in category vs all Rejection)")
    for c in sorted({(r.get("meta") or {}).get("subtask_l1") for r in sub} - {None}):
        cr = [r for r in sub if (r.get("meta") or {}).get("subtask_l1") == c]
        ious = [gt_iou(r, "xyxy") or 0.0 for r in cr if r["groundable"]]
        acc = np.mean([x >= 0.5 for x in ious]) if ious else float("nan")
        corr = [r for r in cr if r["groundable"] and (gt_iou(r, "xyxy") or 0) >= 0.5]
        a1 = auroc([score(r, "verify_full") for r in corr], [score(r, "verify_full") for r in rej]) if corr else float("nan")
        a2 = auroc([score(r, "attn_inbox_frac") for r in corr], [score(r, "attn_inbox_frac") for r in rej]) if corr else float("nan")
        print(f"  {c:<16}{len(cr):>5}{np.mean([r['refused'] for r in cr]):>9.3f}{acc:>9.3f}{a1:>19.3f}{a2:>12.3f}   (n_correct={len(corr)})")


def main():
    rows = load(sys.argv[1:])
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        report(sub, src)
        if src == "groundingme":
            groundingme_breakdown(sub)
    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"\n{len(errs)} errors; first: {errs[0]['error']}")


if __name__ == "__main__":
    main()

"""Analyze atomic-claim runs: AUROC per signal (incl. confound baselines), per category, conformal, examples.

  python scripts/analyze_claims.py results/gme_claims.jsonl [results/rcg_claims.jsonl]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grej.geometry import iou  # noqa: E402
from grej.metrics import auroc, conformal_eval  # noqa: E402

REJ = -1e9
SIGNALS = ["verify_full", "verify_crop", "claim_mean", "claim_min2", "claim_min"]
CONFOUNDS = ["box_area_frac", "desc_words", "n_claims"]


def val(r, s):
    if s == "n_claims":
        return float(len(r.get("claims") or []))
    if s == "desc_words":
        return float(r.get("desc_words") or 0)
    if r.get("refused") and s != "verify_full":
        return REJ
    v = r.get(s)
    return REJ if v is None else float(v)


def correct(r):
    return bool(r.get("box_orig") and r.get("gt_bbox") and iou(r["box_orig"], r["gt_bbox"]) >= 0.5)


def table(pos, neg, title):
    print(f"\n{title}: {len(pos)} pos / {len(neg)} neg")
    for s in SIGNALS + CONFOUNDS:
        a = auroc([val(r, s) for r in pos], [val(r, s) for r in neg])
        tag = "  (confound check: should be ~0.5)" if s in CONFOUNDS else ""
        print(f"  {s:<14} AUROC {a:.3f}{tag}")


def main():
    rows = [json.loads(l) for p in sys.argv[1:] for l in open(p) if l.strip()]
    errs = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    if errs:
        print(f"{len(errs)} errors; first: {errs[0]['error']}")
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        pos = [r for r in sub if r["groundable"]]
        neg = [r for r in sub if not r["groundable"]]
        cor = [r for r in pos if correct(r)]
        print(f"\n===== {src} =====  acc@0.5={len(cor) / max(1, len(pos)):.3f}  refused(neg)={np.mean([r['refused'] for r in neg]):.3f}")
        print(f"desc words: pos mean {np.mean([r['desc_words'] for r in pos]):.1f}  neg mean {np.mean([r['desc_words'] for r in neg]):.1f}; "
              f"claims/query: {np.mean([len(r.get('claims') or []) for r in sub]):.1f}")
        table(pos, neg, "all groundable vs ungroundable")
        table(cor, neg, "correctly localized vs ungroundable")
        if src == "groundingme":
            for c in sorted({(r.get("meta") or {}).get("subtask_l1") for r in pos} - {None}):
                cp = [r for r in cor if r["meta"].get("subtask_l1") == c]
                a = {s: auroc([val(r, s) for r in cp], [val(r, s) for r in neg]) for s in ["verify_full", "claim_min", "claim_mean"]}
                print(f"  [{c:<14}] n_correct={len(cp):<4} " + "  ".join(f"{k}={v:.3f}" for k, v in a.items()))
        print("\n  conformal (accept iff score > tau), 200 splits:")
        for s in ["verify_full", "claim_mean", "claim_min"]:
            for al in (0.05, 0.10):
                c = conformal_eval([val(r, s) for r in pos], [val(r, s) for r in neg], al)
                print(f"    {s:<12} a={al:.2f} FAR={c['FAR_mean']:.3f}  TAR={c['TAR_mean']:.3f}")
        print("\n  examples (ungroundable): weakest claim per query")
        for r in neg[:6]:
            sc = r.get("claim_scores") or []
            if sc:
                j = int(np.argmin(sc))
                print(f"   - {r['qid']}: {r['expression'][:160]!r}\n       weakest: {r['claims'][j]!r} ({sc[j]:+.2f}); "
                      f"verify_full {r['verify_full']:+.2f}; n_claims {len(sc)}")


if __name__ == "__main__":
    main()

"""Atomic-claim verification run (resumable, one JSON line per query).

  python scripts/run_claims.py --dataset groundingme --n 0 --max_pixels 1003520 --out results/gme_claims.jsonl
  python scripts/run_claims.py --dataset refcocog_rej --n 300 --device_map cuda:1 --out results/rcg_claims.jsonl
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grej.claims import ClaimVerifier  # noqa: E402
from grej.data import iter_groundingme, iter_refcocog_rej  # noqa: E402
from grej.model import Grounder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["refcocog_rej", "groundingme"], required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device_map", default="cuda:0")
    ap.add_argument("--max_pixels", type=int, default=640 * 28 * 28)
    ap.add_argument("--max_claims", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["qid"])
            except Exception:
                pass
    g = Grounder(a.model, dtype=a.dtype, max_pixels=a.max_pixels, device_map=a.device_map,
                 attn_implementation="sdpa")
    cv = ClaimVerifier(g, max_claims=a.max_claims)
    it = iter_refcocog_rej(a.n) if a.dataset == "refcocog_rej" else iter_groundingme(limit=a.n if a.n > 0 else None)
    t0, k = time.time(), 0
    with open(a.out, "a") as f:
        for q in it:
            if q.qid in done:
                continue
            try:
                r = cv.run(q.image, q.expression)
            except Exception as e:
                r = {"error": repr(e), "trace": traceback.format_exc()[-1200:]}
            f.write(json.dumps({"qid": q.qid, "pair_id": q.pair_id, "source": q.source, "expression": q.expression,
                                "groundable": q.groundable, "gt_bbox": q.gt_bbox, "meta": q.meta, **r}) + "\n")
            f.flush()
            k += 1
            if k % 25 == 0 or k <= 2:
                print(f"[{k}] {q.qid} {(time.time() - t0) / k:.1f}s/q claims={len(r.get('claims') or [])} "
                      f"err={r.get('error')}", flush=True)
    print(f"[done] {k} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

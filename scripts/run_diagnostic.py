"""Run the grounding-rejection diagnostic and append one JSON line per query.

Resumable: queries whose qid is already in --out are skipped.

Example (Kaggle T4):
  python scripts/run_diagnostic.py --dataset refcocog_rej --n 300 --out results/rcg_qwen25vl3b.jsonl
  python scripts/run_diagnostic.py --dataset groundingme --out results/gme_qwen25vl3b.jsonl
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grej.data import iter_groundingme, iter_refcocog_rej  # noqa: E402
from grej.model import Grounder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["refcocog_rej", "groundingme"], required=True)
    ap.add_argument("--n", type=int, default=300, help="pairs (refcocog_rej) or rows (groundingme)")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device_map", default="cuda:0")
    ap.add_argument("--max_pixels", type=int, default=640 * 28 * 28)
    ap.add_argument("--no_verify", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["qid"])
                except Exception:
                    pass
    print(f"[resume] {len(done)} queries already done", flush=True)

    g = Grounder(args.model, dtype=args.dtype, max_pixels=args.max_pixels, device_map=args.device_map)
    print(f"[model] {args.model} layers={g.n_layers} band={g.layer_band} yes={g.yes_ids} no={g.no_ids}", flush=True)

    it = (iter_refcocog_rej(args.n, skip=args.skip) if args.dataset == "refcocog_rej"
          else iter_groundingme(limit=args.n if args.n > 0 else None))
    t0, k = time.time(), 0
    with open(args.out, "a") as f:
        for q in it:
            if q.qid in done:
                continue
            try:
                r = g.run(q.image, q.expression, do_verify=not args.no_verify)
            except Exception as e:  # keep going; record the failure
                r = {"error": repr(e), "trace": traceback.format_exc()[-1500:]}
            rec = {"qid": q.qid, "pair_id": q.pair_id, "source": q.source, "expression": q.expression,
                   "groundable": q.groundable, "gt_bbox": q.gt_bbox, "meta": q.meta, "model": args.model, **r}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            k += 1
            if k % 10 == 0 or k <= 3:
                dt = (time.time() - t0) / k
                print(f"[{k}] {q.qid} groundable={q.groundable} refused={r.get('refused')} "
                      f"out={str(r.get('raw_output'))[:80]!r} {dt:.1f}s/q", flush=True)
    print(f"[done] {k} new queries in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

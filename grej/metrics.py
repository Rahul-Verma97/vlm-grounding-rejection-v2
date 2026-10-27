"""AUROC and split-conformal false-accept control."""
from __future__ import annotations

import math

import numpy as np


def auroc(scores_pos, scores_neg) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie). Higher score = 'accept / groundable'."""
    p = np.asarray(scores_pos, dtype=float)
    n = np.asarray(scores_neg, dtype=float)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    # average ranks for ties
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    rp = ranks[: len(p)].sum()
    return float((rp - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def conformal_threshold(neg_cal, alpha: float) -> float:
    """Split-conformal threshold on ungroundable-query scores.

    Accept iff score > tau.  For an exchangeable new ungroundable query,
    P(score > tau) <= alpha  (marginal, finite-sample)."""
    s = np.sort(np.asarray(neg_cal, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def conformal_eval(pos, neg, alpha: float, n_splits: int = 200, cal_frac: float = 0.5, seed: int = 0):
    """Random calibration/test splits. Returns mean/std of test false-accept rate (FAR)
    on ungroundable queries and true-accept rate (TAR) on groundable ones."""
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    fars, tars = [], []
    for _ in range(n_splits):
        idx = rng.permutation(len(neg))
        ncal = int(len(neg) * cal_frac)
        tau = conformal_threshold(neg[idx[:ncal]], alpha)
        fars.append(float(np.mean(neg[idx[ncal:]] > tau)))
        pidx = rng.permutation(len(pos))[: max(1, len(pos) - int(len(pos) * cal_frac))]
        tars.append(float(np.mean(pos[pidx] > tau)))
    return {"alpha": alpha, "FAR_mean": float(np.mean(fars)), "FAR_std": float(np.std(fars)),
            "FAR_violation_rate": float(np.mean(np.array(fars) > alpha)),
            "TAR_mean": float(np.mean(tars)), "TAR_std": float(np.std(tars))}

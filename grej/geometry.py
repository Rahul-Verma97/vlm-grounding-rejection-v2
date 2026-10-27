"""Box parsing, coordinate conversion, IoU and image-token masks."""
from __future__ import annotations

import json
import re
from typing import Optional

import numpy as np

_NONE_PAT = re.compile(r"\b(none|no such|not (present|exist|found|visible)|there is no|doesn't exist|does not exist)\b", re.I)
_BOX_PAT = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def explicit_refusal(text: str) -> bool:
    return bool(_NONE_PAT.search(text))


def parse_box(text: str) -> tuple[Optional[list[float]], bool]:
    """Return (box_xyxy_or_None, refused). Box is in the model's coordinate frame.
    Any output without a parsable box counts as a refusal."""
    m = _BOX_PAT.search(text)
    if m:
        box = [float(v) for v in m.groups()]
        x1, y1, x2, y2 = box
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [x1, y1, x2, y2], False
    return None, True


def scale_box(box, from_wh, to_wh):
    sx = to_wh[0] / from_wh[0]
    sy = to_wh[1] / from_wh[1]
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def xywh_to_xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def token_mask_for_box(box_resized, grid_h: int, grid_w: int, patch: int) -> np.ndarray:
    """Boolean mask over the (grid_h x grid_w) merged visual-token grid (row-major)
    marking tokens whose cell overlaps the box (box in resized-image pixels)."""
    x1, y1, x2, y2 = box_resized
    c1 = int(np.clip(np.floor(x1 / patch), 0, grid_w - 1))
    c2 = int(np.clip(np.ceil(x2 / patch) - 1, 0, grid_w - 1))
    r1 = int(np.clip(np.floor(y1 / patch), 0, grid_h - 1))
    r2 = int(np.clip(np.ceil(y2 / patch) - 1, 0, grid_h - 1))
    m = np.zeros((grid_h, grid_w), dtype=bool)
    if c2 >= c1 and r2 >= r1:
        m[r1:r2 + 1, c1:c2 + 1] = True
    return m.reshape(-1)


def crop_with_pad(img, box_xyxy, pad: float = 0.1, min_side: int = 56):
    W, H = img.size
    x1, y1, x2, y2 = box_xyxy
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1, x2 = x1 - pad * w, x2 + pad * w
    y1, y2 = y1 - pad * h, y2 + pad * h
    # enforce a minimum crop size so tiny boxes stay readable
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = max((x2 - x1) / 2, min_side / 2), max((y2 - y1) / 2, min_side / 2)
    box = [max(0, cx - hw), max(0, cy - hh), min(W, cx + hw), min(H, cy + hh)]
    return img.crop(tuple(int(round(v)) for v in box))

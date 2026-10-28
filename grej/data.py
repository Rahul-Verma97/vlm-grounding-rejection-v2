"""Dataset loaders.

RefCOCOg_rej (lirang04/RefCOCOg_rej): each row has one image, one target box, the
original referring expression (groundable) and a minimally edited "rejected"
expression (usually one attribute changed) that should NOT ground to anything.
This gives a perfectly paired, controlled test: same image, same object, one word
different.

GroundingME (lirang04/GroundingME, CVPR 2026): 1,005 hard grounding cases in four
L1 categories; the Rejection category has bbox = None.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from PIL import Image


@dataclass
class Query:
    qid: str
    image: Image.Image
    expression: str
    groundable: bool           # True if a matching object exists
    gt_bbox: Optional[list]    # raw bbox as stored in the dataset (format resolved in analysis)
    source: str
    pair_id: Optional[str] = None
    meta: dict = field(default_factory=dict)


def _rgb(img) -> Image.Image:
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    return img.convert("RGB")


def _load_checked(raw, width, height):
    """Decode an image and make sure it is in the same pixel frame as the dataset's boxes.

    Returns (image, info). If the decoded size differs from the stored width/height, try applying
    the EXIF orientation; record what happened so analysis can flag affected rows."""
    from PIL import ImageOps

    img = raw if isinstance(raw, Image.Image) else Image.open(raw)
    info = {"decoded_wh": list(img.size), "size_mismatch": False, "exif_fixed": False}
    if width and height and tuple(img.size) != (int(width), int(height)):
        info["size_mismatch"] = True
        t = ImageOps.exif_transpose(img)
        if tuple(t.size) == (int(width), int(height)):
            img, info["exif_fixed"] = t, True
    return img.convert("RGB"), info


def iter_refcocog_rej(n_pairs: int, split: str = "validation", skip: int = 0) -> Iterator[Query]:
    from datasets import load_dataset

    ds = load_dataset("lirang04/RefCOCOg_rej", split=split, streaming=True)
    for i, row in enumerate(ds):
        if i < skip:
            continue
        if i >= skip + n_pairs:
            break
        img, info = _load_checked(row["image"], row.get("width"), row.get("height"))
        pid = f"rcg{row['id']}"
        common = dict(image=img, gt_bbox=list(row["original_bbox"]), source="refcocog_rej", pair_id=pid,
                      meta={"width": row.get("width"), "height": row.get("height"),
                            "original_exp": row["original_exp"], "rejected_exp": row["rejected_exp"], **info})
        yield Query(qid=pid + "_pos", expression=row["original_exp"], groundable=True, **common)
        yield Query(qid=pid + "_neg", expression=row["rejected_exp"], groundable=False, **common)


def iter_groundingme(limit: Optional[int] = None) -> Iterator[Query]:
    from datasets import load_dataset

    ds = load_dataset("lirang04/GroundingME", split="test", streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        img, info = _load_checked(row["image"], row.get("width"), row.get("height"))
        bbox = row.get("bbox")
        groundable = bbox is not None and len(bbox) == 4
        yield Query(
            qid=f"gme{row['id']}", image=img, expression=row["description"],
            groundable=groundable, gt_bbox=list(bbox) if groundable else None, source="groundingme",
            meta={"subtask_l1": row.get("subtask_l1"), "subtask_l2": row.get("subtask_l2"),
                  "detection_type": row.get("detection_type"),
                  "width": row.get("width"), "height": row.get("height"), **info},
        )

import json, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grej.geometry import parse_box, iou, token_mask_for_box, scale_box, crop_with_pad, explicit_refusal
from grej.metrics import auroc, conformal_threshold, conformal_eval

def test_parse():
    b, r = parse_box('```json\n[{"bbox_2d": [10, 20, 110, 220], "label": "cup"}]\n```')
    assert b == [10, 20, 110, 220] and not r
    b, r = parse_box("None"); assert b is None and r and explicit_refusal("None")
    b, r = parse_box("[50, 60, 10, 5]"); assert b == [10, 5, 50, 60]

def test_iou():
    assert abs(iou([0,0,10,10],[0,0,10,10]) - 1) < 1e-9
    assert abs(iou([0,0,10,10],[5,0,15,10]) - 1/3) < 1e-9
    assert iou([0,0,1,1],[2,2,3,3]) == 0

def test_mask():
    m = token_mask_for_box([28, 0, 84, 28], 3, 4, 28).reshape(3, 4)
    assert m[0].tolist() == [False, True, True, False] and not m[1:].any()
    assert scale_box([10,10,20,20],(100,100),(200,50)) == [20,5,40,10]
    c = crop_with_pad(Image.new("RGB",(100,100)), [40,40,42,42]); assert min(c.size) >= 56

def test_metrics():
    assert auroc([2,3],[0,1]) == 1.0 and auroc([0],[1]) == 0.0 and auroc([1],[1]) == 0.5
    rng = np.random.default_rng(0)
    neg = rng.normal(0,1,2000); pos = rng.normal(2,1,2000)
    from sklearn.metrics import roc_auc_score
    a = auroc(pos, neg); b = roc_auc_score([1]*2000+[0]*2000, np.r_[pos,neg]); assert abs(a-b)<1e-9
    c = conformal_eval(pos, neg, 0.1, n_splits=100)
    assert c["FAR_mean"] <= 0.11, c        # validity
    assert conformal_threshold([1,2,3], 0.01) == float("inf")

def test_analyze_end_to_end():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(60):
        for g in (True, False):
            refused = (not g) and rng.random() < 0.1
            rows.append(dict(qid=f"p{i}{g}", pair_id=f"p{i}", source="refcocog_rej", expression="x", groundable=g,
                gt_bbox=[10,10,60,60], meta={}, model="m", refused=refused, has_nan=False,
                box_orig=None if refused else [10,10,60,60], output_box=0 if refused else 1,
                box_logprob_mean=None if refused else rng.normal(-.1,.05), box_logprob_min=None if refused else -0.5,
                attn_inbox_frac=None if refused else rng.normal(.3,.05), attn_inbox_ratio=None if refused else 2.0,
                verify_crop=None if refused else rng.normal(2 if g else -1, 1), verify_full=rng.normal(1 if g else 0,1)))
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    out = subprocess.run([sys.executable, "scripts/analyze.py", f.name], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "GT bbox format: xyxy" in out.stdout and "verify_crop" in out.stdout
    print(out.stdout)

def test_load_checked_exif():
    from PIL import Image
    from grej.data import _load_checked
    img = Image.new("RGB", (40, 20))
    exif = img.getexif(); exif[0x0112] = 6  # rotate 90 CW on display
    import io; b = io.BytesIO(); img.save(b, "JPEG", exif=exif); b.seek(0)
    out, info = _load_checked(Image.open(b), 20, 40)
    assert info["size_mismatch"] and info["exif_fixed"] and out.size == (20, 40)
    out, info = _load_checked(Image.new("RGB", (40, 20)), 40, 20)
    assert not info["size_mismatch"]

def test_known_bbox_format():
    sys.path.insert(0, "scripts")
    import analyze
    rows = [dict(source="groundingme", groundable=True, gt_bbox=[1376, 839, 1425, 976], box_orig=[1376, 839, 1425, 976])]
    assert analyze.pick_bbox_format(rows) == "xyxy"
    assert abs(analyze.gt_iou(rows[0], "xyxy") - 1.0) < 1e-9

def test_clean_claims():
    from grej.claims import clean_claims
    t = "Here are the facts:\n- orange handles\n2. dark metallic jaws\n* orange handles\n\nshape of pliers"
    assert clean_claims(t, "d", 8) == ["orange handles", "dark metallic jaws", "shape of pliers"]
    assert clean_claims("", "desc", 8) == ["desc"]
    assert len(clean_claims("\n".join(f"fact {i}" for i in range(20)), "d", 5)) == 5

def test_analyze_claims_runs(tmp_path):
    import subprocess
    rng = np.random.default_rng(0); rows = []
    for i in range(40):
        for g in (True, False):
            sc = list(rng.normal(1, 1, 4)) + ([] if g else [-3.0])
            rows.append(dict(qid=f"q{i}{g}", pair_id=None, source="groundingme", expression="a b c", groundable=g,
                gt_bbox=[0,0,10,10] if g else None, meta={"subtask_l1": "Limited" if g else "Rejection"},
                refused=False, box_orig=[0,0,10,10], desc_words=3, claims=["c"]*len(sc), claim_scores=sc,
                claim_min=min(sc), claim_min2=sorted(sc)[1], claim_mean=float(np.mean(sc)),
                verify_full=float(rng.normal()), verify_crop=float(rng.normal()), box_area_frac=0.1))
    p = tmp_path / "c.jsonl"; p.write_text("\n".join(json.dumps(r) for r in rows))
    out = subprocess.run([sys.executable, "scripts/analyze_claims.py", str(p)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    line = [l for l in out.stdout.splitlines() if "claim_min " in l and "AUROC" in l][0]
    assert float(line.split()[-1]) > 0.9, out.stdout

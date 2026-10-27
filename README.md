# Knowing when it isn't there: calibrated rejection for visual grounding in VLMs

Vision-language models asked to locate "the red cup" draw a box even when the cup is blue or missing. On GroundingME (CVPR 2026), most models score **0%** on rejection. This repo tests:

- **H1.** Do training-free grounding-confidence signals (token log-probs, MTLA-style localized attention) detect attribute-flipped, ungroundable expressions?
- **H2.** Does decoupled crop verification recover the reject signal?
- **H3.** With split-conformal calibration, how many true groundings can we keep while guaranteeing a false-accept rate ≤ α?

See [docs/SURVEY.md](docs/SURVEY.md) for related work and the gap analysis.

## Run (Kaggle, GPU T4, Internet on)
```bash
git clone https://github.com/rahulver551/vlm-grounding-rejection && cd vlm-grounding-rejection
pip install -q -r requirements.txt
python scripts/run_diagnostic.py --dataset refcocog_rej --n 300 --out results/rcg_qwen25vl3b.jsonl
python scripts/run_diagnostic.py --dataset groundingme --n 0 --out results/gme_qwen25vl3b.jsonl
python scripts/analyze.py results/*.jsonl
```
Or open `notebooks/kaggle_run.ipynb`. Runs are resumable: re-running skips finished queries.

## Layout
- `grej/data.py`: RefCOCOg_rej (paired pos/neg) and GroundingME loaders (HF streaming)
- `grej/model.py`: Qwen2.5-VL grounding plus signals (box log-prob, localized attention, crop/full verification)
- `grej/metrics.py`: AUROC, split-conformal false-accept control
- `scripts/`: run and analysis entry points; `tests/`: CPU unit tests

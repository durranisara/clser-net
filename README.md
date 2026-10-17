# CLSER-Net

Cross-Lingual Speech Emotion Recognition via Staged Layer-Wise Transfer Learning
with an Adaptive Temporal Encoder and Emotion-Prototype Relational Attention.

## Setup
    pip install -r requirements.txt

## Configure
Edit `configs/default.yaml`:
- set `data.*_root` to your local dataset paths
- confirm `model.d_h`, `loss.lambda_*`, `training.*` match the paper

## Run
    # Phase 1: source pretraining
    python scripts/pretrain_source.py --config configs/default.yaml

    # Phases 2-3: target adaptation
    python scripts/adapt_target.py --config configs/default.yaml --scenario EN_DE

    # Evaluate
    python scripts/evaluate.py --config configs/default.yaml --scenario EN_DE

## Notes
- `data/datasets.py` parsers are stubs for IEMOCAP/Urdu; fill in your local
  file layouts (TODO comments mark the exact spots).
- Five-fold CV: wrap the three scripts in a loop over fold seeds and average
  the metrics in `utils/metrics.py`.
- `training/transfer.py` implements Algorithm 1: mask [FE, CLATE, EPRA, CLS]
  = [0,0,1,1], discriminative LR eta_l = eta_base * gamma^(L-l), progressive
  unfreezing every `unfreeze_interval` epochs.

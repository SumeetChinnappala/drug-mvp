# Reproducibility

## Environment
- Python 3.11
- Install: `pip install -r requirements.txt`

## Data splits
- Random seed: 42
- Stratified on label for val split; future work: group splits by cell line & drug class.

## Determinism
- CPU run; FFNN epochs=10, early stop patience=3
- Save: artifacts/metrics.json, val_preds.csv, edges.csv, candidates.csv

## Commands
- Train: `python src/drug_discovery_mvp.py`
- App (local): `streamlit run app_viewer.py`

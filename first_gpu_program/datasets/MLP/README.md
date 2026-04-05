# MLP Dataset Bundle

This folder contains all assets related to the current MLP dataset workflow.

## Layout

- `raw/`: source dataset files used by preprocessing.
- `processed/`: train/val/test splits and all generated artifacts.
  - `train.csv`, `val.csv`, `test.csv`, `scaler.json`
  - `models/`, `results/`, `figures/`
- `scripts/`: dataset-specific preprocessing script.
  - `preprocess_manufacturing_defects.py`

## Usage

From repository root:

```powershell
python datasets/MLP/scripts/preprocess_manufacturing_defects.py
```

Defaults now point to:

- input: `datasets/MLP/raw/manufacturing_defect_dataset.csv`
- output-dir: `datasets/MLP/processed`

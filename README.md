# Export bundle (visuals + tables)

This folder is a **standalone bundle** to reproduce thesis-style evaluation visuals and summary tables **offline** from sanitized artifacts under `data/reports/**/v=*/`.

## What’s inside

- `data/reports/…`: copied evaluation artifacts (sanitized CSV/JSON; may also include PNGs)
- `scripts/generate_visuals.py`: regenerates ROC / PR / Confusion Matrix PNGs from
  - `valid_predictions_sanitized.csv` / `test_predictions_sanitized.csv` (columns: `y_true;score;y_pred`)
  - `metrics_sanitized.json` (reads the chosen `threshold` to match the report)
- `scripts/summarize_tables.py`: regenerates thesis-style comparison tables from `metrics_table.csv`

## Quickstart

1) Create a fresh environment and install deps (Windows-friendly with `py`):

From inside this folder:

`py -m venv .venv`

`./.venv/Scripts/python -m pip install -r requirements.txt`

1) The sanitized report for the different experiments mentioned in the thesis are folders are under `data/reports/`.

2) Generate plots for all runs:

`./.venv/Scripts/python scripts/generate_visuals.py`

Plots are written under `out/plots/...`, mirroring the input folder structure, with filenames:
`roc_valid.png`, `pr_valid.png`, `cm_valid.png`, `metrics_valid.png`, `roc_test.png`, `pr_test.png`, `cm_test.png`, `metrics_test.png`.

## Notebook option

If you prefer a notebook-style workflow (easy to tweak variables and rerun), open:

`Generate_Visuals.ipynb`

## Styling

Edit the defaults in `scripts/generate_visuals.py` (the `Style` dataclass) to tweak colors.

## Summary tables

You can regenerate the thesis-style comparison tables from the per-run `metrics_table.csv` files:

`./.venv/Scripts/python scripts/summarize_tables.py`

This writes:

- `out/tables/table_21_like.csv` (model comparison)
- `out/tables/cti_cascade_delta_like.csv` (before/after CTI cascade)

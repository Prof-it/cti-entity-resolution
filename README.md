# Export Bundle: Visuals and Tables

This repository contains the sanitized evaluation artifacts and reproducibility scripts accompanying the following publication:

## Publication

**Haßlinger, T., Lu, T. and Zöller, T. (2026).** *The Signal in the Noise: Semantic Entity Resolution for Cyber Threat Intelligence with Relational Context.* The Second International Conference on Advances in AI-based Systems and Applications, Lisbon, Portugal, 1–5 November 2026. **Accepted.**

## Purpose

This repository is a **standalone export bundle** for reproducing the evaluation visuals and summary tables reported in the paper **offline** from sanitized artifacts under `data/reports/**/v=*/`.

The bundle is designed to reproduce the reported evaluation outputs without requiring access to the original or non-public experiment data.

## What’s Inside

* `data/reports/…`: Sanitized evaluation artifacts, including CSV/JSON files and, where available, PNGs
* `scripts/generate_visuals.py`: Regenerates ROC, PR, confusion-matrix, and metrics visualizations from:

  * `valid_predictions_sanitized.csv` / `test_predictions_sanitized.csv`
  * expected columns: `y_true`, `score`, `y_pred`
  * `metrics_sanitized.json`, using the stored `threshold` to match the corresponding report
* `scripts/summarize_tables.py`: Regenerates thesis-style comparison tables from `metrics_table.csv`

## Quickstart

### 1. Create the environment

From inside this folder:

```bash
py -m venv .venv
./.venv/Scripts/python -m pip install -r requirements.txt
```

### 2. Locate the sanitized reports

The sanitized reports for the experiments discussed in the paper are organized under:

```text
data/reports/
```

### 3. Generate visualizations

Generate plots for all available runs:

```bash
./.venv/Scripts/python scripts/generate_visuals.py
```

Plots are written under:

```text
out/plots/...
```

The output directory mirrors the input report structure.

Generated files include:

```text
roc_valid.png
pr_valid.png
cm_valid.png
metrics_valid.png
roc_test.png
pr_test.png
cm_test.png
metrics_test.png
```

## Notebook Option

For an interactive workflow, open:

```text
Generate_Visuals.ipynb
```

This allows variables and visualization parameters to be inspected and adjusted interactively.

## Styling

Visualization defaults can be adjusted in:

```text
scripts/generate_visuals.py
```

The `Style` dataclass contains the default visualization settings.

## Summary Tables

Thesis-style comparison tables can be regenerated from the per-run `metrics_table.csv` files:

```bash
./.venv/Scripts/python scripts/summarize_tables.py
```

This writes:

* `out/tables/table_21_like.csv` — model comparison
* `out/tables/cti_cascade_delta_like.csv` — before/after CTI cascade comparison

## Reproducibility and Data Scope

This repository contains **sanitized export artifacts** intended to support reproducibility of the evaluation reported in the publication.

Original experiment data, sensitive information, and other non-public artifacts are intentionally excluded from this bundle.

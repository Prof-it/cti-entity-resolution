# The Signal in the Noise: Semantic Entity Resolution for Cyber Threat Intelligence

Reproducibility artifacts for the paper *The Signal in the Noise: Semantic Entity Resolution for Cyber Threat Intelligence with Relational Context* (AINOW 2026, Lisbon).

The paper compares feature-based models (XGBoost), fine-tuned cross-encoders (Ditto/RoBERTa) and few-shot LLM prompting for entity resolution (entity matching, EM) on real-world CTI records, and proposes the **CTI Cascade**: resolving upstream entities (sectors) first and propagating their canonical form as relational context into downstream identity matching.

This repository contains the sanitized evaluation artifacts and the scripts that regenerate the paper's figures and summary tables.

## Publication

**Haßlinger, T., Lu, T. and Zöller, T. (2026).** *The Signal in the Noise: Semantic Entity Resolution for Cyber Threat Intelligence with Relational Context.* The Second International Conference on Advances in AI-based Systems and Applications, Lisbon, Portugal, 1–5 November 2026. **To appear.**

```bibtex
@inproceedings{hasslinger_lu_zoller_2026_cti,
  author    = {Ha{\ss}linger, Timo and Lu, Tianxiang and Z{\"o}ller, Thomas},
  title     = {The Signal in the Noise: Semantic Entity Resolution for Cyber Threat Intelligence with Relational Context},
  booktitle = {The Second International Conference on Advances in AI-based Systems and Applications},
  year      = {2026},
  address   = {Lisbon, Portugal},
  month     = nov,
  note      = {To appear}
}
```
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

From inside this folder.

Linux / macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The commands below use the Linux/macOS interpreter path `.venv/bin/python`; on Windows use `.\.venv\Scripts\python` instead.

### 2. Locate the sanitized reports

The sanitized reports for the experiments discussed in the paper are organized under:

```text
data/reports/
```

### 3. Generate visualizations

Generate plots for all available runs:

```bash
.venv/bin/python scripts/generate_visuals.py
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
.venv/bin/python scripts/summarize_tables.py
```

This writes:

* `out/tables/table_21_like.csv` — model comparison
* `out/tables/cti_cascade_delta_like.csv` — before/after CTI cascade comparison

## Reproducibility and Data Scope

This repository contains **sanitized export artifacts** intended to support reproducibility of the evaluation reported in the publication.

Original experiment data, sensitive information, and other non-public artifacts are intentionally excluded from this bundle.

The artifacts are numeric only: gold labels (`y_true`), model scores, predictions, split statistics and metrics. They contain no source records, entity names or sector labels.

## Data Sources

The experiments in the paper used records from the following sources. **None of their records are included in this repository.** Use of the original data is governed by each provider's own terms.

* [Ransomware.live](https://www.ransomware.live) (sector and identity records)
* [eCrime.ch](https://ecrime.ch) (sector and identity records)
* A GICS-based reference taxonomy (sector records). GICS® is the exclusive property of MSCI and S&P Dow Jones Indices.
* An organizations supplier database (identity records)

## Related Work

The same design principle, evaluating and designing a detection pipeline stage by stage rather than only end to end, is applied to hybrid intrusion detection in:

* **Bilinski, G. and Lu, T. (2026).** *Beyond E2E Metrics: Evaluating Incremental Contributions in Cascaded Hybrid IDS.* IEEE ETECOM 2026, Paris. Code: [Prof-it/stagewise-ids-eval](https://github.com/Prof-it/stagewise-ids-eval)

## Licence

A licence for this repository is being agreed with the co-authors. Until a `LICENSE` file is added, all rights are reserved by the authors.

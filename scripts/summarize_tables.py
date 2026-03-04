from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RunMetrics:
    run_dir: Path
    fbeta: float
    f1: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    accuracy: float


def _read_test_metrics(run_dir: Path) -> RunMetrics | None:
    p = run_dir / "metrics_table.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, sep=";")
    if "split" not in df.columns:
        return None
    test = df[df["split"].astype(str).str.upper() == "TEST"]
    if test.empty:
        return None
    row = test.iloc[0]
    return RunMetrics(
        run_dir=run_dir,
        fbeta=float(row["f_beta(0.8)"]),
        f1=float(row["f1"]),
        roc_auc=float(row.get("roc_auc", float("nan"))),
        pr_auc=float(row.get("pr_auc", float("nan"))),
        precision=float(row.get("precision", float("nan"))),
        recall=float(row.get("recall", float("nan"))),
        accuracy=float(row.get("accuracy", float("nan"))),
    )


def _best_run_under(model_dir: Path) -> RunMetrics | None:
    # Choose the best run by TEST f_beta(0.8)
    candidates: list[RunMetrics] = []
    for run_dir in sorted(model_dir.glob("v=*/")):
        m = _read_test_metrics(run_dir)
        if m is not None:
            candidates.append(m)
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.fbeta)


def _fmt_delta(v: float, base: float) -> str:
    d = v - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _r3(v: float | None) -> float | None:
    return None if v is None else round(float(v), 3)


def _table21(reports_dir: Path) -> pd.DataFrame:
    # Map model-folder names -> labels used in the thesis table
    row_order = [
        ("roberta-base", "Ditto (RoBERTa, Baseline)"),
        ("roberta-base+dapt", "RoBERTa–DAPT"),
        (("roberta-base+dapt+tapt", "roberta-base+tapt+tapt"), "RoBERTa–DAPT+TAPT"),
        ("llms", "LLM (Prompting)"),
        ("xgboost", "XGBoost"),
        ("roberta-base-cti-cascade", "Ditto + CTI-Kaskade"),
        ("llms-cti-cascade", "LLM + CTI-Kaskade"),
    ]

    def _entity_block(entity: str) -> dict[str, tuple[float | None, float | None]]:
        entity_dir = reports_dir / entity
        out: dict[str, tuple[float | None, float | None]] = {}
        for model_key, _ in row_order:
            keys = (model_key,) if isinstance(model_key, str) else tuple(model_key)
            best: RunMetrics | None = None
            for k in keys:
                model_dir = entity_dir / k
                cand = _best_run_under(model_dir) if model_dir.exists() else None
                if cand is None:
                    continue
                if best is None or cand.fbeta > best.fbeta:
                    best = cand
            # store under a stable joined key for later lookup
            joined = keys[0] if len(keys) == 1 else "|".join(keys)
            out[joined] = (best.fbeta, best.f1) if best else (None, None)
        return out

    sec = _entity_block("sectors")
    ids = _entity_block("identities")

    base_sec_fbeta, base_sec_f1 = sec.get("roberta-base", (None, None))
    base_ids_fbeta, base_ids_f1 = ids.get("roberta-base", (None, None))

    base_sec_fbeta_r = _r3(base_sec_fbeta)
    base_sec_f1_r = _r3(base_sec_f1)
    base_ids_fbeta_r = _r3(base_ids_fbeta)
    base_ids_f1_r = _r3(base_ids_f1)

    rows = []
    for model_key, label in row_order:
        keys = (model_key,) if isinstance(model_key, str) else tuple(model_key)
        joined = keys[0] if len(keys) == 1 else "|".join(keys)
        s_fbeta, s_f1 = sec.get(joined, (None, None))
        i_fbeta, i_f1 = ids.get(joined, (None, None))

        s_fbeta_r = _r3(s_fbeta)
        s_f1_r = _r3(s_f1)
        i_fbeta_r = _r3(i_fbeta)
        i_f1_r = _r3(i_f1)

        rows.append(
            {
                "model": label,
                "sectors_fbeta(0.8)": s_fbeta_r,
                "sectors_Δ_fbeta": None
                if (s_fbeta_r is None or base_sec_fbeta_r is None)
                else _fmt_delta(float(s_fbeta_r), float(base_sec_fbeta_r)),
                "sectors_f1": s_f1_r,
                "sectors_Δ_f1": None
                if (s_f1_r is None or base_sec_f1_r is None)
                else _fmt_delta(float(s_f1_r), float(base_sec_f1_r)),
                "identities_fbeta(0.8)": i_fbeta_r,
                "identities_Δ_fbeta": None
                if (i_fbeta_r is None or base_ids_fbeta_r is None)
                else _fmt_delta(float(i_fbeta_r), float(base_ids_fbeta_r)),
                "identities_f1": i_f1_r,
                "identities_Δ_f1": None
                if (i_f1_r is None or base_ids_f1_r is None)
                else _fmt_delta(float(i_f1_r), float(base_ids_f1_r)),
            }
        )

    return pd.DataFrame(rows)


def _cascade_delta(reports_dir: Path) -> pd.DataFrame:
    # Identities only (as in your screenshot)
    ids = reports_dir / "identities"

    pairs = [
        ("Baseline", ids / "roberta-base", ids / "roberta-base-cti-cascade"),
        ("LLM", ids / "llms", ids / "llms-cti-cascade"),
    ]

    metrics = [
        ("f_beta(0.8)", "fbeta"),
        ("f1", "f1"),
        ("accuracy", "accuracy"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("pr_auc", "pr_auc"),
        ("roc_auc", "roc_auc"),
    ]

    rows: list[dict] = []
    for label, before_dir, after_dir in pairs:
        before = _best_run_under(before_dir) if before_dir.exists() else None
        after = _best_run_under(after_dir) if after_dir.exists() else None

        if before is None or after is None:
            continue

        before_map = {
            "fbeta": before.fbeta,
            "f1": before.f1,
            "accuracy": before.accuracy,
            "precision": before.precision,
            "recall": before.recall,
            "pr_auc": before.pr_auc,
            "roc_auc": before.roc_auc,
        }
        after_map = {
            "fbeta": after.fbeta,
            "f1": after.f1,
            "accuracy": after.accuracy,
            "precision": after.precision,
            "recall": after.recall,
            "pr_auc": after.pr_auc,
            "roc_auc": after.roc_auc,
        }

        for metric_label, key in metrics:
            b = float(before_map[key])
            a = float(after_map[key])
            rows.append(
                {
                    "group": label,
                    "metric": metric_label,
                    "before": round(b, 3),
                    "after": round(a, 3),
                    "delta": _fmt_delta(a, b),
                    "before_run": str(before.run_dir.as_posix()),
                    "after_run": str(after.run_dir.as_posix()),
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate thesis-style summary tables from metrics_table.csv")
    parser.add_argument("--reports-dir", type=Path, default=Path("data/reports"), help="Root reports dir (default: data/reports)")
    parser.add_argument("--out-dir", type=Path, default=Path("out/tables"), help="Output directory (default: out/tables)")
    args = parser.parse_args()

    reports_dir = args.reports_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t21 = _table21(reports_dir)
    t21_path = out_dir / "table_21_like.csv"
    t21.to_csv(t21_path, index=False)

    cascade = _cascade_delta(reports_dir)
    cascade_path = out_dir / "cti_cascade_delta_like.csv"
    cascade.to_csv(cascade_path, index=False)

    print(f"Wrote: {t21_path}")
    print(f"Wrote: {cascade_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

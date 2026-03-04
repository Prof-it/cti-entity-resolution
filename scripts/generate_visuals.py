from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Style:
    curve_color: str = "C0"
    baseline_color: str = "0.5"
    cm_cmap: str = "Blues"


DEFAULT_REPORTS_DIR = Path("data/reports")
DEFAULT_OUT_DIR = Path("out/plots")


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    df = pd.read_csv(path, sep=";")
    y_true = df["y_true"].to_numpy(dtype=int)
    y_score = df["score"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=int) if "y_pred" in df.columns else None
    return y_true, y_score, y_pred


def _confusion_at(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _roc_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    s = y_score[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    change = np.where(np.r_[True, s[1:] < s[:-1]])[0]
    tpr = np.r_[0.0, tp[change] / P, 1.0]
    fpr = np.r_[0.0, fp[change] / N, 1.0]
    return fpr, tpr


def _roc_auc_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Thesis ROC-AUC (plot legend) implementation.

    This matches `src/thesis_experiments/thesis_experiments/utils/visualization.py::roc_auc_from_scores`.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    r = np.argsort(np.argsort(np.r_[neg, pos]))
    r_pos = r[len(neg) :].sum()
    auc = (r_pos - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def _pr_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    P = int((y_true == 1).sum())
    if P == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])

    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    s = y_score[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    change = np.where(np.r_[True, s[1:] < s[:-1]])[0]
    prec = tp[change] / np.maximum(tp[change] + fp[change], 1)
    rec = tp[change] / P
    rec = np.r_[0.0, rec]
    prec = np.r_[1.0, prec]
    return rec, prec


def _pr_auc_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Thesis AP (plot legend) implementation.

    This matches `src/thesis_experiments/thesis_experiments/utils/visualization.py::pr_auc_from_scores`.
    """
    rec, prec = _pr_curve_points(y_true, y_score)
    prec_hull = np.maximum.accumulate(prec[::-1])[::-1]
    drec = np.diff(rec)
    ap = float(np.sum(drec * prec_hull[1:])) if len(rec) > 1 else 0.0
    return ap


def _pr_auc_metrics_table_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """PR-AUC/AP implementation matching the experiment `metrics_table.csv`.

    This follows `src/thesis_experiments/thesis_experiments/algos/eval/metrics.py::pr_auc` and
    the inlined implementation in `confusion_and_metrics(...)`.

    Notes:
    - Uses an AP-like Riemann sum over per-example recall increments.
    - Does NOT apply a precision hull / monotonic envelope.
    - Does NOT prepend an explicit (recall=0, precision=1) anchor point.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    prec = tp / np.maximum(tp + fp, 1)
    P = max(int((y_true == 1).sum()), 1)
    rec = tp / P

    if len(rec) <= 1:
        return 0.0

    drec = np.r_[0.0, rec[1:]] - np.r_[0.0, rec[:-1]]
    ap = float((prec * drec).sum())
    return ap


def _plot_roc(
    out_png: Path,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    chosen_t: float | None,
    title: str,
    style: Style,
    auc_label: float | None,
) -> None:
    fpr, tpr = _roc_curve_points(y_true, y_score)
    auc = float(auc_label) if auc_label is not None else _roc_auc_from_scores(y_true, y_score)

    plt.figure(figsize=(4.5, 4.5), dpi=140)
    plt.step(fpr, tpr, where="post", color=style.curve_color, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=style.baseline_color)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if chosen_t is not None:
        y_pred = (np.asarray(y_score) >= float(chosen_t)).astype(int)
        P = max(int((y_true == 1).sum()), 1)
        N = max(int((y_true == 0).sum()), 1)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tpr_star = tp / P
        fpr_star = fp / N
        plt.scatter([fpr_star], [tpr_star], s=40, zorder=3, color=style.curve_color)
        plt.annotate(
            f" t*={float(chosen_t):.3f}\nTPR={tpr_star:.2f}, FPR={fpr_star:.2f}",
            (fpr_star, tpr_star),
            textcoords="offset points",
            xytext=(10, -10),
        )

    plt.legend(loc="lower right")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def _plot_pr(
    out_png: Path,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    chosen_t: float | None,
    title: str,
    style: Style,
    ap_label: float,
) -> None:
    rec, prec = _pr_curve_points(y_true, y_score)
    ap = float(ap_label)

    plt.figure(figsize=(4.5, 4.5), dpi=140)
    plt.step(rec, prec, where="post", color=style.curve_color, label=f"PR (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if chosen_t is not None:
        y_pred = (np.asarray(y_score) >= float(chosen_t)).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        P = max(int((y_true == 1).sum()), 1)
        prec_star = tp / max(tp + fp, 1)
        rec_star = tp / P
        plt.scatter([rec_star], [prec_star], s=40, zorder=3, color=style.curve_color)
        plt.annotate(
            f" t*={float(chosen_t):.3f}\nPrec={prec_star:.2f}, Rec={rec_star:.2f}",
            (rec_star, prec_star),
            textcoords="offset points",
            xytext=(10, -10),
        )

    plt.legend(loc="lower left")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def _plot_confusion(out_png: Path, cf: dict[str, int], *, title: str, style: Style) -> None:
    mat = np.array([[cf["tp"], cf["fn"]], [cf["fp"], cf["tn"]]], dtype=float)
    labels = [["TP", "FN"], ["FP", "TN"]]

    plt.figure(figsize=(3, 3), dpi=160)
    ax = plt.gca()
    im = ax.imshow(mat, cmap=style.cm_cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred=1", "pred=0"])
    ax.set_yticklabels(["actual=1", "actual=0"])
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{int(mat[i, j])}", ha="center", va="center", fontsize=10, color="black")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def _f_beta(precision: float, recall: float, *, beta: float) -> float:
    b2 = float(beta) ** 2
    return _safe_div((1 + b2) * precision * recall, (b2 * precision) + recall)


def _compute_metrics_from_data(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 0.8,
    pr_auc_override: float | None = None,
    roc_auc_override: float | None = None,
) -> dict[str, float]:
    cf = _confusion_at(y_true, y_pred)
    tp, fp, fn, tn = cf["tp"], cf["fp"], cf["fn"], cf["tn"]

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _f_beta(precision, recall, beta=1.0)
    fbeta = _f_beta(precision, recall, beta=beta)
    accuracy = _safe_div(tp + tn, tp + fp + fn + tn)

    roc_auc = float(roc_auc_override) if roc_auc_override is not None else _roc_auc_from_scores(y_true, y_score)
    pr_auc = float(pr_auc_override) if pr_auc_override is not None else _pr_auc_metrics_table_from_scores(y_true, y_score)

    return {
        "f_beta(0.8)": float(fbeta),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
    }


def _load_auc_from_metrics_table(run_dir: Path, *, split: str) -> tuple[float | None, float | None]:
    p = run_dir / "metrics_table.csv"
    if not p.exists():
        return None, None
    try:
        df = pd.read_csv(p, sep=";")
        row = df[df["split"].astype(str).str.upper() == split.upper()]
        if row.empty:
            return None, None
        r0 = row.iloc[0]
        pr_auc = r0.get("pr_auc", None)
        roc_auc = r0.get("roc_auc", None)
        pr_auc_f = float(pr_auc) if pr_auc is not None and pd.notna(pr_auc) else None
        roc_auc_f = float(roc_auc) if roc_auc is not None and pd.notna(roc_auc) else None
        return pr_auc_f, roc_auc_f
    except Exception:
        return None, None


def _plot_metric_table(out_png: Path, metrics: dict[str, float], *, split: str) -> None:
    order = [
        ("f_beta(0.8)", r"$f_{\beta}(0.8)$"),
        ("f1", "f1"),
        ("accuracy", "accuracy"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("pr_auc", "pr_auc"),
        ("roc_auc", "roc_auc"),
    ]

    rows: list[list[str]] = []
    for key, label in order:
        if key not in metrics:
            continue
        rows.append([label, f"{metrics[key]:.4f}"])

    plt.figure(figsize=(2.6, 3.1), dpi=180)
    ax = plt.gca()
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metrik", split.upper()],
        cellLoc="left",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.15)

    # Minimal styling: bold header row, remove heavy cell borders.
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.0)
        if r == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def _iter_run_dirs(reports_dir: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for path in reports_dir.rglob("*"):
        if not path.is_dir():
            continue
        if (path / "metrics_sanitized.json").exists() or (path / "metrics_table.csv").exists():
            run_dirs.append(path)
    return sorted(set(run_dirs))


def _load_threshold(run_dir: Path) -> float | None:
    p = run_dir / "metrics_sanitized.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        t = data.get("threshold", None)
        return float(t) if t is not None else None
    except Exception:
        return None


def _generate_for_split(*, reports_dir: Path, run_dir: Path, split: str, threshold: float | None, style: Style, out_root: Path) -> None:
    split_lower = split.lower()
    pred_file = run_dir / f"{split_lower}_predictions_sanitized.csv"
    if not pred_file.exists():
        return

    y_true, y_score, y_pred_file = _read_predictions(pred_file)
    if threshold is None:
        y_pred = y_pred_file if y_pred_file is not None else (y_score >= 0.5).astype(int)
        chosen_t = None
    else:
        y_pred = (y_score >= float(threshold)).astype(int)
        chosen_t = float(threshold)

    cf = _confusion_at(y_true, y_pred)
    pr_auc_label, roc_auc_label = _load_auc_from_metrics_table(run_dir, split=split)
    auc_label = roc_auc_label if roc_auc_label is not None else _roc_auc_from_scores(y_true, y_score)
    ap_label = pr_auc_label if pr_auc_label is not None else _pr_auc_metrics_table_from_scores(y_true, y_score)

    out_dir = out_root / run_dir.relative_to(reports_dir)

    _plot_confusion(out_dir / f"cm_{split_lower}.png", cf, title=f"Confusion Matrix – {split.upper()}", style=style)

    metrics_row = _compute_metrics_from_data(
        y_true=y_true,
        y_score=y_score,
        y_pred=y_pred,
        pr_auc_override=pr_auc_label,
        roc_auc_override=roc_auc_label,
    )
    _plot_metric_table(out_dir / f"metrics_{split_lower}.png", metrics_row, split=split)

    _plot_roc(
        out_dir / f"roc_{split_lower}.png",
        y_true,
        y_score,
        chosen_t=chosen_t,
        title=f"ROC – {split.upper()}",
        style=style,
        auc_label=auc_label,
    )
    _plot_pr(
        out_dir / f"pr_{split_lower}.png",
        y_true,
        y_score,
        chosen_t=chosen_t,
        title=f"PR – {split.upper()}",
        style=style,
        ap_label=ap_label,
    )


def generate_all(*, reports_dir: Path = DEFAULT_REPORTS_DIR, out_dir: Path = DEFAULT_OUT_DIR, style: Style = Style()) -> int:
    reports_dir = Path(reports_dir).resolve()
    out_root = Path(out_dir).resolve()

    run_dirs = _iter_run_dirs(reports_dir)
    if not run_dirs:
        raise SystemExit(f"No run dirs found under {reports_dir}")

    for run_dir in run_dirs:
        threshold = _load_threshold(run_dir)
        _generate_for_split(reports_dir=reports_dir, run_dir=run_dir, split="valid", threshold=threshold, style=style, out_root=out_root)
        _generate_for_split(reports_dir=reports_dir, run_dir=run_dir, split="test", threshold=threshold, style=style, out_root=out_root)

    print(f"Generated visuals for {len(run_dirs)} runs")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate ROC/PR/CM visuals from sanitized prediction CSVs. "
            "Defaults match the thesis plotting code (AUC/AP computed like utils/visualization.py)."
        )
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Root dir containing reports (default: data/reports)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output root (default: out/plots)",
    )
    args = parser.parse_args()

    return generate_all(reports_dir=args.reports_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations
import argparse
import json
import os
import sys
import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# --- Project imports: use your existing FinBERT pipeline ---
# Ensures `src/` is importable when running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sentiment_analyzer import load_sentiment_model, analyze_sentiment  # noqa: E402


def plot_confusion(cm: np.ndarray, classes: List[str], out_png: Path) -> None:
    """Save confusion matrix heatmap (absolute counts)."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to labeled CSV (headline,label)")
    ap.add_argument("--text-col", default="headline", help="Text column name")
    ap.add_argument("--label-col", default="label", help="Label column name")
    ap.add_argument("--test-size", type=float, default=0.2, help="Held-out test fraction")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out", default="runs/baseline_finbert", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(
            f"Missing required columns: {args.text_col=!r} or {args.label_col=!r}. "
            f"Found: {df.columns.tolist()}"
        )
    df = df[[args.text_col, args.label_col]].dropna()
    df = df.rename(columns={args.text_col: "text", args.label_col: "label"})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    if df.empty:
        raise SystemExit("No valid rows after cleaning input DataFrame.")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values,
        df["label"].values,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"].values,
    )

    # Load FinBERT pipeline (download happens on first run)
    classifier = load_sentiment_model()

    # Predict on test set
    norm = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
    y_pred: List[str] = []
    for t in X_test:
        res = analyze_sentiment(str(t), classifier) or {}
        raw = (res.get("label") or "").lower()
        y_pred.append(norm.get(raw, "neutral"))

    # Metrics
    classes = sorted(list(set(list(y_test) + list(y_pred))))
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, labels=classes)

    # Save classification report
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_png = out_dir / "confusion_matrix.png"
    plot_confusion(cm, classes, cm_png)

    # Predictions CSV
    pd.DataFrame(
        {"text": X_test, "true_label": y_test, "pred_label": y_pred}
    ).to_csv(out_dir / "predictions.csv", index=False)

    # metrics.json
    metrics = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "model": "ProsusAI/finbert",
        "test_size": float(args.test_size),
        "classes": classes,
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "confusion_matrix_png": str(cm_png),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Console summary
    print("=== FinBERT Baseline Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro-F1:  {f1m:.4f}")
    print("Classes:   ", classes)
    print(f"Artifacts: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

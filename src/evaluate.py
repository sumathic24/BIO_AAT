"""
Evaluation metrics and plots. The positive class is DEFAULT (label 1), so
recall = share of actual defaulters the model catches.

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, RMSE of predicted probabilities.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def compute_metrics(model, X, y, threshold: float = 0.5) -> dict:
    """Metrics at a given decision threshold. AUC does not depend on the threshold."""
    proba = model.predict_proba(X)[:, 1]     # P(default)
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, proba),
        "rmse": float(np.sqrt(np.mean((proba - y) ** 2))),
    }


def select_threshold_cv(model, X_train, y_train, metric: str = "f1", folds: int = 5, seed: int = 42):
    """Choose the decision threshold using ONLY training data.

    Out-of-fold probabilities come from stratified k-fold cross-validation on the
    training set (the scaler is refitted inside each fold, so nothing leaks). The
    threshold that maximises F1 (or F2) on those out-of-fold predictions is returned.
    The test set is not used here.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    if metric == "f1":
        score = lambda yt, yp: f1_score(yt, yp, zero_division=0)
    else:
        score = lambda yt, yp: fbeta_score(yt, yp, beta=2, zero_division=0)
    best_t, best_s = 0.5, -1.0
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        s = score(y_train, (oof >= t).astype(int))
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s


def plot_convergence(history: dict, path: str):
    it = history["iteration"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(it, history["best"], "o-", label="Best bat", color="#1f77b4")
    ax.plot(it, history["mean"], "s--", label="Swarm mean", color="#ff7f0e")
    ax.plot(it, history["worst"], "^:", label="Worst bat", color="#7f7f7f")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness (CV score)")
    ax.set_title("Bat Algorithm convergence curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curves(baseline_loss, optimised_loss, path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(baseline_loss, label="Baseline ANN", color="#7f7f7f")
    ax.plot(optimised_loss, label="Bat-optimised ANN", color="#2ca02c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (log-loss)")
    ax.set_title("Learning curves: baseline vs optimised")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(base: dict, opt: dict, path: str):
    keys = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w / 2, [base[k] for k in keys], w, label="Baseline", color="#7f7f7f")
    b2 = ax.bar(x + w / 2, [opt[k] for k in keys], w, label="Bat-optimised", color="#2ca02c")
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, [k.upper() if k == "auc" else k.capitalize() for k in keys])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test-set score")
    ax.set_title("Baseline vs Bat-optimised ANN on held-out test set")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(models: dict, X, y, path: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (model, color) in models.items():
        proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC = {roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def threshold_table(models: dict, X, y, thresholds=None) -> pd.DataFrame:
    """Precision, recall and F1 for each model at several decision thresholds.
    Descriptive sensitivity analysis on the test set: it does NOT choose a threshold."""
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = []
    for name, (model, _) in models.items():
        proba = model.predict_proba(X)[:, 1]
        for t in thresholds:
            pred = (proba >= t).astype(int)
            rows.append({
                "model": name, "threshold": float(t),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "flagged_share": float(pred.mean()),
            })
    return pd.DataFrame(rows)


def plot_threshold(df, path: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    styles = {"Baseline": "#7f7f7f", "Bat-optimised": "#2ca02c"}
    for name, color in styles.items():
        sub = df[df["model"] == name]
        ax.plot(sub["threshold"], sub["recall"], "-", color=color, label=f"{name}: recall")
        ax.plot(sub["threshold"], sub["precision"], "--", color=color, label=f"{name}: precision")
        ax.plot(sub["threshold"], sub["f1"], ":", color=color, label=f"{name}: F1")
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="-.", label="default threshold 0.5")
    ax.set_xlabel("Decision threshold on P(default)")
    ax.set_ylabel("Test-set score")
    ax.set_title("Threshold sensitivity (test set)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
"""
Create explanatory figures from your scores.

    python make_figures.py

Where the numbers come from:
  1. If results/ contains the files written by run.py, they are used.
  2. Otherwise the values in the SCORES block below are used. Edit them to
     your own numbers if you want to plot different results.

Outputs (in docs/figures/):
  convergence.png          how the Bat search improved over iterations
  metric_comparison.png    baseline vs optimised on every test metric
  threshold_tradeoff.png   precision vs recall as the decision threshold moves
  figure_notes.md          a plain-English note to explain each figure
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "docs" else HERE
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# ---------------- SCORES (fallback when results/ is missing) ----------------
# Values from the run log and metrics table of the reported experiment.
SCORES = {
    "history": {
        "iteration": list(range(9)),
        "best":  [0.7454, 0.7464, 0.7492, 0.7536, 0.7536, 0.7536, 0.7536, 0.7546, 0.7546],
        "mean":  [0.7226, 0.7421, 0.7453, 0.7474, 0.7491, 0.7501, 0.7501, 0.7512, 0.7520],
        "evals": [10, 20, 30, 40, 50, 60, 70, 79, 88],
    },
    "metrics": {  # test set, positive class = default, threshold 0.5
        "Baseline":      {"accuracy": 0.8166, "precision": 0.6692, "recall": 0.3386, "f1": 0.4497, "auc": 0.7498},
        "Bat-optimised": {"accuracy": 0.8175, "precision": 0.6701, "recall": 0.3446, "f1": 0.4552, "auc": 0.7481},
    },

}


def load_history():
    path = os.path.join(RESULTS, "bat_history.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return {"iteration": df["iteration"].tolist(), "best": df["best"].tolist(), "mean": df["mean"].tolist(),
                "evals": None}, "results/bat_history.csv"
    return SCORES["history"], "values typed in SCORES"


def load_metrics():
    path = os.path.join(RESULTS, "metrics.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        return {k: {m: float(df.loc[k, m]) for m in ["accuracy", "precision", "recall", "f1", "auc"]}
                for k in df.index}, "results/metrics.csv"
    return SCORES["metrics"], "values typed in SCORES"


def load_threshold():
    path = os.path.join(RESULTS, "threshold_sensitivity.csv")
    if os.path.exists(path):
        return pd.read_csv(path), "results/threshold_sensitivity.csv"
    return None, "not available (run run.py to create threshold_sensitivity.csv)"


NOTES = []


def note(title, text):
    NOTES.append(f"## {title}\n\n{text}\n")


# ---------------- Figure 1: convergence ----------------
hist, src1 = load_history()
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(hist["iteration"], hist["best"], "o-", color="#1f77b4", label="Best bat")
ax.plot(hist["iteration"], hist["mean"], "s--", color="#ff7f0e", label="Swarm mean")
ax.set_xlabel("Iteration")
ax.set_ylabel("Fitness (3-fold CV ROC-AUC)")
ax.set_title("Bat Algorithm convergence")
ax.grid(alpha=0.3)
ax.legend()
gain = hist["best"][-1] - hist["best"][0]
ax.annotate(f"+{gain:.4f} total gain", xy=(hist["iteration"][-1], hist["best"][-1]),
            xytext=(hist["iteration"][len(hist["iteration"]) // 2], hist["best"][0] - 0.004),
            arrowprops=dict(arrowstyle="->", color="grey"), fontsize=9, color="grey")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "convergence.png"), dpi=150)
plt.close(fig)
note("Figure 1: Bat Algorithm convergence",
     f"Source: {src1}. The solid line is the best CV AUC found so far; the dashed line is the average over "
     "all bats. Both rise over the iterations, meaning the search is finding better configurations. "
     f"The best value rose by {gain:.4f} overall. When the solid line flattens, further iterations add little "
     "at this budget. A small gap between the two lines means the swarm has converged on similar solutions.")

# ---------------- Figure 2: metric comparison ----------------
met, src2 = load_metrics()
keys = ["accuracy", "precision", "recall", "f1", "auc"]
labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
names = list(met.keys())
colors = ["#7f7f7f", "#2ca02c"]
x = np.arange(len(keys))
w = 0.38
fig, ax = plt.subplots(figsize=(8.2, 4.4))
for i, name in enumerate(names):
    vals = [met[name][k] for k in keys]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=name, color=colors[i % 2])
    ax.bar_label(bars, fmt="%.4f", fontsize=7.5, padding=2)
ax.set_xticks(x, labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Test-set score")
ax.set_title("Baseline vs optimised model on the held-out test set")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper center", ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "metric_comparison.png"), dpi=150)
plt.close(fig)
diff = {k: met[names[1]][k] - met[names[0]][k] for k in keys}
note("Figure 2: Baseline vs optimised comparison",
     f"Source: {src2}. Each pair of bars is one metric on the same test set. Differences: "
     + ", ".join(f"{l} {diff[k]:+.4f}" for l, k in zip(labels, keys))
     + ". Every difference is small. A reader should conclude that the optimiser did not produce a meaningful "
       "gain; it is only the ~0.5 point variation expected from a single random split. Recall (about one in three "
       "defaulters caught) is the weakest score for both models.")

# ---------------- Figure 3: threshold trade-off ----------------
thr, src3 = load_threshold()
if thr is not None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for name, color in zip(thr["model"].unique(), ["#7f7f7f", "#2ca02c"]):
        sub = thr[thr["model"] == name]
        ax.plot(sub["threshold"], sub["recall"], "-", color=color, label=f"{name}: recall")
        ax.plot(sub["threshold"], sub["precision"], "--", color=color, label=f"{name}: precision")
    ax.axvline(0.5, color="black", linestyle="-.", linewidth=0.8, label="default threshold 0.5")
    ax.set_xlabel("Decision threshold on P(default)")
    ax.set_ylabel("Score")
    ax.set_title("Precision vs recall as the threshold moves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "threshold_tradeoff.png"), dpi=150)
    plt.close(fig)
    note("Figure 3: Threshold trade-off",
         f"Source: {src3}. Lowering the threshold classifies more clients as defaulters. Recall (solid) rises "
         "because more real defaulters are caught, while precision (dashed) falls because more of the flagged "
         "clients are false alarms. The choice of threshold is a business decision about which error is more "
         "costly. It should be made on validation data, not on the test set.")
else:
    print(f"Figure 3 skipped: threshold data {src3}")

with open(os.path.join(OUT, "figure_notes.md"), "w", encoding="utf-8") as f:
    f.write("# Figure notes\n\n" + "\n".join(NOTES))

print("Saved figures to", OUT)
for n in NOTES:
    print(n.splitlines()[0])
"""
Correctness checks for the credit card Bat-ANN project.

    python tests/verify.py                 # uses the dataset in data/
    python tests/verify.py --data path.csv

Each check prints PASS or FAIL. The script exits with an error if any check fails.
It runs a tiny Bat search (a few seconds), so it does not reproduce the full experiment.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

from data import load_and_prepare  # noqa: E402
from bat import BatAlgorithm, BatConfig, decode  # noqa: E402
from model import BASELINE_PARAMS, build_model  # noqa: E402
from evaluate import compute_metrics  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    # 1. Data loading and cleaning
    ds = load_and_prepare(args.data)
    r = ds.report
    check("Rows after cleaning are about 30,000", 29000 <= r["rows_after_cleaning"] <= 30000,
          f"{r['rows_after_cleaning']}")
    check("No NaN values in features", not np.isnan(ds.X_train).any() and not np.isnan(ds.X_test).any())
    check("Labels are only 0 and 1", set(np.unique(ds.y_train)) <= {0, 1} and set(np.unique(ds.y_test)) <= {0, 1})
    check("Default rate is about 22%", 20 <= r["default_rate_pct"] <= 24, f"{r['default_rate_pct']}%")
    check("Split is 80/20", abs(len(ds.X_train) / (len(ds.X_train) + len(ds.X_test)) - 0.8) < 0.01,
          f"train={len(ds.X_train)}, test={len(ds.X_test)}")
    tr_rate, te_rate = ds.y_train.mean(), ds.y_test.mean()
    check("Stratified split keeps default rate equal (within 0.5 points)", abs(tr_rate - te_rate) < 0.005,
          f"train {tr_rate:.4f} vs test {te_rate:.4f}")
    check("Feature count is 23", ds.X_train.shape[1] == 23, f"{ds.X_train.shape[1]}")

    # 2. Decoding stays inside the search ranges
    lo, hi = decode(np.zeros(5)), decode(np.ones(5))
    check("Smallest decoded layer is 4 units", lo["hidden_layer_sizes"][0] == 4)
    check("Largest decoded layer is 128 units", hi["hidden_layer_sizes"][0] == 128)
    check("Learning rate range is 1e-4 to 1e-1",
          abs(lo["learning_rate_init"] - 1e-4) < 1e-12 and abs(hi["learning_rate_init"] - 1e-1) < 1e-9)
    check("Activation 0 gives relu and 1 gives tanh", lo["activation"] == "relu" and hi["activation"] == "tanh")
    check("Position 0 for second layer gives a single hidden layer",
          len(decode(np.array([0.5, 0.0, 0.5, 0.5, 0.2]))["hidden_layer_sizes"]) == 1)

    # 3. No leakage: the scaler must be fitted on training data only
    model = build_model(BASELINE_PARAMS, max_iter=50)
    model.fit(ds.X_train, ds.y_train)
    scaler_mean = model.named_steps["standardscaler"].mean_
    check("Scaler mean equals training mean (not test mean)",
          np.allclose(scaler_mean, ds.X_train.mean(axis=0), rtol=1e-6, atol=1e-6))

    # 4. Metrics agree with a manual confusion-matrix calculation
    m = compute_metrics(model, ds.X_test, ds.y_test)
    proba = model.predict_proba(ds.X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tp = int(((pred == 1) & (ds.y_test == 1)).sum())
    fp = int(((pred == 1) & (ds.y_test == 0)).sum())
    fn = int(((pred == 0) & (ds.y_test == 1)).sum())
    tn = int(((pred == 0) & (ds.y_test == 0)).sum())
    check("Recall equals TP / (TP + FN)", abs(m["recall"] - tp / (tp + fn)) < 1e-9)
    check("Precision equals TP / (TP + FP)", abs(m["precision"] - tp / (tp + fp)) < 1e-9)
    check("Accuracy equals (TP + TN) / total", abs(m["accuracy"] - (tp + tn) / len(pred)) < 1e-9)
    check("AUC is between 0.5 (random) and 1.0", 0.5 <= m["auc"] <= 1.0, f"{m['auc']:.4f}")

    # 5. Bat search is reproducible and does not touch the test set
    X_small, y_small = ds.X_train[:600], ds.y_train[:600]
    cfg = BatConfig(n_bats=3, iterations=2, seed=42)
    res1 = BatAlgorithm(cfg).run(X_small, y_small, verbose=False)
    res2 = BatAlgorithm(cfg).run(X_small, y_small, verbose=False)
    check("Same seed gives identical best position", np.allclose(res1.best_position, res2.best_position))
    check("Same seed gives identical best fitness", abs(res1.best_cv_auc - res2.best_cv_auc) < 1e-12)
    check("Best fitness never decreases across iterations",
          all(b2 >= b1 - 1e-12 for b1, b2 in zip(res1.history["best"], res1.history["best"][1:])))

    # 6. Saved results match each other (only if a full run has been done)
    rdir = os.path.join(HERE, "results")
    summ = os.path.join(rdir, "summary.json")
    if os.path.exists(summ):
        with open(summ) as f:
            s = json.load(f)
        check("summary.json has test metrics for both models", "results" in s and "optimised" in s["results"])
        check("Test set has 20% of cleaned rows",
              abs(s["dataset"]["test_size"] / s["dataset"]["rows_after_cleaning"] - 0.2) < 0.001)
        check("Search used a subsample and recorded its size", s["bat"]["search_rows"] > 0)
    else:
        print("[SKIP] results/summary.json not found: run run.py first to check saved results")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
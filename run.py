"""
Reproduce the full experiment with one command:

    python run.py                          # default: 0.5 threshold (reported numbers)
    python run.py --threshold cv           # cut-off chosen on training data only
    python run.py --threshold cv --threshold-metric f2
    python run.py --bats 20 --iters 15 --search-max-iter 200
    python run.py --search-size 0          # search on the full training set (slowest)
    python run.py --data path/to.csv       # use another copy of the file

Writes into results/:
    metrics.csv                  baseline vs optimised test-set metrics (with threshold used)
    summary.json                 dataset report, Bat settings, best params, thresholds
    bat_history.csv              per-iteration swarm statistics
    convergence.png              Bat Algorithm convergence curve
    learning_curves.png          training loss, baseline vs optimised
    metric_comparison.png        test metric bar chart
    roc_curves.png               ROC curves on the test set
    threshold_sensitivity.csv    precision/recall/F1 across decision thresholds
    threshold_sensitivity.png    threshold sensitivity plot
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from data import DEFAULT_PATH, SEED, load_and_prepare  # noqa: E402
from model import BASELINE_PARAMS, build_model  # noqa: E402
from bat import DIM_NAMES, BatAlgorithm, BatConfig  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics, plot_convergence, plot_learning_curves,
    plot_metric_comparison, plot_roc, threshold_table, plot_threshold,
    select_threshold_cv,
)

RESULTS = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="Path to the CSV. Default: data/default_of_credit_card_clients.csv "
                         "or data/UCI_Credit_Card.csv")
    ap.add_argument("--bats", type=int, default=10, help="Number of bats (search population).")
    ap.add_argument("--iters", type=int, default=8, help="Number of Bat iterations.")
    ap.add_argument("--metric", choices=["auc", "f1", "f2"], default="auc",
                    help="Search fitness: auc (default), f1, or f2 (weights recall more).")
    ap.add_argument("--search-max-iter", type=int, default=100,
                    help="Epochs per candidate during search. 200 matches the final models.")
    ap.add_argument("--search-size", type=int, default=6000,
                    help="Stratified subsample of the training set used ONLY inside the Bat search "
                         "(default 6000). Final models always use the full training data. "
                         "Use 0 to search on the full training set (slower).")
    ap.add_argument("--threshold", choices=["fixed", "cv"], default="fixed",
                    help="fixed: use 0.5 (default, reported numbers). cv: choose the cut-off on "
                         "training-data cross-validation, then apply it once to the test set.")
    ap.add_argument("--threshold-metric", choices=["f1", "f2"], default="f1",
                    help="Score maximised when choosing the cut-off in --threshold cv mode.")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    summary = {}

    # ---------- Rubric 1: dataset selection and problem definition ----------
    print("[1/5] Loading and cleaning dataset")
    ds = load_and_prepare(args.data)
    summary["dataset"] = ds.report

    # ---------- Rubric 2: ML model and baseline (no optimisation) -----------
    print("[2/5] Baseline ANN (default hyperparameters, not tuned)")
    baseline = build_model(BASELINE_PARAMS, max_iter=200)
    baseline.fit(ds.X_train, ds.y_train)
    summary["baseline"] = {"params": {k: str(v) for k, v in BASELINE_PARAMS.items()}}

    # ---------- Rubric 3: nature-inspired optimisation (Bat Algorithm) ------
    print("[3/5] Bat Algorithm tuning on training data (CV fitness)")
    X_search, y_search = ds.X_train, ds.y_train
    if 0 < args.search_size < len(ds.X_train):
        X_search, _, y_search, _ = train_test_split(
            ds.X_train, ds.y_train, train_size=args.search_size,
            random_state=SEED, stratify=ds.y_train)
    cfg = BatConfig(n_bats=args.bats, iterations=args.iters, metric=args.metric,
                    search_max_iter=args.search_max_iter, seed=SEED)
    bat = BatAlgorithm(cfg)
    bat_result = bat.run(X_search, y_search, verbose=True)
    summary["bat"] = {
        "config": cfg.__dict__,
        "dimensions": DIM_NAMES,
        "search_rows": int(len(X_search)),
        "operators": {
            "frequency": f"f in [{cfg.f_min}, {cfg.f_max}]",
            "velocity_update": "v = v + (x - x_best) * f",
            "local_walk": "x_best + 0.01 * mean(A) * N(0,1) when rand > r",
            "acceptance": f"rand < A and better fitness; A *= {cfg.alpha}; r grows with t",
        },
        "unique_evaluations": bat_result.n_evaluations,
        "best_cv_auc": bat_result.best_cv_auc,
        "best_params": {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in bat_result.best_params.items()},
    }
    pd.DataFrame(bat_result.history).drop(columns=["best_params"]).to_csv(
        os.path.join(RESULTS, "bat_history.csv"), index=False)

    # ---------- Rubric 3 (cont.): retrain optimised model and evaluate ------
    print("[4/5] Retraining optimised ANN on full training data and testing")
    optimised = build_model(bat_result.best_params, max_iter=200)
    optimised.fit(ds.X_train, ds.y_train)

    # Decision threshold: 0.5 by default, or chosen on TRAINING data only (cv mode)
    if args.threshold == "cv":
        t_base, _ = select_threshold_cv(build_model(BASELINE_PARAMS, max_iter=200),
                                        ds.X_train, ds.y_train, metric=args.threshold_metric)
        t_opt, _ = select_threshold_cv(build_model(bat_result.best_params, max_iter=200),
                                       ds.X_train, ds.y_train, metric=args.threshold_metric)
    else:
        t_base, t_opt = 0.5, 0.5
    print(f"Decision thresholds used: baseline {t_base:.2f}, optimised {t_opt:.2f} "
          f"({args.threshold} mode)")

    base_metrics = compute_metrics(baseline, ds.X_test, ds.y_test, threshold=t_base)
    opt_metrics = compute_metrics(optimised, ds.X_test, ds.y_test, threshold=t_opt)
    base_metrics["threshold"] = t_base
    opt_metrics["threshold"] = t_opt
    summary["thresholds"] = {"mode": args.threshold, "baseline": t_base, "optimised": t_opt,
                             "selection_metric": args.threshold_metric if args.threshold == "cv" else None}

    # ---------- Rubric 4: results, analysis, visualisation -------------------
    print("[5/5] Writing tables and figures")
    metrics_df = pd.DataFrame({"Baseline": base_metrics, "Bat-optimised": opt_metrics}).T
    metrics_df.round(4).to_csv(os.path.join(RESULTS, "metrics.csv"))

    plot_convergence(bat_result.history, os.path.join(RESULTS, "convergence.png"))
    plot_learning_curves(
        baseline.named_steps["mlpclassifier"].loss_curve_,
        optimised.named_steps["mlpclassifier"].loss_curve_,
        os.path.join(RESULTS, "learning_curves.png"))
    plot_metric_comparison(base_metrics, opt_metrics, os.path.join(RESULTS, "metric_comparison.png"))
    plot_roc({"Baseline": (baseline, "#7f7f7f"), "Bat-optimised": (optimised, "#2ca02c")},
             ds.X_test, ds.y_test, os.path.join(RESULTS, "roc_curves.png"))
    thr = threshold_table({"Baseline": (baseline, None), "Bat-optimised": (optimised, None)},
                          ds.X_test, ds.y_test)
    thr.round(4).to_csv(os.path.join(RESULTS, "threshold_sensitivity.csv"), index=False)
    plot_threshold(thr, os.path.join(RESULTS, "threshold_sensitivity.png"))

    summary["results"] = {"baseline": base_metrics, "optimised": opt_metrics}
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print("\nTest-set results (positive class = default):")
    print(metrics_df.round(4).to_string())
    print(f"\nRuntime: {summary['runtime_seconds']} s")


if __name__ == "__main__":
    main()
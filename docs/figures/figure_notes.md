# Figure notes

## Figure 1: Bat Algorithm convergence

Source: results/bat_history.csv. The solid line is the best CV AUC found so far; the dashed line is the average over all bats. Both rise over the iterations, meaning the search is finding better configurations. The best value rose by 0.0091 overall. When the solid line flattens, further iterations add little at this budget. A small gap between the two lines means the swarm has converged on similar solutions.

## Figure 2: Baseline vs optimised comparison

Source: results/metrics.csv. Each pair of bars is one metric on the same test set. Differences: Accuracy +0.0009, Precision +0.0009, Recall +0.0060, F1 +0.0055, ROC-AUC -0.0017. Every difference is small. A reader should conclude that the optimiser did not produce a meaningful gain; it is only the ~0.5 point variation expected from a single random split. Recall (about one in three defaulters caught) is the weakest score for both models.

## Figure 3: Threshold trade-off

Source: results/threshold_sensitivity.csv. Lowering the threshold classifies more clients as defaulters. Recall (solid) rises because more real defaulters are caught, while precision (dashed) falls because more of the flagged clients are false alarms. The choice of threshold is a business decision about which error is more costly. It should be made on validation data, not on the test set.

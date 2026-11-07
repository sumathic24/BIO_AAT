"""
Nature-inspired optimiser: the Bat Algorithm (Yang, 2010) tuning ANN hyperparameters.

Inspiration: microbats find prey by emitting loud pulses and listening to the
echoes. Each virtual bat has a position (a candidate ANN configuration), a
velocity, a frequency, a loudness A, and a pulse emission rate r.

Position encoding: 5 values in [0, 1], decoded into:
    dim 0 -> hidden layer 1 units      integer in [4, 128]
    dim 1 -> hidden layer 2 units      integer in [0, 64]   (0 = no second layer)
    dim 2 -> learning rate             10^x, x in [-4, -1]
    dim 3 -> L2 penalty alpha          10^x, x in [-6, -1]
    dim 4 -> activation                relu (<0.5) or tanh (>=0.5)

Update rules (per bat i, iteration t):
    Frequency  : f_i = f_min + (f_max - f_min) * beta,  beta ~ U(0, 1)   [echolocation]
    Velocity   : v_i = v_i + (x_i - x_best) * f_i                          [move toward best]
    Position   : x_new = x_i + v_i
    Local walk : if rand > r_i, x_new = x_best + eps * mean(A) * N(0,1)   [exploit near best]
    Acceptance : if rand < A_i and fitness(x_new) > fitness(x_i), accept x_new
                 then A_i <- alpha * A_i  (loudness decreases as the bat homes in)
                      r_i <- r0 * (1 - exp(-gamma * t))  (pulse rate increases)

Fitness: mean ROC-AUC over stratified k-fold cross-validation on the TRAINING
data only (AUC is used because the classes are imbalanced). The test set is
never used during the search.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

from model import build_model

DIM_NAMES = ["hidden1", "hidden2", "learning_rate", "alpha", "activation"]


def decode(x: np.ndarray) -> dict:
    g = np.clip(x, 0.0, 1.0)
    hidden1 = int(round(4 + g[0] * (128 - 4)))
    hidden2 = int(round(g[1] * 64))
    lr = 10 ** (-4 + g[2] * 3)
    alpha = 10 ** (-6 + g[3] * 5)
    activation = "relu" if g[4] < 0.5 else "tanh"
    hidden = (hidden1,) if hidden2 == 0 else (hidden1, hidden2)
    return {
        "hidden_layer_sizes": hidden,
        "learning_rate_init": lr,
        "alpha": alpha,
        "activation": activation,
    }


@dataclass
class BatConfig:
    n_bats: int = 10
    iterations: int = 8
    f_min: float = 0.0
    f_max: float = 2.0
    loudness0: float = 1.0      # A_0
    pulse_rate0: float = 0.5    # r_0
    alpha: float = 0.9          # loudness decay
    gamma: float = 0.9          # pulse rate growth
    cv_folds: int = 3
    seed: int = 42
    metric: str = "auc"          # "auc" (ROC-AUC), "f1", or "f2" (F-beta with beta = 2, weights recall)
    search_max_iter: int = 100   # epochs per candidate during search; 200 matches the final models


@dataclass
class BatResult:
    best_params: dict
    best_position: np.ndarray
    best_cv_auc: float
    history: dict = field(default_factory=dict)
    n_evaluations: int = 0


class BatAlgorithm:
    def __init__(self, cfg: BatConfig, dim: int = len(DIM_NAMES)):
        self.cfg = cfg
        self.dim = dim
        self.rng = np.random.default_rng(cfg.seed)
        self._cache: dict = {}
        self._cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
        self._scoring = {
            "auc": "roc_auc",
            "f1": "f1",
            "f2": make_scorer(fbeta_score, beta=2),
        }[cfg.metric]

    def fitness(self, x: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        key = tuple(np.round(x, 6))
        if key not in self._cache:
            model = build_model(decode(x), seed=self.cfg.seed, max_iter=self.cfg.search_max_iter)
            scores = cross_val_score(model, X, y, cv=self._cv, scoring=self._scoring, n_jobs=1)
            self._cache[key] = float(scores.mean())
        return self._cache[key]

    def run(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> BatResult:
        c, rng = self.cfg, self.rng
        pos = rng.random((c.n_bats, self.dim))
        vel = np.zeros((c.n_bats, self.dim))
        loud = np.full(c.n_bats, c.loudness0)
        pulse = np.full(c.n_bats, 0.0)          # r_i starts at 0 and rises with accepted moves (Yang, 2010)
        fit = np.array([self.fitness(p, X, y) for p in pos])

        best_idx = int(np.argmax(fit))
        best_pos, best_fit = pos[best_idx].copy(), float(fit[best_idx])
        history = {"iteration": [0], "best": [best_fit], "mean": [float(fit.mean())],
                   "worst": [float(fit.min())], "best_params": [decode(best_pos)]}
        if verbose:
            print(f"  iter  0 | best CV {self.cfg.metric} {best_fit:.4f} | mean {fit.mean():.4f}")

        for t in range(1, c.iterations + 1):
            for i in range(c.n_bats):
                beta = rng.random()
                freq = c.f_min + (c.f_max - c.f_min) * beta
                vel[i] = vel[i] + (pos[i] - best_pos) * freq
                cand = np.clip(pos[i] + vel[i], 0.0, 1.0)

                # Local random walk around the best bat
                if rng.random() > pulse[i]:
                    cand = np.clip(best_pos + 0.01 * loud.mean() * rng.normal(size=self.dim), 0.0, 1.0)

                cand_fit = self.fitness(cand, X, y)
                # Accept if loud enough and better (AUC is maximised)
                if rng.random() < loud[i] and cand_fit > fit[i]:
                    pos[i], fit[i] = cand, cand_fit
                    loud[i] *= c.alpha
                    pulse[i] = c.pulse_rate0 * (1 - np.exp(-c.gamma * t))

                if fit[i] > best_fit:
                    best_pos, best_fit = pos[i].copy(), float(fit[i])

            history["iteration"].append(t)
            history["best"].append(best_fit)
            history["mean"].append(float(fit.mean()))
            history["worst"].append(float(fit.min()))
            history["best_params"].append(decode(best_pos))
            if verbose:
                print(f"  iter {t:2d} | best CV {self.cfg.metric} {best_fit:.4f} | mean {fit.mean():.4f} "
                      f"| evals {len(self._cache)}")

        return BatResult(
            best_params=decode(best_pos),
            best_position=best_pos,
            best_cv_auc=best_fit,
            history=history,
            n_evaluations=len(self._cache),
        )

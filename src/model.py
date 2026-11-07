"""
ML/DL model: feed-forward Artificial Neural Network (multilayer perceptron).

Why an ANN: default risk depends on non-linear interactions between repayment
history, bill amounts, and payments. An MLP can learn these interactions, and
its hyperparameters strongly affect performance, which makes it a good target
for a nature-inspired optimiser.

Pipeline: StandardScaler -> MLPClassifier. Scaling is fitted inside the
pipeline, so it only ever sees training folds.
"""

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

# Un-optimised baseline: sensible defaults chosen a priori, NOT tuned.
BASELINE_PARAMS = {
    "hidden_layer_sizes": (32,),
    "activation": "relu",
    "learning_rate_init": 1e-3,
    "alpha": 1e-4,
}


def build_model(params: dict, seed: int = SEED, max_iter: int = 200):
    hidden = tuple(int(h) for h in params["hidden_layer_sizes"] if int(h) > 0)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation=params["activation"],
        learning_rate_init=float(params["learning_rate_init"]),
        alpha=float(params["alpha"]),
        solver="adam",
        batch_size=256,
        max_iter=max_iter,
        early_stopping=True,          # stop when validation score stops improving
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=seed,
    )
    return make_pipeline(StandardScaler(), clf)

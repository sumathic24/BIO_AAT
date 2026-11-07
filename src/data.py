"""
Dataset loading, cleaning, and splitting.

Dataset: Default of Credit Card Clients (UCI / Kaggle). 30,000 Taiwanese credit
card holders, 23 explanatory variables, binary target:
    default = 1 if the client defaulted on the next monthly payment, else 0.

Place the CSV in the project's data/ folder as:
    data/default_of_credit_card_clients.csv
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.20
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
# Accepted file names (the Kaggle and UCI downloads use different names)
ACCEPTED_NAMES = ["default_of_credit_card_clients.csv", "UCI_Credit_Card.csv"]
DEFAULT_PATH = os.path.join(DATA_DIR, ACCEPTED_NAMES[0])


def find_dataset() -> str:
    for name in ACCEPTED_NAMES:
        candidate = os.path.join(DATA_DIR, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Dataset not found. Copy the credit card CSV into the data/ folder and name it "
        "default_of_credit_card_clients.csv (or UCI_Credit_Card.csv)."
    )


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    report: dict  # cleaning / preprocessing summary, written to results


def _find_target_column(df: pd.DataFrame) -> str:
    # Kaggle names it 'default.payment.next.month', UCI 'default payment next month'
    for col in df.columns:
        if "default" in col.lower():
            return col
    raise ValueError("Could not find the target column (expected a name containing 'default').")


def load_and_prepare(path: str = None, seed: int = SEED) -> Dataset:
    path = path or find_dataset()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}.")

    df = pd.read_csv(path)
    n_raw = len(df)

    # --- Cleaning -------------------------------------------------------
    # Drop the record identifier: it carries no predictive information.
    id_cols = [c for c in df.columns if c.strip().lower() in ("id", "unnamed: 0")]
    df = df.drop(columns=id_cols)

    target = _find_target_column(df)
    df = df.apply(pd.to_numeric, errors="coerce")
    n_missing = int(df.isna().sum().sum())
    df = df.dropna()

    n_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # Undocumented categorical codes (documented in the UCI description):
    #   EDUCATION: 0, 5, 6 are "unknown" -> recoded to 4 ("other")
    #   MARRIAGE : 0 is "unknown"        -> recoded to 3 ("other")
    # PAY_* codes -2, -1, 0 have valid meanings (no consumption / paid duly /
    # revolving credit), so they are kept as they are.
    n_edu_fixed = n_mar_fixed = 0
    if "EDUCATION" in df.columns:
        mask = df["EDUCATION"].isin([0, 5, 6])
        n_edu_fixed = int(mask.sum())
        df.loc[mask, "EDUCATION"] = 4
    if "MARRIAGE" in df.columns:
        mask = df["MARRIAGE"] == 0
        n_mar_fixed = int(mask.sum())
        df.loc[mask, "MARRIAGE"] = 3

    y = df[target].astype(int).to_numpy()
    X_df = df.drop(columns=[target])
    X = X_df.to_numpy(dtype=float)

    # --- Splitting: stratified, because about 22% of clients default ------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    report = {
        "rows_raw": n_raw,
        "rows_after_cleaning": int(len(df)),
        "n_features": int(X.shape[1]),
        "id_columns_dropped": id_cols,
        "missing_values_dropped_rows": n_missing,
        "duplicate_rows_removed": n_dupes,
        "education_codes_recoded": n_edu_fixed,
        "marriage_codes_recoded": n_mar_fixed,
        "class_counts": {"no_default (0)": int((y == 0).sum()), "default (1)": int((y == 1).sum())},
        "default_rate_pct": round(100 * float(y.mean()), 2),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "split": "stratified 80/20",
        "transformations": "StandardScaler fitted on training data only (inside model pipeline)",
    }

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X_df.columns),
        report=report,
    )

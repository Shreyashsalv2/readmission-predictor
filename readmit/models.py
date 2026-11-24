"""Model pipelines: preprocessing + classifier, all scikit-learn."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [("num", StandardScaler(), numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)],
        verbose_feature_names_out=False,
    )


def logistic(numeric, categorical, C: float = 0.1) -> Pipeline:
    return Pipeline([("pre", preprocessor(numeric, categorical)),
                     ("clf", LogisticRegression(C=C, max_iter=5000))])


def gradient_boosting(numeric, categorical, seed: int = 0) -> Pipeline:
    return Pipeline([("pre", preprocessor(numeric, categorical)),
                     ("clf", HistGradientBoostingClassifier(
                         learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=100,
                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=30, random_state=seed))])


def baseline(numeric, categorical) -> Pipeline:
    return Pipeline([("pre", preprocessor(numeric, categorical)),
                     ("clf", DummyClassifier(strategy="prior"))])


def all_models(numeric, categorical, seed: int = 0, C: float = 0.1) -> dict[str, Pipeline]:
    return {
        "Baseline (always the average rate)": baseline(numeric, categorical),
        "Logistic regression": logistic(numeric, categorical, C),
        "Gradient boosting": gradient_boosting(numeric, categorical, seed),
    }

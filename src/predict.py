"""Shared inference: identical columns and history rules as training."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import FEATURE_COLUMNS, MODEL_PATH, PREPROCESSOR_INFO_PATH
from src.feature_engineering import build_feature_row, history_before_date


def load_model():
    pipeline = joblib.load(MODEL_PATH)
    with open(PREPROCESSOR_INFO_PATH, encoding="utf-8") as f:
        spec = json.load(f)
    return pipeline, spec


def predict_return_risk(pipeline, spec, orders: pd.DataFrame, customer_id: str, order_date, order_fields: dict) -> dict:
    history = history_before_date(orders, customer_id, order_date)
    X = build_feature_row(order_fields, history, order_date)
    X = X[FEATURE_COLUMNS]
    proba = float(pipeline.predict_proba(X)[0, 1])
    threshold = float(spec["threshold"])

    pre = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    Xt = pre.transform(X)
    names = list(pre.get_feature_names_out())
    clf_name = type(clf).__name__
    if clf_name in {"XGBClassifier", "RandomForestClassifier"}:
        values = shap.TreeExplainer(clf).shap_values(Xt)
        if isinstance(values, list):
            values = values[1]
        values = np.ravel(values)
    elif hasattr(clf, "coef_"):
        # Linear contribution to the log-odds: coefficient × transformed feature.
        values = np.ravel(clf.coef_) * np.ravel(Xt)
    else:
        explainer = shap.Explainer(clf.predict_proba, Xt)
        values = np.ravel(explainer(Xt).values[:, 1] if explainer(Xt).values.ndim == 3 else explainer(Xt).values)

    shap_df = pd.DataFrame({"feature": names, "contribution": values})
    shap_df["abs"] = shap_df["contribution"].abs()
    shap_df = shap_df.sort_values("abs", ascending=False).head(8)

    return {
        "probability": proba,
        "threshold": threshold,
        "classification": "High return risk" if proba >= threshold else "Lower return risk",
        "history": {
            k: history[k]
            for k in [
                "previous_order_count",
                "previous_return_count",
                "historical_return_rate",
                "historical_average_order_value",
                "days_since_previous_order",
                "is_first_order",
            ]
        },
        "shap": [
            {
                "feature": r.feature.replace("num__", "").replace("cat__", ""),
                "contribution": float(r.contribution),
                "direction": "increase" if r.contribution > 0 else "decrease",
            }
            for r in shap_df.itertuples(index=False)
        ],
        "shap_note": (
            "For this logistic regression, bars are coefficient × transformed feature value "
            "(contribution to log-odds). That is the same idea SHAP uses for linear models: "
            "how each feature moved the score. It does not prove causation."
        ),
    }

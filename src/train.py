"""Train baseline + LR + RF + XGBoost with a chronological split.

Model selection uses validation PR-AUC (threshold-free). The decision
threshold is then chosen on validation only and locked before test evaluation.
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    ARTIFACTS,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURES_PATH,
    FIGURES,
    MODEL_PATH,
    MODELS,
    NUMERIC_FEATURES,
    PREPROCESSOR_INFO_PATH,
    RANDOM_STATE,
    TRAIN_END,
    VAL_END,
)


def load_features(path=FEATURES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    return df


def temporal_split(df: pd.DataFrame):
    train = df[df["order_date"] <= TRAIN_END].copy()
    val = df[(df["order_date"] > TRAIN_END) & (df["order_date"] <= VAL_END)].copy()
    test = df[df["order_date"] > VAL_END].copy()
    return train, val, test


def xy(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df["returned"].astype(int).values
    return X, y


def preprocessor(scale_numeric: bool) -> ColumnTransformer:
    num_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", num_step, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def metrics_at_threshold(y_true, probs, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probs)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def sweep_threshold(y_true, probs) -> tuple[float, list[dict]]:
    rows = []
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        m = metrics_at_threshold(y_true, probs, float(t))
        rows.append(
            {
                "threshold": float(t),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "accuracy": m["accuracy"],
            }
        )
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
    return best_t, rows


def curve_payload(y_true, probs) -> dict:
    fpr, tpr, _ = roc_curve(y_true, probs)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr": {"precision": prec.tolist(), "recall": rec.tolist()},
        "calibration": {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()},
    }


def time_series_cv_pr_auc(pipeline, X, y, n_splits=3) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        pipeline.fit(X.iloc[tr], y[tr])
        if len(np.unique(y[te])) < 2:
            continue
        probs = pipeline.predict_proba(X.iloc[te])[:, 1]
        scores.append({"fold": fold, "pr_auc": float(average_precision_score(y[te], probs))})
    return {
        "strategy": "TimeSeriesSplit on training period only",
        "n_splits": n_splits,
        "folds": scores,
        "mean_pr_auc": float(np.mean([s["pr_auc"] for s in scores])) if scores else None,
        "reason": (
            "Orders are time-ordered. Random k-fold would mix later customer behavior "
            "into earlier folds. Expanding-window TimeSeriesSplit respects chronology."
        ),
    }


def build_models(y_train: np.ndarray) -> dict:
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = float(neg / pos) if pos else 1.0
    return {
        "Majority-class baseline": {
            "estimator": DummyClassifier(strategy="prior"),
            "scale": False,
        },
        "Logistic Regression": {
            "estimator": LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
            ),
            "scale": True,
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scale": False,
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=spw,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "scale": False,
        },
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df = load_features()
    train, val, test = temporal_split(df)
    split_info = {
        "train_period": f"2018-01-01 to {TRAIN_END}",
        "validation_period": f"2020-01-01 to {VAL_END}",
        "test_period": f"2021-01-01 to {df['order_date'].max().date()}",
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "train_return_rate": float(train["returned"].mean()),
        "val_return_rate": float(val["returned"].mean()),
        "test_return_rate": float(test["returned"].mean()),
        "rationale": (
            "The labeled table spans 2018–2021. Earlier years train the model; 2020 is used "
            "for model selection and threshold locking; 2021 is touched once for final evaluation."
        ),
    }
    with open(ARTIFACTS / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    X_train, y_train = xy(train)
    X_val, y_val = xy(val)
    X_test, y_test = xy(test)

    models = build_models(y_train)
    val_rows = []
    fitted = {}

    for name, spec in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor(spec["scale"])),
                ("classifier", clone(spec["estimator"])),
            ]
        )
        t0 = time.time()
        pipe.fit(X_train, y_train)
        elapsed = time.time() - t0
        val_probs = pipe.predict_proba(X_val)[:, 1]
        # Threshold-free metrics for selection; 0.5 only as a comparable operating point
        m = metrics_at_threshold(y_val, val_probs, 0.5)
        m["model"] = name
        m["train_seconds"] = float(elapsed)
        val_rows.append(m)
        fitted[name] = (pipe, val_probs)
        print(f"{name}: val PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} F1@0.5={m['f1']:.4f}")

    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(ARTIFACTS / "validation_comparison.csv", index=False)

    # Select by validation PR-AUC (not by assuming XGBoost wins).
    best_name = max(val_rows, key=lambda r: r["pr_auc"])["model"]
    best_pipe, best_val_probs = fitted[best_name]
    locked_threshold, tuning_rows = sweep_threshold(y_val, best_val_probs)
    pd.DataFrame(tuning_rows).to_csv(ARTIFACTS / "threshold_tuning.csv", index=False)

    selection = {
        "selected_model": best_name,
        "selection_metric": "validation PR-AUC",
        "validation_pr_auc": float(max(r["pr_auc"] for r in val_rows if r["model"] == best_name)),
        "locked_threshold": locked_threshold,
        "threshold_rule": (
            "Maximize F1 on the validation set over thresholds 0.05–0.95. "
            "The threshold is locked before any test evaluation."
        ),
        "imbalance_handling": (
            f"Positive class is {float(y_train.mean()):.1%} in training. "
            "Used class_weight='balanced' / scale_pos_weight. "
            "SMOTE was not used because interpolating neighbors can mix customers across time."
        ),
        "calibration_note": (
            "Class weighting improves recall but typically pushes probabilities upward, "
            "so Brier score can be worse than a constant base-rate predictor. "
            "Model selection used PR-AUC (ranking). Displayed probabilities should be treated as risk scores, "
            "not as perfectly calibrated frequencies."
        ),
    }

    # Time-aware CV on training data for the selected architecture (refit later on full train).
    cv_spec = models[best_name]
    cv_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor(cv_spec["scale"])),
            ("classifier", clone(cv_spec["estimator"])),
        ]
    )
    cv_result = time_series_cv_pr_auc(cv_pipe, X_train, y_train)
    # Refit selected model on full training set (already fitted above; keep that fit).
    with open(ARTIFACTS / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(cv_result, f, indent=2)

    test_probs = best_pipe.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at_threshold(y_test, test_probs, locked_threshold)
    test_metrics["model"] = best_name
    val_at_locked = metrics_at_threshold(y_val, best_val_probs, locked_threshold)
    val_at_locked["model"] = best_name

    curves = {
        "validation": curve_payload(y_val, best_val_probs),
        "test": curve_payload(y_test, test_probs),
    }
    with open(ARTIFACTS / "curve_data.json", "w", encoding="utf-8") as f:
        json.dump(curves, f)
    with open(ARTIFACTS / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    with open(ARTIFACTS / "val_metrics_locked_threshold.json", "w", encoding="utf-8") as f:
        json.dump(val_at_locked, f, indent=2)
    with open(ARTIFACTS / "model_selection.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    joblib.dump(best_pipe, MODEL_PATH)
    spec = {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "selected_model": best_name,
        "threshold": locked_threshold,
    }
    with open(PREPROCESSOR_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    print(f"Selected: {best_name} | locked threshold: {locked_threshold}")
    print(f"Test PR-AUC={test_metrics['pr_auc']:.4f} ROC-AUC={test_metrics['roc_auc']:.4f} "
          f"F1={test_metrics['f1']:.4f} Brier={test_metrics['brier_score']:.4f}")


if __name__ == "__main__":
    main()

"""SHAP explanations for the selected model. Association, not causation."""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import ARTIFACTS, FEATURES_PATH, FIGURES, MODEL_PATH, TRAIN_END, VAL_END
from src.train import temporal_split, xy


def transformed_feature_names(pipeline) -> list[str]:
    pre = pipeline.named_steps["preprocessor"]
    return list(pre.get_feature_names_out())


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURES_PATH)
    df["order_date"] = pd.to_datetime(df["order_date"])
    train, val, _test = temporal_split(df)
    X_val, _ = xy(val)

    pre = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    sample = X_val.sample(n=min(400, len(X_val)), random_state=42)
    Xt = pre.transform(sample)
    names = transformed_feature_names(pipeline)

    clf_name = type(clf).__name__
    if clf_name in {"XGBClassifier", "RandomForestClassifier"}:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(Xt)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = float(np.ravel(expected)[-1])
        else:
            expected = float(expected)
    else:
        explainer = shap.LinearExplainer(clf, Xt)
        shap_values = explainer.shap_values(Xt)
        expected = float(np.ravel(explainer.expected_value)[0])

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(15)
    )
    importance.to_csv(ARTIFACTS / "shap_global.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importance["feature"][::-1], importance["mean_abs_shap"][::-1], color="#0f3d5e")
    ax.set_xlabel("Mean |SHAP| (validation sample)")
    ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_global.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(8, 5.5))
    shap.summary_plot(shap_values, Xt, feature_names=names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()

    # Example local explanation: first validation row (real order)
    one = X_val.iloc[[0]]
    meta = val.iloc[[0]][["order_id", "customer_id", "order_date", "returned"]]
    Xt1 = pre.transform(one)
    if clf_name in {"XGBClassifier", "RandomForestClassifier"}:
        local = shap.TreeExplainer(clf).shap_values(Xt1)
        if isinstance(local, list):
            local = local[1]
        local = np.ravel(local)
    else:
        local = np.ravel(shap.LinearExplainer(clf, Xt).shap_values(Xt1))

    local_df = pd.DataFrame({"feature": names, "shap": local})
    local_df["abs"] = local_df["shap"].abs()
    local_df = local_df.sort_values("abs", ascending=False).head(8)
    example = {
        "order_id": str(meta["order_id"].iloc[0]),
        "customer_id": str(meta["customer_id"].iloc[0]),
        "order_date": str(pd.to_datetime(meta["order_date"].iloc[0]).date()),
        "actual_returned": int(meta["returned"].iloc[0]),
        "base_value": expected,
        "contributions": [
            {
                "feature": r.feature,
                "shap": float(r.shap),
                "direction": "increase" if r.shap > 0 else "decrease",
            }
            for r in local_df.itertuples(index=False)
        ],
        "plain_language": (
            "SHAP shows how each feature contributed to moving this prediction higher or lower "
            "relative to the model's baseline. It does not prove that a feature caused a return."
        ),
    }
    with open(ARTIFACTS / "shap_example.json", "w", encoding="utf-8") as f:
        json.dump(example, f, indent=2)

    global_payload = {
        "model_type": clf_name,
        "sample_size": int(len(sample)),
        "sample_source": "validation set (not test)",
        "top_features": importance.to_dict(orient="records"),
        "plain_language": (
            "Global SHAP ranks features by average absolute contribution to predictions. "
            "This is model attribution, not a causal ranking."
        ),
    }
    with open(ARTIFACTS / "shap_global.json", "w", encoding="utf-8") as f:
        json.dump(global_payload, f, indent=2)
    print("SHAP artifacts written")


if __name__ == "__main__":
    main()

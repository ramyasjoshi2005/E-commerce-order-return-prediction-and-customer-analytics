"""Flask API + static frontend for the return-prediction project."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, send_file

from src.config import ARTIFACTS, FIGURES, ORDERS_CLEAN_PATH, PREPROCESSOR_INFO_PATH
from src.predict import load_model, predict_return_risk

FRONTEND = ROOT / "frontend"

# Do not use static_url_path="". That registers /<path:filename> and intercepts /api/*.
app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="/static")

_ORDERS = None
_MODEL = None
_SPEC = None


def orders() -> pd.DataFrame:
    global _ORDERS
    if _ORDERS is None:
        df = pd.read_csv(ORDERS_CLEAN_PATH)
        df["order_date"] = pd.to_datetime(df["order_date"])
        _ORDERS = df
    return _ORDERS


def model_bundle():
    global _MODEL, _SPEC
    if _MODEL is None:
        _MODEL, _SPEC = load_model()
    return _MODEL, _SPEC


def read_json(name: str):
    with open(ARTIFACTS / name, encoding="utf-8") as f:
        return json.load(f)


@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/figures/<path:filename>")
def figures(filename: str):
    return send_file(FIGURES / filename)


@app.get("/api/overview")
def api_overview():
    quality = read_json("data_quality.json")
    eda = read_json("eda.json")
    split = read_json("split_info.json")
    sel = read_json("model_selection.json")
    test = read_json("test_metrics.json")
    return jsonify(
        {
            "title": "E-Commerce Return Prediction & Customer Analytics",
            "business_problem": (
                "Can we predict whether a newly placed e-commerce order will be returned "
                "using only information available at the time the order is placed, and "
                "identify factors associated with returns?"
            ),
            "objective": (
                "Estimate return probability at order placement, compare simple models fairly, "
                "and explain predictions without claiming causation."
            ),
            "prediction_point": (
                "Prediction is made when the customer places the order. "
                "Shipping cost, ship date, profit, return date, and return reason are not used."
            ),
            "dataset": quality,
            "overall_return_rate": eda["overall"]["return_rate"],
            "split": split,
            "selected_model": sel["selected_model"],
            "threshold": sel["locked_threshold"],
            "test_pr_auc": test["pr_auc"],
            "methodology": [
                "Inspect data quality and remove duplicate order_id rows that leaked labels into history.",
                "Engineer current-order fields and point-in-time customer history.",
                "Test whether prior returns are associated with later return rates (two-proportion z-test).",
                "Train baseline, logistic regression, random forest, and XGBoost on 2018–2019.",
                "Select on 2020 PR-AUC; lock an F1-maximizing threshold; evaluate 2021 once.",
                "Explain with SHAP; serve a small Flask app for customer-level what-if prediction.",
            ],
        }
    )


@app.get("/api/eda")
def api_eda():
    return jsonify(
        {
            "quality": read_json("data_quality.json"),
            "eda": read_json("eda.json"),
            "leakage_audit": pd.read_csv(ARTIFACTS / "leakage_audit.csv").to_dict(orient="records"),
            "sql": read_json("sql_results.json"),
        }
    )


@app.get("/api/statistics")
def api_statistics():
    return jsonify(read_json("hypothesis_test.json"))


@app.get("/api/model")
def api_model():
    val = pd.read_csv(ARTIFACTS / "validation_comparison.csv")
    keep = [
        "model",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "brier_score",
    ]
    keep = [c for c in keep if c in val.columns]
    return jsonify(
        {
            "validation_comparison": val[keep].to_dict(orient="records"),
            "selection": read_json("model_selection.json"),
            "validation_locked": read_json("val_metrics_locked_threshold.json"),
            "test": read_json("test_metrics.json"),
            "threshold_tuning": pd.read_csv(ARTIFACTS / "threshold_tuning.csv").to_dict(orient="records"),
            "curves": read_json("curve_data.json"),
            "cv": read_json("cv_results.json"),
            "split": read_json("split_info.json"),
        }
    )


@app.get("/api/explainability")
def api_explain():
    return jsonify(
        {
            "global": read_json("shap_global.json"),
            "example": read_json("shap_example.json"),
            "insights": read_json("business_insights.json"),
        }
    )


@app.get("/api/interview")
def api_interview():
    return jsonify(read_json("interview_guide.json"))


@app.get("/api/options")
def api_options():
    df = orders()
    return jsonify(
        {
            "regions": sorted(df["region"].unique().tolist()),
            "segments": sorted(df["segment"].unique().tolist()),
            "priorities": sorted(df["order_priority"].unique().tolist()),
            "countries": sorted(df["country"].unique().tolist()),
            "markets": sorted(df["market"].unique().tolist()),
            "date_min": df["order_date"].min().strftime("%Y-%m-%d"),
            "date_max": df["order_date"].max().strftime("%Y-%m-%d"),
            "india_available": bool((df["country"] == "India").any()),
        }
    )


@app.get("/api/customers")
def api_customers():
    df = orders()
    q = (request.args.get("q") or "").strip().lower()
    agg = (
        df.groupby("customer_id", as_index=False)
        .agg(n_orders=("order_id", "count"), n_returns=("returned", "sum"), last_order=("order_date", "max"))
        .sort_values(["n_orders", "customer_id"], ascending=[False, True])
    )
    if q:
        agg = agg[agg["customer_id"].str.lower().str.contains(q, regex=False)]
    agg["last_order"] = pd.to_datetime(agg["last_order"]).dt.strftime("%Y-%m-%d")
    records = agg.head(400).to_dict(orient="records")
    return jsonify({"customers": records, "n_total": int(df["customer_id"].nunique())})


@app.get("/api/customer-history")
def api_history():
    customer_id = request.args.get("customer_id", "")
    order_date = request.args.get("date")
    if not customer_id or not order_date:
        return jsonify({"error": "customer_id and date are required"}), 400
    df = orders()
    if customer_id not in set(df["customer_id"]):
        return jsonify({"error": "Unknown customer_id"}), 404
    from src.feature_engineering import history_before_date

    hist = history_before_date(df, customer_id, order_date)
    profile = df[df["customer_id"] == customer_id].iloc[0]
    hist["customer_id"] = customer_id
    hist["segment"] = str(profile["segment"])
    hist["region"] = str(profile["region"])
    hist["country"] = str(profile["country"])
    hist["market"] = str(profile["market"])
    return jsonify(hist)


@app.post("/api/predict")
def api_predict():
    data = request.get_json(force=True)
    customer_id = data.get("customer_id")
    order_date = data.get("order_date")
    if not customer_id or not order_date:
        return jsonify({"error": "customer_id and order_date are required"}), 400
    df = orders()
    if customer_id not in set(df["customer_id"]):
        return jsonify({"error": "Unknown customer_id"}), 404
    pipeline, spec = model_bundle()
    fields = {
        "region": data["region"],
        "segment": data["segment"],
        "order_priority": data["order_priority"],
        "order_value": data["order_value"],
        "quantity": data["quantity"],
        "discount": data["discount"],
        "max_discount": data.get("max_discount", data["discount"]),
        "number_of_products": data.get("number_of_products", 1),
        "number_of_categories": data.get("number_of_categories", 1),
    }
    result = predict_return_risk(pipeline, spec, df, customer_id, order_date, fields)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

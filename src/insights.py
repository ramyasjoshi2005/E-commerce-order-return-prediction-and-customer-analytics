"""Write a small insights JSON from computed artifacts (no invented numbers)."""

from __future__ import annotations

import json

from src.config import ARTIFACTS, FEATURES_PATH
import pandas as pd


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    with open(ARTIFACTS / "hypothesis_test.json", encoding="utf-8") as f:
        test = json.load(f)
    with open(ARTIFACTS / "eda.json", encoding="utf-8") as f:
        eda = json.load(f)
    with open(ARTIFACTS / "model_selection.json", encoding="utf-8") as f:
        sel = json.load(f)
    with open(ARTIFACTS / "test_metrics.json", encoding="utf-8") as f:
        test_m = json.load(f)

    def top_group(rows, higher=True):
        rows = sorted(rows, key=lambda r: r["return_rate"], reverse=higher)
        return rows[0], rows[-1]

    hi_disc, lo_disc = top_group(eda["return_rate_by_discount"])
    hi_val, lo_val = top_group(eda["return_rate_by_order_value"])
    hi_reg, lo_reg = top_group(eda["return_rate_by_region"])

    insights = {
        "items": [
            {
                "title": "US region",
                "finding": (
                    f"{hi_reg['group']} had the highest observed regional return rate "
                    f"({hi_reg['return_rate']:.1%}, n={hi_reg['n_orders']:,}) versus "
                    f"{lo_reg['group']} ({lo_reg['return_rate']:.1%}, n={lo_reg['n_orders']:,})."
                ),
                "association_not_causation": (
                    "This is a descriptive geographic association in this US Superstore table. "
                    "It does not identify an operational cause (mix, policy, or reporting differences could all contribute)."
                ),
            },
            {
                "title": "Prior returns and later orders",
                "finding": (
                    f"Orders placed after at least one previous return had an observed return rate of "
                    f"{test['p1']:.1%} (n={test['n1']:,}), compared with {test['p2']:.1%} (n={test['n2']:,}) "
                    f"for orders with no previous returns. Two-proportion z = {test['z_statistic']:.2f}, "
                    f"p = {test['p_value']:.4g}, risk ratio {test['risk_ratio']:.2f}."
                ),
                "association_not_causation": (
                    "This is an observational association. Previous returns may mark product mix, "
                    "customer context, or other factors — they are not shown to cause later returns."
                ),
            },
            {
                "title": "Discount at order placement",
                "finding": (
                    f"The highest observed return rate among discount bins was {hi_disc['group']} "
                    f"({hi_disc['return_rate']:.1%}, n={hi_disc['n_orders']:,}). "
                    f"The lowest was {lo_disc['group']} ({lo_disc['return_rate']:.1%}, n={lo_disc['n_orders']:,})."
                ),
                "association_not_causation": (
                    "Discount bins can overlap with different products and promotions. "
                    "The model may use discount as a correlate, not a proven lever."
                ),
            },
            {
                "title": "Order value",
                "finding": (
                    f"Among value bins, {hi_val['group']} had the highest observed return rate "
                    f"({hi_val['return_rate']:.1%}) and {lo_val['group']} the lowest "
                    f"({lo_val['return_rate']:.1%})."
                ),
                "association_not_causation": "Order value is known at checkout; this does not imply that changing price would change returns.",
            },
        ],
        "recommendation": (
            "The model can flag higher-risk orders at placement for targeted review or customer-experience "
            "follow-up. It should not be used to automatically block customers. Any operational intervention "
            "(holds, outreach, policy changes) would need a controlled experiment / A/B test to estimate "
            "causal impact on return rate or cost. This project does not claim a percentage reduction in returns."
        ),
        "model_context": {
            "selected_model": sel["selected_model"],
            "test_pr_auc": test_m["pr_auc"],
            "test_roc_auc": test_m["roc_auc"],
            "locked_threshold": sel["locked_threshold"],
            "overall_return_rate": float(df["returned"].mean()),
        },
    }
    with open(ARTIFACTS / "business_insights.json", "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)

    with open(ARTIFACTS / "split_info.json", encoding="utf-8") as f:
        split = json.load(f)
    with open(ARTIFACTS / "data_quality.json", encoding="utf-8") as f:
        quality = json.load(f)
    with open(ARTIFACTS / "cv_results.json", encoding="utf-8") as f:
        cv = json.load(f)
    with open(ARTIFACTS / "shap_global.json", encoding="utf-8") as f:
        shap_g = json.load(f)

    val = pd.read_csv(ARTIFACTS / "validation_comparison.csv")
    val_lines = "; ".join(
        f"{r.model}: PR-AUC {r.pr_auc:.3f}, ROC-AUC {r.roc_auc:.3f}"
        for r in val.itertuples(index=False)
    )
    top_shap = ", ".join(f"{r['feature']}" for r in shap_g["top_features"][:5])

    guide = {
        "answers": [
            {
                "q": "Problem statement",
                "a": "Predict whether a newly placed order will be returned using only information available at order placement, and identify associated factors.",
            },
            {
                "q": "Why this problem?",
                "a": "Returns consume logistics cost after checkout. A probability at placement is early enough to review high-risk orders without waiting for shipment data.",
            },
            {
                "q": "Why this target?",
                "a": "Returned vs not returned is the operational decision unit. The business action (review, outreach) is at order level, so the label is order-level.",
            },
            {
                "q": "Prediction point",
                "a": "The moment the customer places the order. Profit, shipping cost, ship date, return date, and return reason are excluded.",
            },
            {
                "q": "How leakage was prevented",
                "a": (
                    "Duplicate order_id rows that counted an order as its own history were dropped. "
                    "Customer history uses only earlier orders (date, then order_id). "
                    "The 2021 test set was not used for model selection or threshold tuning."
                ),
            },
            {
                "q": "Feature engineering",
                "a": (
                    "Current-order fields (value, quantity, discount, segment, region, priority, calendar) "
                    "plus point-in-time history (prior orders, prior returns, historical return rate, "
                    "average prior order value, days since previous order)."
                ),
            },
            {
                "q": "Why the statistical test was selected",
                "a": (
                    f"The research question is a comparison of two proportions. A two-sided two-proportion z-test "
                    f"matches that question. Result: z={test['z_statistic']:.2f}, p={test['p_value']:.4g}. "
                    "Significance is not causation."
                ),
            },
            {
                "q": "Why these ML models were selected",
                "a": "A majority baseline shows the imbalance. Logistic regression is a linear, inspectable model. Random forest and XGBoost capture non-linear interactions. The set is small on purpose.",
            },
            {
                "q": "Why the final model was selected",
                "a": (
                    f"Validation PR-AUC (2020), not a pre-chosen algorithm. Comparison: {val_lines}. "
                    f"Selected: {sel['selected_model']}."
                ),
            },
            {
                "q": "Evaluation metrics",
                "a": (
                    f"On the untouched 2021 test set: ROC-AUC {test_m['roc_auc']:.3f}, PR-AUC {test_m['pr_auc']:.3f}, "
                    f"precision {test_m['precision']:.3f}, recall {test_m['recall']:.3f}, F1 {test_m['f1']:.3f}, "
                    f"accuracy {test_m['accuracy']:.3f} (secondary), Brier {test_m['brier_score']:.3f}."
                ),
            },
            {
                "q": "Why PR-AUC matters",
                "a": (
                    f"Overall return rate is {float(df['returned'].mean()):.1%}. Accuracy is easy to inflate by always predicting no return. "
                    "PR-AUC focuses on ranking the positive class."
                ),
            },
            {
                "q": "Threshold selection",
                "a": (
                    f"Threshold {sel['locked_threshold']} was chosen to maximize F1 on validation, then locked. "
                    "The test set was not used to pick it."
                ),
            },
            {
                "q": "SHAP explanation",
                "a": (
                    f"SHAP attributes how features move a prediction up or down versus a baseline. "
                    f"It is not causal. Top global |SHAP| features (validation sample): {top_shap}."
                ),
            },
            {
                "q": "Main business insight",
                "a": insights["items"][0]["finding"],
            },
            {
                "q": "Business recommendation",
                "a": insights["recommendation"],
            },
            {
                "q": "Limitations",
                "a": (
                    f"Labeled data is US Superstore orders {quality['date_min']} to {quality['date_max']} "
                    f"({quality['n_orders']:,} orders). Product category is not in the labeled table. "
                    f"Train {split['n_train']:,} / val {split['n_val']:,} / test {split['n_test']:,}. "
                    f"TimeSeriesSplit mean PR-AUC on train folds: {cv.get('mean_pr_auc')}. "
                    "No experiment measures causal return reduction. India is not in this labeled table."
                ),
            },
        ]
    }
    with open(ARTIFACTS / "interview_guide.json", "w", encoding="utf-8") as f:
        json.dump(guide, f, indent=2)
    print("Wrote artifacts/business_insights.json and interview_guide.json")


if __name__ == "__main__":
    main()

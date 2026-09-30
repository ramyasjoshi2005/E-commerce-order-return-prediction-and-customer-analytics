"""Save evaluation plots and a leakage audit from trained artifacts."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import ARTIFACTS, FIGURES

sns.set_theme(style="whitegrid")


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_curves() -> None:
    with open(ARTIFACTS / "curve_data.json", encoding="utf-8") as f:
        curves = json.load(f)
    test = curves["test"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(test["roc"]["fpr"], test["roc"]["tpr"], color="#0f3d5e")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test)")
    _save(fig, "roc_test.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(test["pr"]["recall"], test["pr"]["precision"], color="#0f3d5e")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test)")
    _save(fig, "pr_test.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(test["calibration"]["prob_pred"], test["calibration"]["prob_true"], "o-", color="#0f3d5e")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed return rate")
    ax.set_title("Calibration curve (test)")
    _save(fig, "calibration_test.png")


def plot_threshold() -> None:
    tuning = pd.read_csv(ARTIFACTS / "threshold_tuning.csv")
    with open(ARTIFACTS / "model_selection.json", encoding="utf-8") as f:
        sel = json.load(f)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(tuning["threshold"], tuning["precision"], label="Precision")
    ax.plot(tuning["threshold"], tuning["recall"], label="Recall")
    ax.plot(tuning["threshold"], tuning["f1"], label="F1")
    ax.axvline(sel["locked_threshold"], color="black", linestyle="--", label="Locked threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold analysis (validation only)")
    ax.legend()
    _save(fig, "threshold_validation.png")


def plot_confusion() -> None:
    with open(ARTIFACTS / "test_metrics.json", encoding="utf-8") as f:
        m = json.load(f)
    cm = m["confusion_matrix"]
    mat = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(
        mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Pred no return", "Pred return"],
        yticklabels=["Actual no return", "Actual return"],
    )
    ax.set_title("Confusion matrix (test, locked threshold)")
    _save(fig, "confusion_test.png")


def plot_eda() -> None:
    with open(ARTIFACTS / "eda.json", encoding="utf-8") as f:
        eda = json.load(f)

    def bar(key, title, filename, xlabel="Return rate"):
        rows = eda[key]
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        labels = [r["group"] for r in rows]
        rates = [r["return_rate"] * 100 for r in rows]
        ax.barh(labels, rates, color="#1f6f8b")
        ax.set_xlabel("Observed return rate (%)")
        ax.set_title(title)
        _save(fig, filename)

    bar("return_rate_by_segment", "Return rate by customer segment", "eda_segment.png")
    bar("return_rate_by_discount", "Return rate by discount range", "eda_discount.png")
    bar("return_rate_by_order_value", "Return rate by order value", "eda_value.png")
    bar("return_rate_by_prior_returns", "Return rate by prior-return history", "eda_history.png")
    bar("return_rate_by_region", "Return rate by US region", "eda_region.png")


def leakage_audit() -> None:
    rows = [
        {
            "feature": "order_value / quantity / discount",
            "available_at_order_placement": "Yes",
            "safe": "Yes",
            "reason": "Known at checkout.",
        },
        {
            "feature": "region, segment, order_priority, calendar fields",
            "available_at_order_placement": "Yes",
            "safe": "Yes",
            "reason": "Known when the order is placed.",
        },
        {
            "feature": "previous_order_count, previous_return_count, historical_return_rate, historical_average_order_value, days_since_previous_order",
            "available_at_order_placement": "Yes",
            "safe": "Yes",
            "reason": "Computed from orders strictly before the current order (date, then order_id).",
        },
        {
            "feature": "returned (target)",
            "available_at_order_placement": "No",
            "safe": "No (label only)",
            "reason": "Outcome after the order. Never used as a feature.",
        },
        {
            "feature": "profit, shipping_cost, ship_date",
            "available_at_order_placement": "No",
            "safe": "Excluded",
            "reason": "Post-order / settlement information.",
        },
        {
            "feature": "return reason / return date",
            "available_at_order_placement": "No",
            "safe": "Excluded",
            "reason": "Not present; would be post-return leakage if used.",
        },
    ]
    pd.DataFrame(rows).to_csv(ARTIFACTS / "leakage_audit.csv", index=False)


def main() -> None:
    plot_curves()
    plot_threshold()
    plot_confusion()
    plot_eda()
    leakage_audit()
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()

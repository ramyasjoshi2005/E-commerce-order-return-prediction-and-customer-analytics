"""EDA summaries used by the application. Every chart answers a return-rate question."""

from __future__ import annotations

import json

import pandas as pd

from src.config import ARTIFACTS, FEATURES_PATH


def _rate_table(df: pd.DataFrame, col: str) -> list[dict]:
    g = df.groupby(col, dropna=False, observed=True)["returned"].agg(["count", "sum", "mean"]).reset_index()
    g.columns = ["group", "n_orders", "n_returns", "return_rate"]
    return [
        {
            "group": str(r.group),
            "n_orders": int(r.n_orders),
            "n_returns": int(r.n_returns),
            "return_rate": float(r.return_rate),
        }
        for r in g.itertuples(index=False)
    ]


def _bin_rate(df: pd.DataFrame, series: pd.Series, bins, labels, name: str) -> list[dict]:
    tmp = df.copy()
    tmp[name] = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    return _rate_table(tmp, name)


def build_eda(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    hist_group = df["previous_return_count"].gt(0).map(
        {True: "Had previous return(s)", False: "No previous return"}
    )
    tmp = df.copy()
    tmp["history_group"] = hist_group

    return {
        "overall": {
            "n_orders": int(len(df)),
            "return_rate": float(df["returned"].mean()),
            "n_returns": int(df["returned"].sum()),
        },
        "return_rate_by_segment": _rate_table(df, "segment"),
        "return_rate_by_region": _rate_table(df, "region"),
        "return_rate_by_priority": _rate_table(df, "order_priority"),
        "return_rate_by_discount": _bin_rate(
            df,
            df["discount"],
            bins=[-0.01, 0.0, 0.2, 0.4, 1.0],
            labels=["0%", "0–20%", "20–40%", ">40%"],
            name="discount_bin",
        ),
        "return_rate_by_order_value": _bin_rate(
            df,
            df["order_value"],
            bins=[-0.01, 50, 150, 400, df["order_value"].max() + 1],
            labels=["≤$50", "$50–150", "$150–400", ">$400"],
            name="value_bin",
        ),
        "return_rate_by_prior_returns": _rate_table(tmp, "history_group"),
        "return_rate_by_year": _rate_table(df.assign(year=df["order_date"].dt.year), "year"),
        "interpretations": {
            "segment": "Observed return rates by customer segment. Differences are associations, not causal effects of segment.",
            "discount": "Observed return rates by discount at order placement. Higher discounts may coincide with different products or promotions.",
            "order_value": "Observed return rates by order value. Value is known at checkout.",
            "history": "Point-in-time prior returns versus subsequent return rate — the same grouping used in the hypothesis test.",
        },
    }


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    payload = build_eda(df)
    with open(ARTIFACTS / "eda.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("EDA written to artifacts/eda.json")


if __name__ == "__main__":
    main()

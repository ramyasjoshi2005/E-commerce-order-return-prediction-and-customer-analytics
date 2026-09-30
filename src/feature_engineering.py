"""Point-in-time feature engineering used by both training and the web app.

History for order t uses only orders with:
  (order_date < t.order_date)
  OR (order_date == t.order_date AND order_id < t.order_id)

The current order is never included in its own historical statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURES_PATH, FIRST_ORDER_DAYS, ORDERS_CLEAN_PATH


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["order_date"] = pd.to_datetime(out["order_date"])
    out["order_month"] = out["order_date"].dt.month
    out["order_day_of_week"] = out["order_date"].dt.dayofweek
    out["is_weekend"] = out["order_day_of_week"].isin([5, 6]).astype(int)
    return out


def add_customer_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized, leakage-safe customer history.

    Rows are sorted by customer, date, then order_id so same-day orders have a
    deterministic order and cannot leak backward.
    """
    out = df.copy()
    out["order_date"] = pd.to_datetime(out["order_date"])
    out = out.sort_values(["customer_id", "order_date", "order_id"]).reset_index(drop=True)

    grouped = out.groupby("customer_id", sort=False)

    out["previous_order_count"] = grouped.cumcount()
    out["previous_return_count"] = grouped["returned"].cumsum() - out["returned"]
    prev_value_cumsum = grouped["order_value"].cumsum() - out["order_value"]
    prev_n = out["previous_order_count"].to_numpy()
    out["historical_average_order_value"] = np.where(
        prev_n > 0, prev_value_cumsum.to_numpy() / np.maximum(prev_n, 1), 0.0
    )
    out["historical_return_rate"] = np.where(
        prev_n > 0, out["previous_return_count"].to_numpy() / np.maximum(prev_n, 1), 0.0
    )

    prev_date = grouped["order_date"].shift(1)
    out["days_since_previous_order"] = (out["order_date"] - prev_date).dt.days
    out["is_first_order"] = out["previous_order_count"].eq(0).astype(int)
    out["days_since_previous_order"] = out["days_since_previous_order"].fillna(FIRST_ORDER_DAYS)

    return out


def history_before_date(orders: pd.DataFrame, customer_id: str, as_of_date) -> dict:
    """Customer history strictly before as_of_date. Used by the Prediction Studio."""
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    as_of = pd.to_datetime(as_of_date)
    hist = orders[
        (orders["customer_id"] == customer_id) & (orders["order_date"] < as_of)
    ].sort_values(["order_date", "order_id"])

    n = int(len(hist))
    n_ret = int(hist["returned"].sum()) if n else 0
    avg_value = float(hist["order_value"].mean()) if n else 0.0
    rate = float(n_ret / n) if n else 0.0
    if n:
        days = int((as_of - hist["order_date"].max()).days)
        is_first = 0
    else:
        days = FIRST_ORDER_DAYS
        is_first = 1

    table = hist[
        ["order_id", "order_date", "region", "segment", "order_value", "quantity", "discount", "returned"]
    ].copy()
    table["order_date"] = table["order_date"].dt.strftime("%Y-%m-%d")
    table["returned"] = table["returned"].map({1: "Yes", 0: "No"})

    return {
        "previous_order_count": n,
        "previous_return_count": n_ret,
        "historical_return_rate": rate,
        "historical_average_order_value": avg_value,
        "days_since_previous_order": days,
        "is_first_order": is_first,
        "history_table": table.to_dict(orient="records"),
    }


def build_feature_row(order_fields: dict, history: dict, order_date) -> pd.DataFrame:
    """Single-row frame with the same columns used in training."""
    dt = pd.to_datetime(order_date)
    row = {
        "order_value": float(order_fields["order_value"]),
        "quantity": int(order_fields["quantity"]),
        "discount": float(order_fields["discount"]),
        "max_discount": float(order_fields.get("max_discount", order_fields["discount"])),
        "number_of_products": int(order_fields.get("number_of_products", 1)),
        "number_of_categories": int(order_fields.get("number_of_categories", 1)),
        "is_weekend": int(dt.dayofweek in (5, 6)),
        "previous_order_count": int(history["previous_order_count"]),
        "previous_return_count": int(history["previous_return_count"]),
        "historical_return_rate": float(history["historical_return_rate"]),
        "historical_average_order_value": float(history["historical_average_order_value"]),
        "days_since_previous_order": int(history["days_since_previous_order"]),
        "is_first_order": int(history["is_first_order"]),
        "region": order_fields["region"],
        "segment": order_fields["segment"],
        "order_priority": order_fields["order_priority"],
        "order_month": int(dt.month),
        "order_day_of_week": int(dt.dayofweek),
    }
    return pd.DataFrame([row])


def main() -> None:
    df = pd.read_csv(ORDERS_CLEAN_PATH)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df = add_calendar_features(df)
    df = add_customer_history_features(df)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"Wrote features {df.shape} -> {FEATURES_PATH}")
    print(
        "History check (should be 0): current return counted in previous_return_count "
        f"on first occurrence per customer: "
        f"{int(((df['previous_order_count'] == 0) & (df['previous_return_count'] > 0)).sum())}"
    )


if __name__ == "__main__":
    main()

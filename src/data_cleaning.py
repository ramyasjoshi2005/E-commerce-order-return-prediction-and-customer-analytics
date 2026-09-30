"""Load the labeled order table, audit data quality, and write SQLite + clean CSV.

The return target comes from the existing labeled Superstore order table
(`data/processed/features.csv`). The Global Superstore files in `data/raw`
were inspected and are not used for modeling because `returns.csv` does not
join to `Global_Superstore2.csv` (zero overlapping Order IDs).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import (
    ARTIFACTS,
    DATA_RAW,
    DATA_PROCESSED,
    LABELED_ORDERS_PATH,
    ORDERS_CLEAN_PATH,
    SQLITE_PATH,
)


def inspect_raw_global_superstore() -> dict:
    """Record why Global Superstore is not the modeling dataset."""
    orders_path = DATA_RAW / "Global_Superstore2.csv"
    returns_path = DATA_RAW / "returns.csv"
    report = {
        "orders_file": str(orders_path.name) if orders_path.exists() else None,
        "returns_file": str(returns_path.name) if returns_path.exists() else None,
        "used_for_modeling": False,
        "reason": (
            "Order IDs do not overlap between Global_Superstore2.csv and returns.csv. "
            "Returns use a different ID scheme and include 2015 dates, while the "
            "order file covers 2011-2014. Using it would require fabricating return labels."
        ),
    }
    if orders_path.exists() and returns_path.exists():
        orders = pd.read_csv(orders_path, encoding="latin-1", usecols=["Order ID", "Country"])
        returns = pd.read_csv(returns_path, encoding="latin-1")
        overlap = set(orders["Order ID"]) & set(returns["Order ID"])
        report.update(
            {
                "global_line_items": int(len(orders)),
                "global_unique_orders": int(orders["Order ID"].nunique()),
                "global_countries": int(orders["Country"].nunique()),
                "india_line_items": int((orders["Country"] == "India").sum()),
                "return_rows": int(len(returns)),
                "order_id_overlap": int(len(overlap)),
            }
        )
    return report


def load_labeled_orders(path: Path = LABELED_ORDERS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Labeled orders not found at {path}. This file is the only source "
            "with a valid return target in this repository."
        )
    df = pd.read_csv(path)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def build_clean_orders(raw: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate order_id and keep fields that exist at order placement."""
    df = raw.copy()
    df = df.sort_values(["order_date", "order_id"]).reset_index(drop=True)

    # Duplicate order_id rows were a leakage bug: the second row treated the
    # same order as prior history and included its own return label.
    before = len(df)
    df = df.drop_duplicates(subset=["order_id"], keep="first").reset_index(drop=True)
    dropped_dupes = before - len(df)

    rename = {
        "total_sales": "order_value",
        "total_quantity": "quantity",
        "average_discount": "discount",
    }
    df = df.rename(columns=rename)

    keep = [
        "order_id",
        "customer_id",
        "order_date",
        "market",
        "region",
        "country",
        "segment",
        "order_priority",
        "order_value",
        "quantity",
        "discount",
        "max_discount",
        "number_of_products",
        "number_of_categories",
        "returned",
    ]
    df = df[keep]
    df["returned"] = df["returned"].astype(int)
    df.attrs["dropped_duplicate_orders"] = dropped_dupes
    return df


def data_quality_report(df: pd.DataFrame, dropped_dupes: int, global_inspect: dict) -> dict:
    numeric_cols = ["order_value", "quantity", "discount", "max_discount"]
    report = {
        "dataset_name": "Tableau Sample Superstore (US) — labeled order table already in this repo",
        "modeling_grain": "one row per order_id",
        "n_orders": int(len(df)),
        "n_customers": int(df["customer_id"].nunique()),
        "date_min": df["order_date"].min().strftime("%Y-%m-%d"),
        "date_max": df["order_date"].max().strftime("%Y-%m-%d"),
        "orders_by_year": {int(k): int(v) for k, v in df["order_date"].dt.year.value_counts().sort_index().items()},
        "return_rate": float(df["returned"].mean()),
        "n_returns": int(df["returned"].sum()),
        "n_non_returns": int((df["returned"] == 0).sum()),
        "class_balance": {
            "no_return": float((df["returned"] == 0).mean()),
            "return": float(df["returned"].mean()),
        },
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicate_order_ids_after_clean": int(df["order_id"].duplicated().sum()),
        "duplicate_order_ids_dropped": int(dropped_dupes),
        "invalid_order_value": int((df["order_value"] < 0).sum()),
        "invalid_quantity": int((df["quantity"] <= 0).sum()),
        "invalid_discount": int(((df["discount"] < 0) | (df["discount"] > 1)).sum()),
        "countries": sorted(df["country"].dropna().unique().tolist()),
        "markets": sorted(df["market"].dropna().unique().tolist()),
        "regions": sorted(df["region"].dropna().unique().tolist()),
        "segments": sorted(df["segment"].dropna().unique().tolist()),
        "order_priorities": sorted(df["order_priority"].dropna().unique().tolist()),
        "india_in_modeling_data": bool((df["country"] == "India").any()),
        "constant_columns": [c for c in ["market", "country"] if df[c].nunique() <= 1],
        "numeric_summary": df[numeric_cols].describe().to_dict(),
        "global_superstore_inspection": global_inspect,
        "leakage_notes": [
            "profit, shipping_cost, ship_date are not in this table and are not used.",
            "Duplicate order_id rows were dropped; they previously leaked the current return into history.",
            "Product category/sub-category are not present in the labeled order table.",
        ],
    }
    return report


def write_sqlite(df: pd.DataFrame, path: Path = SQLITE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    out = df.copy()
    out["order_date"] = out["order_date"].dt.strftime("%Y-%m-%d")
    out.to_sql("orders", conn, index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date)")
    conn.commit()
    conn.close()


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    global_inspect = inspect_raw_global_superstore()
    raw = load_labeled_orders()
    clean = build_clean_orders(raw)
    dropped = int(clean.attrs.get("dropped_duplicate_orders", 0))
    report = data_quality_report(clean, dropped, global_inspect)

    clean.to_csv(ORDERS_CLEAN_PATH, index=False)
    write_sqlite(clean)

    with open(ARTIFACTS / "data_quality.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Clean orders: {len(clean):,} | customers: {clean['customer_id'].nunique():,}")
    print(f"Date range: {report['date_min']} to {report['date_max']}")
    print(f"Return rate: {report['return_rate']:.4f}")
    print(f"Dropped duplicate order_id rows: {dropped}")
    print(f"Global Superstore Order ID overlap with returns.csv: {global_inspect.get('order_id_overlap')}")
    print(f"Wrote {ORDERS_CLEAN_PATH} and {SQLITE_PATH}")


if __name__ == "__main__":
    main()

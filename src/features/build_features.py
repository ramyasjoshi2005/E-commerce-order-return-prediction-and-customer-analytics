import sqlite3
import pandas as pd
import numpy as np
import os

def build_features():
    print("Connecting to database...")
    db_path = 'data/processed/superstore.db'
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}.")
        
    conn = sqlite3.connect(db_path)
    
    print("Executing feature extraction query...")
    with open('sql/feature_queries.sql', 'r') as f:
        query = f.read()
        
    df = pd.read_sql_query(query, conn)
    
    print("Performing point-in-time feature engineering...")
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['order_month'] = df['order_date'].dt.month
    df['order_day_of_week'] = df['order_date'].dt.dayofweek
    df['is_weekend'] = df['order_day_of_week'].isin([5, 6]).astype(int)
    
    # Sort chronologically using order_date and a deterministic secondary sorting (order_id)
    df = df.sort_values(['order_date', 'order_id']).reset_index(drop=True)
    
    # ---------------------------------------------------------
    # 1. Point-in-time Historical Features (Customer-level)
    # ---------------------------------------------------------
    # Group by customer_id
    grouped_cust = df.groupby('customer_id')
    
    # Days since previous order
    df['days_since_previous_order'] = grouped_cust['order_date'].diff().dt.days
    df['days_since_previous_order'] = df['days_since_previous_order'].fillna(9999) # 9999 means first order
    
    # Historical orders (must not include the current order)
    df['historical_orders'] = grouped_cust.cumcount()
    
    # Historical returns (must not include the current order's status)
    df['historical_returns'] = grouped_cust['returned'].apply(lambda x: x.shift(1).cumsum().fillna(0)).reset_index(level=0, drop=True)
    
    # Smoothed historical return rate for customers
    # Global return rate is ~14.5% across the dataset. We'll use a Laplace smoothing prior of 1 return per 7 orders (~14%)
    prior_returns = 1.0
    prior_orders = 7.0
    df['customer_historical_return_rate'] = (df['historical_returns'] + prior_returns) / (df['historical_orders'] + prior_orders)
    
    # ---------------------------------------------------------
    # 2. Point-in-time Historical Features (Market/Region level)
    # Since we aggregated products in SQL (number of products), we can't easily track single product history.
    # But wait, we can track 'market' historical return rate as a proxy for category/product.
    # Actually, we can get product_id from order_items, but since the prediction unit is Order, an order can have multiple products.
    # To keep it simple and order-level, we will track the 'market' or 'segment' historical return rate.
    # ---------------------------------------------------------
    grouped_market = df.groupby('market')
    df['market_historical_orders'] = grouped_market.cumcount()
    df['market_historical_returns'] = grouped_market['returned'].apply(lambda x: x.shift(1).cumsum().fillna(0)).reset_index(level=0, drop=True)
    df['market_historical_return_rate'] = (df['market_historical_returns'] + prior_returns) / (df['market_historical_orders'] + prior_orders)
    
    # Output path
    output_path = 'data/processed/features.csv'
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}. Shape: {df.shape}")
    
    conn.close()

if __name__ == "__main__":
    build_features()

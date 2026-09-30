import pandas as pd
import sqlite3
import os
import re

def clean_column_names(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9]', '_', c.strip().lower()) for c in df.columns]
    df.columns = [re.sub(r'_+', '_', c).strip('_') for c in df.columns]
    return df

def main():
    print("Loading data from raw files...")
    # Use the synced superstore.xls which has all sheets properly linked
    xls_path = 'data/raw/superstore.xls'
    orders_df = pd.read_excel(xls_path, sheet_name='Orders')
    returns_df = pd.read_excel(xls_path, sheet_name='Returns')
    
    orders_df = clean_column_names(orders_df)
    returns_df = clean_column_names(returns_df)
    
    # Check if 'market' or 'region' is used in this version
    # The US-only superstore uses 'region' instead of 'market' and doesn't have 'market' column in Returns sometimes
    # Let's inspect columns to handle both
    
    print(f"Returns columns: {returns_df.columns.tolist()}")
    print(f"Orders columns: {orders_df.columns.tolist()}")
    
    if 'customer_id' not in orders_df.columns:
        orders_df.rename(columns={'customer_id': 'customer_id'}, inplace=True) # just in case it's capitalized differently, but clean_column_names handles lowercasing
        
    # 1. Customers
    customers = orders_df[['customer_id', 'customer_name', 'segment']].drop_duplicates()
    
    # 2. Products
    products = orders_df[['product_id', 'category', 'sub_category', 'product_name']].drop_duplicates(subset=['product_id'])
    
    if 'country_region' in orders_df.columns:
        orders_df.rename(columns={'country_region': 'country'}, inplace=True)
    
    # 3. Orders
    order_cols = ['order_id', 'order_date', 'ship_date', 'ship_mode', 'customer_id', 
                  'region', 'country', 'state', 'city', 'postal_code']
    if 'market' in orders_df.columns:
        order_cols.append('market')
    else:
        orders_df['market'] = 'US'
        order_cols.append('market')
        
    if 'order_priority' in orders_df.columns:
        order_cols.append('order_priority')
    else:
        orders_df['order_priority'] = 'Medium'
        order_cols.append('order_priority')
        
    orders = orders_df[order_cols].drop_duplicates(subset=['order_id'])
    
    # 4. Order Items
    item_cols = ['order_id', 'product_id', 'sales', 'quantity', 'discount', 'profit']
    if 'shipping_cost' in orders_df.columns:
        item_cols.append('shipping_cost')
    else:
        orders_df['shipping_cost'] = 0.0
        item_cols.append('shipping_cost')
        
    order_items = orders_df[item_cols].copy()
    
    # Convert dates to ISO format for SQLite
    orders['order_date'] = pd.to_datetime(orders['order_date']).dt.strftime('%Y-%m-%d')
    orders['ship_date'] = pd.to_datetime(orders['ship_date']).dt.strftime('%Y-%m-%d')
    
    print("Connecting to SQLite database...")
    db_path = 'data/processed/superstore.db'
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    
    # Load schema
    with open('sql/schema.sql', 'r') as f:
        schema = f.read()
    schema = schema.replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
    conn.executescript(schema)
    
    print("Inserting data into database...")
    customers.to_sql('customers', conn, if_exists='append', index=False)
    products.to_sql('products', conn, if_exists='append', index=False)
    orders.to_sql('orders', conn, if_exists='append', index=False)
    order_items.to_sql('order_items', conn, if_exists='append', index=False)
    
    returns_df.rename(columns={'returned': 'returned_flag'}, inplace=True)
    if 'market' not in returns_df.columns:
        returns_df['market'] = 'US'
    
    returns_df[['order_id', 'market', 'returned_flag']].to_sql('returns', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()
    print("Database successfully populated!")

if __name__ == "__main__":
    main()

import sqlite3
import pandas as pd

conn = sqlite3.connect('data/processed/superstore.db')

print("Date range:")
print(pd.read_sql('SELECT MIN(order_date) as min_date, MAX(order_date) as max_date FROM orders', conn))

print("\nOrders per year:")
query = """
SELECT 
    strftime('%Y', order_date) as year, 
    count(o.order_id) as total_orders, 
    sum(case when r.order_id is not null then 1 else 0 end) as returns,
    sum(case when r.order_id is not null then 1 else 0 end)*100.0/count(o.order_id) as ret_rate 
FROM orders o 
LEFT JOIN returns r USING(order_id) 
GROUP BY year
"""
print(pd.read_sql(query, conn))

conn.close()

-- sql/data_quality.sql

-- 1. Check for missing critical values in orders
SELECT COUNT(*) AS missing_dates
FROM orders
WHERE order_date IS NULL OR ship_date IS NULL;

-- 2. Check for negative quantities or prices
SELECT COUNT(*) AS invalid_items
FROM order_items
WHERE quantity <= 0 OR sales < 0;

-- 3. Check for invalid dates (ship date before order date)
SELECT COUNT(*) AS invalid_ship_dates
FROM orders
WHERE ship_date < order_date;

-- 4. Check for orphaned order items (no matching order)
SELECT COUNT(*) AS orphaned_items
FROM order_items oi
LEFT JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_id IS NULL;

-- 5. Outlier detection in sales (z-score approximation or simple threshold)
-- Using simple threshold as SQLite doesn't have STDDEV built-in natively without extensions
SELECT order_id, sales
FROM order_items
WHERE sales > 10000;

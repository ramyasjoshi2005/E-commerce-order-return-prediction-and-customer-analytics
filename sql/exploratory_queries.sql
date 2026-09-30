-- sql/exploratory_queries.sql

-- 1. Overall Return Rate
SELECT 
    COUNT(r.order_id) * 100.0 / COUNT(o.order_id) AS overall_return_rate
FROM orders o
LEFT JOIN returns r ON o.order_id = r.order_id;

-- 2. Return Rate by Category
SELECT 
    p.category,
    COUNT(DISTINCT o.order_id) as total_orders,
    COUNT(DISTINCT r.order_id) as returned_orders,
    ROUND(COUNT(DISTINCT r.order_id) * 100.0 / COUNT(DISTINCT o.order_id), 2) AS return_rate
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
LEFT JOIN returns r ON o.order_id = r.order_id
GROUP BY p.category
ORDER BY return_rate DESC;

-- 3. Return Rate by Market
SELECT 
    o.market,
    COUNT(o.order_id) AS total_orders,
    SUM(CASE WHEN r.order_id IS NOT NULL THEN 1 ELSE 0 END) AS returned_orders,
    ROUND(SUM(CASE WHEN r.order_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(o.order_id), 2) AS return_rate
FROM orders o
LEFT JOIN returns r ON o.order_id = r.order_id
GROUP BY o.market
ORDER BY return_rate DESC;

-- 4. Average Delivery Time for Returned vs Non-Returned Orders (SQLite julian day calculation)
SELECT 
    CASE WHEN r.order_id IS NOT NULL THEN 'Returned' ELSE 'Not Returned' END AS status,
    AVG(julianday(o.ship_date) - julianday(o.order_date)) AS avg_processing_days
FROM orders o
LEFT JOIN returns r ON o.order_id = r.order_id
GROUP BY status;

-- 5. Monthly Return Trends
SELECT 
    strftime('%Y-%m', o.order_date) AS order_month,
    COUNT(o.order_id) AS total_orders,
    SUM(CASE WHEN r.order_id IS NOT NULL THEN 1 ELSE 0 END) AS returned_orders,
    ROUND(SUM(CASE WHEN r.order_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(o.order_id), 2) AS return_rate
FROM orders o
LEFT JOIN returns r ON o.order_id = r.order_id
GROUP BY order_month
ORDER BY order_month;

-- 6. High Value Customers with High Return Rates
SELECT 
    c.customer_name,
    COUNT(o.order_id) as total_orders,
    SUM(CASE WHEN r.order_id IS NOT NULL THEN 1 ELSE 0 END) AS returned_orders,
    SUM(oi.sales) as total_sales,
    ROUND(SUM(CASE WHEN r.order_id IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(o.order_id), 2) AS return_rate
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN returns r ON o.order_id = r.order_id
GROUP BY c.customer_id, c.customer_name
HAVING total_orders > 5
ORDER BY return_rate DESC
LIMIT 10;

-- Analytical queries on the labeled order table (SQLite).
-- Grain: one row per order. Target: returned (1/0).

-- Overall return rate
SELECT
    COUNT(*) AS n_orders,
    SUM(returned) AS n_returns,
    AVG(returned) AS return_rate
FROM orders;

-- Return rate by customer segment
SELECT
    segment,
    COUNT(*) AS n_orders,
    SUM(returned) AS n_returns,
    AVG(returned) AS return_rate
FROM orders
GROUP BY segment
ORDER BY return_rate DESC;

-- Return rate by US region
SELECT
    region,
    COUNT(*) AS n_orders,
    SUM(returned) AS n_returns,
    AVG(returned) AS return_rate
FROM orders
GROUP BY region
ORDER BY return_rate DESC;

-- Average order value by return status
SELECT
    CASE WHEN returned = 1 THEN 'Returned' ELSE 'Not returned' END AS return_status,
    COUNT(*) AS n_orders,
    AVG(order_value) AS avg_order_value,
    AVG(discount) AS avg_discount,
    AVG(quantity) AS avg_quantity
FROM orders
GROUP BY returned;

-- Customer-level order and return counts (customers with at least 3 orders)
SELECT
    customer_id,
    COUNT(*) AS n_orders,
    SUM(returned) AS n_returns,
    AVG(returned) AS customer_return_rate,
    AVG(order_value) AS avg_order_value
FROM orders
GROUP BY customer_id
HAVING COUNT(*) >= 3
ORDER BY customer_return_rate DESC, n_orders DESC
LIMIT 15;

-- Discount buckets vs return rate
SELECT
    CASE
        WHEN discount = 0 THEN '0%'
        WHEN discount <= 0.2 THEN '0-20%'
        WHEN discount <= 0.4 THEN '20-40%'
        ELSE '>40%'
    END AS discount_bucket,
    COUNT(*) AS n_orders,
    AVG(returned) AS return_rate
FROM orders
GROUP BY discount_bucket
ORDER BY return_rate DESC;

-- Customers whose return rate exceeds the overall rate (HAVING)
SELECT
    o.customer_id,
    COUNT(*) AS n_orders,
    SUM(o.returned) AS n_returns,
    AVG(o.returned) AS customer_return_rate
FROM orders o
GROUP BY o.customer_id
HAVING AVG(o.returned) > (SELECT AVG(returned) FROM orders)
   AND COUNT(*) >= 5
ORDER BY customer_return_rate DESC
LIMIT 10;

-- Point-in-time prior returns vs later outcome using a window (not used as a model feature here)
WITH ordered AS (
    SELECT
        customer_id,
        order_id,
        order_date,
        returned,
        SUM(returned) OVER (
            PARTITION BY customer_id
            ORDER BY order_date, order_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS previous_return_count
    FROM orders
)
SELECT
    CASE WHEN COALESCE(previous_return_count, 0) >= 1 THEN 'had_previous_return' ELSE 'no_previous_return' END AS history_group,
    COUNT(*) AS n_orders,
    AVG(returned) AS subsequent_return_rate
FROM ordered
GROUP BY history_group;

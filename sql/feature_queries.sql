-- sql/feature_queries.sql
-- Extracts a flattened feature table for Machine Learning, AT ORDER PLACEMENT
-- One row per order_id

WITH order_aggregates AS (
    SELECT 
        oi.order_id,
        SUM(oi.sales) AS total_sales,
        SUM(oi.quantity) AS total_quantity,
        AVG(oi.discount) AS average_discount,
        MAX(oi.discount) AS max_discount,
        COUNT(oi.item_id) AS number_of_products,
        COUNT(DISTINCT p.category) AS number_of_categories
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    GROUP BY oi.order_id
)
SELECT 
    o.order_id,
    o.customer_id,
    o.order_date,
    o.market,
    o.region,
    o.country,
    c.segment,
    o.order_priority,
    oa.total_sales,
    oa.total_quantity,
    oa.average_discount,
    oa.max_discount,
    oa.number_of_products,
    oa.number_of_categories,
    CASE WHEN r.order_id IS NOT NULL THEN 1 ELSE 0 END AS returned
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_aggregates oa ON o.order_id = oa.order_id
LEFT JOIN returns r ON o.order_id = r.order_id;

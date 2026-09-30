-- Drop tables if they exist
DROP TABLE IF EXISTS returns;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

-- 1. Customers Table
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_name VARCHAR(255),
    segment VARCHAR(100)
);

-- 2. Products Table
CREATE TABLE products (
    product_id VARCHAR(100) PRIMARY KEY,
    category VARCHAR(100),
    sub_category VARCHAR(100),
    product_name TEXT
);

-- 3. Orders Table
CREATE TABLE orders (
    order_id VARCHAR(100) PRIMARY KEY,
    order_date DATE,
    ship_date DATE,
    ship_mode VARCHAR(100),
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    market VARCHAR(50),
    region VARCHAR(100),
    country VARCHAR(150),
    state VARCHAR(150),
    city VARCHAR(150),
    postal_code VARCHAR(50),
    order_priority VARCHAR(50)
);

-- 4. Order Items Table
-- Note: In this dataset, a single order_id + product_id might appear multiple times or we can use an auto-incrementing ID.
-- We'll use a serial primary key for order_items to be safe.
CREATE TABLE order_items (
    item_id SERIAL PRIMARY KEY,
    order_id VARCHAR(100) REFERENCES orders(order_id),
    product_id VARCHAR(100) REFERENCES products(product_id),
    sales NUMERIC(10, 4),
    quantity INTEGER,
    discount NUMERIC(5, 4),
    profit NUMERIC(10, 4),
    shipping_cost NUMERIC(10, 4)
);

-- 5. Returns Table
CREATE TABLE returns (
    return_id SERIAL PRIMARY KEY,
    order_id VARCHAR(100) REFERENCES orders(order_id),
    market VARCHAR(50),
    returned_flag VARCHAR(10)
);

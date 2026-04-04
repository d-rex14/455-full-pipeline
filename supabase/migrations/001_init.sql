-- Supabase migration: recreate shop.db schema in Postgres
-- Run this in the Supabase SQL Editor (Dashboard > SQL Editor > New query)

-- 1. customers
CREATE TABLE IF NOT EXISTS customers (
  customer_id      SERIAL PRIMARY KEY,
  full_name        TEXT NOT NULL,
  email            TEXT NOT NULL UNIQUE,
  gender           TEXT NOT NULL,
  birthdate        TEXT NOT NULL,
  created_at       TEXT NOT NULL,
  city             TEXT,
  state            TEXT,
  zip_code         TEXT,
  customer_segment TEXT,
  loyalty_tier     TEXT,
  is_active        INTEGER NOT NULL DEFAULT 1
);

-- 2. products
CREATE TABLE IF NOT EXISTS products (
  product_id   SERIAL PRIMARY KEY,
  sku          TEXT NOT NULL UNIQUE,
  product_name TEXT NOT NULL,
  category     TEXT NOT NULL,
  price        REAL NOT NULL,
  cost         REAL NOT NULL,
  is_active    INTEGER NOT NULL DEFAULT 1
);

-- 3. orders
CREATE TABLE IF NOT EXISTS orders (
  order_id           SERIAL PRIMARY KEY,
  customer_id        INTEGER NOT NULL REFERENCES customers(customer_id),
  order_datetime     TEXT NOT NULL,
  billing_zip        TEXT,
  shipping_zip       TEXT,
  shipping_state     TEXT,
  payment_method     TEXT NOT NULL,
  device_type        TEXT NOT NULL,
  ip_country         TEXT NOT NULL,
  promo_used         INTEGER NOT NULL DEFAULT 0,
  promo_code         TEXT,
  order_subtotal     REAL NOT NULL,
  shipping_fee       REAL NOT NULL,
  tax_amount         REAL NOT NULL,
  order_total        REAL NOT NULL,
  risk_score         REAL NOT NULL,
  is_fraud           INTEGER NOT NULL DEFAULT 0
);

-- 4. order_items
CREATE TABLE IF NOT EXISTS order_items (
  order_item_id  SERIAL PRIMARY KEY,
  order_id       INTEGER NOT NULL REFERENCES orders(order_id),
  product_id     INTEGER NOT NULL REFERENCES products(product_id),
  quantity       INTEGER NOT NULL,
  unit_price     REAL NOT NULL,
  line_total     REAL NOT NULL
);

-- 5. shipments
CREATE TABLE IF NOT EXISTS shipments (
  shipment_id        SERIAL PRIMARY KEY,
  order_id           INTEGER NOT NULL UNIQUE REFERENCES orders(order_id),
  ship_datetime      TEXT NOT NULL,
  carrier            TEXT NOT NULL,
  shipping_method    TEXT NOT NULL,
  distance_band      TEXT NOT NULL,
  promised_days      INTEGER NOT NULL,
  actual_days        INTEGER NOT NULL,
  late_delivery      INTEGER NOT NULL DEFAULT 0
);

-- 6. product_reviews
CREATE TABLE IF NOT EXISTS product_reviews (
  review_id       SERIAL PRIMARY KEY,
  customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
  product_id      INTEGER NOT NULL REFERENCES products(product_id),
  rating          INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
  review_datetime TEXT NOT NULL,
  review_text     TEXT,
  UNIQUE(customer_id, product_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_orders_customer   ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_datetime   ON orders(order_datetime);
CREATE INDEX IF NOT EXISTS idx_items_order       ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_items_product     ON order_items(product_id);
CREATE INDEX IF NOT EXISTS idx_shipments_late    ON shipments(late_delivery);
CREATE INDEX IF NOT EXISTS idx_reviews_product   ON product_reviews(product_id);
CREATE INDEX IF NOT EXISTS idx_reviews_customer  ON product_reviews(customer_id);

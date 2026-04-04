-- ML scoring cache + human review flag (apply in Supabase SQL Editor after 001_init.sql)

ALTER TABLE orders ADD COLUMN IF NOT EXISTS ml_fraud_probability REAL;
ALTER TABLE orders ADD COLUMN IF NOT EXISTS ml_predicted_fraud INTEGER;
ALTER TABLE orders ADD COLUMN IF NOT EXISTS admin_reviewed INTEGER NOT NULL DEFAULT 0;

COMMENT ON COLUMN orders.ml_fraud_probability IS 'Last fraud class probability from deployed model';
COMMENT ON COLUMN orders.ml_predicted_fraud IS 'Binary prediction from model at FRAUD_THRESHOLD';
COMMENT ON COLUMN orders.admin_reviewed IS '1 after operator saved manual is_fraud label';

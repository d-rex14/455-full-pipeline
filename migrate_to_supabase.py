"""
One-time migration: push all shop.db data into Supabase Postgres.

Usage:
    1. Create a .env file with SUPABASE_URL and SUPABASE_KEY
    2. Run the SQL in supabase_migration.sql via the Supabase SQL Editor first
    3. python3 migrate_to_supabase.py

Tables are inserted in foreign-key order:
    customers -> products -> orders -> order_items -> shipments -> product_reviews
"""

import os
import sqlite3
import math

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
DB_PATH = os.path.join(os.path.dirname(__file__), "shop.db")
BATCH_SIZE = 500  # Supabase REST API handles ~500 rows per insert reliably

TABLE_ORDER = [
    "customers",
    "products",
    "orders",
    "order_items",
    "shipments",
    "product_reviews",
]


def migrate():
    conn = sqlite3.connect(DB_PATH)
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    for table in TABLE_ORDER:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        records = df.to_dict(orient="records")
        total = len(records)
        batches = math.ceil(total / BATCH_SIZE)

        print(f"\n{'='*60}")
        print(f"  {table}: {total} rows  ({batches} batches)")
        print(f"{'='*60}")

        for i in range(batches):
            start = i * BATCH_SIZE
            end = min(start + BATCH_SIZE, total)
            batch = records[start:end]

            # Clean NaN/None values for Supabase (JSON nulls)
            for row in batch:
                for k, v in row.items():
                    if pd.isna(v) if not isinstance(v, str) else False:
                        row[k] = None

            resp = sb.table(table).insert(batch).execute()
            print(f"  batch {i+1}/{batches}  rows {start+1}-{end}  inserted: {len(resp.data)}")

        print(f"  {table} complete.")

    conn.close()
    print("\nMigration finished. Verify row counts in Supabase SQL Editor:")
    for t in TABLE_ORDER:
        print(f"  SELECT COUNT(*) FROM {t};")


if __name__ == "__main__":
    migrate()

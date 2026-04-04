# Project Context: Fraud Detection ML Pipeline

Read this file before doing anything else. 

You may update this file as you go along. 

## Overview
This project focuses on building a machine learning solution to identify fraudulent transactions. We are strictly adhering to the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework as outlined in the provided textbook documentation.

## Tech Stack
* **Modeling:** Python (Jupyter Notebook)
* **Framework:** CRISP-DM
* **Model Format:** Serialized Pickle file (`model.sav`)
* **Database/Backend:** Supabase
* **Hosting/Deployment:** Vercel

---

## Project Phases (CRISP-DM)
We will document and execute the following phases within the `fraud_detection.ipynb` file:

1.  **Business Understanding:** Define fraud detection goals and success metrics.
2.  **Data Understanding:** Explore the provided dataset and identify quality issues.
3.  **Data Preparation:** Clean, transform, and handle class imbalance (Fraud vs. Non-Fraud).
4.  **Modeling:** Select and train algorithms to predict the target variable.
5.  **Evaluation:** Validate the model against business objectives (Focusing on Recall and F1-Score).
6.  **Deployment:** Exporting `model.sav` for integration with the Supabase/Vercel stack.

---

## File Structure
* `shop.db` — SQLite source data (local); optional after data lives in Supabase.
* `Textbook_Chapters/` — Course reference (CRISP-DM and methods).
* `fraud_detection.ipynb` — Primary notebook (CRISP-DM phases).
* `model.sav` — Trained pipeline for the Vercel API (generate via notebook; must be present at deploy).
* `api/predict.py` — Serverless scorer (Supabase + pickle model).
* `public/` — Static frontend for Vercel (`index.html` and assets).
* `supabase/migrations/` — Postgres schema applied in Supabase.
* `migrate_to_supabase.py` — One-time SQLite → Supabase data load.

---

## Deployment Workflow
1.  **Serialization:** The final model is saved as `model.sav`.
2.  **Backend Integration:** Connect to **Supabase** for data persistence and user management.
3.  **Production:** Deploy a serverless function or API on **Vercel** to serve real-time fraud predictions.
4.  **Frontend:** Static UI in `public/` is routed at `/`; configure `SUPABASE_URL`, `SUPABASE_KEY`, and optionally `FRAUD_THRESHOLD` in the Vercel project environment.
5.  **`model.sav` on Vercel:** Run the notebook through the serialization cell so `model.sav` exists at the **repo root**, **commit and push** it, then redeploy. `vercel.json` uses `includeFiles` so that file is packaged with `api/predict.py` (the serverless bundle does not see your laptop’s filesystem).
6.  **Cron:** `vercel.json` defines a daily job at **00:00 UTC** hitting `GET /api/cron`. Set **`CRON_SECRET`** in Vercel; the handler checks `Authorization: Bearer <CRON_SECRET>`. Implement nightly logic inside `api/cron.py` as needed.
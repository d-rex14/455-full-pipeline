# Project Context: Fraud Detection ML Pipeline

Read this file before doing anything else.

You may update this file as you go along.

## Overview
This project focuses on building a machine learning solution to identify fraudulent transactions. We are strictly adhering to the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework as outlined in the provided textbook documentation.

## Tech Stack
* **Modeling:** Python (Jupyter Notebook)
* **Framework:** CRISP-DM
* **Model Format:** Serialized joblib file (`model.sav`)
* **Database/Backend:** Supabase (already populated — migration complete)
* **Hosting/Deployment:** Vercel (serverless, deploys from this repo via GitHub integration)

---

## Project Phases (CRISP-DM) — Status

| Phase | Status | Notes |
|---|---|---|
| 1. Business Understanding | Complete | Fraud detection; primary metrics are Recall and F1-score |
| 2. Data Understanding | Complete | Explored `shop.db`; fraud rate ~6.36% |
| 3. Data Preparation | Complete | Feature engineering, outlier capping, stratified split |
| 4. Modeling | Complete | LR, DT, RF, GB trained; tuned RF and GB as finalists |
| 5. Evaluation | Complete | Final model selected by F1 @ Recall ≥ 0.75; threshold optimised |
| 6. Deployment | Complete | `model.sav` live on Vercel via `api/predict.py` |

---

## Current File Structure

```
ML-Pipeline/
├── fraud_detection.ipynb       # Main CRISP-DM notebook (Phases 1–6)
├── model.sav                   # Trained model artifact (joblib) — committed to git
├── model_metadata.json         # Final threshold, feature list, training stats
├── metrics.json                # recall, F1, ROC-AUC, accuracy on held-out test set
├── shop.db                     # Original SQLite training data (5,000 rows)
├── pyLibrary.py                # Shared ML utility functions used by the notebook
├── retrain.py                  # Standalone retraining script (run by GitHub Actions)
├── requirements.txt            # Python dependencies for Vercel
├── vercel.json                 # Vercel build + route config
├── .env / .env.example         # Supabase credentials (local use only)
├── CLAUDE.md                   # Instructions for Claude Code
├── context.md                  # This file
├── Textbook_Chapters/          # Reference chapters (Ch1–28)
├── api/
│   ├── predict.py              # POST /api/predict — live fraud scoring endpoint
│   └── retrain.py              # POST /api/retrain — on-demand HTTP retrain trigger
└── .github/
    └── workflows/
        └── retrain.yml         # Nightly cron retrain (3 AM UTC) + manual trigger
```

---

## Automated Retraining Pipeline

### How it works
```
Supabase (new labeled data)
    → GitHub Actions runs retrain.py (nightly 3 AM UTC or manual trigger)
    → Writes model.sav + model_metadata.json + metrics.json
    → Commits back to repo (only if model changed)
    → Vercel detects commit → auto-redeploys
    → api/predict.py loads new model.sav on next cold start
```

### Key files
| File | Role |
|---|---|
| `retrain.py` | Fetches all labeled orders from Supabase, engineers features (mirrors notebook Phase 3), trains RF with `class_weight='balanced'`, finds optimal threshold, saves artifacts |
| `.github/workflows/retrain.yml` | Cron schedule + manual `workflow_dispatch`; commits artifacts back to repo only when model changes (git diff guard) |
| `api/retrain.py` | On-demand HTTP trigger (`POST /api/retrain`, secured by `RETRAIN_SECRET` header); stores model in Supabase Storage for immediate use without a redeploy |
| `api/predict.py` | Loads `model.sav` and reads `final_threshold` from `model_metadata.json` — stays in sync with every retrain automatically |

### Required secrets (GitHub repo Settings → Secrets)
* `SUPABASE_URL`
* `SUPABASE_KEY`

### Required env vars (Vercel project Settings → Environment Variables)
* `SUPABASE_URL`
* `SUPABASE_KEY`
* `RETRAIN_SECRET` — any secret string to lock down `POST /api/retrain`

### Important notes
* `model.sav` must **not** be in `.gitignore` — the GitHub Action commits it back to the repo
* The workflow skips the commit step if `git diff` shows no changes (prevents empty commits)
* `api/predict.py` threshold is always read from `model_metadata.json`; the `FRAUD_THRESHOLD` env var is only a fallback if the file is missing

---

## Deployment Workflow
1. **Serialization:** Final model saved as `model.sav` via `joblib.dump`.
2. **Backend:** Supabase stores transaction data (orders, customers, shipments, order_items).
3. **Production:** Vercel serves `api/predict.py` for real-time fraud predictions.
4. **Retraining:** GitHub Actions retrains nightly from Supabase data and redeploys automatically.

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
* `data/` - Directory containing the raw fraud dataset.
* `CRISP-DM_Textbook.md` - The methodological guide for this project.
* `fraud_detection.ipynb` - The primary notebook for development.
* `model.sav` - The final trained model ready for deployment.

---


# CronJob Retrain nightly workflow

This document outlines the automated CI/CD pipeline responsible for retraining the machine learning model nightly, updating the repository, and triggering a fresh deployment.

---

## 🛠 Workflow Architecture
The system uses **GitHub Actions** as the orchestrator to bridge the gap between the data layer (**Supabase**) and the hosting layer (**Vercel**).

### 1. GitHub Action Configuration
The workflow is defined in `.github/workflows/retrain.yml`. It is scheduled to run every night at 3:00 AM UTC and can also be triggered manually via `workflow_dispatch`.

**Key Pipeline Steps:**
* **Environment Setup:** Provisions an Ubuntu runner with Python 3.11.
* **Dependency Management:** Installs required ML libraries (`scikit-learn`, `pandas`, `imbalanced-learn`, etc.).
* **Execution:** Runs `retrain.py` using repository secrets for Supabase authentication.
* **Automated Commit:** If the model weights or metrics change, the GitHub Actions bot commits the updated `model.sav`, `model_metadata.json`, and `metrics.json` back to the main branch.

---

### 2. The Retraining Script (`retrain.py`)
To ensure consistency, the retraining script must perform the following:
1. **Data Ingestion:** Query Supabase for the latest labeled dataset.
2. **Feature Parity:** Implement identical feature engineering steps as found in the research notebooks (e.g., account age, zip code validation).
3. **Training:** Fit the Random Forest pipeline.
4. **Serialization:** Export the model via `joblib` and update the performance metrics files.

---

### 3. Deployment Loop
Deployment is handled automatically via the **Vercel-GitHub Integration**:
* Once the GitHub Action pushes a new commit to the repository, Vercel detects the change.
* A new deployment is triggered.
* The API endpoint (`api/predict.py`) loads the newly committed `model.sav` upon initialization.

---

### Critical Maintenance Notes
* **Secrets:** Ensure `SUPABASE_URL` and `SUPABASE_KEY` are maintained in the GitHub Repo Secrets.
* **Gitignore:** The file `model.sav` must **not** be ignored by Git; otherwise, the automated push will fail.
* **Commit Logic:** The workflow uses a `git diff` guard to prevent "empty" commits if the retraining results in an identical model file.


# Final Deployment Workflow
1.  **Serialization:** The final model is saved as `model.sav`.
2.  **Backend Integration:** Connect to **Supabase** for data persistence and user management.
3.  **Production:** Deploy a serverless function or API on **Vercel** to serve real-time fraud predictions.


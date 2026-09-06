# 🤖 ADSLM — Adaptive Data Science Language Model

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189F50?style=for-the-badge)
![Cloudflare](https://img.shields.io/badge/Cloudflare_Tunnel-F38020?style=for-the-badge&logo=cloudflare&logoColor=white)

> **ADSLM (Adaptive Data Science Language Model)** is an industrial-grade **AutoML + AI Data Science Copilot** system. It autonomously profiles tabular datasets, detects ML tasks (Classification, Regression, Clustering, Time-Series), executes data-leakage-free preprocessing, trains and benchmarks multiple algorithms, generates explainable AI (XAI) feature importances, performs regulatory compliance audits (EU AI Act, GDPR, ISO 27001), and generates tailored natural-language insights adapted to Beginner, Intermediate, and Expert audiences.

*(Note: ADSLM is an adaptive AutoML and structured natural-language insight generation engine designed for industrial data science workflows, rather than a generative text LLM like GPT/Llama).*

---

## 🌐 Hybrid Deployment Architecture

The production demo follows a hybrid cloud-and-edge architecture:

```
                    INTERNET
                       │
                       ▼
              STREAMLIT CLOUD
               frontend/app.py
                       │
                       │ HTTPS
                       ▼
              CLOUDFLARE TUNNEL
            (https://xxxxx.trycloudflare.com)
                       │
                       ▼
                  LOCAL LAPTOP
                       │
                       ▼
                   FASTAPI
              main.py (0.0.0.0:8000)
                       │
                       ▼
              ADSLM ML PIPELINE
        (AutoML, XAI, Audit, Reports)
```

- **Frontend**: Deployed publicly on **Streamlit Community Cloud** (or executed locally).
- **Backend**: **FastAPI** server running on the developer laptop (`0.0.0.0:8000`).
- **Secure Bridge**: **Cloudflare Tunnel** exposes the local FastAPI backend securely over public HTTPS.
- **Dynamic Configuration**: Streamlit dynamically discovers the backend via `st.secrets["API_URL"]`.
- **Remote Report Retrieval**: Generated PDF, TXT, and JSON reports are hosted by the FastAPI backend via `GET /report/{filename}` and streamed directly to Streamlit Cloud users.

---

## 📋 Table of Contents

- [Overview & Features](#-overview--features)
- [Hybrid Deployment Architecture](#-hybrid-deployment-architecture)
- [Local Quick Start (3 Terminals)](#-local-quick-start-3-terminals)
- [Cloudflare Tunnel Setup](#-cloudflare-tunnel-setup)
- [Streamlit Cloud Deployment](#-streamlit-cloud-deployment)
- [API Reference](#-api-reference)
- [Pipeline Workflow](#-pipeline-workflow)
- [Supported ML Models](#-supported-ml-models)
- [User Expertise Levels](#-user-expertise-levels)
- [Regulatory & AI Governance](#-regulatory--ai-governance)
- [Troubleshooting](#-troubleshooting)
- [Limitations & Architecture Notes](#-limitations--architecture-notes)

---

## 🌟 Overview & Features

1. **Automated Dataset Health Profiling**: Identifies missing cells, data types, duplicate records, outliers, and candidate prediction targets.
2. **Intelligent Task Detection**: Automatically maps data to `Classification`, `Regression`, `Clustering`, or `Time-Series`.
3. **Data Leakage Prevention**: Separates train and test folds *prior* to fitting imputation and scaling transformers.
4. **Context-Aware Model Recommendation & Benchmarking**: Ranks and trains multiple model families (Random Forest, XGBoost, SVM, Linear/Logistic, KMeans).
5. **Explainable AI (XAI)**: Computes feature importance rankings and model decision profiles.
6. **Adaptive Natural-Language Insights**: Generates plain English explanations adapted to 3 expertise tiers.
7. **Big Data & Telemetry Profiler**: Estimates PySpark RAM, worker executor distribution, and Delta Lake/Parquet partitioning.
8. **Automated AI Governance**: EU AI Act risk tiering, GDPR PII scanning, and ISO 27001 audit logging.
9. **Multi-Format Report Generation**: Creates timestamped PDF, TXT, and JSON audit reports accessible via API.

---

## 🚀 Local Quick Start (3 Terminals)

### 1. Prerequisites & Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Local Secrets Configuration
Create `.streamlit/secrets.toml` in the project root (pre-configured):
```toml
API_URL = "http://127.0.0.1:8000"
```

### 3. Launch the 3 Services

#### Terminal 1 — Start FastAPI Backend
```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```
- Interactive API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

#### Terminal 2 — Start Secure Tunnel (Choose Cloudflare or Ngrok)

**Option A: Cloudflare Tunnel (Quick & Free)**
```powershell
# Standard command (if in PATH):
cloudflared tunnel --url http://localhost:8000

# Full Windows executable command:
& "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8000
```
> Copy the generated HTTPS URL (e.g., `https://xxxx.trycloudflare.com`).

**Option B: Ngrok (Permanent Static Domain — URL Never Changes)**
```powershell
# Standard command:
ngrok http --domain=vegetative-companionably-juli.ngrok-free.dev 8000

# Full Windows executable command:
& "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe\ngrok.exe" http --domain=vegetative-companionably-juli.ngrok-free.dev 8000
```

#### Terminal 3 — Start Streamlit Frontend (Local Testing)
```powershell
streamlit run frontend/app.py
```
- Local Streamlit UI: `http://localhost:8501`

---

## 🛡️ Cloudflare Tunnel Setup Details

To connect a public Streamlit Cloud app to your local FastAPI backend using Cloudflare:

### 1. Install Cloudflared on Windows
```powershell
winget install Cloudflare.cloudflared
```
*(Or download `cloudflared-windows-amd64.msi` directly from [Cloudflare GitHub Releases](https://github.com/cloudflare/cloudflared/releases)).*

### 2. Complete Launch Command
```powershell
& "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8000
```

### 3. Copy the Public HTTPS URL
`cloudflared` outputs a public HTTPS address:
```text
https://random-assigned-subdomain.trycloudflare.com
```

---

## ☁️ Streamlit Cloud Deployment

1. **Push your repository** to GitHub (excluding `.streamlit/secrets.toml`, reports, and cache):
   ```bash
   git add .
   git commit -m "Deploy ADSLM to Streamlit Cloud"
   git push origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and click **"New app"**.
3. Select your repository, branch (`main`), and set the main file path:
   ```text
   frontend/app.py
   ```
4. Click **"Advanced settings"** -> **"Secrets"**, and paste your tunnel URL:
   ```toml
   # If using Cloudflare Tunnel:
   API_URL = "https://random-assigned-subdomain.trycloudflare.com"

   # Or if using your permanent Ngrok domain:
   API_URL = "https://vegetative-companionably-juli.ngrok-free.dev"
   ```
5. Click **"Deploy"**. Your Streamlit Cloud app will communicate securely with your local FastAPI pipeline.


---

## 🔌 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Backend status probe for health monitoring |
| `GET` | `/` | Welcome endpoint and API directory |
| `POST` | `/analyze` | Dataset profiling and health check without training |
| `POST` | `/orchestrate` | Full end-to-end ML pipeline (analysis, training, XAI, governance, report) |
| `GET` | `/report/{filename}` | Secure download for generated PDF, TXT, or JSON reports |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc API documentation |

### Example Pipeline Request (`/orchestrate`)
```bash
curl -X POST http://localhost:8000/orchestrate \
  -F "file=@datasets/predictive_maintenance.csv" \
  -F "target_column=Fault" \
  -F "expertise_level=intermediate"
```

### Example Report Download (`/report/{filename}`)
```bash
curl -O http://localhost:8000/report/adslm_report_20260906_131206.pdf
```

---

## 🔄 Pipeline Workflow

```
User uploads CSV (Streamlit Cloud)
        ↓
POST /orchestrate (via Cloudflare Tunnel HTTPS)
        ↓
1. DatasetAnalyzer: Profiling (shape, types, missing, outliers)
        ↓
2. TaskDetector: Rules-based task mapping (Classification / Regression / Clustering / Time-Series)
        ↓
3. PreprocessingEngine: Train/Test split BEFORE transformer fitting (imputation, encoding, scaling)
        ↓
4. ModelRecommendationEngine: Context-ranked algorithms
        ↓
5. TrainingEngine: Multi-model training and metric evaluation
        ↓
6. ExplainabilityEngine: Feature importance & model profiling
        ↓
7. BigDataEngine & RegulatoryComplianceEngine: Telemetry, GDPR, EU AI Act audit
        ↓
8. InsightGenerator & ExpertiseAdapter: Natural language narratives (Beginner / Intermediate / Expert)
        ↓
9. ReportGenerator: PDF / TXT / JSON reports saved to /reports
        ↓
FastAPI returns JSON + Streamlit displays metrics & links to /report/{filename}
```

---

## 📊 Supported ML Models

| Task Type | Implemented Algorithms |
|-----------|------------------------|
| **Classification** | Logistic Regression, Random Forest Classifier, XGBoost Classifier, Support Vector Classifier (SVC) |
| **Regression** | Linear Regression, Random Forest Regressor, XGBoost Regressor |
| **Clustering** | KMeans, DBSCAN |
| **Time-Series** | Tabular temporal models (ARIMA & Prophet feature regression) |

---

## 🎓 User Expertise Levels

- **Beginner**: Strips ML jargon, uses intuitive everyday analogies, and highlights emoji indicators for floor operators.
- **Intermediate**: Natural language insights with balanced technical clarity (default).
- **Expert**: Rigorous statistical summaries, algorithmic details, and engineering caveats for data scientists.

---

## 🏭 Regulatory & AI Governance

- **EU AI Act (2024/2026)**: Categorizes the workload risk tier (e.g. High Risk for safety-critical assets) and lists compliance obligations.
- **GDPR Privacy Audit**: Scans columns and content for Personally Identifiable Information (PII) like emails, IPs, and phone numbers.
- **ISO 27001 Traceability**: Generates an audit compliance score (0–100) reflecting data isolation, logging, and XAI explainability.

---

## 🔧 Troubleshooting

### 1. `[Errno 10048] address already in use: 8000`
If port 8000 is occupied by a previous uvicorn session:
```powershell
# Kill all python processes holding port 8000
Get-Process python | Stop-Process -Force
```

### 2. `cloudflared : The term is not recognized`
Run with the absolute path or add to your session PATH:
```powershell
& "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8000
```

### 3. Streamlit Backend Connection Error
Ensure:
1. `uvicorn` is running in Terminal 1.
2. If on Streamlit Cloud, verify `API_URL` in **Streamlit Settings -> Secrets** matches your active Cloudflare Tunnel URL.

---

## ⚠️ Limitations & Architecture Notes

1. **Local Backend Execution**: For this demo architecture, the FastAPI backend and model training engine run on your local laptop. Your laptop and Cloudflare Tunnel must remain active while accessing the Streamlit Cloud frontend.
2. **Session Persistence**: Cloudflare quick tunnels generate a dynamic URL on restart; update Streamlit Cloud Secrets with the new tunnel URL if the tunnel is restarted.
3. **Report Storage**: Reports are saved to the backend `reports/` folder and streamed securely through the `/report/{filename}` endpoint.

---

## 📜 Verification & Testing

Run the automated test suite locally:
```bash
python test_deployment.py
```
This tests:
- `/health` status probe
- `/orchestrate` full pipeline execution
- `/report/{filename}` file streaming and path traversal prevention
- Empty and invalid file upload rejection
- Local `.streamlit/secrets.toml` configuration

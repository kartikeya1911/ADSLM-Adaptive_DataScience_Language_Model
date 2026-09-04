# 🤖 ADSLM — Adaptive Data Science Language Model
### ABB Innovation Evaluation Project 2026

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189F50?style=for-the-badge)

> An **industrial-grade, AI-powered AutoML + Data Science Copilot** that intelligently adapts to any dataset, automatically detects ML task types, trains and evaluates multiple models, generates explainable AI outputs, and delivers expert-level natural language insights — tailored to three user expertise levels.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Pipeline Workflow](#-pipeline-workflow)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Module Descriptions](#-module-descriptions)
- [Supported Models](#-supported-models)
- [Expertise Levels](#-expertise-levels)
- [Dataset](#-dataset)
- [Industrial Relevance (ABB)](#-industrial-relevance-abb)
- [Tech Stack](#-tech-stack)
- [Scalability & Future Scope](#-scalability--future-scope)
- [Interview Talking Points](#-interview-talking-points)

---

## 🌟 Overview

ADSLM is a fully autonomous end-to-end machine learning system built for ABB's industrial evaluation. Given any CSV dataset, it:

1. **Profiles** the dataset — shape, dtypes, nulls, outliers, duplicates, potential targets
2. **Detects** the correct ML task type — Regression, Classification, Clustering, or Time-Series
3. **Preprocesses** automatically — imputation, encoding, scaling, and stratified splitting
4. **Recommends** the best models ranked by dataset context and feature characteristics
5. **Trains** all candidate models in parallel and selects the best performer
6. **Evaluates** with task-appropriate metrics (F1, RMSE, R², Silhouette)
7. **Explains** predictions via feature importance extraction and model profiling
8. **Generates** natural language insights adapted to Beginner / Intermediate / Expert levels
9. **Reports** full audit trails as PDF, TXT, and JSON

---

## 🏗️ Architecture

```
ABB/
├── main.py                              ← FastAPI application entry point
├── requirements.txt                     ← All pinned dependencies
├── test_pipeline.py                     ← End-to-end smoke test
├── README.md                            ← This file
├── ABB_EVALUATION_SUBMISSION.md         ← Submission guide
│
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                    ← FastAPI endpoints (analyze, orchestrate, health)
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                    ← Central configuration & constants
│   ├── services/                        ← 10 ADSLM engine modules
│   │   ├── dataset_analyzer.py          ← Module 1: Dataset profiling
│   │   ├── task_detection.py            ← Module 2: Task type detection
│   │   ├── preprocessing.py             ← Module 3: Auto preprocessing
│   │   ├── model_recommendation.py      ← Module 4: Context-aware model ranking
│   │   ├── training.py                  ← Module 5: Multi-model training & selection
│   │   ├── evaluation.py               ← Module 6: Task-specific metrics
│   │   ├── explainability.py           ← Module 7: XAI & feature importance
│   │   ├── insight_generator.py        ← Module 8: AI-style NL insights
│   │   ├── expertise_adaptation.py     ← Module 9: Beginner/Intermediate/Expert output
│   │   └── report_generator.py         ← Module 10: PDF + TXT + JSON reports
│   └── utils/
│       ├── logger.py                    ← Centralized UTF-8 logging
│       └── helpers.py                   ← NumpyEncoder & JSON utilities
│
├── frontend/
│   └── app.py                          ← Premium Streamlit UI (Plotly visualizations)
│
├── datasets/
│   └── abb_predictive_maintenance.csv  ← 50-record industrial sample dataset
│
├── saved_models/                       ← Auto-saved best models (.pkl via joblib)
└── reports/                            ← Auto-generated reports (PDF / TXT / JSON)
```

---

## 🔄 Pipeline Workflow

```
CSV Upload
    ↓
[1] DatasetAnalyzer
    Shape, dtypes, nulls, outliers, duplicates, potential targets
    ↓
[2] TaskDetector
    Rules: no target → Clustering | datetime+num → TimeSeries
           cat target → Classification | num target → Regression
    ↓
[3] PreprocessingEngine
    Drop duplicates → median/mode impute → OHE → StandardScaler → 80/20 split
    ↓
[4] ModelRecommendationEngine
    Context-aware ranked list (dataset size, feature mix, task type)
    ↓
[5] TrainingEngine
    Train all models via registry → evaluate → select & save best (joblib)
    ↓
[6] EvaluationEngine
    F1 / Accuracy (Classification) | RMSE / MAE / R² (Regression) | Silhouette (Clustering)
    ↓
[7] ExplainabilityEngine
    feature_importances_ / |coef_| extraction → top-N ranked features
    Model profile: type, strengths, limitations
    ↓
[8] InsightGenerator
    AI-style narratives: dataset health, task rationale, preprocessing summary,
    model choice justification, feature insights, actionable recommendations
    ↓
[9] ExpertiseAdapter
    Beginner   → plain English, analogies, emoji indicators
    Intermediate → natural language with moderate technical detail
    Expert     → statistical depth, architecture notes, engineering caveats
    ↓
[10] ReportGenerator
    Full TXT + JSON + PDF report saved to /reports
    ↓
Streamlit Dashboard
    Metrics cards, Plotly charts, insight panels, download button
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Clone & Install

```bash
git clone <repository-url>
cd ABB
pip install -r requirements.txt
```

### 2. Start the Backend (FastAPI)

```bash
# Terminal 1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start the Frontend (Streamlit)

```bash
# Terminal 2
streamlit run frontend/app.py
```

### 4. Open in Browser

| Service | URL |
|---------|-----|
| **Streamlit UI** | http://localhost:8501 |
| **Swagger API Docs** | http://localhost:8000/docs |
| **ReDoc API Docs** | http://localhost:8000/redoc |
| **API Root** | http://localhost:8000 |

### 5. Run End-to-End Smoke Test

```bash
python test_pipeline.py
```

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message + endpoint map |
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Dataset analysis only (no training) |
| `POST` | `/orchestrate` | Full end-to-end pipeline |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |
| `GET` | `/api/v1/...` | Versioned equivalents of all above |

### `/analyze` — Dataset Analysis

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@datasets/abb_predictive_maintenance.csv"
```

**Response:** Dataset profile — shape, dtypes, null counts, outlier flags, potential target columns, and detected task type.

### `/orchestrate` — Full Pipeline

```bash
curl -X POST http://localhost:8000/orchestrate \
  -F "file=@datasets/abb_predictive_maintenance.csv" \
  -F "target_column=Fault" \
  -F "expertise_level=intermediate"
```

**Parameters:**

| Parameter | Type | Required | Values | Default |
|-----------|------|----------|--------|---------|
| `file` | File (CSV) | ✅ | Any CSV | — |
| `target_column` | string | ❌ | Column name | Auto-detected |
| `expertise_level` | string | ❌ | `beginner` \| `intermediate` \| `expert` | `intermediate` |

**Response:** Complete pipeline results including analysis, preprocessing info, model rankings, evaluation metrics, feature importances, NL insights, and report file paths.

---

## 🧩 Module Descriptions

| # | Module | File | Purpose |
|---|--------|------|---------|
| 1 | **Dataset Analyzer** | `dataset_analyzer.py` | Profiles shape, dtypes, nulls, outliers, duplicates, and potential target columns |
| 2 | **Task Detector** | `task_detection.py` | Rules-based task type detection (Regression / Classification / Clustering / Time-Series) with reason generation |
| 3 | **Preprocessing Engine** | `preprocessing.py` | Median/mode imputation, One-Hot Encoding, StandardScaler, stratified 80/20 split |
| 4 | **Model Recommender** | `model_recommendation.py` | Dataset-aware ranked model recommendations with industrial rationale |
| 5 | **Training Engine** | `training.py` | Trains all candidate models, selects & saves best via joblib |
| 6 | **Evaluation Engine** | `evaluation.py` | Task-specific metrics: F1/Accuracy, RMSE/MAE/R², Silhouette Score |
| 7 | **XAI Module** | `explainability.py` | Feature importance extraction, model profiles, strengths/limitations |
| 8 | **Insight Generator** | `insight_generator.py` | AI-style narratives covering all pipeline stages + actionable recommendations |
| 9 | **Expertise Adapter** | `expertise_adaptation.py` | Rewrites insights for Beginner / Intermediate / Expert audiences |
| 10 | **Report Generator** | `report_generator.py` | Full PDF (via ReportLab) + TXT + JSON audit reports |
| — | **API Layer** | `routes.py` | FastAPI router with full pipeline orchestration and error handling |
| — | **Frontend** | `frontend/app.py` | Premium Streamlit UI with Plotly charts and download buttons |

---

## 📊 Supported Models

| Task | Models |
|------|--------|
| **Regression** | Linear Regression, Random Forest Regressor, XGBoost Regressor |
| **Classification** | Logistic Regression, Random Forest Classifier, XGBoost Classifier, SVM |
| **Clustering** | KMeans, DBSCAN |
| **Time-Series** | ARIMA, Prophet *(extensible)* |
| **Time-Series** | ARIMA, Prophet |

> Adding a new model requires only one line in the `model_registry` — zero changes to the rest of the pipeline.

---

## 🎓 Expertise Levels

| Level | Audience | Output Style |
|-------|----------|-------------|
| **Beginner** | Non-technical stakeholders | Plain English, no jargon, real-world analogies, emoji indicators |
| **Intermediate** | Data-aware professionals | Natural language with moderate technical detail (default) |
| **Expert** | Data scientists / ML engineers | Full statistical depth, architecture notes, engineering caveats |

---

## 📁 Dataset

**File:** `datasets/predictive_maintenance.csv`

An industrial sample dataset simulating equipment sensor telemetry with the following characteristics:

- Sensor readings (temperature, vibration, pressure, etc.)
- Operational labels (`Fault` column: `No Fault`, `Bearing Wear`, `Critical`)
- Mixed feature types — numeric sensors + categorical status flags
- Intentional noise: missing values, outliers, and class imbalance for realistic preprocessing testing

---

## 🏭 Industrial Relevance & Regulatory Compliance

| Domain / Governance | How ADSLM Helps |
|---------------------|----------------|
| **Predictive Maintenance** | Classify faults (`Bearing Wear`, `Critical`, `No Fault`) from high-frequency sensor telemetry |
| **Energy Management** | Regression on power consumption, load forecasting, efficiency prediction |
| **Anomaly Detection** | Clustering unlabeled operational sensor streams (normal / degraded / critical) |
| **EU AI Act (2024/2026)** | Automated Risk Classification (Category 3 High-Risk for Asset Safety) + Mandatory Human Oversight rules |
| **GDPR Privacy Audit** | Automated regex scanning for PII (emails, IPs, phone numbers) with auto-anonymization flags |
| **ISO 27001 Traceability** | 0–100 Compliance Score audit logging, data leakage protection, and reproducible model cards |
| **Big Data Streaming** | Ingestion throughput telemetry (MB/s), Delta Lake / Apache Parquet partitioning strategies |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **API Framework** | FastAPI |
| **Frontend** | Streamlit |
| **Governance Engine** | RegulatoryComplianceEngine |
| **ML & Boosting** | Scikit-learn, XGBoost |
| **Data & Numerics** | Pandas, NumPy |
| **Visualization** | Plotly |
| **Reporting** | ReportLab |

---

## 🚀 Scalability & Future Scope

| Feature | Description |
|---------|-------------|
| **Delta Lake Ingestion** | Connector for streaming telemetry from Azure Event Hubs / Kafka |
| **PySpark MLlib** | Distributed training for multi-terabyte datasets |
| **SHAP Integration** | Model-agnostic feature explanations |
| **MLflow Registry** | Production model versioning and rollback |
| **Kubernetes Scaling** | Azure AKS deployment with Horizontal Pod Autoscaler |

---

## 💼 Interview Talking Points

**Q: How does ADSLM address Big Data requirements?**
> The `BigDataEngine` profiles dataset volume and ingestion throughput (MB/s), calculates PySpark JVM memory requirements, and generates optimal Delta Lake / Apache Parquet partitioning strategies for high-frequency industrial telemetry.

**Q: How does ADSLM ensure Regulatory Compliance?**
> The `RegulatoryComplianceEngine` performs automated governance audits: (1) **EU AI Act**: classifies model risk tiers and mandates human oversight; (2) **GDPR**: scans for PII and flags anonymization requirements; (3) **ISO 27001**: computes an auditability score (0–100) covering logging, XAI lineage, and data leakage isolation.

**Q: What makes this "adaptive"?**
> The system adapts at **three levels**: (1) *task type* — handles Regression, Classification, Clustering, and Time-Series; (2) *dataset scale* — adapts preprocessing (hybrid encoding for high-cardinality ID columns) and model parameters (`n_jobs=-1`, sample caps); (3) *audience* — rewrites insights for Beginner, Intermediate, and Expert stakeholders.

**Q: How is this production-grade?**
> Config-driven constants (`config.py`), centralized UTF-8 logging (`logger.py`), `NumpyEncoder` for safe JSON serialization, stratified splits to prevent data leakage, `joblib` model serialization, modular router architecture, and structured error handling at every pipeline stage.

**Q: How would you scale this?**
> Replace `joblib` with MLflow for experiment tracking, add Optuna for HPO, containerize with Docker, and deploy on Azure AKS with horizontal pod autoscaling. The router-based API design means each endpoint can be independently scaled.

**Q: Why FastAPI + Streamlit?**
> FastAPI is async, fully typed, and auto-documents via Swagger — production-ready from day one. Streamlit eliminates React boilerplate for rapid demo delivery while still supporting Plotly for rich, interactive visualizations. A perfect fit for an innovation evaluation context.

**Q: What design principles does this follow?**
> **Single Responsibility**: each of the 10 service modules owns exactly one pipeline stage. **Open/Closed**: adding a new model needs one line in the model registry — no changes to training, evaluation, or reporting logic. **Dependency Inversion**: all stages communicate through structured dicts, making them independently testable.

---

## 📜 License

This project was developed exclusively for the **ABB Innovation Evaluation 2026**. All rights reserved.

---

*ADSLM v1.0 | Adaptive Data Science Language Model | ABB Evaluation 2026*

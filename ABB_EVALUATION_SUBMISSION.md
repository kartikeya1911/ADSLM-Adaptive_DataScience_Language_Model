# ADSLM — Architecture & Submission Guide
## ABB Innovation Evaluation 2026

---

## 🚀 How to Run (3 Commands)

```bash
# Terminal 1 — Backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
streamlit run frontend/app.py

# Open browser → http://localhost:8501
```

---

## 📁 Complete Project Structure

```
ABB/
├── main.py                              ← FastAPI app entry point
├── requirements.txt                     ← All dependencies
├── test_pipeline.py                     ← End-to-end smoke test
├── README.md                            ← Full documentation
├── ABB_EVALUATION_SUBMISSION.md         ← This file
│
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                    ← All FastAPI endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                    ← Paths, constants, settings
│   ├── services/                        ← All 10 ADSLM engine modules
│   │   ├── dataset_analyzer.py          ← Module 1: Dataset profiling
│   │   ├── task_detection.py            ← Module 2: Task type detection
│   │   ├── preprocessing.py             ← Module 3: Auto preprocessing
│   │   ├── model_recommendation.py      ← Module 4: Model suggestion
│   │   ├── training.py                  ← Module 5: Multi-model training
│   │   ├── evaluation.py               ← Module 6: Metrics computation
│   │   ├── explainability.py           ← Module 7: XAI / feature importance
│   │   ├── insight_generator.py        ← Module 8: NL insight generation
│   │   ├── expertise_adaptation.py     ← Module 9: Beginner/Expert adapt
│   │   └── report_generator.py         ← Module 10: PDF/TXT/JSON reports
│   └── utils/
│       ├── logger.py                    ← Centralized UTF-8 logging
│       └── helpers.py                   ← NumpyEncoder, JSON utilities
│
├── frontend/
│   └── app.py                          ← Premium Streamlit UI
│
├── datasets/
│   └── abb_predictive_maintenance.csv  ← 50-record industrial sample
│
├── saved_models/                       ← Auto-saved best models (.pkl)
└── reports/                            ← Auto-generated reports (PDF/TXT/JSON)
```

---

## 🔄 Full Pipeline Flow (Step-by-Step)

```
User uploads CSV
      ↓
[1] DatasetAnalyzer
    - shape, dtypes, nulls, outliers, duplicates, potential targets
      ↓
[2] TaskDetector
    - Rules: no target→Clustering | datetime+num→TimeSeries
            | cat target→Classification | num target→Regression
      ↓
[3] PreprocessingEngine
    - Drop duplicates → median/mode impute → OHE → StandardScaler → 80/20 split
      ↓
[4] ModelRecommendationEngine
    - Context-aware ranked list (dataset size, feature types, task)
      ↓
[5] TrainingEngine
    - Train all models via registry → evaluate each → save best (joblib)
      ↓
[6] EvaluationEngine
    - F1/Accuracy (Classification) | RMSE/MAE/R² (Regression) | Silhouette (Clustering)
      ↓
[7] ExplainabilityEngine
    - feature_importances_ / coef_ extraction → top-N ranked features
    - Model profile (type, strengths, limitations)
      ↓
[8] InsightGenerator
    - AI-style narratives for: dataset, task, preprocessing, model, features
    - Actionable recommendations list
      ↓
[9] ExpertiseAdapter
    - Beginner: plain English, analogies, emoji
    - Intermediate: natural language (default)
    - Expert: statistical depth, architecture notes, engineering caveats
      ↓
[10] ReportGenerator
    - Full TXT + JSON + PDF report saved to /reports
      ↓
Streamlit Dashboard
    - Metrics cards, Plotly charts, insight cards, download button
```

---

## 🏭 Industrial Relevance for ABB

| ABB Domain | How ADSLM Helps |
|------------|----------------|
| Predictive Maintenance | Classify faults (Bearing Wear, Critical, No Fault) from sensor data |
| Energy Management | Regression on power consumption / efficiency |
| Anomaly Detection | Clustering of unlabeled operational sensor streams |
| Production Quality | Binary classification (Pass/Fail) on production line data |
| Process Forecasting | Time-Series for KPI prediction over shift schedules |

---

## 💼 Interview Talking Points

**Q: What makes this "adaptive"?**
> The system adapts at 3 levels: (1) task type — same pipeline handles 4 different ML problems; (2) model selection — recommendations change based on dataset size/feature mix; (3) output language — same insights rewritten for Beginner/Intermediate/Expert.

**Q: How does explainability work?**
> ExplainabilityEngine extracts feature importances from tree models (feature_importances_) or linear models (|coef_|), ranks them, and generates a narrative. The model profile module provides strengths/limitations text for every supported estimator.

**Q: How is this production-grade?**
> Config-driven constants, centralized UTF-8 logging, NumpyEncoder for safe JSON, stratified splits preventing data leakage, joblib model serialization, modular router architecture, error handling at every pipeline stage.

**Q: How would you scale this?**
> Replace joblib with MLflow for experiment tracking, add Optuna for HPO, containerize with Docker, deploy on Azure AKS with horizontal pod autoscaling. The router-based API design means each endpoint can be independently scaled.

**Q: Why FastAPI + Streamlit?**
> FastAPI is async, typed, auto-documented (Swagger), and production-ready. Streamlit eliminates React boilerplate for rapid demo delivery while still supporting Plotly for rich visualizations. Perfect for an innovation evaluation context.

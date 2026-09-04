"""
app/api/routes.py
==================
FastAPI Router — all ADSLM endpoints.

Endpoints:
    POST /upload        → Analyse a CSV dataset
    POST /orchestrate   → Full end-to-end pipeline (analyze → train → insights)
    GET  /health        → API health check
    GET  /report/{id}   → Retrieve a previously generated report
"""

import io
import json
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.services.bigdata_engine       import BigDataEngine
from app.services.dataset_analyzer    import DatasetAnalyzer
from app.services.expertise_adaptation import ExpertiseAdapter
from app.services.explainability      import ExplainabilityEngine
from app.services.insight_generator   import InsightGenerator
from app.services.model_recommendation import ModelRecommendationEngine
from app.services.preprocessing       import PreprocessingEngine
from app.services.regulatory_compliance import RegulatoryComplianceEngine
from app.services.report_generator    import ReportGenerator
from app.services.task_detection      import TaskDetector
from app.services.training            import TrainingEngine
from app.utils.helpers                import sanitize_for_json
from app.utils.logger                 import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ── Health Check ──────────────────────────────────────────────────────────────

@router.get("/health", tags=["System"])
def health_check():
    """Returns API status — useful for deployment health probes."""
    return {"status": "healthy", "service": "ADSLM", "version": "1.0.0"}


# ── Dataset Analysis Only ─────────────────────────────────────────────────────

@router.post("/analyze", tags=["Pipeline"])
async def analyze_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV and receive a full dataset analysis profile.
    Use this for quick data health checks without running ML training.
    """
    contents = await _read_csv_upload(file)
    try:
        analyzer = DatasetAnalyzer(io.StringIO(contents))
        stats    = analyzer.analyze()
        df       = pd.read_csv(io.StringIO(contents))
        bd       = BigDataEngine(df, len(contents.encode('utf-8'))).analyze_scale()
        rc       = RegulatoryComplianceEngine(df).run_audit()
        stats["big_data_profile"]   = bd
        stats["regulatory_audit"]  = rc
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Analysis failed: {e}")

    return JSONResponse(content=sanitize_for_json(stats))


# ── Full Pipeline Orchestration ───────────────────────────────────────────────

@router.post("/orchestrate", tags=["Pipeline"])
async def run_full_pipeline(
    file:           UploadFile = File(...),
    target_column:  Optional[str] = Form(None),
    expertise_level:str = Form("intermediate"),
):
    """
    Master endpoint — runs the complete ADSLM pipeline:

    1. Dataset Analysis
    2. Task Detection
    3. Preprocessing
    4. Model Recommendation
    5. Model Training & Evaluation
    6. Explainability (Feature Importance)
    7. Natural Language Insight Generation
    8. Expertise-level Adaptation
    9. Big Data Telemetry & Regulatory Compliance Audit
    10. Report Generation
    11. Return consolidated JSON response
    """
    logger.info(f"Pipeline triggered | target='{target_column}' | expertise='{expertise_level}'")

    # ── 0. Read file ──────────────────────────────────────────────────────────
    contents = await _read_csv_upload(file)
    try:
        df = pd.read_csv(io.StringIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    # ── 1. Dataset Analysis ───────────────────────────────────────────────────
    try:
        analyzer     = DatasetAnalyzer(io.StringIO(contents))
        analysis_stats = analyzer.analyze()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Dataset analysis failed: {e}")

    # ── 2. Task Detection ─────────────────────────────────────────────────────
    detector  = TaskDetector(analysis_stats, target_column)
    task_type = detector.detect_task()

    if task_type == "Unknown":
        raise HTTPException(
            status_code=400,
            detail="Could not determine the ML task type. "
                   "Please check the target column — it may not appear in the dataset."
        )

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    try:
        preprocessor = PreprocessingEngine(df, target_column, task_type)
        X_train, X_test, y_train, y_test = preprocessor.preprocess()
        prep_summary = preprocessor.get_summary()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    # ── 4. Model Recommendation ───────────────────────────────────────────────
    recommender   = ModelRecommendationEngine(task_type, analysis_stats)
    recommendations = recommender.recommend()
    models_to_run = [r["model"] for r in recommendations]

    # ── 5. Training & Evaluation ──────────────────────────────────────────────
    try:
        trainer          = TrainingEngine(task_type, X_train, y_train, X_test, y_test, models_to_run)
        training_results = trainer.run()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    best_model_name = training_results["best_model"]
    best_metrics    = training_results["all_results"].get(best_model_name, {})

    # ── 6. Explainability ─────────────────────────────────────────────────────
    feature_importances = {}
    top_features        = []
    model_summary       = {}

    if task_type not in ("Clustering",) and X_train is not None:
        try:
            best_model_obj = trainer.model_registry.get(best_model_name)
            if best_model_obj:
                xai = ExplainabilityEngine(best_model_obj, X_train.columns.tolist())
                feature_importances = xai.get_feature_importance()
                top_features        = xai.get_top_features(n=10)
                model_summary       = xai.get_model_summary()
        except Exception as e:
            logger.warning(f"Explainability step failed: {e}")
            feature_importances = {"_note": "Could not extract importances for this model."}

    # ── 7. Big Data Telemetry & Regulatory Audit ──────────────────────────────
    bigdata_profiler = BigDataEngine(df, len(contents.encode('utf-8')))
    big_data_profile = bigdata_profiler.analyze_scale()

    compliance_engine = RegulatoryComplianceEngine(df, task_type, target_column or "")
    regulatory_audit  = compliance_engine.run_audit(has_xai=bool(top_features))

    # ── 8. Insight Generation ─────────────────────────────────────────────────
    base_insights = {
        "data_profile":  InsightGenerator.generate_data_profile_insights(analysis_stats),
        "task":          InsightGenerator.generate_task_insight(task_type, target_column),
        "model":         InsightGenerator.generate_model_insight(best_model_name, task_type, best_metrics),
        "features":      InsightGenerator.generate_feature_insight(feature_importances),
        "preprocessing": InsightGenerator.generate_preprocessing_insight(prep_summary),
    }

    actionable_recs = InsightGenerator.generate_actionable_recommendations(
        task_type=task_type,
        metrics=best_metrics,
        prep_summary=prep_summary,
        analysis=analysis_stats,
        top_features=top_features,
        regulatory_audit=regulatory_audit,
        big_data_profile=big_data_profile,
    )

    # ── 9. Expertise Adaptation ───────────────────────────────────────────────
    adapted_insights = ExpertiseAdapter.adapt_insights(base_insights, expertise_level)

    # ── 10. Build & Save Report ───────────────────────────────────────────────
    full_response = {
        "metadata": {
            "task_type":     task_type,
            "target_column": target_column,
            "best_model":    best_model_name,
            "best_score":    training_results["best_score_metric"],
            "expertise_level": expertise_level,
        },
        "analysis":                  analysis_stats,
        "preprocessing_summary":     prep_summary,
        "recommendations":           recommendations,
        "all_model_results":         training_results["all_results"],
        "metrics":                   best_metrics,
        "feature_importances":       feature_importances,
        "top_features":              top_features,
        "model_profile":             model_summary,
        "insights":                  adapted_insights,
        "actionable_recommendations":actionable_recs,
        "big_data_profile":          big_data_profile,
        "regulatory_audit":          regulatory_audit,
        "saved_model_path":          training_results.get("saved_model_path"),
    }

    try:
        gen         = ReportGenerator(full_response)
        report_path = gen.generate()
        full_response["report_path"] = report_path
    except Exception as e:
        logger.warning(f"Report generation failed (non-critical): {e}")

    return JSONResponse(content=sanitize_for_json(full_response))


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _read_csv_upload(file: UploadFile) -> str:
    """Reads an uploaded file and returns its content as a UTF-8 string."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    try:
        raw = await file.read()
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("latin-1")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not decode file: {e}")

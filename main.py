"""
main.py
========
ADSLM FastAPI Application Entry Point

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

API Documentation:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

import os

# Silence joblib/loky CPU count detection warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import API_TITLE, API_VERSION, API_DESCRIPTION
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Lifespan Context ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(" ADSLM API starting up …")
    logger.info(f" Version : {API_VERSION}")
    logger.info(" Docs    : http://localhost:8000/docs")
    logger.info("=" * 60)
    yield
    logger.info("ADSLM API shutting down.")


# ── Application Factory ───────────────────────────────────────────────────────

app = FastAPI(
    title       = API_TITLE,
    version     = API_VERSION,
    description = API_DESCRIPTION,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
# Allow Streamlit frontend (localhost:8501) to communicate with FastAPI (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Restrict in production: ["http://localhost:8501"]
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Register Routers ──────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")

# Backwards-compatibility: keep /orchestrate and /analyze at root level
app.include_router(router)


# ── Root Endpoint ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    """Welcome endpoint — confirms API is running."""
    return {
        "message":     "🤖 ADSLM API is running!",
        "version":     API_VERSION,
        "description": API_DESCRIPTION,
        "endpoints": {
            "docs":        "/docs",
            "analyze":     "/analyze",
            "orchestrate": "/orchestrate",
            "health":      "/health",
        },
    }

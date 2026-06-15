"""
backend/app/main.py

FastAPI application factory for the Vehicle Damage Assessment API.

Thin entry point: configures logging, builds the app, mounts routers, and
pre-warms the VLM on startup. Business logic lives in `app/services/`; HTTP
routes in `app/routers/`; in-memory state in `app/state.py`.

Run from the repository root:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .core import config
from .db import init_db
from .routers import accounts, assessment, claims, feedback, health, images, insurance, sessions, settings, vehicles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables, then pre-warm the VLM so the server is ready quickly."""

    try:
        init_db()
        logger.info("Database tables ready.")
    except Exception as e:
        logger.error(f"Database init failed: {e}")

    async def _prewarm():
        try:
            cfg = config.load_config()
            from pipeline.orchestrator import _load_models
            logger.info("Pre-warming VLM on startup...")
            await run_in_threadpool(_load_models, cfg)
            logger.info("VLM pre-warm complete")
        except Exception as e:
            logger.warning(f"VLM pre-warm failed: {e}. Model will load on first request.")

    asyncio.create_task(_prewarm())
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount feature routers.
app.include_router(health.router)
app.include_router(assessment.router)
app.include_router(images.router)
app.include_router(sessions.router)
app.include_router(feedback.router)

# Persistence routers (PostgreSQL-backed).
app.include_router(accounts.router)
app.include_router(vehicles.router)
app.include_router(claims.router)
app.include_router(settings.router)
app.include_router(insurance.router)

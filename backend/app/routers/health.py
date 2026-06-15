"""Service metadata and VLM readiness."""

from fastapi import APIRouter

from ..core import config

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Report whether the VLM has finished warming up."""
    try:
        from pipeline.orchestrator import _model as vlm_model
        loaded = vlm_model is not None
    except ImportError:
        loaded = False
    return {"status": "ready" if loaded else "warming_up", "vlm_loaded": loaded}


@router.get("/")
async def root():
    """Service banner with the primary endpoints."""
    return {
        "service": config.API_TITLE,
        "version": config.API_VERSION,
        "endpoints": {
            "POST /assess": "Submit vehicle image for damage assessment",
            "GET  /job/{id}": "Poll assessment job status / result",
            "GET  /health": "Check VLM load status",
            "GET  /session/{id}": "Get HITL session state",
            "POST /session/{id}/approve": "Approve with human corrections",
            "POST /recalculate": "Recompute costs without re-running pipeline",
            "POST /api/feedback": "Direct feedback log write",
        },
    }

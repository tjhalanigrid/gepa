"""
backend/app.py

FastAPI application entry point.
Endpoints:
  POST /assess              — accept image, run pipeline, return report or session
  GET  /health              — VLM load status
  GET  /session/{id}        — retrieve session state
  POST /session/{id}/approve — human correction, finalise, write feedback log
  POST /recalculate         — recompute costs from COST_DB without re-running pipeline
  POST /api/feedback        — direct feedback log write
"""

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline.schema import (
    FinalDamageReport,
    DamagePartEntry,
    SessionState,
    RecalculateRequest,
    RecalculateResponse,
    ApproveRequest,
    FeedbackEntry,
)
from models.vlm_reasoning.sandbox import COST_DB

# Configure logging before any other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── In-memory session store — MVP only, lost on restart ──────────────────────
_sessions: dict[str, SessionState] = {}

FEEDBACK_LOG = Path("data/feedback/feedback_log.jsonl")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path("configs/global_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at {config_path}. "
            "Run from repo root: uvicorn backend.app:app"
        )
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _apply_cost_lookup(
    entries: list[DamagePartEntry],
) -> tuple[list[DamagePartEntry], int, int]:
    """
    Recalculate cost_min/cost_max for each entry using COST_DB.
    Falls back to severity-weighted average for unknown (damage, part) pairs.
    Returns (updated_entries, total_min, total_max).
    """
    updated = []
    for e in entries:
        costs = COST_DB.get(e.damage, {}).get(e.part)
        if costs:
            cost_min, cost_max = costs
        else:
            base = COST_DB.get(e.damage, {})
            if base:
                avg_min = int(sum(v[0] for v in base.values()) / len(base))
                avg_max = int(sum(v[1] for v in base.values()) / len(base))
            else:
                avg_min, avg_max = 5000, 15000
            multipliers = {"minor": 0.6, "moderate": 1.0, "severe": 1.6}
            m = multipliers.get(e.severity, 1.0)
            cost_min = int(avg_min * m)
            cost_max = int(avg_max * m)
        updated.append(DamagePartEntry(
            damage=e.damage,
            part=e.part,
            severity=e.severity,
            cost_min=cost_min,
            cost_max=cost_max,
        ))
    total_min = sum(e.cost_min for e in updated)
    total_max = sum(e.cost_max for e in updated)
    return updated, total_min, total_max


def _write_feedback(entry: FeedbackEntry) -> None:
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG, "a") as f:
        f.write(entry.model_dump_json() + "\n")
    logger.info(f"Feedback written: session={entry.session_id}")


# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm VLM on startup so first request is not penalised."""
    try:
        config = _load_config()
        from pipeline.orchestrator import _load_models
        logger.info("Warming up VLM on startup...")
        _load_models(config)
        logger.info("VLM warmup complete — server ready.")
    except Exception as e:
        logger.error(f"VLM warmup failed: {e}. First request will trigger load.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Vehicle Damage Assessment API",
    description="Thinking with Images — Qwen2-VL orchestrated damage detection and cost estimation.",
    version="0.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        from pipeline.orchestrator import _model as vlm_model
        loaded = vlm_model is not None
    except ImportError:
        loaded = False
    return {
        "status": "ready" if loaded else "warming_up",
        "vlm_loaded": loaded
    }


@app.post("/assess")
async def assess_damage(
    image: UploadFile = File(...),
    claim_id: str = Form(default=None),
    vehicle_id: str = Form(default=None),
):
    """
    Accept a vehicle image and return a FinalDamageReport.

    If approval_decision is ESCALATE_TO_HUMAN, stores session and returns:
      {session_id, report, status: "pending_review"}

    Otherwise returns FinalDamageReport JSON directly.
    """
    allowed_types = ("image/jpeg", "image/png", "image/webp")
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: '{image.content_type}'. "
                f"Accepted: {', '.join(allowed_types)}"
            )
        )

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(image.filename).suffix if image.filename else ".jpg"
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    save_path = upload_dir / unique_name

    try:
        with save_path.open("wb") as f:
            shutil.copyfileobj(image.file, f)
    except Exception as e:
        logger.error(f"Failed to save uploaded image: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded image.")

    logger.info(
        f"Image saved: {save_path.name} | "
        f"claim_id={claim_id} | vehicle_id={vehicle_id}"
    )

    try:
        config = _load_config()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    claim_metadata = {}
    if claim_id:
        claim_metadata["claim_id"] = claim_id
    if vehicle_id:
        claim_metadata["vehicle_id"] = vehicle_id

    try:
        from pipeline.orchestrator import run as orchestrator_run
        report = orchestrator_run(
            image_path=str(save_path.resolve()),
            config=config,
            claim_metadata=claim_metadata if claim_metadata else None
        )
    except ValueError as e:
        logger.warning(f"Validation error during assessment: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Model runtime error: {e}")
        raise HTTPException(status_code=503, detail=f"Model error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during assessment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal assessment error.")

    if report.get("approval_decision") == "ESCALATE_TO_HUMAN":
        session_id = uuid.uuid4().hex
        session = SessionState(
            session_id=session_id,
            status="pending_review",
            report=FinalDamageReport(**report),
            created_at=datetime.now(timezone.utc).isoformat(),
            claim_id=claim_id or None,
        )
        _sessions[session_id] = session
        logger.info(f"Session created: {session_id} (ESCALATE_TO_HUMAN)")
        return {"session_id": session_id, "report": report, "status": "pending_review"}

    return report


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Return current state of a HITL session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return _sessions[session_id].model_dump()


@app.post("/session/{session_id}/approve")
async def approve_session(session_id: str, request: ApproveRequest):
    """
    Accept human corrections, recalculate costs, finalise report,
    write to feedback log, return finalised report dict.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    session = _sessions[session_id]
    if session.status != "pending_review":
        raise HTTPException(
            status_code=409,
            detail=f"Session status is '{session.status}', expected 'pending_review'"
        )

    updated_entries, total_min, total_max = _apply_cost_lookup(request.damage_part_map)

    session.status = "approved"
    session.corrected_map = updated_entries
    session.correction_notes = request.correction_notes

    feedback = FeedbackEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id=session_id,
        image_path=session.report.image_path,
        claim_id=session.claim_id,
        original_report=session.report,
        human_corrections={
            "damage_part_map": [e.model_dump() for e in updated_entries],
            "correction_notes": request.correction_notes,
        },
        final_total_min=total_min,
        final_total_max=total_max,
    )
    _write_feedback(feedback)

    finalised = session.report.model_dump()
    finalised.update({
        "damage_part_map": [e.model_dump() for e in updated_entries],
        "total_min": total_min,
        "total_max": total_max,
        "approval_decision": "HUMAN_APPROVED",
    })
    return finalised


@app.post("/recalculate", response_model=RecalculateResponse)
async def recalculate(request: RecalculateRequest):
    """
    Recompute cost_min/cost_max for each damage-part entry using COST_DB.
    Does NOT re-run the VLM or CV models — pure math, synchronous.
    """
    updated_entries, total_min, total_max = _apply_cost_lookup(request.damage_part_map)
    return RecalculateResponse(
        damage_part_map=updated_entries,
        total_min=total_min,
        total_max=total_max,
    )


@app.post("/api/feedback")
async def submit_feedback(entry: FeedbackEntry):
    """Direct feedback log write. Used for external or programmatic submissions."""
    _write_feedback(entry)
    return {"status": "written"}


@app.get("/")
async def root():
    return {
        "service": "Vehicle Damage Assessment API",
        "version": "0.2.0",
        "endpoints": {
            "POST /assess": "Submit vehicle image for damage assessment",
            "GET  /health": "Check VLM load status",
            "GET  /session/{id}": "Get HITL session state",
            "POST /session/{id}/approve": "Approve with human corrections",
            "POST /recalculate": "Recompute costs without re-running pipeline",
            "POST /api/feedback": "Direct feedback log write",
        }
    }

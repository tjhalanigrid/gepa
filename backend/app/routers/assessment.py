"""
Assessment intake and job polling.

  POST /assess            — save image, queue async pipeline job, return job_id
  GET  /job/{id}          — poll job status / result
  GET  /job/{id}/iterations — per-tool-call iteration log for the UI
"""

import asyncio
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import state
from ..core import config
from ..services.assessment import run_assessment_job

logger = logging.getLogger(__name__)
router = APIRouter(tags=["assessment"])


@router.post("/assess")
async def assess_damage(
    image: UploadFile = File(...),
    claim_id: str = Form(default=None),
    vehicle_id: str = Form(default=None),
):
    """
    Accept a vehicle image, save it, and immediately return a job_id.
    The pipeline runs asynchronously — poll GET /job/{job_id} for the result.
    """
    if image.content_type not in config.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: '{image.content_type}'. "
                f"Accepted: {', '.join(config.ALLOWED_IMAGE_TYPES)}"
            ),
        )

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename).suffix if image.filename else ".jpg"
    save_path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    try:
        with save_path.open("wb") as f:
            shutil.copyfileobj(image.file, f)
    except Exception as e:
        logger.error(f"Failed to save uploaded image: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded image.")

    logger.info(
        f"Image saved: {save_path.name} | claim_id={claim_id} | vehicle_id={vehicle_id}"
    )

    try:
        cfg = config.load_config()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    job_id = uuid.uuid4().hex
    state.jobs[job_id] = {
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    asyncio.create_task(
        run_assessment_job(job_id, save_path, cfg, claim_id, vehicle_id)
    )
    logger.info(f"Job queued: {job_id}")
    return {"job_id": job_id, "status": "processing"}


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Poll assessment job status. Returns one of:
      {status: "queued" | "processing", elapsed_s: int}
      {status: "complete", result: <FinalDamageReport or HITL session dict>}
      {status: "failed", error: str}
    """
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = state.jobs[job_id]

    # Surface elapsed time and auto-fail jobs stuck past the wall clock.
    if job["status"] == "processing" and "started_at" in job:
        started = datetime.fromisoformat(job["started_at"])
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        job["elapsed_s"] = round(elapsed)
        if elapsed > 600:
            job["status"] = "failed"
            job["error"] = (
                "Job timed out after 600 seconds. The VLM inference exceeded the "
                "maximum allowed time. Restart the server and try again."
            )
            logger.warning(f"Job {job_id} auto-timed-out after {elapsed:.0f}s")

    return job


@router.get("/job/{job_id}/iterations")
async def get_job_iterations(job_id: str):
    """Return the per-tool-call iteration log for the UI iteration-logs panel."""
    report = state.get_completed_report(job_id)
    return {
        "iterations": report.get("iterations", []),
        "merged_detections": report.get("merged_detections", []),
        "approval_decision": report.get("approval_decision"),
    }

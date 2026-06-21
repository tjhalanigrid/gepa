"""
Assessment intake and job polling.

  POST /assess            — save image to DB, queue async pipeline job, return job_id
  GET  /job/{id}          — poll job status / result
  GET  /job/{id}/iterations — per-tool-call iteration log for the UI

Images are stored as BYTEA in the claim_images table — no filesystem dependency.
The pipeline still needs a file path, so we write a temp file for its duration
and delete it once the job completes.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session as DBSession

from .. import state
from ..core import config
from ..db import get_db
from ..models import ClaimImage
from ..services.assessment import run_assessment_job

logger = logging.getLogger(__name__)
router = APIRouter(tags=["assessment"])


@router.post("/assess")
async def assess_damage(
    image: UploadFile = File(...),
    claim_id: str = Form(default=None),
    vehicle_id: str = Form(default=None),
    db: DBSession = Depends(get_db),
):
    """
    Accept a vehicle image, persist it to the DB, and immediately return a job_id.
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

    image_bytes = await image.read()
    mime_type = image.content_type or "image/jpeg"
    suffix = Path(image.filename).suffix if image.filename else ".jpg"

    try:
        cfg = config.load_config()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    job_id = uuid.uuid4().hex

    # Persist the upload to a PERMANENT path (not a temp file). The image_path that
    # ends up in corrections_log.jsonl must stay valid after the job, because GEPA
    # re-reads the original image to score prompt candidates. (Also stored in DB below.)
    config.NEW_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.NEW_UPLOAD_DIR / f"{job_id}{suffix}"
    try:
        save_path.write_bytes(image_bytes)
    except Exception as e:
        logger.error(f"Failed to save upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to prepare image for processing.")

    # Persist the original image to DB as well (redundant with the file; survives any
    # later filesystem cleanup).
    def _save_original():
        db.add(ClaimImage(
            job_id=job_id,
            image_type="original",
            mime_type=mime_type,
            data=image_bytes,
        ))
        db.commit()

    await run_in_threadpool(_save_original)

    state.jobs[job_id] = {
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        # No "_tmp_path": the permanent upload is intentionally kept so GEPA can read
        # it later as ground truth. (force_human_review collection mode.)
    }

    asyncio.create_task(
        run_assessment_job(job_id, save_path, cfg, claim_id, vehicle_id)
    )
    logger.info(
        f"Job queued: {job_id} | original image saved to DB ({len(image_bytes):,} bytes)"
    )
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

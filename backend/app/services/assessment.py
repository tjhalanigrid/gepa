"""
Background assessment job runner.

`run_assessment_job` executes the pipeline orchestrator in a threadpool so the
event loop stays responsive, records the result on the shared `jobs` store, and
persists all pipeline-generated images (annotated, masked, merged) to the DB.
Temp files are deleted after the job completes — no filesystem residue.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi.concurrency import run_in_threadpool

from pipeline.schema import FinalDamageReport, SessionState

from .. import state

logger = logging.getLogger(__name__)

JOB_TIMEOUT_S = 360  # 6 min — leaves ~40s buffer before the 600s client wall


async def run_assessment_job(
    job_id: str,
    save_path: Path,
    config: dict,
    claim_id: str | None,
    vehicle_id: str | None,
) -> None:
    """Run orchestrator.run(), persist generated images to DB, and clean up."""
    try:
        state.jobs[job_id]["status"] = "processing"
        state.jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

        from pipeline.orchestrator import run as orchestrator_run

        claim_metadata: dict = {}
        if claim_id:
            claim_metadata["claim_id"] = claim_id
        if vehicle_id:
            claim_metadata["vehicle_id"] = vehicle_id

        report = await asyncio.wait_for(
            run_in_threadpool(
                orchestrator_run,
                str(save_path.resolve()),
                config,
                claim_metadata or None,
            ),
            timeout=JOB_TIMEOUT_S,
        )

        # Save annotated / masked / merged images to DB then delete their files.
        await _persist_generated_images(job_id, report)

        if report.get("approval_decision") == "ESCALATE_TO_HUMAN":
            session_id = uuid.uuid4().hex
            state.sessions[session_id] = SessionState(
                session_id=session_id,
                status="pending_review",
                report=FinalDamageReport(**report),
                created_at=datetime.now(timezone.utc).isoformat(),
                claim_id=claim_id or None,
                job_id=job_id,
            )
            logger.info(f"Session created: {session_id} (ESCALATE_TO_HUMAN)")
            state.jobs[job_id]["status"] = "complete"
            state.jobs[job_id]["result"] = {
                "session_id": session_id,
                "report": report,
                "status": "pending_review",
            }
        else:
            state.jobs[job_id]["status"] = "complete"
            state.jobs[job_id]["result"] = report

        state.jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(f"Job {job_id} completed successfully")

    except asyncio.TimeoutError:
        logger.error(f"Job {job_id} hard timeout after {JOB_TIMEOUT_S}s")
        state.jobs[job_id]["status"] = "failed"
        state.jobs[job_id]["error"] = (
            f"Job timed out after {JOB_TIMEOUT_S}s. "
            "VLM did not complete within the time budget."
        )
        try:
            import gc as _gc
            import torch as _torch
            _gc.collect()
            if _torch.backends.mps.is_available():
                _torch.mps.empty_cache()
        except Exception:
            pass
    except MemoryError as e:
        state.jobs[job_id]["status"] = "failed"
        state.jobs[job_id]["error"] = (
            "Out of memory during inference. Try a smaller image or restart the server."
        )
        logger.error(f"Job {job_id} failed with MemoryError: {e}")
    except RuntimeError as e:
        state.jobs[job_id]["status"] = "failed"
        state.jobs[job_id]["error"] = str(e)
        logger.error(f"Job {job_id} failed with RuntimeError: {e}")
    except Exception as e:
        state.jobs[job_id]["status"] = "failed"
        state.jobs[job_id]["error"] = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Job {job_id} failed with unexpected error: {e}", exc_info=True)
    finally:
        # Always delete the temp upload file regardless of success/failure.
        _delete_temp_file(job_id)


async def _persist_generated_images(job_id: str, report: dict) -> None:
    """Read pipeline output images, write BYTEA rows to DB, delete the files."""
    image_keys = [
        ("annotated", "annotated_image_path"),
        ("masked",    "masked_image_path"),
        ("merged",    "merged_image_path"),
    ]

    def _write_sync():
        from ..db import SessionLocal
        from ..models import ClaimImage

        db = SessionLocal()
        try:
            saved = []
            for img_type, path_key in image_keys:
                path = report.get(path_key)
                if path and Path(path).exists():
                    data = Path(path).read_bytes()
                    db.add(ClaimImage(
                        job_id=job_id,
                        image_type=img_type,
                        mime_type="image/jpeg",
                        data=data,
                    ))
                    Path(path).unlink(missing_ok=True)
                    saved.append(img_type)
            db.commit()
            if saved:
                logger.info(f"Job {job_id}: saved images to DB: {saved}")
        except Exception as e:
            logger.warning(f"Job {job_id}: failed to persist images to DB: {e}")
            db.rollback()
        finally:
            db.close()

    await run_in_threadpool(_write_sync)


def _delete_temp_file(job_id: str) -> None:
    """Delete the original temp upload file tracked in state.jobs."""
    tmp_path = state.jobs.get(job_id, {}).pop("_tmp_path", None)
    if tmp_path:
        try:
            Path(tmp_path).unlink(missing_ok=True)
            logger.debug(f"Job {job_id}: temp file deleted: {tmp_path}")
        except Exception as e:
            logger.warning(f"Job {job_id}: could not delete temp file {tmp_path}: {e}")

"""
Background assessment job runner.

`run_assessment_job` executes the pipeline orchestrator in a threadpool so the
event loop stays responsive, and records the result (or a typed error) on the
shared `jobs` store. It never lets a job hang in "processing" forever.
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

# 6 min — leaves a ~40s buffer before the 600s client/poll wall.
JOB_TIMEOUT_S = 360


async def run_assessment_job(
    job_id: str,
    save_path: Path,
    config: dict,
    claim_id: str | None,
    vehicle_id: str | None,
) -> None:
    """Run orchestrator.run() and update state.jobs[job_id] with the outcome."""
    try:
        state.jobs[job_id]["status"] = "processing"
        state.jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

        from pipeline.orchestrator import run as orchestrator_run
        import torch as _torch

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

        if report.get("approval_decision") == "ESCALATE_TO_HUMAN":
            session_id = uuid.uuid4().hex
            state.sessions[session_id] = SessionState(
                session_id=session_id,
                status="pending_review",
                report=FinalDamageReport(**report),
                created_at=datetime.now(timezone.utc).isoformat(),
                claim_id=claim_id or None,
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
        import gc as _gc
        _gc.collect()
        try:
            import torch as _torch
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

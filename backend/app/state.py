"""
In-memory application state — MVP only, lost on restart.

`sessions` and `jobs` are module-level singletons shared across every router and
service (importing this module gives all of them the same dict objects). For
production these would move to Redis or a database.
"""

from fastapi import HTTPException

from pipeline.schema import SessionState

# job_id → {status, result?, error?, started_at?, ...}
jobs: dict[str, dict] = {}

# session_id → SessionState (ESCALATE_TO_HUMAN reports awaiting review)
sessions: dict[str, SessionState] = {}


def get_completed_report(job_id: str) -> dict:
    """Return the FinalDamageReport dict for a completed job, or raise 404/400."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job.get("status") != "complete":
        raise HTTPException(status_code=400, detail="Job not complete yet")
    result = job.get("result", {})
    # Escalated jobs nest the report under "report"; approved jobs return it directly.
    return result.get("report", result)

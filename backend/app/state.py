"""
Application state.

`jobs` stays in memory (transient per-run job tracking). `sessions` is disk-backed:
ESCALATE_TO_HUMAN reviews must survive a backend restart so a reviewer can still
submit a correction after the server reloads (otherwise the session_id 404s and the
review — and its ground truth for GEPA — is lost). Sessions persist as JSON files
under data/sessions/ and are reloaded into memory on startup.
"""

import logging
from pathlib import Path

from fastapi import HTTPException

from pipeline.schema import SessionState

logger = logging.getLogger(__name__)

# job_id → {status, result?, error?, started_at?, ...}  (transient — fine to lose on restart)
jobs: dict[str, dict] = {}

_SESSIONS_DIR = Path("data/sessions")


class _SessionStore:
    """
    Dict-like store for SessionState that mirrors every write to disk and lazily
    loads from disk on read. Supports the access patterns used across the routers:
    `sid in store`, `store[sid]`, `store[sid] = s`, `store.get(sid)`, `del store[sid]`,
    and `store.values()`.
    """

    def __init__(self, directory: Path):
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, SessionState] = {}
        # Eager-load any sessions persisted by a previous run.
        loaded = 0
        for f in self._dir.glob("*.json"):
            try:
                self._mem[f.stem] = SessionState.model_validate_json(f.read_text())
                loaded += 1
            except Exception as e:
                logger.warning(f"Could not load persisted session {f.name}: {e}")
        if loaded:
            logger.info(f"Restored {loaded} pending session(s) from {self._dir}")

    def _path(self, sid: str) -> Path:
        return self._dir / f"{sid}.json"

    def __setitem__(self, sid: str, session: SessionState) -> None:
        self._mem[sid] = session
        try:
            self._path(sid).write_text(session.model_dump_json())
        except Exception as e:
            logger.warning(f"Could not persist session {sid}: {e}")

    def __getitem__(self, sid: str) -> SessionState:
        if sid not in self._mem:
            f = self._path(sid)
            if f.exists():
                self._mem[sid] = SessionState.model_validate_json(f.read_text())
        return self._mem[sid]

    def __contains__(self, sid: str) -> bool:
        return sid in self._mem or self._path(sid).exists()

    def __delitem__(self, sid: str) -> None:
        self._mem.pop(sid, None)
        self._path(sid).unlink(missing_ok=True)

    def get(self, sid: str, default=None):
        try:
            return self[sid]
        except KeyError:
            return default

    def values(self):
        return self._mem.values()

    def save(self, sid: str) -> None:
        """Re-persist a session after an in-place mutation (status, report, …)."""
        if sid in self._mem:
            self[sid] = self._mem[sid]


# session_id → SessionState (ESCALATE_TO_HUMAN reports awaiting review) — disk-backed.
sessions = _SessionStore(_SESSIONS_DIR)


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

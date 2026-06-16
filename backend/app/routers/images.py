"""
Image-serving endpoints — all images are read from the claim_images DB table.
No filesystem dependency. Every endpoint falls back gracefully:
  masked → annotated → original
  merged → annotated → original
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session as DBSession

from .. import state
from ..db import get_db
from ..models import ClaimImage

logger = logging.getLogger(__name__)
router = APIRouter(tags=["images"])


def _get_image(db: DBSession, job_id: str, image_type: str) -> ClaimImage | None:
    return (
        db.query(ClaimImage)
        .filter(ClaimImage.job_id == job_id, ClaimImage.image_type == image_type)
        .first()
    )


def _image_response(row: ClaimImage) -> Response:
    return Response(content=row.data, media_type=row.mime_type)


def _job_id_for_session(session_id: str) -> str:
    """Resolve a session_id to its originating job_id."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = state.sessions[session_id]
    if not session.job_id:
        raise HTTPException(status_code=404, detail="Session has no associated job")
    return session.job_id


def _require_job(job_id: str) -> None:
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")


# ── Session-scoped (HITL) ─────────────────────────────────────────────────────

@router.get("/session/{session_id}/plain_image")
async def get_plain_image(session_id: str, db: DBSession = Depends(get_db)):
    job_id = _job_id_for_session(session_id)
    row = _get_image(db, job_id, "original")
    if not row:
        raise HTTPException(status_code=404, detail="Original image not in database")
    return _image_response(row)


@router.get("/session/{session_id}/annotated_image")
async def get_annotated_image(session_id: str, db: DBSession = Depends(get_db)):
    job_id = _job_id_for_session(session_id)
    row = _get_image(db, job_id, "annotated") or _get_image(db, job_id, "original")
    if not row:
        raise HTTPException(status_code=404, detail="No annotated image in database")
    return _image_response(row)


@router.get("/session/{session_id}/masked_image")
async def get_masked_image(session_id: str, db: DBSession = Depends(get_db)):
    job_id = _job_id_for_session(session_id)
    row = (
        _get_image(db, job_id, "masked")
        or _get_image(db, job_id, "annotated")
        or _get_image(db, job_id, "original")
    )
    if not row:
        raise HTTPException(status_code=404, detail="No image in database for this session")
    return _image_response(row)


# ── Job-scoped ────────────────────────────────────────────────────────────────

@router.get("/job/{job_id}/plain_image")
async def get_job_plain_image(job_id: str, db: DBSession = Depends(get_db)):
    _require_job(job_id)
    row = _get_image(db, job_id, "original")
    if not row:
        raise HTTPException(status_code=404, detail="Original image not in database")
    return _image_response(row)


@router.get("/job/{job_id}/annotated_image")
async def get_job_annotated_image(job_id: str, db: DBSession = Depends(get_db)):
    _require_job(job_id)
    row = _get_image(db, job_id, "annotated") or _get_image(db, job_id, "original")
    if not row:
        raise HTTPException(status_code=404, detail="No annotated image in database")
    return _image_response(row)


@router.get("/job/{job_id}/masked_image")
async def get_job_masked_image(job_id: str, db: DBSession = Depends(get_db)):
    _require_job(job_id)
    row = (
        _get_image(db, job_id, "masked")
        or _get_image(db, job_id, "annotated")
        or _get_image(db, job_id, "original")
    )
    if not row:
        raise HTTPException(status_code=404, detail="No image in database for this job")
    return _image_response(row)


@router.get("/job/{job_id}/merged_image")
async def get_job_merged_image(job_id: str, db: DBSession = Depends(get_db)):
    _require_job(job_id)
    row = (
        _get_image(db, job_id, "merged")
        or _get_image(db, job_id, "annotated")
        or _get_image(db, job_id, "original")
    )
    if not row:
        raise HTTPException(status_code=404, detail="No merged image in database")
    return _image_response(row)

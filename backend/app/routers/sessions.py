"""
Human-in-the-loop (HITL) session endpoints for escalated reports.

  GET  /session/{id}                  — current session state
  POST /session/{id}/approve          — approve with corrections, finalise, log
  POST /session/{id}/update_detections — apply Step-2 bbox corrections in place
  POST /session/{id}/save_correction  — persist a full correction record + YOLO labels
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from pipeline.schema import (
    ApproveRequest,
    BBoxCorrectionRequest,
    CorrectionEntry,
    DamagePartEntry,
    FeedbackEntry,
    SaveCorrectionRequest,
)

from .. import state
from ..core import config
from ..services.cost import apply_cost_lookup
from ..services.feedback import write_feedback
from ..services.imaging import generate_annotated_image, write_yolo_labels

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/session", tags=["sessions"])


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Return the current state of a HITL session."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return state.sessions[session_id].model_dump()


@router.post("/{session_id}/approve")
async def approve_session(session_id: str, request: ApproveRequest):
    """Accept corrections, recalculate costs, finalise the report, write feedback."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    session = state.sessions[session_id]
    if session.status != "pending_review":
        raise HTTPException(
            status_code=409,
            detail=f"Session status is '{session.status}', expected 'pending_review'",
        )

    updated_entries, total_min, total_max = apply_cost_lookup(request.damage_part_map)

    session.status = "approved"
    session.corrected_map = updated_entries
    session.correction_notes = request.correction_notes
    state.sessions.save(session_id)   # persist the approved status across restarts

    write_feedback(
        FeedbackEntry(
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
    )

    finalised = session.report.model_dump()
    finalised.update(
        {
            "damage_part_map": [e.model_dump() for e in updated_entries],
            "total_min": total_min,
            "total_max": total_max,
            "approval_decision": "HUMAN_APPROVED",
        }
    )
    return finalised


@router.post("/{session_id}/update_detections")
async def update_detections(session_id: str, request: BBoxCorrectionRequest):
    """Apply Step-2 bbox corrections in place and regenerate the annotated image."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    report = state.sessions[session_id].report
    report.detections_with_bbox = request.corrected_detections

    updated_entries, total_min, total_max = apply_cost_lookup(
        [
            DamagePartEntry(
                damage=d.damage,
                part=d.part,
                severity=d.severity,
                cost_min=d.cost_min,
                cost_max=d.cost_max,
            )
            for d in request.corrected_detections
        ]
    )
    report.damage_part_map = updated_entries
    report.total_min = total_min
    report.total_max = total_max
    state.sessions.save(session_id)   # persist the edited report across restarts

    try:
        generate_annotated_image(report.image_path, request.corrected_detections)
    except Exception as e:
        logger.warning(f"Annotated image regeneration failed: {e}")

    return {
        "status": "updated",
        "total_min": total_min,
        "total_max": total_max,
        "detection_count": len(request.corrected_detections),
    }


@router.post("/{session_id}/save_correction")
async def save_correction(session_id: str, request: SaveCorrectionRequest):
    """Persist a full correction record (per-item diffs + bbox annotations + YOLO labels)."""
    from PIL import Image as PILImage

    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = state.sessions[session_id]
    report = session.report
    image_path = report.image_path

    img_w, img_h = 1920, 1080
    try:
        with PILImage.open(image_path) as img:
            img_w, img_h = img.size
    except Exception:
        pass

    actions = request.correction_actions
    items_kept = sum(1 for a in actions if a.action == "keep")
    items_edited = sum(1 for a in actions if a.action == "edit")
    items_removed = sum(1 for a in actions if a.action == "remove")
    items_added = sum(1 for a in actions if a.action == "add")

    has_notes = any(a.reason for a in actions if a.reason)
    has_bboxes = len(request.bbox_annotations) > 0
    quality_score = min(1.0, 0.5 + (0.2 if has_notes else 0) + (0.3 if has_bboxes else 0))

    _, final_total_min, final_total_max = apply_cost_lookup(request.final_damage_map)

    entry = CorrectionEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id=session_id,
        image_path=image_path,
        image_width=img_w,
        image_height=img_h,
        claim_id=session.claim_id,
        annotated_by=request.annotated_by,
        original_damage_map=report.damage_part_map,
        original_total_min=report.total_min,
        original_total_max=report.total_max,
        correction_actions=request.correction_actions,
        bbox_annotations=request.bbox_annotations,
        final_damage_map=request.final_damage_map,
        final_total_min=final_total_min,
        final_total_max=final_total_max,
        items_kept=items_kept,
        items_edited=items_edited,
        items_removed=items_removed,
        items_added=items_added,
        had_missed_damages=items_added > 0,
        had_false_positives=items_removed > 0,
        correction_quality_score=quality_score,
    )

    config.CORRECTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CORRECTIONS_LOG, "a") as f:
        f.write(entry.model_dump_json() + "\n")
    logger.info(f"Correction saved: session={session_id} quality={quality_score:.2f}")

    if request.bbox_annotations:
        write_yolo_labels(image_path, request.bbox_annotations, img_w, img_h)

    return {"status": "saved", "session_id": session_id, "quality_score": quality_score}

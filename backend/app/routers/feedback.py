"""
Cost recalculation and feedback endpoints.

  POST /recalculate     — recompute costs from COST_DB (no model inference)
  GET  /feedback/stats  — aggregated correction stats for the dashboard
  POST /api/feedback    — direct feedback-log write
"""

import json
import logging
from collections import Counter

from fastapi import APIRouter

from pipeline.schema import FeedbackEntry, RecalculateRequest, RecalculateResponse

from ..core import config
from ..services.cost import apply_cost_lookup
from ..services.feedback import write_feedback

logger = logging.getLogger(__name__)
router = APIRouter(tags=["feedback"])


@router.post("/recalculate", response_model=RecalculateResponse)
async def recalculate(request: RecalculateRequest):
    """Recompute cost_min/cost_max per entry using COST_DB — pure math, synchronous."""
    updated_entries, total_min, total_max = apply_cost_lookup(request.damage_part_map)
    return RecalculateResponse(
        damage_part_map=updated_entries,
        total_min=total_min,
        total_max=total_max,
    )


@router.get("/feedback/stats")
async def get_feedback_stats():
    """Aggregated stats from the corrections log. Used by the dashboard sidebar."""
    if not config.CORRECTIONS_LOG.exists():
        return {"total_corrections": 0}

    entries = []
    with open(config.CORRECTIONS_LOG) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                continue

    class_corrections: Counter = Counter()
    for e in entries:
        for action in e.get("correction_actions", []):
            if action.get("action") in ("edit", "add"):
                item = action.get("corrected") or action.get("original") or {}
                class_corrections[item.get("damage", "unknown")] += 1

    return {
        "total_corrections": len(entries),
        "total_missed_damages_found": sum(e.get("items_added", 0) for e in entries),
        "total_false_positives_removed": sum(e.get("items_removed", 0) for e in entries),
        "total_bbox_annotations": sum(len(e.get("bbox_annotations", [])) for e in entries),
        "corrections_by_damage_class": dict(class_corrections),
        "high_quality_corrections": sum(
            1 for e in entries if e.get("correction_quality_score", 0) >= 0.8
        ),
    }


@router.post("/api/feedback")
async def submit_feedback(entry: FeedbackEntry):
    """Direct feedback-log write for external or programmatic submissions."""
    write_feedback(entry)
    return {"status": "written"}

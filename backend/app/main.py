"""
backend/app/main.py

FastAPI application entry point.
Endpoints:
  POST /assess              — accept image, run pipeline, return report or session
  GET  /health              — VLM load status
  GET  /session/{id}        — retrieve session state
  POST /session/{id}/approve — human correction, finalise, write feedback log
  POST /recalculate         — recompute costs from COST_DB without re-running pipeline
  POST /api/feedback        — direct feedback log write
"""

import asyncio
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from pipeline.schema import (
    ApproveRequest,
    DamagePartEntry,
    FeedbackEntry,
    FinalDamageReport,
    RecalculateRequest,
    RecalculateResponse,
    SessionState,
    SaveCorrectionRequest,
    CorrectionEntry,
    DetectionWithBBox,
    BBoxCorrectionRequest,
)
import cv2
from models.vlm_reasoning.sandbox import COST_DB

# Configure logging before any other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── In-memory stores — MVP only, lost on restart ─────────────────────────────
_sessions: dict[str, SessionState] = {}
_jobs: dict[str, dict] = {}  # job_id → {status, result?, error?}

FEEDBACK_LOG = Path("data/feedback/feedback_log.jsonl")
CORRECTIONS_LOG = Path("data/feedback/corrections_log.jsonl")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path("configs/global_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at {config_path}. "
            "Run from repo root: uvicorn backend.app:app"
        )
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


def _generate_annotated_image(
    image_path: str,
    detections: list,
    output_dir: str = "data/uploads/annotated",
) -> str:
    """
    Draws numbered bounding boxes on image, colour-coded by damage class.
    Saves result as JPEG and returns output path.
    """
    CLASS_COLORS = {
        "dent":          (221, 138,  55),
        "scratch":       (117, 158,  29),
        "crack":         ( 23, 117, 186),
        "glass_shatter": (126,  83, 212),
        "lamp_broken":   ( 48,  90, 216),
        "tire_flat":     (128, 135, 136),
    }
    DEFAULT_COLOR = (128, 128, 128)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]

    for det in detections:
        if hasattr(det, "bbox"):
            bbox, cls, idx, conf, src = det.bbox, det.damage, det.index, det.confidence, det.source
        else:
            bbox = det.get("bbox", [0, 0, 0, 0])
            cls  = det.get("damage", "dent")
            idx  = det.get("index", 0)
            conf = det.get("confidence", 0.0)
            src  = det.get("source", "yolo")

        if all(v == 0.0 for v in bbox):
            continue

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        thickness = 3 if src == "human" else 2

        if conf < 0.5 and src != "human":
            corner_len = min(20, max(1, (x2 - x1) // 4), max(1, (y2 - y1) // 4))
            for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(img, (cx, cy), (cx + dx * corner_len, cy), color, 3)
                cv2.line(img, (cx, cy), (cx, cy + dy * corner_len), color, 3)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        badge_r = 14
        badge_x = min(x1 + badge_r + 2, w - badge_r - 2)
        badge_y = (y1 - badge_r - 2) if y1 > badge_r + 10 else (y1 + badge_r + 2)
        badge_y = max(badge_r + 2, min(h - badge_r - 2, badge_y))
        cv2.circle(img, (badge_x, badge_y), badge_r, color, -1)
        cv2.circle(img, (badge_x, badge_y), badge_r, (255, 255, 255), 1)
        label = str(idx)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 if idx < 10 else 0.4
        text_size = cv2.getTextSize(label, font, font_scale, 2)[0]
        cv2.putText(img, label,
                    (badge_x - text_size[0] // 2, badge_y + text_size[1] // 2),
                    font, font_scale, (255, 255, 255), 2)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{Path(image_path).stem}_annotated.jpg")
    cv2.imwrite(out_path, img)
    return out_path


def _write_yolo_labels(
    image_path: str,
    bbox_annotations: list,
    img_w: int,
    img_h: int,
) -> None:
    """Writes YOLO-format label file and copies image for fine-tune dataset."""
    YOLO_CLASS_MAP = {
        "dent": 0, "scratch": 1, "crack": 2,
        "glass_shatter": 3, "lamp_broken": 4, "tire_flat": 5,
    }
    src = Path(image_path)
    if not src.exists():
        return
    img_dir = Path("data/yolo_corrections/images")
    lbl_dir = Path("data/yolo_corrections/labels")
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, img_dir / src.name)
    lines = []
    for ann in bbox_annotations:
        if hasattr(ann, "damage_class"):
            cls_name = ann.damage_class
            x1, y1, x2, y2 = ann.x1, ann.y1, ann.x2, ann.y2
        else:
            cls_name = ann.get("damage_class", "dent")
            x1 = ann.get("x1", 0)
            y1 = ann.get("y1", 0)
            x2 = ann.get("x2", 0)
            y2 = ann.get("y2", 0)
        cls_id = YOLO_CLASS_MAP.get(cls_name, 0)
        lines.append(
            f"{cls_id} {((x1+x2)/2)/img_w:.6f} {((y1+y2)/2)/img_h:.6f} "
            f"{(x2-x1)/img_w:.6f} {(y2-y1)/img_h:.6f}"
        )
    (lbl_dir / (src.stem + ".txt")).write_text("\n".join(lines))


def _write_feedback(entry: FeedbackEntry) -> None:
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG, "a") as f:
        f.write(entry.model_dump_json() + "\n")
    logger.info(f"Feedback written: session={entry.session_id}")


async def _run_assessment_job(
    job_id: str,
    save_path: Path,
    config: dict,
    claim_id: str | None,
    vehicle_id: str | None,
) -> None:
    """
    Runs pipeline/orchestrator.run() in a threadpool so the event loop stays
    unblocked. Updates _jobs[job_id] when done or on any exception.
    All exceptions are caught — the job will never stay "processing" forever.
    """
    JOB_TIMEOUT_S = 360  # 6 min — 40s buffer before the 600s wall

    try:
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

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
            session = SessionState(
                session_id=session_id,
                status="pending_review",
                report=FinalDamageReport(**report),
                created_at=datetime.now(timezone.utc).isoformat(),
                claim_id=claim_id or None,
            )
            _sessions[session_id] = session
            logger.info(f"Session created: {session_id} (ESCALATE_TO_HUMAN)")
            _jobs[job_id]["status"] = "complete"
            _jobs[job_id]["result"] = {"session_id": session_id, "report": report, "status": "pending_review"}
        else:
            _jobs[job_id]["status"] = "complete"
            _jobs[job_id]["result"] = report

        _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(f"Job {job_id} completed successfully")

    except asyncio.TimeoutError:
        logger.error(f"Job {job_id} hard timeout after {JOB_TIMEOUT_S}s")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = (
            f"Job timed out after {JOB_TIMEOUT_S}s. "
            "VLM did not complete within time budget. "
            "YOLO detections may still be available — check warnings."
        )
        import gc as _gc
        _gc.collect()
        if _torch.backends.mps.is_available():
            _torch.mps.empty_cache()
    except MemoryError as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = "Out of memory during inference. Try a smaller image or restart the server."
        logger.error(f"Job {job_id} failed with MemoryError: {e}")
    except RuntimeError as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
        logger.error(f"Job {job_id} failed with RuntimeError: {e}")
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Job {job_id} failed with unexpected error: {e}", exc_info=True)


# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm VLM as background task — server accepts requests immediately."""
    async def _prewarm():
        try:
            config = _load_config()
            from pipeline.orchestrator import _load_models
            logger.info("Pre-warming VLM on startup...")
            await run_in_threadpool(_load_models, config)
            logger.info("VLM pre-warm complete")
        except Exception as e:
            logger.warning(
                f"VLM pre-warm failed: {e}. Model will load on first request."
            )

    asyncio.create_task(_prewarm())
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
        "vlm_loaded": loaded,
    }


@app.post("/assess")
async def assess_damage(
    image: UploadFile = File(...),
    claim_id: str = Form(default=None),
    vehicle_id: str = Form(default=None),
):
    """
    Accept a vehicle image, save it, and immediately return a job_id.
    Pipeline runs asynchronously in a threadpool — poll GET /job/{job_id} for result.

    Returns: {job_id: str, status: "processing"}
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

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    asyncio.create_task(
        _run_assessment_job(job_id, save_path, config, claim_id, vehicle_id)
    )
    logger.info(f"Job queued: {job_id}")
    return {"job_id": job_id, "status": "processing"}


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Poll assessment job status.

    Returns one of:
      {status: "queued" | "processing", elapsed_s: int}
      {status: "complete", result: <FinalDamageReport or HITL session dict>}
      {status: "failed", error: str}
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs[job_id]

    # Auto-timeout jobs stuck in processing
    if job["status"] == "processing" and "started_at" in job:
        started = datetime.fromisoformat(job["started_at"])
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        job["elapsed_s"] = round(elapsed)

        if elapsed > 600:
            job["status"] = "failed"
            job["error"] = (
                "Job timed out after 600 seconds. "
                "The VLM inference exceeded the maximum allowed time. "
                "This usually means the model ran out of memory silently. "
                "Restart the server and try again."
            )
            logger.warning(f"Job {job_id} auto-timed-out after {elapsed:.0f}s")

    return job


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


@app.get("/session/{session_id}/plain_image")
async def get_plain_image(session_id: str):
    """Serves the original uploaded image for use as canvas background."""
    from PIL import Image as PILImage

    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    image_path = _sessions[session_id].report.image_path
    if not Path(image_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    headers = {}
    try:
        with PILImage.open(image_path) as img:
            w, h = img.size
            headers = {"X-Image-Width": str(w), "X-Image-Height": str(h)}
    except Exception:
        pass

    return FileResponse(image_path, media_type="image/jpeg", headers=headers)


@app.get("/session/{session_id}/masked_image")
async def get_masked_image(session_id: str):
    """Returns SAM2 mask overlays. Returns 503 if weights missing. Falls back to annotated image."""
    from shared.sam_mask import generate_masked_image, _sam_failed

    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    report  = session.report

    if _sam_failed:
        raise HTTPException(
            status_code=503,
            detail=(
                "SAM2 weights not available. "
                "Run: python3 scripts/download_sam2_weights.py"
            ),
        )

    detections = report.detections_with_bbox
    if not detections:
        ann = getattr(report, "annotated_image_path", None)
        if ann and Path(ann).exists():
            return FileResponse(ann, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail="No detections available for masking")

    try:
        cfg = _load_config()
        weights = cfg.get("part_segmentation", {}).get("sam2", {}).get(
            "weights_path", "weights/sam2.1_hiera_base_plus.pt"
        )
        out_path = generate_masked_image(
            image_path   = report.image_path,
            detections   = detections,
            weights_path = weights,
        )
        return FileResponse(out_path, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Masked image generation failed: {e}")
        ann = getattr(report, "annotated_image_path", None)
        if ann and Path(ann).exists():
            return FileResponse(ann, media_type="image/jpeg")
        if Path(report.image_path).exists():
            return FileResponse(report.image_path, media_type="image/jpeg")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job/{job_id}/plain_image")
async def get_job_plain_image(job_id: str):
    """Returns original uploaded image for canvas background."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    result = _jobs[job_id].get("result", {})
    image_path = result.get("image_path") or result.get("report", {}).get("image_path", "")
    if image_path and Path(image_path).exists():
        return FileResponse(image_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/job/{job_id}/masked_image")
async def get_job_masked_image(job_id: str):
    """Returns SAM2 masks for any completed job. Falls back to annotated image."""
    from shared.sam_mask import generate_masked_image, _sam_failed

    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    if job.get("status") != "complete":
        raise HTTPException(status_code=400, detail="Job not complete")

    result = job.get("result", {})
    report = result.get("report", result)
    image_path = report.get("image_path", "")
    annotated  = report.get("annotated_image_path", "")
    raw_dets   = report.get("detections_with_bbox", [])

    if _sam_failed or not raw_dets:
        if annotated and Path(annotated).exists():
            return FileResponse(annotated, media_type="image/jpeg")
        if image_path and Path(image_path).exists():
            return FileResponse(image_path, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail="No image available")

    try:
        det_objects = []
        for d in raw_dets:
            if isinstance(d, dict):
                det_objects.append(DetectionWithBBox(**d))
            else:
                det_objects.append(d)

        cfg = _load_config()
        weights = cfg.get("part_segmentation", {}).get("sam2", {}).get(
            "weights_path", "weights/sam2.1_hiera_base_plus.pt"
        )
        out = generate_masked_image(
            image_path   = image_path,
            detections   = det_objects,
            weights_path = weights,
        )
        return FileResponse(out, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Job masked image failed: {e}")
        if annotated and Path(annotated).exists():
            return FileResponse(annotated, media_type="image/jpeg")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/annotated_image")
async def get_annotated_image(session_id: str):
    """Returns annotated image with numbered bounding boxes for Step 2 correction UI."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    report  = session.report

    detections = report.detections_with_bbox
    plain_path = report.image_path

    if not detections:
        if Path(plain_path).exists():
            return FileResponse(plain_path, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail="No detections with bbox and no image")

    try:
        annotated_path = _generate_annotated_image(plain_path, detections)
        return FileResponse(annotated_path, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Annotated image generation failed: {e}")
        if Path(plain_path).exists():
            return FileResponse(plain_path, media_type="image/jpeg")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job/{job_id}/annotated_image")
async def get_job_annotated_image(job_id: str):
    """Serves the YOLO-annotated image for any completed job (approved or escalated)."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    if job.get("status") != "complete":
        raise HTTPException(status_code=400, detail="Job not complete yet")

    result = job.get("result", {})
    # Escalated jobs: result["report"]. Approved jobs: result directly.
    report = result.get("report", result)

    annotated = report.get("annotated_image_path")
    if annotated and Path(annotated).exists():
        return FileResponse(annotated, media_type="image/jpeg")

    image_path = report.get("image_path", "")
    if image_path:
        stem = Path(image_path).stem
        ann_dir = Path("data/uploads/yolo_annotated")
        if ann_dir.exists():
            candidates = list(ann_dir.glob(f"{stem}_yolo_*.jpg"))
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                return FileResponse(str(latest), media_type="image/jpeg")

    if image_path and Path(image_path).exists():
        return FileResponse(image_path, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="No annotated image available")


@app.post("/session/{session_id}/update_detections")
async def update_detections(session_id: str, request: BBoxCorrectionRequest):
    """
    Called when intern finishes Step 2 bbox correction.
    Updates session in-place with corrected detections and regenerates annotated image.
    Does NOT save to corrections_log — that happens on final approve.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    report  = session.report

    report.detections_with_bbox = request.corrected_detections

    updated_entries, total_min, total_max = _apply_cost_lookup([
        DamagePartEntry(
            damage=d.damage,
            part=d.part,
            severity=d.severity,
            cost_min=d.cost_min,
            cost_max=d.cost_max,
        )
        for d in request.corrected_detections
    ])
    report.damage_part_map = updated_entries
    report.total_min = total_min
    report.total_max = total_max

    try:
        _generate_annotated_image(report.image_path, request.corrected_detections)
    except Exception as e:
        logger.warning(f"Annotated image regeneration failed: {e}")

    return {
        "status": "updated",
        "total_min": total_min,
        "total_max": total_max,
        "detection_count": len(request.corrected_detections),
    }


@app.post("/session/{session_id}/save_correction")
async def save_correction(session_id: str, request: SaveCorrectionRequest):
    """
    Saves a full correction record including per-item diffs and bbox annotations.
    Also writes YOLO label files for any bbox_annotations provided.
    Called by dashboard before final approval.
    """
    from PIL import Image as PILImage

    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    report = session.report
    image_path = report.image_path

    img_w, img_h = 1920, 1080
    try:
        with PILImage.open(image_path) as img:
            img_w, img_h = img.size
    except Exception:
        pass

    actions = request.correction_actions
    items_kept    = sum(1 for a in actions if a.action == "keep")
    items_edited  = sum(1 for a in actions if a.action == "edit")
    items_removed = sum(1 for a in actions if a.action == "remove")
    items_added   = sum(1 for a in actions if a.action == "add")

    has_notes  = any(a.reason for a in actions if a.reason)
    has_bboxes = len(request.bbox_annotations) > 0
    quality_score = min(1.0, 0.5 + (0.2 if has_notes else 0) + (0.3 if has_bboxes else 0))

    _, final_total_min, final_total_max = _apply_cost_lookup(request.final_damage_map)

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

    CORRECTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CORRECTIONS_LOG, "a") as f:
        f.write(entry.model_dump_json() + "\n")
    logger.info(f"Correction saved: session={session_id} quality={quality_score:.2f}")

    if request.bbox_annotations:
        _write_yolo_labels(image_path, request.bbox_annotations, img_w, img_h)

    return {"status": "saved", "session_id": session_id, "quality_score": quality_score}


@app.get("/feedback/stats")
async def get_feedback_stats():
    """Aggregated stats from corrections_log.jsonl. Used by dashboard sidebar."""
    import json as _json
    from collections import Counter

    if not CORRECTIONS_LOG.exists():
        return {"total_corrections": 0}

    entries = []
    with open(CORRECTIONS_LOG) as f:
        for line in f:
            try:
                entries.append(_json.loads(line))
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

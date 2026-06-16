#!/usr/bin/env python3
"""
Pydantic Schema Validation Contract
Defines and validates relational models for damages, segmentations, and claims.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field

class DamageItem(BaseModel):
    """
    Structured visual damage item returned by models and VLMs.
    """
    part: str = Field(..., description="Affected vehicle part name")
    damage_type: str = Field(..., description="Type of damage detected (e.g. dent, scratch)")
    severity: str = Field("Moderate", description="Severity level: Pristine, Minor, Moderate, Severe")
    supporting_images: List[str] = Field(default_factory=list, description="Associated view filenames")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Model prediction confidence score")
    reasoning: Optional[str] = Field(None, description="Visual rationale justifying severity")

class ClaimAnalysisSchema(BaseModel):
    """
    Executive verified vehicle damage insurance claim report.
    """
    claim_id: str = Field(..., description="Unique alphanumeric claims identifier")
    overall_summary: str = Field(..., description="High-level claims summary")
    overall_severity: str = Field(..., description="Pristine | Minor | Moderate | Severe")
    damages: List[DamageItem] = Field(default_factory=list, description="Visual detections list")
    view_consistency_notes: Optional[str] = Field(None, description="Spatial correlation notes")
    cost_estimation: Optional[dict] = Field(None, description="Detailed repair and refinish cost breakdown sheet")


# ── Appended by instruct_follow.md ──────────────────────────────────────────

from typing import Dict  # noqa: F811 — List and Optional already imported above


class ToolCallRecord(BaseModel):
    """Records a single tool invocation by the VLM during orchestration."""
    tool: str
    args_summary: str          # short string, not full args — avoid logging image paths twice
    elapsed_s: float
    result_keys: List[str]
    success: bool


class DamagePartEntry(BaseModel):
    """A single damage-to-part mapping with cost estimate."""
    damage: str                # dent | scratch | crack | glass_shatter | lamp_broken | tire_flat
    part: str                  # see CLAUDE.md for valid part label list
    severity: str              # minor | moderate | severe
    cost_min: int              # INR
    cost_max: int              # INR


class FinalDamageReport(BaseModel):
    """Top-level output of pipeline/orchestrator.py::run()."""
    image_path: str
    damage_part_map: List[DamagePartEntry]
    detections_with_bbox: List["DetectionWithBBox"] = Field(default_factory=list)
    # Merged union (VLM ∪ SAM2) findings as source-tagged boxes for the UI.
    merged_detections: List[Dict[str, Any]] = Field(default_factory=list)
    total_min: int
    total_max: int
    currency: str = "INR"
    approval_decision: str             # AUTO_APPROVED | ESCALATE_TO_HUMAN | UNKNOWN
    tool_call_log: List[ToolCallRecord]
    # Per-tool-call iteration log for the UI: {turn, tool, reason, summary, elapsed_s, ok}.
    iterations: List[Dict[str, Any]] = Field(default_factory=list)
    total_inference_s: float
    warnings: List[str]
    raw_vlm_response: Optional[str]    # last assistant message — keep for MVP debugging
    annotated_image_path: Optional[str] = None
    merged_image_path: Optional[str] = None
    masked_image_path: Optional[str] = None    # pre-generated SAM2 mask overlay


# ── HITL Layer ────────────────────────────────────────────────────────────────

class SessionState(BaseModel):
    """In-memory session for ESCALATE_TO_HUMAN reports awaiting human review."""
    session_id: str
    status: str                              # pending_review | approved | rejected
    report: FinalDamageReport
    created_at: str                          # ISO8601
    claim_id: Optional[str] = None
    job_id: Optional[str] = None            # links to claim_images rows in DB
    corrected_map: Optional[List[DamagePartEntry]] = None
    correction_notes: Optional[str] = None


class RecalculateRequest(BaseModel):
    damage_part_map: List[DamagePartEntry]


class RecalculateResponse(BaseModel):
    damage_part_map: List[DamagePartEntry]
    total_min: int
    total_max: int
    currency: str = "INR"


class ApproveRequest(BaseModel):
    damage_part_map: List[DamagePartEntry]
    correction_notes: Optional[str] = None


class FeedbackEntry(BaseModel):
    """One approved HITL correction written to feedback_log.jsonl."""
    timestamp: str
    session_id: str
    image_path: str
    claim_id: Optional[str]
    original_report: FinalDamageReport
    human_corrections: dict
    final_total_min: int
    final_total_max: int
    approval_decision: str = "HUMAN_APPROVED"


# ── Correction and Feedback Loop Schemas ──────────────────────────────────────

class CorrectionAction(BaseModel):
    """Records a single correction action made by an intern on one damage entry."""
    action: str                                 # "keep" | "edit" | "remove" | "add"
    original: Optional[DamagePartEntry] = None  # None for "add" actions
    corrected: Optional[DamagePartEntry] = None # None for "remove" actions
    reason: Optional[str] = None


class BBoxAnnotation(BaseModel):
    """
    Bounding box drawn by intern for a missed damage.
    Pixel coordinates relative to original image dimensions.
    Used to generate YOLO training labels.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    damage_class: str    # dent|scratch|crack|glass_shatter|lamp_broken|tire_flat
    part: str
    severity: str
    annotated_by: Optional[str] = None


class CorrectionEntry(BaseModel):
    """
    Full correction record for one pipeline output.
    Written to corrections_log.jsonl (separate from feedback_log.jsonl).
    Used as source for VLM few-shot injection and YOLO fine-tune data.
    """
    timestamp: str
    session_id: str
    image_path: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    claim_id: Optional[str] = None
    annotated_by: Optional[str] = None

    original_damage_map: List[DamagePartEntry]
    original_total_min: int
    original_total_max: int

    correction_actions: List[CorrectionAction]
    bbox_annotations: List[BBoxAnnotation] = Field(default_factory=list)

    final_damage_map: List[DamagePartEntry]
    final_total_min: int
    final_total_max: int

    items_kept: int = 0
    items_edited: int = 0
    items_removed: int = 0
    items_added: int = 0
    had_missed_damages: bool = False
    had_false_positives: bool = False
    correction_quality_score: float = 1.0


class SaveCorrectionRequest(BaseModel):
    """Request body for POST /session/{session_id}/save_correction"""
    correction_actions: List[CorrectionAction]
    bbox_annotations: List[BBoxAnnotation] = Field(default_factory=list)
    final_damage_map: List[DamagePartEntry]
    annotated_by: Optional[str] = None
    notes: Optional[str] = None


class FewShotExample(BaseModel):
    """A single formatted example for VLM prompt injection."""
    image_description: str
    original_output: List[dict]
    corrected_output: List[dict]
    correction_notes: str
    quality_score: float


class DetectionWithBBox(BaseModel):
    """
    A damage detection entry with full spatial information.
    Used for annotation UI — preserves bbox coords discarded by DamagePartEntry.
    """
    index: int                  # display number shown on annotated image (1-based)
    bbox: List[float]           # [x1, y1, x2, y2] in original image pixel coords
    damage: str
    part: str
    severity: str
    confidence: float           # 0.0–1.0 (0.0 for human-added boxes)
    # source provenance after the untrained-models + VLM-brain rework:
    #   "vlm"  → VLM-only visual claim   "both" → VLM claim grounded in a model region
    #   "model"→ model region with no VLM damage   "human" → dashboard edit
    #   (legacy: "yolo" | "vlm_visual" | "vlm_only" | "vlm_verified")
    source: str = "vlm"
    cost_min: int = 0
    cost_max: int = 0
    # ── Grounded-union evidence tags (untrained-models + VLM-brain design) ──
    grounded: bool = False      # True if this damage falls inside a model-proposed region/ROI
    needs_review: bool = False  # True for ungrounded VLM claims or unconfirmed model regions
    anomaly_score: float = 0.0  # DINOv2 region-vs-vehicle distance (0.0 if unavailable)


class RegionEvidence(BaseModel):
    """
    A spatial region proposed by the untrained CV models (vehicle ROI from stock
    YOLO, or a SAM2+DINOv2 segment). Carries NO damage label — it exists purely to
    ground/anchor the VLM's visual damage claims. The VLM names what (if anything)
    is damaged inside it.
    """
    bbox: List[float]                       # [x1, y1, x2, y2] original-image pixels
    source: str                             # "vehicle_roi" | "sam2_dinov2"
    area_px: int = 0
    anomaly_score: float = 0.0              # DINOv2 distance from vehicle-mean embedding
    cls: Optional[str] = None               # COCO class for vehicle_roi (car/truck/...); None for segments
    confidence: float = 0.0                 # detector confidence (vehicle_roi only)


class BBoxCorrectionRequest(BaseModel):
    """Submitted from dashboard Step 2. Full corrected detection list with bbox coords."""
    corrected_detections: List[DetectionWithBBox]
    intern_name: Optional[str] = None


# ── CodeAct schemas ───────────────────────────────────────────────────────────

class CodeActAction(BaseModel):
    """A single action in a CodeAct turn."""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class CodeActTurn(BaseModel):
    """
    One VLM output in the CodeAct loop.
    Must be valid JSON matching this schema — rejected and retried if not.
    """
    thought: str
    uncertainty: List[str] = Field(default_factory=list)
    actions: List[CodeActAction] = Field(default_factory=list)
    confidence: Optional[float] = None


class TrajectoryStep(BaseModel):
    """One (action, observation) pair in a trajectory."""
    turn_index: int
    action: CodeActAction
    observation_type: str               # "image" | "json" | "error"
    observation_summary: str
    observation_image_path: Optional[str] = None
    observation_data: Optional[Dict] = None
    elapsed_s: float


class Trajectory(BaseModel):
    """
    Complete tool-use trajectory for one image.
    Used as training data for SFT.
    """
    trajectory_id: str
    image_path: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    created_at: str
    model_id: str
    steps: List[TrajectoryStep]
    final_damage_map: List[DamagePartEntry]
    total_min: int
    total_max: int
    total_elapsed_s: float
    filter_status: str = "unfiltered"
    filter_reason: Optional[str] = None
    quality_score: float = 0.0


# Forward reference resolution — DetectionWithBBox is defined after FinalDamageReport
# which references it via a string annotation. model_rebuild() resolves the ref.
FinalDamageReport.model_rebuild()

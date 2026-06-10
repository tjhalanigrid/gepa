"""
pipeline/orchestrator.py

Central entry point for the Vehicle Damage Assessment pipeline.

Architecture (VLM-only, Ollama backend):
  - PiAgent drives a recursive CodeAct loop using qwen3.5:9b via Ollama
  - The VLM is the sole perception model: it sees images and calls tools
  - Tools: run_damage_detection, zoom_region, detect_part, segment_damage,
           estimate_depth, execute_cost_computation, Terminate
  - All VLM calls are HTTP POST to localhost:11434 (Ollama)
  - Cost computation runs in the Monty sandbox (sandbox.py)
  - No YOLO / SAM2 / GroundingDINO in the reasoning hot-path
  - YOLO kept as a silent safety-net fallback only

This module owns:
  - Ollama health-check gate
  - Delegating Stage 1 to PiAgent (models/vlm_reasoning/pi_agent.py)
  - Auto-approve threshold gate
  - FinalDamageReport construction
  - Trajectory saving
"""

import gc
import json
import logging
import re
import signal
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import yaml
from PIL import Image

from pipeline.schema import FinalDamageReport, DamagePartEntry, ToolCallRecord, DetectionWithBBox

logger = logging.getLogger(__name__)


class VLMGenerationTimeout(Exception):
    pass


@contextmanager
def _vlm_timeout(seconds: int):
    # signal.SIGALRM cannot be used in FastAPI's threadpool (non-main thread).
    # Per-call timeout is enforced at the job level (600s) and loop level (540s).
    yield


# ── Ollama health gate ───────────────────────────────────────────────────────
# _model / _processor removed — model is served by Ollama, not loaded in-process.
# _load_models() is kept for API compatibility (called by backend/app.py on startup)
# but now only performs an Ollama health check.

# ── System prompt ────────────────────────────────────────────────────────────
# Update in CLAUDE.md when this changes so changes are tracked.

SYSTEM_PROMPT = """You are an expert vehicle damage assessment AI.

The computer vision pipeline has already run on this image. The user message contains:
  • DAMAGE_DETECTIONS — bounding boxes and classes from YOLOv8 (may be empty)
  • PART_SEGMENTS — vehicle part labels and bounding boxes from Grounding DINO + SAM2 (may be empty)

YOUR JOB:
1. Read DAMAGE_DETECTIONS. For each detection, note its bbox [x1,y1,x2,y2] and class.
2. Read PART_SEGMENTS. For each part, note its bbox and part_label.
3. Cross-reference: for each damage bbox, find the part whose bbox overlaps it most.
   Use numeric overlap — check if damage center (cx,cy) falls inside part bbox, or use IoU.
4. CRITICAL — If DAMAGE_DETECTIONS is empty or has fewer than 2 entries:
   You MUST perform independent visual damage assessment from the image itself.
   Do not rely solely on CV pipeline output. Examine the image carefully for:
   - Deformation, crumpling, or crushing of body panels
   - Paint damage: scratches, chips, deep gouges
   - Cracks or shattering in glass components
   - Broken or displaced lamp assemblies
   - Flat or visibly damaged tires
   If you can visually confirm damage that the CV pipeline missed, include it.
   You are the safety net — CV pipeline misses are expected for complex damage angles.
5. If PART_SEGMENTS is empty, assign parts using your visual assessment of the image.
6. Assign severity: minor=surface only, moderate=panel needs repair, severe=structural/replacement.
7. Call execute_cost_computation with Python code that builds the cost estimate.
   Your code must assign a dict to `result` with keys:
     damage_part_map: list of {damage, part, severity, cost_min, cost_max}
     total_min: int  (INR)
     total_max: int  (INR)
     currency: "INR"
   Your code has access to COST_DB[damage_class][part_label] = (cost_min, cost_max).
   For any (damage, part) pair not in COST_DB, use (3000, 8000) as fallback.
   NEVER submit an empty damage_part_map. If you see visible damage, report it.
8. After execute_cost_computation returns, produce your final JSON report.
  
FINAL REPORT — wrap in ```json ... ```:
{
  "damage_part_map": [{"damage": str, "part": str, "severity": str, "cost_min": int, "cost_max": int}],
  "total_min": int,
  "total_max": int,
  "currency": "INR",
  "warnings": ["note any low-confidence detections or assumptions made"]
}

Valid damage classes: dent, scratch, crack, glass_shatter, lamp_broken, tire_flat
Valid parts: front_bumper, rear_bumper, hood, windshield, rear_windshield,
  front_left_door, front_right_door, rear_left_door, rear_right_door,
  left_fender, right_fender, trunk_lid, roof_panel, left_headlight, right_headlight, tire

IMPORTANT: dent/scratch/crack detections below confidence 0.4 are unreliable.
Include them if you visually confirm the damage, with a warning note."""


VISION_ASSESSMENT_PROMPT = """Examine this vehicle image carefully.

Identify every visible damage. Be thorough — do not skip damage because it is
severe, complex, or overlapping. Crushed panels, structural deformation, and
multi-area damage must all be reported.

For each damage item produce exactly this JSON structure:
{"damage_type": "...", "part": "...", "severity": "...", "confidence": 0.0}

damage_type must be one of: dent, scratch, crack, glass_shatter, lamp_broken, tire_flat
part must be one of: front_bumper, rear_bumper, hood, windshield, rear_windshield,
  front_left_door, front_right_door, rear_left_door, rear_right_door,
  left_fender, right_fender, trunk_lid, roof_panel, headlight, taillight, tire
severity must be one of: minor, moderate, severe
confidence: your visual certainty from 0.0 to 1.0

Severity guide:
  minor = surface only, paintwork affected, no panel deformation
  moderate = panel deformation, needs body shop repair
  severe = structural damage, panel replacement required, safety impact

Respond with ONLY a valid JSON object. No markdown. No explanation. No preamble.
{
  "damage_items": [
    {"damage_type": "dent", "part": "front_bumper", "severity": "severe", "confidence": 0.92}
  ]
}

If no damage is visible respond with: {"damage_items": []}"""


CODEACT_SYSTEM_PROMPT = """You are a vehicle damage assessment agent that thinks with images.

You are given the raw vehicle photo first. You decide which computer vision tools
to call, you look at the images they return, and you reason step by step until you
can produce a final, costed damage assessment.

═══════════════════════════════════════════════════
YOUR OUTPUT FORMAT — MANDATORY, NO EXCEPTIONS
═══════════════════════════════════════════════════
Every single response MUST be exactly one JSON object with this structure.
Do NOT output anything else. No explanations outside "thought". No markdown.

{
  "thought": "what you see and what you will do next",
  "uncertainty": ["open questions, empty list if none"],
  "actions": [
    {"name": "TOOL_NAME", "arguments": {...}}
  ],
  "confidence": 0.0
}
═══════════════════════════════════════════════════

TOOLS YOU CAN CALL (one or more per turn):
  • run_damage_detection — runs trained YOLOv8, returns an annotated image with
        bounding boxes + a structured detection list. Call this early to locate
        damage. arguments: {"confidence_threshold": 0.15, "reason": "..."}
  • zoom_region — crop + magnify a region to inspect it closely.
        arguments: {"bbox": [x1,y1,x2,y2], "reason": "..."}
  • detect_part — highlight a specific vehicle part you are unsure about.
        arguments: {"part_query": "left headlight", "reason": "..."}
  • segment_damage — precise SAM2 mask over a damage region to judge severity.
        arguments: {"bbox": [x1,y1,x2,y2], "reason": "..."}
  • execute_cost_computation — run Python in a sandbox to compute repair cost.
        Your code has COST_DB[damage_class][part_label] = (cost_min, cost_max).
        For pairs not in COST_DB use (3000, 8000). You MUST write real Python that
        reads COST_DB and assigns a dict to `result`. Do NOT paste a price table.
        `result` keys: damage_part_map (list of {damage, part, severity, cost_min,
        cost_max}), total_min (int), total_max (int), currency ("INR").
        Copy this template and edit only the `items` list:
            items = [{"damage": "dent", "part": "front bumper", "severity": "severe"}]
            result = {"damage_part_map": [], "total_min": 0, "total_max": 0, "currency": "INR"}
            for d in items:
                lo, hi = COST_DB.get(d["damage"], {}).get(d["part"], (3000, 8000))
                result["damage_part_map"].append({**d, "cost_min": lo, "cost_max": hi})
                result["total_min"] += lo
                result["total_max"] += hi
        arguments: {"code": "<the python above, as a single string>"}
  • Terminate — finish and return the final damage list.
        arguments: {"damage_items": [{damage_type, part, severity, confidence}]}

RECOMMENDED WORKFLOW (THINKING WITH IMAGES):
1. Examine the raw image. Form an initial hypothesis in "thought".
2. Call run_damage_detection to locate damage precisely. Read the returned image
   and detection list.
3. For any region you cannot classify confidently (confidence < 0.50) or whose
   severity is unclear, call zoom_region or segment_damage and look again.
4. If you are unsure which part is affected, call detect_part.
5. YOLO is weak on dent/scratch/crack — if you visually see damage YOLO missed,
   include it anyway and note it in "uncertainty".
6. Once you know every (damage, part, severity), call execute_cost_computation to
   price it. Read the returned cost result.
7. Call Terminate with your final verified damage_items.

TERMINATION FORMAT:
{
  "thought": "Assessment complete. Costs computed.",
  "uncertainty": [],
  "actions": [{"name": "Terminate", "arguments": {"damage_items": [
    {"damage_type": "dent", "part": "front_bumper", "severity": "severe", "confidence": 0.91}
  ]}}],
  "confidence": 0.91
}

Valid damage_type: dent scratch crack glass_shatter lamp_broken tire_flat
Valid part: front_bumper rear_bumper hood windshield rear_windshield
  front_left_door front_right_door rear_left_door rear_right_door
  left_fender right_fender trunk_lid roof_panel headlight taillight tire
Valid severity: minor moderate severe

Be conservative with severity. Only call tools you actually need.
NEVER output anything outside the single JSON object. No markdown. No preamble."""

CODEACT_TOOL_DEFINITIONS = [
    {
        "name": "run_damage_detection",
        "description": (
            "Run the trained YOLOv8 damage detection model on the vehicle image. "
            "Returns an annotated image with bounding boxes drawn over all detected "
            "damage regions, colour-coded by class. Also returns the raw detection "
            "list as structured data. "
            "Call this first to get an initial structured view of damage locations "
            "before using zoom_region or segment_damage to inspect uncertain areas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "confidence_threshold": {
                    "type": "number",
                    "description": (
                        "Minimum confidence score to include a detection (0.0-1.0). "
                        "Use 0.15 to cast wide and catch weak detections. "
                        "Use 0.40 for high-confidence only. Default: 0.15"
                    )
                },
                "reason": {
                    "type": "string",
                    "description": "Why you are calling the detector at this point"
                }
            },
            "required": ["reason"]
        }
    },
    {
        "name": "zoom_region",
        "description": "Zoom into a specific region for closer inspection. Call when a damage region is too small or unclear to classify confidently. Returns a cropped magnified sub-image.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x1, y1, x2, y2] pixel coordinates of region to zoom"
                },
                "reason": {"type": "string", "description": "What you cannot determine from the full image"}
            },
            "required": ["bbox", "reason"]
        }
    },
    {
        "name": "detect_part",
        "description": "Detect and highlight a specific vehicle part. Call when you cannot confidently identify which part is damaged. Returns image with detected part annotated.",
        "input_schema": {
            "type": "object",
            "properties": {
                "part_query": {"type": "string", "description": "Natural language part description, e.g. 'left headlight'"},
                "reason": {"type": "string", "description": "Why you are uncertain about the part location"}
            },
            "required": ["part_query", "reason"]
        }
    },
    {
        "name": "segment_damage",
        "description": "Generate precise SAM2 mask over a damage region. Call when you need exact damage boundaries to assess severity. Returns image with coloured mask overlay.",
        "input_schema": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x1, y1, x2, y2] rough bbox around damage to segment"
                },
                "reason": {"type": "string", "description": "What severity question this resolves"}
            },
            "required": ["bbox", "reason"]
        }
    },
    {
        "name": "Terminate",
        "description": "End the reasoning loop and return the final damage assessment. Only call when confident about all visible damage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "damage_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "damage_type": {"type": "string"},
                            "part": {"type": "string"},
                            "severity": {"type": "string"},
                            "confidence": {"type": "number"}
                        },
                        "required": ["damage_type", "part", "severity", "confidence"]
                    }
                }
            },
            "required": ["damage_items"]
        }
    }
]

VALID_DAMAGE_CLASSES = {"dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"}
VALID_PARTS = {
    "front_bumper", "rear_bumper", "hood", "windshield", "rear_windshield",
    "front_left_door", "front_right_door", "rear_left_door", "rear_right_door",
    "left_fender", "right_fender", "trunk_lid", "roof_panel",
    "headlight", "taillight", "tire"
}
VALID_SEVERITY = {"minor", "moderate", "severe"}

# Canonical tool names the CodeAct loop accepts. Small VLMs frequently emit
# casing/spacing variants (e.g. "terminate", "Run Damage Detection") — normalise
# them so a correct action is not rejected purely on formatting.
_CANONICAL_TOOLS = {
    "run_damage_detection",
    "zoom_region",
    "detect_part",
    "segment_damage",
    "execute_cost_computation",
    "Terminate",
}
_CANONICAL_TOOL_LOOKUP = {t.lower().replace(" ", "_"): t for t in _CANONICAL_TOOLS}


def _canonicalize_action_names(turn) -> None:
    """Rewrite each action.name to its canonical form in place (case/space-insensitive)."""
    for action in turn.actions:
        key = (action.name or "").strip().lower().replace(" ", "_")
        if key in _CANONICAL_TOOL_LOOKUP:
            action.name = _CANONICAL_TOOL_LOOKUP[key]


def _bbox_to_part(bbox: list, img_w: int, img_h: int) -> str:
    """
    Maps a bounding box [x1, y1, x2, y2] to a vehicle part label.
    Uses centre point relative position within the image frame.
    Assumes standard front-3/4 or front view vehicle photo.
    Returns closest matching valid part label.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h

    if cy < 0.25:
        if cx < 0.4:
            return "hood"
        elif cx > 0.6:
            return "roof_panel"
        else:
            return "hood"
    elif cy > 0.78:
        if cx < 0.35:
            return "tire"
        elif cx > 0.65:
            return "tire"
        else:
            return "front_bumper"
    else:
        if cx < 0.25:
            return "left_fender"
        elif cx > 0.75:
            return "right_fender"
        elif cx < 0.45:
            return "front_left_door"
        elif cx > 0.55:
            return "front_right_door"
        else:
            return "front_bumper"


def _yolo_to_damage_entries(detections: list, config: dict) -> list:
    """
    Converts YOLO detections directly to costed DamagePartEntry objects.
    Deterministic fallback when VLM produces no output.
    Severity derived from confidence score and damage class.
    """
    from pipeline.schema import DamagePartEntry
    from models.vlm_reasoning.cost_db import lookup_cost

    if not detections:
        return []

    entries = []
    for det in detections:
        cls  = det.get("class", "dent")
        conf = det.get("confidence", 0.5)
        bbox = det.get("bbox", [0, 0, 100, 100])

        if conf >= 0.70:
            severity = "moderate"
        elif conf >= 0.40:
            severity = "minor"
        else:
            severity = "minor"

        if cls in ("glass_shatter", "lamp_broken"):
            severity = "moderate" if conf >= 0.50 else "minor"
        elif cls == "tire_flat":
            severity = "severe" if conf >= 0.70 else "moderate"

        part = _bbox_to_part(bbox, 1920, 1080)
        cost_min, cost_max = lookup_cost(cls, part)

        entries.append(DamagePartEntry(
            damage=cls,
            part=part,
            severity=severity,
            cost_min=cost_min,
            cost_max=cost_max,
        ))

    return entries


def _run_vlm_vision_pass(image_path: str, config: dict) -> list:
    """
    Runs a single VLM forward pass on the raw image with no CV context.
    Returns a list of damage dicts: [{damage_type, part, severity, confidence}]
    Returns [] on any failure — caller handles fallback.
    """
    from pipeline.feedback_reader import get_few_shot_examples
    few_shot = get_few_shot_examples()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{Path(image_path).resolve()}"},
                {"type": "text", "text": VISION_ASSESSMENT_PROMPT + few_shot},
            ],
        }
    ]

    try:
        raw = _call_vlm(
            messages=messages,
            config=config,
            tools=None,
            max_new_tokens=config["vlm"].get("max_new_tokens_tool", 120),
        )
    except Exception as e:
        logger.error(f"VLM vision pass failed: {e}")
        return []

    clean = raw.strip()
    clean = re.sub(r"^```json\s*", "", clean)
    clean = re.sub(r"```\s*$", "", clean)

    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if not match:
        logger.warning(f"VLM vision pass: no JSON object found. Raw: {raw[:300]}")
        return []

    try:
        parsed = json.loads(match.group())
        items = parsed.get("damage_items", [])
        if not isinstance(items, list):
            return []
        required = {"damage_type", "part", "severity", "confidence"}
        valid = []
        for item in items:
            if required.issubset(item.keys()):
                valid.append(item)
            else:
                logger.warning(f"VLM vision pass: dropping malformed item: {item}")
        return valid
    except json.JSONDecodeError as e:
        logger.error(f"VLM vision pass JSON parse failed: {e}. Raw: {raw[:300]}")
        return []


def _merge_damage_sources(
    yolo_detections: list,
    vlm_visual: list,
    img_w: int,
    img_h: int,
) -> list:
    """
    Merges YOLO detections and VLM visual assessment into a single damage list.

    VLM visual items added first (accurate part labels from image context).
    YOLO items added with bbox-derived part labels.
    Dedup: same (part, damage_type) → keep higher confidence item.
    Result sorted by confidence descending.

    Returns list of dicts: [{damage_type, part, severity, confidence, source}]
    """
    def normalise(s: str) -> str:
        return s.lower().replace("_", " ").strip()

    merged: dict = {}

    for item in vlm_visual:
        key = (normalise(item["damage_type"]), normalise(item["part"]))
        merged[key] = {**item, "source": "vlm_visual", "bbox": [0.0, 0.0, 0.0, 0.0]}

    for det in yolo_detections:
        part = _bbox_to_part(det["bbox"], img_w, img_h)
        severity = (
            "minor" if det["confidence"] < 0.4
            else "moderate" if det["confidence"] < 0.7
            else "severe"
        )
        key = (normalise(det["class"]), normalise(part))
        existing = merged.get(key)
        if not existing or det["confidence"] > existing.get("confidence", 0):
            merged[key] = {
                "damage_type": det["class"],
                "part": part,
                "severity": severity,
                "confidence": det["confidence"],
                "source": "yolo",
                "bbox": det.get("bbox", [0.0, 0.0, 0.0, 0.0]),
            }

    return sorted(merged.values(), key=lambda x: x["confidence"], reverse=True)


def _load_models(config: dict) -> None:
    """
    Formerly loaded HuggingFace Qwen2-VL weights into GPU/MPS RAM.

    Now replaced by an Ollama health check — the model is served by the
    Ollama process (localhost:11434) and does NOT need to be loaded in-process.

    Function name kept for backward compatibility with backend/app.py startup
    which calls _load_models(config) on /health warm-up.

    Raises RuntimeError if Ollama is unreachable or the configured model is absent.
    """
    from models.vlm_reasoning.ollama_client import check_health

    vlm_cfg  = config.get("vlm", {})
    base_url = vlm_cfg.get("ollama_base_url", "http://localhost:11434")
    model    = vlm_cfg.get("model_id", "qwen3.5:9b")

    logger.info(f"Checking Ollama: model='{model}' at {base_url}")
    if not check_health(base_url, model):
        raise RuntimeError(
            f"Ollama is not running or model '{model}' is not available at {base_url}. "
            f"Start Ollama with: ollama serve | Then pull model with: ollama pull {model}"
        )
    logger.info(f"Ollama ready: {model} @ {base_url}")


def _apply_cost_lookup(damage_items: list) -> list:
    """
    Applies COST_DB pricing to each damage item.
    Returns list of DamagePartEntry-compatible dicts.
    """
    from models.vlm_reasoning.cost_db import lookup_cost

    result = []
    for item in damage_items:
        cost_min, cost_max = lookup_cost(item["damage_type"], item["part"])
        result.append({
            "damage": item["damage_type"],
            "part": item["part"],
            "severity": item["severity"],
            "cost_min": cost_min,
            "cost_max": cost_max,
        })
    return result






def _preprocess_image(image_path: str) -> str:
    """
    Validate and normalise image before sending to VLM.
    Converts grayscale and RGBA to RGB.
    Returns the (possibly converted) image path.

    Raises:
        ValueError: if image cannot be opened or is too small
    """
    path = Path(image_path)
    if not path.exists():
        raise ValueError(f"Image file not found: {image_path}")

    try:
        img = Image.open(path)
    except Exception as e:
        raise ValueError(f"Cannot open image at {image_path}: {e}") from e

    w, h = img.size
    if w < 100 or h < 100:
        raise ValueError(
            f"Image too small for segmentation: {w}x{h}px. "
            f"Minimum 100x100px required."
        )

    # Convert non-RGB modes
    if img.mode in ("L", "RGBA", "P", "LA"):
        logger.info(f"Converting image from mode '{img.mode}' to RGB")
        converted = img.convert("RGB")
        out_path = path.parent / f"_rgb_{path.name}"
        converted.save(out_path)
        return str(out_path)

    return image_path


def _call_vlm(
    messages: list,
    config: dict,
    is_final_turn: bool = False,
    tools=None,
    max_new_tokens: Optional[int] = None,
) -> str:
    """
    Run a single VLM forward pass and return the decoded response string.

    Args:
        messages: Full conversation history
        config: Pipeline config dict
        is_final_turn: If True, use larger max_new_tokens for synthesis
        tools: Tool definitions to inject (pass None for plain structured-output calls)
        max_new_tokens: Override token budget (ignores is_final_turn if provided)
    """
    from qwen_vl_utils import process_vision_info

    vlm_cfg = config.get("vlm", {})
    if max_new_tokens is not None:
        max_tokens = max_new_tokens
    elif is_final_turn:
        max_tokens = vlm_cfg.get("max_new_tokens_final", 160)
    else:
        max_tokens = vlm_cfg.get("max_new_tokens_tool", 120)

    text_input = _processor.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = _processor(
        text=[text_input],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt"
    )
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    n_prompt_tokens = int(inputs["input_ids"].shape[1])
    n_images = len(image_inputs) if image_inputs else 0
    device = str(_model.device)
    logger.info(
        f"VLM.generate START | device={device} | prompt_tokens={n_prompt_tokens} "
        f"| images={n_images} | max_new_tokens={max_tokens} "
        f"(this can take a while on MPS/CPU)"
    )
    t_gen = time.time()

    # NOTE: _vlm_timeout() uses signal.alarm(), which only fires on the main
    # thread. Background job workers are NOT the main thread, so this guard is a
    # best-effort no-op there; the backend's own hard job timeout is the real cap.
    with torch.inference_mode():
        with _vlm_timeout(90):
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=vlm_cfg.get("temperature", 0.1),
                do_sample=vlm_cfg.get("do_sample", False),
            )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    n_new = int(new_tokens.shape[0])
    gen_s = time.time() - t_gen
    tok_per_s = (n_new / gen_s) if gen_s > 0 else 0.0
    logger.info(
        f"VLM.generate DONE  | new_tokens={n_new} | elapsed={gen_s:.1f}s "
        f"| {tok_per_s:.2f} tok/s"
    )

    # skip_special_tokens=True prevents <|im_end|>/<|endoftext|> leaking into the
    # parsed CodeAct JSON (regression noted in architecture_verification.md).
    result = _processor.decode(new_tokens, skip_special_tokens=True)
    logger.debug(f"VLM raw output: {result[:300]}")

    del inputs, output_ids, new_tokens
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()

    return result


def _extract_json_objects(text: str) -> list:
    """
    Bracket-matching extractor. Returns all top-level JSON objects from text.
    Safer than greedy regex — handles nested structures correctly.
    """
    candidates = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            start = i
            in_string = False
            escape_next = False
            for j in range(i, len(text)):
                c = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if c == '\\' and in_string:
                    escape_next = True
                    continue
                if c == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            candidates.append(text[start:j+1])
                            i = j + 1
                            break
            else:
                i += 1
        else:
            i += 1
    return candidates


def _repair_json(text: str) -> Optional[str]:
    """Attempt common JSON repairs: trailing commas, single quotes on keys."""
    try:
        repaired = re.sub(r',\s*([}\]])', r'\1', text)
        repaired = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', repaired)
        return repaired
    except Exception:
        return None


def _parse_codeact_turn(raw: str) -> "tuple[Optional[object], str]":
    """
    Parse VLM raw output into CodeActTurn.
    Returns (turn, error_message). error_message is '' on success.

    Uses three strategies:
    1. Bracket-matching — finds all JSON objects, picks first with "actions" key
    2. JSON repair then retry strategy 1
    3. Regex extraction of actions array + thought field separately
    """
    from pipeline.schema import CodeActTurn, CodeActAction

    if not raw or not raw.strip():
        return None, "Empty response from VLM"

    clean = raw.strip()
    clean = re.sub(r'^```json\s*', '', clean, flags=re.MULTILINE)
    clean = re.sub(r'^```\s*$', '', clean, flags=re.MULTILINE).strip()

    # Strategy 1: bracket-matching
    for candidate in _extract_json_objects(clean):
        try:
            data = json.loads(candidate)
            if "actions" not in data or not isinstance(data.get("actions"), list):
                continue
            return CodeActTurn(**data), ""
        except Exception:
            continue

    # Strategy 2: repair then retry
    repaired = _repair_json(clean)
    if repaired:
        for candidate in _extract_json_objects(repaired):
            try:
                data = json.loads(candidate)
                if "actions" not in data or not isinstance(data.get("actions"), list):
                    continue
                return CodeActTurn(**data), ""
            except Exception:
                continue

    # Strategy 3: extract actions array and thought via regex
    actions_match = re.search(r'"actions"\s*:\s*(\[.*?\])', clean, re.DOTALL)
    thought_match = re.search(r'"thought"\s*:\s*"([^"]*)"', clean)
    if actions_match:
        try:
            actions = json.loads(actions_match.group(1))
            thought = thought_match.group(1) if thought_match else "recovered"
            turn = CodeActTurn(
                thought=thought,
                uncertainty=[],
                actions=[CodeActAction(**a) for a in actions],
            )
            return turn, ""
        except Exception:
            pass

    return None, (
        f"JSON parse error after all recovery strategies. "
        f"Raw (first 300 chars): {raw[:300]}"
    )


def _enforce_turn_policy(
    turn,
    iteration: int,
    tool_calls_made: int,
) -> "tuple[bool, str]":
    """
    Returns (is_valid, rejection_reason).
    Rejection reason injected back as corrective message on retry.
    """
    actions = turn.actions
    is_terminate = any(a.name == "Terminate" for a in actions)

    if is_terminate:
        conf = turn.confidence or 0.0

        # Relaxed pre-Terminate rule: a confident pure-visual assessment is allowed
        # without a prior tool call, so clean/obvious images do not burn retries.
        # Only force a tool call when the VLM terminates immediately AND is unsure.
        if tool_calls_made == 0 and iteration == 0 and conf < 0.70:
            return False, (
                "You terminated immediately with low confidence and no tool calls. "
                "Call run_damage_detection (or zoom_region / segment_damage) to gather "
                "evidence first, then terminate."
            )
        if conf < 0.70:
            return False, (
                f"Your confidence is {conf:.2f}, below the required 0.70. "
                f"Call a tool to resolve: {turn.uncertainty}"
            )
        if turn.uncertainty:
            return False, (
                f"You have {len(turn.uncertainty)} unresolved uncertainties: "
                f"{turn.uncertainty}. Call zoom_region or detect_part to resolve them."
            )
        term_action = next(a for a in actions if a.name == "Terminate")
        items = term_action.arguments.get("damage_items", [])
        # Empty damage_items is valid only for a high-confidence "no damage" verdict.
        if not items and conf < 0.90:
            return False, (
                "Terminate requires at least one damage_item, or confidence >= 0.90 "
                "if you are certain the vehicle is undamaged."
            )
        for item in items:
            if item.get("damage_type") not in VALID_DAMAGE_CLASSES:
                return False, f"Invalid damage_type: {item.get('damage_type')}. Must be one of {VALID_DAMAGE_CLASSES}"
            if item.get("part") not in VALID_PARTS:
                return False, f"Invalid part: {item.get('part')}. Must be one of {VALID_PARTS}"
            if item.get("severity") not in VALID_SEVERITY:
                return False, f"Invalid severity: {item.get('severity')}. Must be minor|moderate|severe"
    else:
        if not actions:
            return False, (
                "Your response has no actions. Either call a tool or call Terminate. "
                "If you need more information, call zoom_region on the most uncertain region."
            )
        valid_tools = {
            "run_damage_detection", "zoom_region", "detect_part",
            "segment_damage", "execute_cost_computation", "Terminate",
        }
        for a in actions:
            if a.name not in valid_tools:
                return False, f"Unknown tool: {a.name}. Valid tools: {valid_tools}"

    return True, ""


def _zoom_region(image_path: str, bbox: list, padding: float = 0.12) -> str:
    """PIL crop + upscale to 320px min, capped at 512px max. Returns path to cropped image."""
    with Image.open(image_path) as img:
        w, h = img.size
        x1, y1, x2, y2 = bbox
        pw = (x2 - x1) * padding
        ph = (y2 - y1) * padding
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw)
        y2 = min(h, y2 + ph)
        crop = img.crop((x1, y1, x2, y2))
        min_dim = min(crop.size)
        if min_dim < 320:
            scale = 320 / min_dim
            crop = crop.resize(
                (int(crop.width * scale), int(crop.height * scale)),
                Image.LANCZOS,
            )
        max_crop_dim = max(crop.size)
        if max_crop_dim > 512:
            scale = 512 / max_crop_dim
            crop = crop.resize(
                (int(crop.width * scale), int(crop.height * scale)),
                Image.LANCZOS,
            )
        out = f"/tmp/zoom_{uuid.uuid4().hex[:8]}.jpg"
        crop.save(out, quality=95)
    return out


def _detect_part_grounding_dino(image_path: str, query: str, config: dict) -> str:
    """
    Runs GroundingDINO for part_query. Draws bbox on image.
    Falls back to full image with text label if GroundingDINO unavailable.
    """
    import cv2

    try:
        from groundingdino.util.inference import load_model, load_image, predict
        from groundingdino.util import box_ops
        import torch as _torch

        gdino_cfg = config.get("part_segmentation", {}).get("grounding_dino", {})
        cfg_path = gdino_cfg.get("config_path", "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        wts_path = gdino_cfg.get("weights_path", "weights/groundingdino_swint_ogc.pth")
        box_thresh = gdino_cfg.get("box_threshold", 0.30)
        txt_thresh = gdino_cfg.get("text_threshold", 0.25)

        model = load_model(cfg_path, wts_path)
        image_source, image_tensor = load_image(image_path)
        h, w = image_source.shape[:2]

        boxes, logits, _ = predict(
            model=model,
            image=image_tensor,
            caption=query,
            box_threshold=box_thresh,
            text_threshold=txt_thresh,
        )

        img_bgr = cv2.imread(image_path)
        if len(boxes) > 0:
            boxes_px = box_ops.box_cxcywh_to_xyxy(boxes) * _torch.tensor([w, h, w, h])
            for box in boxes_px:
                bx1, by1, bx2, by2 = [int(v) for v in box]
                cv2.rectangle(img_bgr, (bx1, by1), (bx2, by2), (55, 138, 221), 2)
                cv2.putText(img_bgr, query, (bx1, max(by1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 138, 221), 2)
        else:
            cv2.putText(img_bgr, f"Not found: {query}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 220), 2)

        out = f"/tmp/detect_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(out, img_bgr)
        return out

    except ImportError:
        img_bgr = cv2.imread(image_path)
        cv2.putText(img_bgr, f"[GroundingDINO unavailable] query: {query}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 220), 2)
        out = f"/tmp/detect_fallback_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(out, img_bgr)
        return out


def _segment_damage_sam2(image_path: str, bbox: list, config: dict) -> str:
    """
    Runs SAM2 with bbox prompt. Draws mask overlay on image.
    Falls back to bbox rectangle if SAM2 unavailable.
    """
    import cv2

    try:
        from shared.sam_mask import generate_masked_image
        from pipeline.schema import DetectionWithBBox

        det = DetectionWithBBox(
            index=1, bbox=bbox, damage="damage", part="unknown",
            severity="unknown", confidence=1.0, source="tool",
        )
        out = generate_masked_image(
            image_path=image_path,
            detections=[det],
            weights_path=config.get("part_segmentation", {}).get("sam2", {}).get(
                "weights_path", "weights/sam2.1_hiera_base_plus.pt"
            ),
            output_dir="/tmp/seg_out",
        )
        return out
    except Exception as e:
        logger.warning(f"segment_damage SAM2 failed: {e} — falling back to bbox outline")
        img_bgr = cv2.imread(image_path)
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (186, 117, 23), 3)
        out = f"/tmp/seg_fallback_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(out, img_bgr)
        return out


def _run_yolo_detection(
    image_path: str,
    config: dict,
    confidence_threshold: float = 0.15,
) -> dict:
    """
    Runs the trained YOLOv8 best.pt on the image.
    Returns dict with:
      - annotated_image_path: str  (image with bboxes drawn, for VLM visual)
      - detections: list[dict]     (raw detection data)
      - total_detections: int
      - summary: str               (human-readable summary for logging)

    Annotated image is colour-coded by damage class:
      dent          → blue   (55, 138, 221)
      scratch       → green  (29, 158, 117)
      crack         → amber  (186, 117, 23)
      glass_shatter → pink   (212, 83, 126)
      lamp_broken   → coral  (216, 90, 48)
      tire_flat     → gray   (136, 135, 128)
    """
    import cv2
    import numpy as np
    from models.damage_detection import run as yolo_run

    CLASS_COLORS_BGR = {
        "dent":          (221, 138,  55),
        "scratch":       (117, 158,  29),
        "crack":         ( 23, 117, 186),
        "glass_shatter": (126,  83, 212),
        "lamp_broken":   ( 48,  90, 216),
        "tire_flat":     (128, 135, 136),
    }
    DEFAULT_COLOR = (128, 128, 128)

    det_config = {
        "weights_path": config.get("damage_detection", {}).get(
            "weights_path", "models/damage_detection/models/best.pt"
        ),
        "confidence_threshold": confidence_threshold,
        "device": config.get("damage_detection", {}).get("device", "cpu"),
    }

    try:
        result = yolo_run(image_path, det_config)
    except Exception as e:
        logger.error(f"YOLO run() failed: {e}")
        raise

    detections = result.get("detections", [])
    total = result.get("total_detections", 0)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    for i, det in enumerate(detections):
        bbox = det.get("bbox", [])
        cls  = det.get("class", "dent")
        conf = det.get("confidence", 0.0)

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        color = CLASS_COLORS_BGR.get(cls, DEFAULT_COLOR)

        if conf < 0.40:
            corner = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            for cx, cy, dx, dy in [
                (x1, y1,  1,  1), (x2, y1, -1,  1),
                (x1, y2,  1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(img, (cx, cy), (cx + dx * corner, cy), color, 2)
                cv2.line(img, (cx, cy), (cx, cy + dy * corner), color, 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{i+1}. {cls} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        lbl_y1 = max(y1 - th - 6, 0)
        cv2.rectangle(img, (x1, lbl_y1), (x1 + tw + 6, lbl_y1 + th + 4), color, -1)
        cv2.putText(
            img, label,
            (x1 + 3, lbl_y1 + th + 1),
            font, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    out_dir = Path("data/uploads/yolo_annotated")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{Path(image_path).stem}_yolo_{uuid.uuid4().hex[:6]}.jpg")
    cv2.imwrite(out_path, img)

    if detections:
        class_counts: dict = {}
        for d in detections:
            c = d.get("class", "unknown")
            class_counts[c] = class_counts.get(c, 0) + 1
        summary_parts = [f"{v}x {k}" for k, v in sorted(class_counts.items())]
        summary = f"Detected {total} damage region(s): {', '.join(summary_parts)}"
    else:
        summary = "No damage detected above confidence threshold"

    return {
        "annotated_image_path": out_path,
        "detections": detections,
        "total_detections": total,
        "summary": summary,
    }


def _merge_overlapping_detections(
    detections: list,
    iou_threshold: float = 0.30,
    same_class_only: bool = True,
) -> list:
    """
    Merges overlapping detections of the same class using greedy IoU-based algorithm.

    Algorithm: sort by confidence desc; for each detection, if IoU >= threshold
    with an already-accepted same-class detection, merge (union bbox, keep higher
    confidence). Otherwise add as new detection.

    Args:
        detections: list of dicts with 'bbox', 'class', 'confidence'
        iou_threshold: minimum IoU to trigger merge (0.30 = 30% overlap)
        same_class_only: only merge detections of the same class

    Returns: deduplicated list of dicts
    """
    if not detections:
        return []

    def _iou(a: list, b: list) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _union_bbox(a: list, b: list) -> list:
        return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
    merged = []

    for det in sorted_dets:
        det_bbox = det.get("bbox", [0, 0, 0, 0])
        det_cls  = det.get("class", "")
        merged_into = None

        for existing in merged:
            if same_class_only and det_cls != existing.get("class", ""):
                continue
            if _iou(det_bbox, existing.get("bbox", [0, 0, 0, 0])) >= iou_threshold:
                existing["bbox"] = _union_bbox(existing["bbox"], det_bbox)
                existing["_merge_count"] = existing.get("_merge_count", 1) + 1
                merged_into = existing
                break

        if merged_into is None:
            new_det = dict(det)
            new_det["_merge_count"] = 1
            merged.append(new_det)

    result = [{k: v for k, v in d.items() if not k.startswith("_")} for d in merged]

    logger.info(
        f"Merged {len(detections)} detections → {len(result)} "
        f"(removed {len(detections) - len(result)} overlapping boxes, "
        f"IoU threshold={iou_threshold})"
    )

    return result


def _run_yolo_eagerly(image_path: str, config: dict) -> dict:
    """
    Runs YOLO best.pt unconditionally on the image before the VLM loop.
    NOT a CodeAct tool — called directly from run().

    Returns:
    {
        "detections": [{"bbox": [...], "class": str, "confidence": float}],
        "total_detections": int,
        "annotated_image_path": str,
        "summary": str,
        "success": bool,
        "error": str | None,
    }
    """
    import uuid as _uuid
    import cv2
    from pathlib import Path as _Path
    from collections import Counter
    from models.damage_detection import run as yolo_run

    CLASS_COLORS_BGR = {
        "dent":          (221, 138,  55),
        "scratch":       (117, 158,  29),
        "crack":         ( 23, 117, 186),
        "glass_shatter": (126,  83, 212),
        "lamp_broken":   ( 48,  90, 216),
        "tire_flat":     (128, 135, 136),
    }

    det_config = {
        "weights_path": config.get("damage_detection", {}).get(
            "weights_path", "models/damage_detection/models/best.pt"
        ),
        "confidence_threshold": config.get("damage_detection", {}).get(
            "confidence_threshold", 0.15
        ),
        "device": config.get("damage_detection", {}).get("device", "cpu"),
    }

    try:
        result = yolo_run(image_path, det_config)
    except Exception as e:
        logger.error(f"YOLO eager run failed: {e}")
        return {
            "detections": [], "total_detections": 0,
            "annotated_image_path": image_path,
            "summary": f"YOLO failed: {e}",
            "success": False, "error": str(e),
        }

    detections = result.get("detections", [])
    total      = result.get("total_detections", 0)

    if len(detections) > 1:
        iou_thresh = config.get("damage_detection", {}).get("merge_iou_threshold", 0.30)
        same_only  = config.get("damage_detection", {}).get("merge_same_class_only", True)
        detections = _merge_overlapping_detections(detections, iou_thresh, same_only)
        total = len(detections)

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"cv2 cannot read: {image_path}")

        h, w = img.shape[:2]

        for i, det in enumerate(detections):
            bbox = det.get("bbox", [])
            cls  = det.get("class", "dent")
            conf = det.get("confidence", 0.0)
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            color = CLASS_COLORS_BGR.get(cls, (128, 128, 128))

            if conf >= 0.40:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            else:
                c = min(18, (x2-x1)//4, (y2-y1)//4)
                for cx, cy, dx, dy in [
                    (x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)
                ]:
                    cv2.line(img,(cx,cy),(cx+dx*c,cy),color,2)
                    cv2.line(img,(cx,cy),(cx,cy+dy*c),color,2)

            badge_r = 13
            bx = min(x1 + badge_r + 2, w - badge_r - 2)
            by = max(y1 - badge_r - 2, badge_r + 2)
            cv2.circle(img, (bx, by), badge_r, color, -1)
            cv2.circle(img, (bx, by), badge_r, (255,255,255), 1)
            lbl = str(i + 1)
            fs  = 0.5 if (i+1) < 10 else 0.4
            ts  = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
            cv2.putText(img, lbl,
                (bx - ts[0]//2, by + ts[1]//2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 2)

            label = f"{cls} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ly = max(y1 - th - 4, 0)
            cv2.rectangle(img, (x1, ly), (x1+tw+4, ly+th+4), color, -1)
            cv2.putText(img, label, (x1+2, ly+th+1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)

        out_dir = _Path("data/uploads/yolo_annotated")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(
            out_dir / f"{_Path(image_path).stem}_yolo_{_uuid.uuid4().hex[:6]}.jpg"
        )
        cv2.imwrite(out_path, img)
        annotated_path = out_path

    except Exception as e:
        logger.warning(f"Annotated image draw failed: {e} — using plain image")
        annotated_path = image_path

    if detections:
        counts  = Counter(d.get("class", "?") for d in detections)
        parts   = ", ".join(f"{v}x {k}" for k, v in sorted(counts.items()))
        summary = f"YOLO detected {total} damage region(s): {parts}"
    else:
        summary = (
            "YOLO found no detections above threshold "
            f"({det_config['confidence_threshold']}). "
            "VLM will perform independent visual assessment."
        )

    logger.info(f"YOLO eager: {summary}")

    return {
        "detections":           detections,
        "total_detections":     total,
        "annotated_image_path": annotated_path,
        "summary":              summary,
        "success":              True,
        "error":                None,
    }


def _execute_codeact_tool(
    action,
    image_path: str,
    config: dict,
) -> dict:
    """
    Executes a CodeAct vision tool call.
    Returns dict with keys:
      - type: "image" | "error"
      - image_path: str (if type == "image")
      - error: str (if type == "error")
      - summary: str
    """
    name = action.name
    args = action.arguments

    if name == "run_damage_detection":
        conf = float(args.get("confidence_threshold", 0.15))
        conf = max(0.05, min(0.95, conf))
        try:
            yolo_result = _run_yolo_detection(image_path, config, conf)
            return {
                "type": "image",
                "image_path": yolo_result["annotated_image_path"],
                "summary": yolo_result["summary"],
                "detections": yolo_result["detections"],
                "total_detections": yolo_result["total_detections"],
            }
        except Exception as e:
            logger.error(f"YOLO tool failed: {e}")
            return {"type": "error", "error": str(e), "summary": f"YOLO detection failed: {e}"}

    elif name == "zoom_region":
        bbox = args.get("bbox", [])
        if len(bbox) != 4:
            return {"type": "error", "error": "zoom_region requires bbox [x1,y1,x2,y2]", "summary": "bad args"}
        try:
            out_path = _zoom_region(image_path, bbox)
            return {
                "type": "image",
                "image_path": out_path,
                "summary": f"Zoomed into region {[int(v) for v in bbox]}",
            }
        except Exception as e:
            return {"type": "error", "error": str(e), "summary": "zoom failed"}

    elif name == "detect_part":
        query = args.get("part_query", "")
        try:
            out_path = _detect_part_grounding_dino(image_path, query, config)
            return {
                "type": "image",
                "image_path": out_path,
                "summary": f"Detected part: {query}",
            }
        except Exception as e:
            return {"type": "error", "error": str(e), "summary": f"detect_part failed: {e}"}

    elif name == "segment_damage":
        bbox = args.get("bbox", [])
        try:
            out_path = _segment_damage_sam2(image_path, bbox, config)
            return {
                "type": "image",
                "image_path": out_path,
                "summary": f"Segmented damage at {[int(v) for v in bbox]}",
            }
        except Exception as e:
            return {"type": "error", "error": str(e), "summary": f"segment failed: {e}"}

    elif name == "execute_cost_computation":
        from models.vlm_reasoning.sandbox import execute_sandboxed

        code = args.get("code", "")
        if not code or not str(code).strip():
            return {
                "type": "error",
                "error": "execute_cost_computation requires non-empty 'code' argument",
                "summary": "no code provided",
            }
        engine = config.get("vlm", {}).get("sandbox_engine", "monty")
        sandbox_out = execute_sandboxed(str(code), engine=engine)
        if "error" in sandbox_out:
            return {
                "type": "error",
                "error": sandbox_out["error"],
                "summary": f"cost computation failed: {sandbox_out['error']}",
            }
        cost_result = sandbox_out["result"]
        n_items = len(cost_result.get("damage_part_map", []))
        return {
            "type": "json",
            "data": cost_result,
            "summary": (
                f"Cost computed: {n_items} item(s), "
                f"total INR {cost_result.get('total_min', 0)}–{cost_result.get('total_max', 0)}"
            ),
        }

    else:
        return {"type": "error", "error": f"Unknown tool: {name}", "summary": "unknown tool"}


def _resize_for_vlm(image_path: str, max_dim: int = 640) -> str:
    """
    Resizes image so longest dimension <= max_dim before passing to VLM.
    Saves resized copy to /tmp. Returns path to resized image.
    Original image is not modified — YOLO always receives the original.

    640px → ~200 visual tokens in Qwen2-VL (vs ~2000 for full 1920px).
    """
    from PIL import Image as PILImage
    import uuid as _uuid2

    try:
        with PILImage.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size

            if max(w, h) <= max_dim:
                logger.info(f"Image already small ({w}x{h}), no resize needed")
                return image_path

            scale = max_dim / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = img.resize((new_w, new_h), PILImage.LANCZOS)

            out_path = f"/tmp/vlm_resize_{_uuid2.uuid4().hex[:8]}.jpg"
            resized.save(out_path, quality=90, optimize=True)

            logger.info(
                f"Resized for VLM: {w}x{h} → {new_w}x{new_h} "
                f"(max_dim={max_dim})"
            )
            return out_path

    except Exception as e:
        logger.warning(f"Image resize failed: {e} — using original")
        return image_path


def _run_codeact_loop(
    image_path: str,
    config: dict,
    trajectory_steps: list,
) -> dict:
    """
    Thinking-with-images CodeAct loop. The VLM sees the RAW image first (no eager
    YOLO, no pre-injected detections) and drives every tool call itself.

    The VLM may call run_damage_detection, zoom_region, detect_part, segment_damage,
    execute_cost_computation, or Terminate. Image-returning tools are fed back as new
    image turns ("thinking with images"); cost results are fed back as text.

    Tool execution always uses the ORIGINAL full-resolution image. Only the copy
    shown to the VLM is downscaled (image_max_dim) to keep visual-token count low.

    Returns dict:
      {
        "damage_items": list,             # from Terminate (may be empty)
        "cost_result": dict | None,       # last valid execute_cost_computation output
        "yolo_detections": list,          # captured from any run_damage_detection call
        "annotated_image_path": str|None, # YOLO-annotated image if YOLO was called
        "tool_calls": int,
        "warnings": list,
      }
    """
    from pipeline.schema import TrajectoryStep

    max_iter  = config["vlm"].get("max_iterations", 6)
    max_retry = config["vlm"].get("codeact_max_retries", 2)
    max_dim   = config["vlm"].get("image_max_dim", 640)

    warnings: list = []
    tool_calls = 0
    cost_result = None
    yolo_detections: list = []
    annotated_image_path = None
    last_raw = None

    vlm_image_path = _resize_for_vlm(image_path, max_dim)

    model_id = config["vlm"].get("model_id", "?")
    device   = config["vlm"].get("device", "?")
    logger.info(
        f"CodeAct loop START | model={model_id} | device={device} "
        f"| max_iterations={max_iter} | vlm_image={Path(vlm_image_path).name}"
    )

    messages = [
        {"role": "system", "content": CODEACT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{Path(vlm_image_path).resolve()}"},
                {"type": "text", "text": (
                    "Assess all visible damage on this vehicle. Use your tools to "
                    "locate and inspect damage, compute repair cost with "
                    "execute_cost_computation, then call Terminate with the final "
                    "verified damage list. Respond with ONLY the JSON object."
                )},
            ],
        },
    ]

    t_loop = time.time()

    for iteration in range(max_iter):
        if time.time() - t_loop > 480:
            warnings.append(f"Loop timeout after {iteration} iterations")
            break

        logger.info(
            f"--- CodeAct iteration {iteration + 1}/{max_iter} "
            f"| tool_calls_so_far={tool_calls} "
            f"| loop_elapsed={time.time() - t_loop:.1f}s ---"
        )
        turn = None
        for attempt in range(max_retry + 1):
            try:
                raw = _call_vlm(
                    messages       = messages,
                    config         = config,
                    tools          = None,
                    max_new_tokens = config["vlm"].get("max_new_tokens_tool", 512),
                )
            except VLMGenerationTimeout:
                warnings.append(f"VLM timeout iter={iteration} attempt={attempt}")
                raw = None
                break
            except Exception as e:
                warnings.append(f"VLM call error: {e}")
                raw = None
                break

            if raw is None:
                break

            turn, parse_err = _parse_codeact_turn(raw)
            if parse_err:
                if attempt < max_retry:
                    messages.append({"role": "user", "content": [{
                        "type": "text",
                        "text": (
                            f"Your output was not valid JSON. Error: {parse_err}. "
                            "Respond with ONLY the JSON object, no other text."
                        )
                    }]})
                    turn = None
                    continue
                warnings.append(f"JSON parse failed after {max_retry} retries: {parse_err}")
                break

            _canonicalize_action_names(turn)
            valid, reject_reason = _enforce_turn_policy(turn, iteration, tool_calls)
            if not valid:
                if attempt < max_retry:
                    messages.append({"role": "user", "content": [{
                        "type": "text",
                        "text": f"Policy violation: {reject_reason} Try again."
                    }]})
                    turn = None
                    continue
                warnings.append(f"Policy failed after {max_retry} retries: {reject_reason}")
                break
            break

        if turn is None:
            break

        logger.info(f"[codeact iter {iteration}] thought: {turn.thought[:120]}")
        if turn.uncertainty:
            logger.info(f"[codeact iter {iteration}] uncertainty: {turn.uncertainty}")

        # Record the model's reasoning turn so it stays in context for the next turn.
        last_raw = raw
        messages.append({"role": "assistant", "content": raw})

        for action in turn.actions:
            t_action = time.time()

            if action.name == "Terminate":
                damage_items = action.arguments.get("damage_items", [])
                trajectory_steps.append(TrajectoryStep(
                    turn_index          = iteration,
                    action              = action,
                    observation_type    = "json",
                    observation_summary = f"Terminated with {len(damage_items)} items",
                    observation_data    = {"damage_items": damage_items},
                    elapsed_s           = round(time.time() - t_action, 2),
                ))
                logger.info(f"VLM terminated: {len(damage_items)} damage items")
                return {
                    "damage_items":         damage_items,
                    "cost_result":          cost_result,
                    "yolo_detections":      yolo_detections,
                    "annotated_image_path": annotated_image_path,
                    "tool_calls":           tool_calls,
                    "warnings":             warnings,
                    "raw_vlm_response":     last_raw,
                }

            logger.info(f"[codeact iter {iteration}] tool call: {action.name} args={action.arguments}")
            result   = _execute_codeact_tool(action, image_path, config)
            tool_calls += 1
            elapsed  = round(time.time() - t_action, 2)
            logger.info(
                f"[codeact iter {iteration}] tool {action.name} -> "
                f"{result.get('type')} in {elapsed}s | {result.get('summary', '')[:120]}"
            )

            trajectory_steps.append(TrajectoryStep(
                turn_index             = iteration,
                action                 = action,
                observation_type       = result["type"],
                observation_summary    = result.get("summary", ""),
                observation_image_path = result.get("image_path"),
                observation_data       = result.get("data"),
                elapsed_s              = elapsed,
            ))

            # Capture side-channel data for the final report / annotation UI.
            if action.name == "run_damage_detection" and result["type"] == "image":
                yolo_detections      = result.get("detections", yolo_detections)
                annotated_image_path = result.get("image_path", annotated_image_path)
            if action.name == "execute_cost_computation" and result["type"] == "json":
                cost_result = result.get("data", cost_result)

            # Feed the observation back to the VLM.
            if result["type"] == "image" and result.get("image_path"):
                obs_img = _resize_for_vlm(result["image_path"], max_dim)
                if action.name == "run_damage_detection":
                    dets = result.get("detections", [])
                    if dets:
                        det_lines = "\n".join(
                            f"  Box {i+1}: {d.get('class','?').replace('_',' ')} "
                            f"({int(d.get('confidence',0)*100)}%) at "
                            f"{[int(v) for v in d.get('bbox', [])]}"
                            for i, d in enumerate(dets)
                        )
                        obs_text = (
                            f"run_damage_detection result: {result.get('summary','')}\n"
                            f"{det_lines}\n\n"
                            "The annotated image above shows numbered boxes. Inspect any "
                            "uncertain region with zoom_region or segment_damage, then "
                            "compute cost and Terminate."
                        )
                    else:
                        obs_text = (
                            f"run_damage_detection result: {result.get('summary','')}\n"
                            "YOLO found nothing above threshold. Assess the image visually "
                            "for any damage it may have missed."
                        )
                else:
                    obs_text = (
                        f"Tool result for {action.name}: {result.get('summary','')}. "
                        "The image above shows the result. Continue your assessment."
                    )
                messages.append({"role": "user", "content": [
                    {"type": "image", "image": f"file://{Path(obs_img).resolve()}"},
                    {"type": "text", "text": obs_text},
                ]})

            elif result["type"] == "json":
                messages.append({"role": "user", "content": [{
                    "type": "text",
                    "text": (
                        f"Tool result for {action.name}: {result.get('summary','')}.\n"
                        f"Cost data: {json.dumps(result.get('data', {}), default=str)}\n"
                        "If this looks correct, call Terminate with your final damage list."
                    )
                }]})

            elif result["type"] == "error":
                messages.append({"role": "user", "content": [{
                    "type": "text",
                    "text": (
                        f"Tool {action.name} failed: {result['error']}. "
                        "Continue with available information."
                    )
                }]})

    warnings.append(f"VLM loop ended without Terminate after {max_iter} iterations")
    return {
        "damage_items":         [],
        "cost_result":          cost_result,
        "yolo_detections":      yolo_detections,
        "annotated_image_path": annotated_image_path,
        "tool_calls":           tool_calls,
        "warnings":             warnings,
        "raw_vlm_response":     last_raw,
    }


def _save_trajectory(
    image_path: str,
    img_w: int,
    img_h: int,
    steps: list,
    costed: list,
    total_min: int,
    total_max: int,
    elapsed: float,
    model_id: str,
) -> None:
    """
    Saves raw trajectory to data/trajectories/raw/.
    filter_status set to 'unfiltered' — trajectory_filter.py promotes to approved/.
    """
    from pipeline.schema import Trajectory

    traj = Trajectory(
        trajectory_id=uuid.uuid4().hex,
        image_path=image_path,
        image_width=img_w,
        image_height=img_h,
        created_at=datetime.now(timezone.utc).isoformat(),
        model_id=model_id,
        steps=steps,
        final_damage_map=costed,
        total_min=total_min,
        total_max=total_max,
        total_elapsed_s=elapsed,
        filter_status="unfiltered",
    )

    raw_dir = Path("data/trajectories/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{traj.trajectory_id}.json"
    out.write_text(traj.model_dump_json(indent=2))
    logger.info(f"Trajectory saved: {out}")


def run(
    image_path: str,
    config: dict,
    claim_metadata: Optional[dict] = None,
) -> dict:
    """
    Main pipeline entry point. Called by FastAPI backend.

    Stage 1 [always]:  VLM CodeAct loop drives tool calls on the RAW image
                       (thinking with images). The VLM decides when to call
                       run_damage_detection, zoom_region, detect_part,
                       segment_damage, and execute_cost_computation.
    Stage 2 [always]:  Build the costed damage map — prefer the VLM's sandboxed
                       execute_cost_computation result, else cost-lookup the VLM's
                       Terminate damage_items.
    Stage 3 [fallback]: if the VLM produced nothing usable → run YOLO and use its
                        detections directly (safety net).
    Stage 4 [always]:  approval gate → FinalDamageReport.

    The UI contract (detections_with_bbox + annotated_image_path) is always
    populated: from the VLM's run_damage_detection call if it made one, otherwise
    from a single YOLO pass that does not influence VLM reasoning.

    Returns:
        FinalDamageReport as dict (via .model_dump())

    Raises:
        ValueError: if image_path does not exist or image is invalid
        RuntimeError: if VLM fails to load
    """
    from models.vlm_reasoning.cost_db import lookup_cost

    t_start = time.time()
    warnings_list: List[str] = []
    trajectory_steps: list = []

    image_path = _preprocess_image(image_path)

    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except Exception:
        img_w, img_h = 1920, 1080
        warnings_list.append("Could not read image dimensions — using fallback 1920x1080")

    from pipeline.schema import DetectionWithBBox as _DetWithBBox
    from models.vlm_reasoning.cost_db import lookup_cost as _lc

    # ── Stage 1: PiAgent recursive CodeAct loop (VLM-only, Ollama backend) ──────
    _load_models(config)   # Ollama health check (raises if not reachable)

    logger.info("Stage 1: Starting PiAgent CodeAct loop (qwen3.5:9b via Ollama)")
    from models.vlm_reasoning.pi_agent import PiAgent
    agent    = PiAgent(config)
    loop_out = agent.run(
        image_path       = image_path,
        trajectory_steps = trajectory_steps,
    )
    warnings_list.extend(loop_out["warnings"])

    vlm_damage_items = loop_out["damage_items"]
    cost_result      = loop_out["cost_result"]
    yolo_detections  = loop_out["yolo_detections"]
    annotated_path   = loop_out["annotated_image_path"]

    vlm_produced = bool(vlm_damage_items) or bool(
        cost_result and cost_result.get("damage_part_map")
    )

    # Ensure YOLO detections exist for the annotation UI and the fallback path.
    # If the VLM never called run_damage_detection, run YOLO once now (UI-only —
    # it does not influence the VLM reasoning that already finished above).
    yolo_success = True
    if not yolo_detections or not annotated_path:
        logger.info("VLM did not call run_damage_detection — running YOLO for UI/fallback")
        yolo_ui = _run_yolo_eagerly(image_path, config)
        yolo_success = yolo_ui["success"]
        if not yolo_success:
            warnings_list.append(f"YOLO (UI/fallback) failed: {yolo_ui['error']}")
        if not yolo_detections:
            yolo_detections = yolo_ui["detections"]
        if not annotated_path:
            annotated_path = yolo_ui["annotated_image_path"]

    # Build DetectionWithBBox from YOLO results — preserves spatial bbox information
    detections_with_bbox = []
    for _i, _det in enumerate(yolo_detections):
        _bbox = _det.get("bbox", [0.0, 0.0, 0.0, 0.0])
        _cls  = _det.get("class", "dent")
        _conf = _det.get("confidence", 0.0)
        _part = _bbox_to_part(_bbox, img_w, img_h)
        if _cls in ("glass_shatter", "lamp_broken"):
            _sev = "moderate" if _conf >= 0.5 else "minor"
        elif _cls == "tire_flat":
            _sev = "severe" if _conf >= 0.7 else "moderate"
        elif _conf >= 0.7:
            _sev = "moderate"
        else:
            _sev = "minor"
        _cmin, _cmax = _lc(_cls, _part)
        detections_with_bbox.append(_DetWithBBox(
            index      = _i + 1,
            bbox       = [float(v) for v in _bbox],
            damage     = _cls,
            part       = _part,
            severity   = _sev,
            confidence = _conf,
            source     = "yolo",
            cost_min   = _cmin,
            cost_max   = _cmax,
        ))

    # Merge VLM classifications into detections_with_bbox when VLM produced output
    if vlm_damage_items:
        _vlm_by_class = {}
        for _item in vlm_damage_items:
            _c = _item.get("damage_type", "")
            if _c not in _vlm_by_class:
                _vlm_by_class[_c] = _item

        _updated = []
        for _d in detections_with_bbox:
            if _d.damage in _vlm_by_class:
                _vi = _vlm_by_class[_d.damage]
                _d = _d.model_copy(update={
                    "severity":   _vi.get("severity", _d.severity),
                    "confidence": _vi.get("confidence", _d.confidence),
                    "source":     "vlm_verified",
                })
            _updated.append(_d)

        _yolo_classes = {_d.damage for _d in detections_with_bbox}
        for _item in vlm_damage_items:
            if _item.get("damage_type") not in _yolo_classes:
                from models.vlm_reasoning.cost_db import lookup_cost as _lc2
                _cm, _cx = _lc2(_item.get("damage_type", ""), _item.get("part", ""))
                _updated.append(_DetWithBBox(
                    index      = len(_updated) + 1,
                    bbox       = [0.0, 0.0, 0.0, 0.0],
                    damage     = _item.get("damage_type", ""),
                    part       = _item.get("part", ""),
                    severity   = _item.get("severity", "minor"),
                    confidence = _item.get("confidence", 0.7),
                    source     = "vlm_only",
                    cost_min   = _cm,
                    cost_max   = _cx,
                ))
        detections_with_bbox = _updated

    # ── Stage 2: Build the costed damage map ──────────────────────────────────
    # Preference order:
    #   1. VLM's sandboxed execute_cost_computation result (already priced)
    #   2. VLM's Terminate damage_items, priced via COST_DB lookup
    #   3. YOLO fallback (safety net)
    costed: List[DamagePartEntry] = []

    if cost_result and cost_result.get("damage_part_map"):
        logger.info("Using VLM execute_cost_computation result for cost map")
        for e in cost_result["damage_part_map"]:
            try:
                costed.append(DamagePartEntry(
                    damage   = str(e.get("damage", "unknown")),
                    part     = str(e.get("part", "unknown")),
                    severity = str(e.get("severity", "minor")),
                    cost_min = int(e.get("cost_min", 0)),
                    cost_max = int(e.get("cost_max", 0)),
                ))
            except Exception as _e:
                warnings_list.append(f"Skipped malformed cost entry {e}: {_e}")

    if not costed and vlm_damage_items:
        logger.info("Pricing VLM Terminate damage_items via COST_DB lookup")
        for item in vlm_damage_items:
            cost_min, cost_max = lookup_cost(
                item.get("damage_type", ""),
                item.get("part", ""),
            )
            costed.append(DamagePartEntry(
                damage   = item.get("damage_type", ""),
                part     = item.get("part", ""),
                severity = item.get("severity", "minor"),
                cost_min = cost_min,
                cost_max = cost_max,
            ))

    # ── Stage 3: YOLO fallback if VLM produced nothing usable ─────────────────
    if not costed:
        vlm_produced = False
        if yolo_detections:
            warnings_list.append(
                "VLM produced no usable output — falling back to YOLO detections directly"
            )
            logger.warning("VLM fallback: using YOLO detections with heuristic severity")
            costed = _yolo_to_damage_entries(yolo_detections, config)
        else:
            warnings_list.append(
                "EMPTY_DAMAGE_MAP: Both VLM and YOLO found no damage. "
                "Escalating to human review."
            )

    # ── Stage 4: Costs and approval ───────────────────────────────────────────
    total_min = sum(e.cost_min for e in costed)
    total_max = sum(e.cost_max for e in costed)

    threshold = config.get("approval", {}).get("auto_approve_threshold_inr", 50000)
    if not costed:
        approval = "ESCALATE_TO_HUMAN"
    elif not vlm_produced:
        approval = "ESCALATE_TO_HUMAN"
        warnings_list.append(
            "YOLO_ONLY_RESULT: VLM did not produce a verified assessment. "
            "Escalating for human review regardless of cost."
        )
    elif total_max <= threshold:
        approval = "AUTO_APPROVED"
    else:
        approval = "ESCALATE_TO_HUMAN"

    # ── Tool call log ─────────────────────────────────────────────────────────
    tool_log = [
        ToolCallRecord(
            tool         = step.action.name,
            args_summary = str(step.action.arguments)[:80],
            elapsed_s    = step.elapsed_s,
            result_keys  = list(step.observation_data.keys()) if step.observation_data else [],
            success      = step.observation_type != "error",
        )
        for step in trajectory_steps
    ]
    if not yolo_success:
        tool_log.append(ToolCallRecord(
            tool         = "run_damage_detection",
            args_summary = "ui_fallback",
            elapsed_s    = 0.0,
            result_keys  = [],
            success      = False,
        ))

    elapsed = round(time.time() - t_start, 2)

    _save_trajectory(
        image_path = image_path,
        img_w      = img_w,
        img_h      = img_h,
        steps      = trajectory_steps,
        costed     = costed,
        total_min  = total_min,
        total_max  = total_max,
        elapsed    = elapsed,
        model_id   = config["vlm"]["model_id"],
    )

    return FinalDamageReport(
        image_path           = image_path,
        damage_part_map      = costed,
        detections_with_bbox = detections_with_bbox,
        total_min            = total_min,
        total_max            = total_max,
        currency             = "INR",
        approval_decision    = approval,
        tool_call_log        = tool_log,
        total_inference_s    = elapsed,
        warnings             = list(dict.fromkeys(warnings_list)),
        raw_vlm_response     = loop_out.get("raw_vlm_response"),
        annotated_image_path = annotated_path,
    ).model_dump()

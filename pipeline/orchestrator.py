"""
pipeline/orchestrator.py

The central entry point for the Thinking with Images pipeline.

Architecture:
  - Qwen2-VL-7B-Instruct receives the raw image first
  - VLM uses native multimodal vision to form a damage hypothesis
  - VLM calls CV tools (damage_detection, part_segmentation) as needed
  - VLM generates Python cost computation code
  - Code runs in sandbox.py
  - VLM synthesizes FinalDamageReport JSON

This module owns:
  - VLM model loading (lazy singleton, thread-safe)
  - The tool-calling agentic loop
  - Tool call JSON parsing from Qwen2-VL response format
  - Auto-approve threshold gate
  - FinalDamageReport construction

This module does NOT own:
  - Individual model inference (delegated to tool_registry.py)
  - Sandboxed execution (delegated to sandbox.py)
  - Multi-turn context management (delegated to context_manager.py)
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

from models.vlm_reasoning.tool_registry import TOOL_DEFINITIONS, get_tool_executor
from pipeline.schema import FinalDamageReport, DamagePartEntry, ToolCallRecord, DetectionWithBBox

logger = logging.getLogger(__name__)


class VLMGenerationTimeout(Exception):
    pass


@contextmanager
def _vlm_timeout(seconds: int):
    # signal.SIGALRM cannot be used in FastAPI's threadpool (non-main thread).
    # Per-call timeout is enforced at the job level (600s) and loop level (540s).
    yield


# ── Module-level singletons ──────────────────────────────────────────────────
_model = None
_processor = None
_tool_executor = None
_model_lock = threading.Lock()

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


CODEACT_SYSTEM_PROMPT = """You are a vehicle damage assessment agent.

═══════════════════════════════════════════════════
YOUR OUTPUT FORMAT — MANDATORY, NO EXCEPTIONS
═══════════════════════════════════════════════════
Every single response MUST be this exact JSON structure.
Do NOT output anything else. No explanations. No markdown.
No detection lists. Just this JSON object:

{
  "thought": "your reasoning here",
  "uncertainty": [],
  "actions": [
    {"name": "TOOL_NAME", "arguments": {}}
  ]
}

To terminate: use Terminate as the tool name.
To zoom in:   use zoom_region as the tool name.
To find part: use detect_part as the tool name.
To segment:   use segment_damage as the tool name.
═══════════════════════════════════════════════════

CONTEXT:
The YOLOv8 model has ALREADY run. Detections are shown to you.
Do NOT re-output the detection data. Use it to reason.

YOUR TASKS:
1. Look at the YOLO detections already provided
2. Verify each detection (class + part) is correct
3. For detections below confidence 0.50, call zoom_region
4. Assign severity: minor/moderate/severe
5. Call Terminate with final damage list

TERMINATION FORMAT:
{
  "thought": "Assessment complete",
  "uncertainty": [],
  "actions": [{"name": "Terminate", "arguments": {"damage_items": [
    {"damage_type": "dent", "part": "front_bumper",
     "severity": "severe", "confidence": 0.91}
  ]}}],
  "confidence": 0.91
}

Valid damage_type: dent scratch crack glass_shatter lamp_broken tire_flat
Valid part: front_bumper rear_bumper hood windshield rear_windshield
  front_left_door front_right_door rear_left_door rear_right_door
  left_fender right_fender trunk_lid roof_panel headlight taillight tire
Valid severity: minor moderate severe

NEVER output anything outside the JSON object. No markdown. No preamble.
Do NOT call run_damage_detection — YOLO has already run."""

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


def _load_models(config: dict) -> None:
    """
    Lazy-load Qwen2-VL-7B-Instruct. Thread-safe via _model_lock.
    Called on first request and on /health startup warmup.
    """
    global _model, _processor, _tool_executor

    if _model is not None and _processor is not None:
        return

    with _model_lock:
        if _model is not None and _processor is not None:
            return

        vlm_cfg = config.get("vlm", {})
        model_id = vlm_cfg.get("model_id", "Qwen/Qwen2-VL-7B-Instruct")
        device = vlm_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning(
                "MPS requested but not available. Falling back to CPU. "
                "Inference will be significantly slower."
            )
            device = "cpu"

        logger.info(f"Loading VLM: {model_id} on {device} ...")
        t0 = time.time()

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info

            _processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            _model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map=None,
                trust_remote_code=True
            )
            _model = _model.to(device)
            _model.eval()

        except Exception as e:
            logger.error(f"Failed to load VLM {model_id}: {e}")
            raise RuntimeError(f"VLM load failed: {e}") from e

        _tool_executor = get_tool_executor(config)

        elapsed = round(time.time() - t0, 1)
        logger.info(f"VLM loaded in {elapsed}s. Device: {device}")
        if torch.backends.mps.is_available():
            allocated = round(torch.mps.driver_allocated_memory() / 1e9, 2)
            logger.info(f"MPS memory after model load: {allocated}GB")


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


# DEPRECATED — no longer called by run(). Kept for reference.
def _run_cv_tools_eagerly(
    image_path: str,
    config: dict,
) -> Tuple[dict, dict, List[ToolCallRecord], List[str]]:
    """
    Run damage detection and part segmentation before the VLM loop.

    Returns:
        (damage_result, seg_result, tool_call_log, warnings)
    Both results are raw dicts from the tool dispatcher.
    Failures return {"error": ...} — never raises.
    """
    tool_call_log: List[ToolCallRecord] = []
    warnings: List[str] = []

    for tool_name in ("run_damage_detection", "run_part_segmentation"):
        logger.info(f"Pre-running CV tool: {tool_name}")
        t0 = time.time()
        result = _tool_executor(tool_name, {"image_path": image_path})
        elapsed = round(time.time() - t0, 3)

        tool_call_log.append(ToolCallRecord(
            tool=tool_name,
            args_summary=f"image_path={Path(image_path).name}",
            elapsed_s=elapsed,
            result_keys=list(result.keys()) if isinstance(result, dict) else [],
            success="error" not in result
        ))

        if "error" in result:
            warn = f"CV tool '{tool_name}' error: {result['error']}"
            logger.warning(warn)
            warnings.append(warn)

        if tool_name == "run_damage_detection":
            damage_result = result
        else:
            seg_result = result

    return damage_result, seg_result, tool_call_log, warnings


# DEPRECATED — no longer called by run(). Kept for reference.
def _build_initial_messages(
    image_path: str,
    claim_metadata: Optional[dict],
    damage_result: dict,
    seg_result: dict,
) -> list:
    """Construct the first user message with image, CV results, and claim context."""
    meta_text = (
        f"Claim metadata: {json.dumps(claim_metadata, indent=2)}"
        if claim_metadata
        else "No claim metadata provided."
    )

    damage_json = json.dumps(damage_result, indent=2, default=str)
    seg_json = json.dumps(seg_result, indent=2, default=str)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{Path(image_path).resolve()}"
                },
                {
                    "type": "text",
                    "text": (
                        f"{meta_text}\n\n"
                        f"DAMAGE_DETECTIONS (from YOLOv8):\n{damage_json}\n\n"
                        f"PART_SEGMENTS (from Grounding DINO + SAM2):\n{seg_json}\n\n"
                        f"Now cross-reference the bboxes, assign each damage to a part, "
                        f"call execute_cost_computation, and return the final JSON report."
                    )
                }
            ]
        }
    ]


def _call_vlm(
    messages: list,
    config: dict,
    is_final_turn: bool = False,
    tools=TOOL_DEFINITIONS,
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

    with torch.inference_mode():
        with _vlm_timeout(90):
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=vlm_cfg.get("temperature", 0.1),
                do_sample=vlm_cfg.get("do_sample", False),
            )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    result = _processor.decode(new_tokens, skip_special_tokens=False)

    del inputs, output_ids, new_tokens
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()

    return result


# DEPRECATED — no longer called by run(). Kept for reference.
def _extract_tool_call(response_text: str) -> Optional[dict]:
    """
    Parse a tool call from Qwen2-VL response text.

    Qwen2-VL wraps tool calls in <tool_call>...</tool_call> tags.
    Falls back to raw JSON detection if tags are absent.

    Returns:
        dict with 'name' and 'arguments' keys, or None if no tool call found
    """
    # Primary: tag-based extraction (Qwen2-VL native format)
    tag_pattern = r"<tool_call>(.*?)</tool_call>"
    match = re.search(tag_pattern, response_text, re.DOTALL)

    if match:
        raw = match.group(1).strip()
    else:
        # Fallback: look for a raw JSON object that looks like a tool call
        json_pattern = r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:.*?\}'
        match = re.search(json_pattern, response_text, re.DOTALL)
        if not match:
            return None
        raw = match.group(0)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"Tool call JSON parse failed: {e}. Raw: {raw[:200]}")
        return None

    if "name" not in parsed:
        logger.warning(f"Tool call JSON missing 'name' key: {parsed}")
        return None

    # Normalise arguments — sometimes returned as a JSON string instead of dict
    args = parsed.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse tool arguments as JSON: {args[:200]}")
            args = {}
    parsed["arguments"] = args

    return parsed


# DEPRECATED — no longer called by run(). Kept for reference.
def _extract_final_report(messages: list) -> dict:
    """
    Extract the structured JSON report from the last assistant message.

    Tries:
    1. ```json ... ``` fenced block
    2. Raw JSON object in response text
    3. Returns raw text under 'raw_vlm_response' if neither works
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue

        # Try fenced JSON block
        fenced = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try raw JSON object
        raw_json = re.search(r"\{.*\}", content, re.DOTALL)
        if raw_json:
            try:
                return json.loads(raw_json.group(0))
            except json.JSONDecodeError:
                pass

        # Give up — return raw text for debugging
        return {"raw_vlm_response": content}

    return {"error": "No assistant message found in conversation history"}


# DEPRECATED — no longer called by run(). Kept for reference.
def _run_tool_loop(
    messages: list,
    config: dict,
) -> Tuple[list, List[ToolCallRecord], List[str]]:
    """
    Agentic tool-calling loop.

    Continues until:
    - VLM produces a response with no tool call (final answer)
    - max_iterations is reached (logged as warning)

    Returns:
        (final_messages, tool_call_log, warnings)
    """
    vlm_cfg = config.get("vlm", {})
    max_iterations = vlm_cfg.get("max_iterations", 6)
    tool_call_log: List[ToolCallRecord] = []
    warnings: List[str] = []
    t_loop_start = time.time()

    for iteration in range(max_iterations):
        logger.info(f"Orchestrator loop — iteration {iteration + 1}/{max_iterations}")

        # Check total elapsed time and abort if over limit
        loop_elapsed = time.time() - t_loop_start
        if loop_elapsed > 540:  # 9 minutes hard cap for entire loop
            warnings.append(f"Tool loop aborted: exceeded 540s time limit at iteration {iteration + 1}")
            logger.warning(f"Tool loop timeout at iteration {iteration + 1} after {loop_elapsed:.0f}s")
            break

        is_final = (iteration == max_iterations - 1)
        t0 = time.time()

        try:
            response_text = _call_vlm(messages, config, is_final_turn=is_final)
        except VLMGenerationTimeout as e:
            warnings.append(f"VLM generate() timed out on iteration {iteration + 1}: {e}")
            logger.error(f"VLM generate() timed out: {e}")
            break
        vlm_elapsed = round(time.time() - t0, 2)
        logger.debug(f"VLM response ({vlm_elapsed}s): {response_text[:300]}...")

        tool_call = _extract_tool_call(response_text)

        if tool_call is None:
            # No tool call — VLM produced final answer
            messages.append({"role": "assistant", "content": response_text})
            logger.info(f"VLM produced final answer at iteration {iteration + 1}")
            break

        # Tool call found — dispatch it
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
        image_arg = tool_args.get("image_path", "")

        logger.info(f"VLM calls tool: {tool_name}")
        t1 = time.time()
        tool_result = _tool_executor(tool_name, tool_args)
        tool_elapsed = round(time.time() - t1, 3)

        # Record the call
        tool_call_log.append(ToolCallRecord(
            tool=tool_name,
            args_summary=(
                f"image_path={Path(image_arg).name}"
                if image_arg
                else str(list(tool_args.keys()))
            ),
            elapsed_s=tool_elapsed,
            result_keys=list(tool_result.keys()) if isinstance(tool_result, dict) else [],
            success="error" not in tool_result
        ))

        if "error" in tool_result:
            warn_msg = f"Tool '{tool_name}' returned error: {tool_result['error']}"
            logger.warning(warn_msg)
            warnings.append(warn_msg)

        # Append assistant tool call message and tool result to history
        messages.append({"role": "assistant", "content": response_text})
        messages.append({
            "role": "tool",
            "name": tool_name,
            "content": json.dumps(tool_result, default=str)
        })

    else:
        warn = f"Tool-calling loop reached max_iterations={max_iterations} without final answer"
        logger.warning(warn)
        warnings.append(warn)

    return messages, tool_call_log, warnings


# DEPRECATED — no longer called by run(). Kept for reference.
def _build_final_report(
    raw_report: dict,
    image_path: str,
    tool_call_log: List[ToolCallRecord],
    warnings: List[str],
    total_elapsed: float,
    approval_threshold: int,
    last_assistant_message: str,
    config: dict
) -> FinalDamageReport:
    """
    Validate and construct FinalDamageReport from VLM raw output.
    Handles missing keys gracefully — never raises.
    """
    raw_map = raw_report.get("damage_part_map", [])
    damage_part_map = []

    for entry in raw_map:
        try:
            damage_part_map.append(DamagePartEntry(
                damage=str(entry.get("damage", "unknown")),
                part=str(entry.get("part", "unknown")),
                severity=str(entry.get("severity", "minor")),
                cost_min=int(entry.get("cost_min", 0)),
                cost_max=int(entry.get("cost_max", 0)),
            ))
        except Exception as e:
            warnings.append(f"Skipped malformed damage_part_map entry: {entry}. Error: {e}")

    total_min = int(raw_report.get("total_min", sum(e.cost_min for e in damage_part_map)))
    total_max = int(raw_report.get("total_max", sum(e.cost_max for e in damage_part_map)))

    # Merge warnings from VLM response into our warnings list
    vlm_warnings = raw_report.get("warnings", [])
    if isinstance(vlm_warnings, list):
        warnings.extend([str(w) for w in vlm_warnings])

    # Auto-approve gate
    if not damage_part_map:
        approval_decision = "UNKNOWN"
        warnings.append("No damage-part mappings found — cannot compute approval decision")
    elif total_max < approval_threshold:
        approval_decision = "AUTO_APPROVED"
    else:
        approval_decision = "ESCALATE_TO_HUMAN"

    logger.info(
        f"Report built: {len(damage_part_map)} damage entries, "
        f"total INR {total_min}–{total_max}, decision={approval_decision}"
    )

    return FinalDamageReport(
        image_path=image_path,
        damage_part_map=damage_part_map,
        total_min=total_min,
        total_max=total_max,
        currency=raw_report.get("currency", "INR"),
        approval_decision=approval_decision,
        tool_call_log=tool_call_log,
        total_inference_s=total_elapsed,
        warnings=list(dict.fromkeys(warnings)),   # deduplicate, preserve order
        raw_vlm_response=last_assistant_message
    )


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
        if tool_calls_made == 0 and iteration == 0:
            return False, (
                "You must call at least one vision tool before terminating. "
                "Call zoom_region, detect_part, or segment_damage first."
            )
        conf = turn.confidence or 0.0
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
        if not items:
            return False, (
                "Terminate requires at least one damage_item. "
                "If no damage is visible, state confidence=1.0 and empty items with explicit reasoning."
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
        valid_tools = {"run_damage_detection", "zoom_region", "detect_part", "segment_damage", "Terminate"}
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


# DEPRECATED — replaced by _run_codeact_loop_with_yolo_context
def _run_codeact_loop_deprecated(
    image_path: str,
    config: dict,
    trajectory_steps: list,
) -> "tuple[list, list]":
    """
    Runs the CodeAct reasoning loop.
    Returns (damage_items, warnings).
    Appends TrajectoryStep objects to trajectory_steps (mutated in place).
    """
    from pipeline.schema import TrajectoryStep, CodeActAction

    max_iter = config["vlm"].get("max_iterations", 4)
    max_retries = config["vlm"].get("codeact_max_retries", 2)
    warnings = []
    tool_calls_made = 0

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{Path(image_path).resolve()}"},
                {"type": "text", "text": (
                    "Assess all vehicle damage visible in this image. "
                    "Follow the output format exactly."
                )},
            ],
        }
    ]

    t_loop = time.time()

    for iteration in range(max_iter):
        if time.time() - t_loop > 480:
            warnings.append(f"Loop timeout after {iteration} iterations")
            break

        turn = None
        for attempt in range(max_retries + 1):
            try:
                raw = _call_vlm(
                    messages=messages,
                    config=config,
                    tools=None,  # CodeAct uses structured JSON, not Qwen native tool-call format
                    max_new_tokens=config["vlm"].get("max_new_tokens_tool", 120),
                )
            except VLMGenerationTimeout:
                warnings.append(f"VLM timeout on iter {iteration} attempt {attempt}")
                raw = None
                break

            if raw is None:
                break

            turn, parse_err = _parse_codeact_turn(raw)
            if parse_err:
                if attempt < max_retries:
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text":
                            f"Your output was not valid JSON. Error: {parse_err}. "
                            f"Respond with only the JSON object, no other text."}]
                    })
                    turn = None
                    continue
                else:
                    warnings.append(f"Failed to parse CodeAct JSON after {max_retries} retries: {parse_err}")
                    turn = None
                    break

            valid, reject_reason = _enforce_turn_policy(turn, iteration, tool_calls_made)
            if not valid:
                if attempt < max_retries:
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text":
                            f"Policy violation: {reject_reason} Try again."}]
                    })
                    turn = None
                    continue
                else:
                    warnings.append(f"Policy enforcement failed after {max_retries} retries: {reject_reason}")
                    turn = None
                    break
            break  # valid turn

        if turn is None:
            break

        logger.info(f"[iter {iteration}] thought: {turn.thought[:120]}")
        if turn.uncertainty:
            logger.info(f"[iter {iteration}] uncertainty: {turn.uncertainty}")

        for action in turn.actions:
            t_action = time.time()

            if action.name == "Terminate":
                damage_items = action.arguments.get("damage_items", [])
                trajectory_steps.append(TrajectoryStep(
                    turn_index=iteration,
                    action=action,
                    observation_type="json",
                    observation_summary=f"Terminated with {len(damage_items)} damage items",
                    observation_data={"damage_items": damage_items},
                    elapsed_s=round(time.time() - t_action, 2),
                ))
                return damage_items, warnings

            result = _execute_codeact_tool(action, image_path, config)
            tool_calls_made += 1
            elapsed = round(time.time() - t_action, 2)

            trajectory_steps.append(TrajectoryStep(
                turn_index=iteration,
                action=action,
                observation_type=result["type"],
                observation_summary=result.get("summary", ""),
                observation_image_path=result.get("image_path"),
                observation_data=result.get("data"),
                elapsed_s=elapsed,
            ))

            if result["type"] == "image" and result.get("image_path"):
                img_p = result["image_path"]

                if action.name == "run_damage_detection":
                    dets = result.get("detections", [])
                    if dets:
                        det_lines = []
                        for i, d in enumerate(dets):
                            bbox = [int(v) for v in d.get("bbox", [])]
                            det_lines.append(
                                f"  [{i+1}] {d['class']} "
                                f"conf={d['confidence']:.2f} "
                                f"bbox={bbox}"
                            )
                        det_text = "YOLO detections:\n" + "\n".join(det_lines)
                    else:
                        det_text = "YOLO found no detections above the confidence threshold."

                    obs_text = (
                        f"Tool result: run_damage_detection\n"
                        f"{result.get('summary', '')}\n\n"
                        f"{det_text}\n\n"
                        f"The annotated image above shows numbered bounding boxes for each "
                        f"detection. Use the bbox coordinates above to call zoom_region or "
                        f"segment_damage on any regions you need to inspect more closely."
                    )
                else:
                    obs_text = (
                        f"Tool result for {action.name}: "
                        f"{result.get('summary', '')}. "
                        f"The image above shows the result. "
                        f"Continue your assessment based on what you now see."
                    )

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{Path(img_p).resolve()}"},
                        {"type": "text", "text": obs_text},
                    ],
                })
            elif result["type"] == "error":
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text":
                        f"Tool {action.name} failed: {result['error']}. "
                        f"Continue assessment with available information."}]
                })

    warnings.append(f"Loop ended without Terminate after {max_iter} iterations")
    return [], warnings


def _run_codeact_loop_with_yolo_context(
    image_path: str,
    annotated_path: str,
    yolo_detections: list,
    yolo_summary: str,
    config: dict,
    trajectory_steps: list,
) -> "tuple[list, list]":
    """
    CodeAct reasoning loop with YOLO results pre-injected into the first message.

    The VLM receives:
      - Original image (raw)
      - YOLO annotated image (bboxes drawn) — only if different from original
      - Structured detection list as text

    The VLM's job: verify detections, assign severity, zoom uncertain regions, Terminate.
    Returns (damage_items, warnings).
    """
    import time as _time
    from pathlib import Path as _Path
    from pipeline.schema import TrajectoryStep, CodeActAction  # noqa: F401

    max_iter  = config["vlm"].get("max_iterations", 2)
    max_retry = config["vlm"].get("codeact_max_retries", 2)
    warnings  = []
    tool_calls = 0

    # Resize images before passing to VLM to reduce visual token count
    max_dim = config["vlm"].get("image_max_dim", 640)
    vlm_original_path  = _resize_for_vlm(image_path, max_dim)
    vlm_annotated_path = _resize_for_vlm(annotated_path, max_dim)

    if yolo_detections:
        det_lines = []
        for i, d in enumerate(yolo_detections):
            bbox = [int(v) for v in d.get("bbox", [])]
            det_lines.append(
                f"  Box {i+1}: {d['class'].replace('_', ' ')} "
                f"({int(d['confidence'] * 100)}% confidence) "
                f"at location {bbox}"
            )
        det_text = "YOLO already detected these damage regions:\n" + "\n".join(det_lines)
        task_text = (
            "IMPORTANT: Do NOT copy or repeat the detection data below. "
            "Only output the JSON format shown in your system instructions.\n\n"
            f"{det_text}\n\n"
            "Review each detection. Verify class and part are correct. "
            "Assign severity. Call Terminate with your verified damage list."
        )
    else:
        task_text = (
            "YOLO found no detections above the confidence threshold. "
            "Examine the image carefully and identify any visible vehicle damage yourself. "
            "Look for: deformation, scratches, cracks, broken lamps, flat tyres, "
            "shattered glass. If you see damage, include it in your Terminate call. "
            "If the vehicle is genuinely undamaged, call Terminate with an empty list "
            "and confidence >= 0.90."
        )

    first_content = [
        {"type": "image", "image": f"file://{_Path(vlm_original_path).resolve()}"},
    ]
    if vlm_annotated_path != vlm_original_path and _Path(vlm_annotated_path).exists():
        first_content.append(
            {"type": "image", "image": f"file://{_Path(vlm_annotated_path).resolve()}"}
        )
    first_content.append({"type": "text", "text": task_text})

    messages = [{"role": "user", "content": first_content}]
    t_loop = _time.time()

    for iteration in range(max_iter):
        if _time.time() - t_loop > 480:
            warnings.append(f"Loop timeout after {iteration} iterations")
            break

        turn = None
        for attempt in range(max_retry + 1):
            try:
                with _vlm_timeout(90):
                    raw = _call_vlm(
                        messages       = messages,
                        max_new_tokens = config["vlm"].get("max_new_tokens_tool", 120),
                        config         = config,
                        tools          = None,
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
                else:
                    warnings.append(f"JSON parse failed after {max_retry} retries: {parse_err}")
                    break

            valid, reject_reason = _enforce_turn_policy(turn, iteration, tool_calls)
            if not valid:
                if attempt < max_retry:
                    messages.append({"role": "user", "content": [{
                        "type": "text",
                        "text": f"Policy violation: {reject_reason} Try again."
                    }]})
                    turn = None
                    continue
                else:
                    warnings.append(f"Policy failed after {max_retry} retries: {reject_reason}")
                    break
            break

        if turn is None:
            break

        logger.info(f"[iter {iteration}] thought: {turn.thought[:100]}")

        for action in turn.actions:
            t_action = _time.time()

            if action.name == "Terminate":
                damage_items = action.arguments.get("damage_items", [])
                trajectory_steps.append(TrajectoryStep(
                    turn_index          = iteration,
                    action              = action,
                    observation_type    = "json",
                    observation_summary = f"Terminated with {len(damage_items)} items",
                    observation_data    = {"damage_items": damage_items},
                    elapsed_s           = round(_time.time() - t_action, 2),
                ))
                logger.info(f"VLM terminated: {len(damage_items)} damage items")
                return damage_items, warnings

            if action.name == "run_damage_detection":
                messages.append({"role": "user", "content": [{
                    "type": "text",
                    "text": (
                        "run_damage_detection has already run and results are "
                        "in your context. Do not call it again. "
                        "Use zoom_region or detect_part if you need more detail, "
                        "or call Terminate if you are confident."
                    )
                }]})
                continue

            result   = _execute_codeact_tool(action, image_path, config)
            tool_calls += 1
            elapsed  = round(_time.time() - t_action, 2)

            trajectory_steps.append(TrajectoryStep(
                turn_index             = iteration,
                action                 = action,
                observation_type       = result["type"],
                observation_summary    = result.get("summary", ""),
                observation_image_path = result.get("image_path"),
                observation_data       = result.get("data"),
                elapsed_s              = elapsed,
            ))

            if result["type"] == "image" and result.get("image_path"):
                img_p = result["image_path"]
                messages.append({"role": "user", "content": [
                    {"type": "image",
                     "image": f"file://{_Path(img_p).resolve()}"},
                    {"type": "text",
                     "text": (
                         f"Tool result for {action.name}: "
                         f"{result.get('summary', '')}. "
                         "The image above shows the result. "
                         "Continue your assessment."
                     )},
                ]})
            elif result["type"] == "error":
                messages.append({"role": "user", "content": [{
                    "type": "text",
                    "text": (
                        f"Tool {action.name} failed: {result['error']}. "
                        "Continue with available information."
                    )
                }]})

    warnings.append(
        f"VLM loop ended without Terminate after {max_iter} iterations"
    )
    return [], warnings


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

    Stage 1 [always]:  YOLO runs unconditionally → detections + annotated image
    Stage 2 [always]:  VLM CodeAct loop with YOLO context pre-injected
    Stage 3 [fallback]: if VLM returns empty → use YOLO detections directly
    Stage 4 [always]:  cost lookup → FinalDamageReport

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

    # ── Stage 1: YOLO always runs first ──────────────────────────────────────
    logger.info(f"Stage 1: Running YOLO on {image_path}")
    yolo_result = _run_yolo_eagerly(image_path, config)

    if not yolo_result["success"]:
        warnings_list.append(f"YOLO failed: {yolo_result['error']}")

    yolo_detections = yolo_result["detections"]
    annotated_path  = yolo_result["annotated_image_path"]
    logger.info(f"YOLO: {yolo_result['summary']}")

    # Build DetectionWithBBox from YOLO results — preserves spatial bbox information
    from pipeline.schema import DetectionWithBBox as _DetWithBBox
    from models.vlm_reasoning.cost_db import lookup_cost as _lc

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

    # ── Stage 2: VLM CodeAct loop with YOLO context ──────────────────────────
    _load_models(config)

    logger.info("Stage 2: Starting VLM CodeAct loop")
    vlm_damage_items, loop_warnings = _run_codeact_loop_with_yolo_context(
        image_path      = image_path,
        annotated_path  = annotated_path,
        yolo_detections = yolo_detections,
        yolo_summary    = yolo_result["summary"],
        config          = config,
        trajectory_steps= trajectory_steps,
    )
    warnings_list.extend(loop_warnings)

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

    # ── Stage 3: Fallback if VLM returned nothing ─────────────────────────────
    if not vlm_damage_items:
        if yolo_detections:
            warnings_list.append(
                "VLM produced no output — falling back to YOLO detections directly"
            )
            logger.warning("VLM fallback: using YOLO detections with heuristic severity")
            costed: List[DamagePartEntry] = _yolo_to_damage_entries(yolo_detections, config)
        else:
            warnings_list.append(
                "EMPTY_DAMAGE_MAP: Both YOLO and VLM found no damage. "
                "Escalating to human review."
            )
            costed = []
    else:
        # VLM succeeded — apply cost lookup to its output
        costed = []
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

    # ── Stage 4: Costs and approval ───────────────────────────────────────────
    total_min = sum(e.cost_min for e in costed)
    total_max = sum(e.cost_max for e in costed)

    threshold = config.get("approval", {}).get("auto_approve_threshold_inr", 50000)
    vlm_verified = not any(
        "falling back to YOLO" in w or "VLM produced no output" in w
        for w in warnings_list
    )
    if not costed:
        approval = "ESCALATE_TO_HUMAN"
    elif not vlm_verified:
        approval = "ESCALATE_TO_HUMAN"
        warnings_list.append(
            "YOLO_ONLY_RESULT: VLM did not verify detections. "
            "Escalating for human review regardless of cost."
        )
    elif total_max <= threshold:
        approval = "AUTO_APPROVED"
    else:
        approval = "ESCALATE_TO_HUMAN"

    # ── Tool call log ─────────────────────────────────────────────────────────
    tool_log = [
        ToolCallRecord(
            tool         = "run_damage_detection",
            args_summary = f"conf={config.get('damage_detection', {}).get('confidence_threshold', 0.15)}",
            elapsed_s    = 0.0,
            result_keys  = ["detections", "total_detections", "annotated_image_path"],
            success      = yolo_result["success"],
        )
    ] + [
        ToolCallRecord(
            tool         = step.action.name,
            args_summary = str(step.action.arguments)[:80],
            elapsed_s    = step.elapsed_s,
            result_keys  = list(step.observation_data.keys()) if step.observation_data else [],
            success      = step.observation_type != "error",
        )
        for step in trajectory_steps
    ]

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
        raw_vlm_response     = None,
        annotated_image_path = annotated_path,
    ).model_dump()

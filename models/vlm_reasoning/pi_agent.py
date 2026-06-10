"""
models/vlm_reasoning/pi_agent.py

Pi-style recursive agentic loop for vehicle damage assessment.

The VLM (qwen3.5:9b via Ollama) is the SOLE perception intelligence.
All detection, part identification, severity reasoning, and depth estimation
are done by the VLM directly seeing images. No YOLO, no SAM2, no GroundingDINO
in the hot-path decision loop.

Tool dispatch:
  run_damage_detection   → VLM vision pass → JSON detections + annotated image
  zoom_region            → PIL crop/upscale → image fed back to VLM
  detect_part            → VLM targeted question → annotated image fed back
  segment_damage         → SAM2 (if weights present) or PIL mask fallback
  estimate_depth         → PIL luminance gradient heatmap → fed back to VLM
  execute_cost_computation → Monty sandbox (unchanged) → cost dict as text
  Terminate              → extract final damage_items, exit loop

Recursion contract:
  max_iterations = 6   (hard cap, no infinite loops)
  max_retries    = 2   (JSON parse / policy violation recovery per iteration)
  Wall timeout   = 480s per full run

The PiAgent class is imported and called from pipeline/orchestrator.py::run().
It has NO imports from orchestrator.py — circular dependency is avoided.
"""

import json
import logging
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from models.vlm_reasoning.ollama_client import chat as ollama_chat, encode_image
from models.vlm_reasoning.sandbox import execute_sandboxed
from pipeline.schema import CodeActTurn, CodeActAction, TrajectoryStep

logger = logging.getLogger(__name__)


# ── Valid vocabulary ──────────────────────────────────────────────────────────

VALID_DAMAGE_CLASSES = frozenset({
    "dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat",
})
VALID_PARTS = frozenset({
    "front_bumper", "rear_bumper", "hood", "windshield", "rear_windshield",
    "front_left_door", "front_right_door", "rear_left_door", "rear_right_door",
    "left_fender", "right_fender", "trunk_lid", "roof_panel",
    "headlight", "taillight", "tire",
})
VALID_SEVERITY = frozenset({"minor", "moderate", "severe"})

_CANONICAL_TOOLS = frozenset({
    "run_damage_detection", "zoom_region", "detect_part",
    "segment_damage", "estimate_depth", "execute_cost_computation", "Terminate",
})
_CANONICAL_TOOL_LOOKUP = {t.lower().replace(" ", "_"): t for t in _CANONICAL_TOOLS}

# PIL RGB colours per damage class (used when drawing detection boxes)
CLASS_COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    "dent":          (55,  138, 221),
    "scratch":       (29,  158, 117),
    "crack":         (186, 117,  23),
    "glass_shatter": (212,  83, 126),
    "lamp_broken":   (216,  90,  48),
    "tire_flat":     (128, 135, 136),
}
_DEFAULT_COLOR_RGB: Tuple[int, int, int] = (128, 128, 128)


# ── System prompt (CODEACT mode) ──────────────────────────────────────────────

CODEACT_SYSTEM_PROMPT = """\
You are an expert vehicle damage assessment AI.
You are the SOLE intelligence in this system — your vision is the only perception tool.
No other CV models exist. You see the raw vehicle image and decide which tools to call.

══════════════════════════════════════════════════════════
MANDATORY OUTPUT FORMAT — EVERY SINGLE RESPONSE MUST BE:
══════════════════════════════════════════════════════════
Exactly ONE JSON object. No markdown. No preamble. No text outside this JSON.

{
  "thought": "what you see and what you plan to do next",
  "uncertainty": ["any open questions; empty list [] if none"],
  "actions": [
    {"name": "TOOL_NAME", "arguments": {...}}
  ],
  "confidence": 0.0
}
══════════════════════════════════════════════════════════

TOOLS YOU CAN CALL:

• run_damage_detection
  Use your vision to locate and classify ALL damage in the image.
  Returns a structured JSON detection list + an annotated image with coloured boxes.
  arguments: {"reason": "why you are calling this"}

• zoom_region
  Crop and magnify a specific image region for close inspection.
  Use when a damage region is too small or unclear to classify from the full image.
  arguments: {"bbox": [x1, y1, x2, y2], "reason": "what you cannot determine"}
  Note: bbox values are PIXEL coordinates in the original image.

• detect_part
  Ask your vision to pinpoint a specific vehicle part.
  Use when you cannot confidently identify which part is damaged.
  arguments: {"part_query": "e.g. left headlight", "reason": "why you are unsure"}

• segment_damage
  Generate a precise segmentation mask over a damage region.
  Use when you need exact damage boundaries to judge severity (minor vs severe).
  arguments: {"bbox": [x1, y1, x2, y2], "reason": "what severity question this resolves"}

• estimate_depth
  Generate a luminance-based deformation heatmap of the entire image.
  RED = high-gradient regions (pushed-in / deformed panels).
  BLUE = smooth flat surfaces.
  Use when judging structural vs surface-only damage.
  arguments: {"reason": "why depth information is needed"}

• execute_cost_computation
  Execute Python code in a SECURE Monty sandbox to compute repair costs.
  Available variable: COST_DB[damage_class][part_label] = (cost_min_INR, cost_max_INR)
  Unknown (damage, part) pairs → use (3000, 8000) as fallback.

  COPY THIS TEMPLATE EXACTLY — only edit the `items` list:
    items = [
        {"damage": "dent",    "part": "front_bumper", "severity": "moderate"},
        {"damage": "scratch", "part": "hood",         "severity": "minor"},
    ]
    result = {"damage_part_map": [], "total_min": 0, "total_max": 0, "currency": "INR"}
    for d in items:
        lo, hi = COST_DB.get(d["damage"], {}).get(d["part"], (3000, 8000))
        result["damage_part_map"].append({**d, "cost_min": lo, "cost_max": hi})
        result["total_min"] += lo
        result["total_max"] += hi

  arguments: {"code": "<python code as a single string>"}

• Terminate
  End the assessment loop and return your final findings.
  ONLY call when ALL of these are true:
    1. You have called execute_cost_computation and it returned a valid result.
    2. confidence >= 0.70
    3. uncertainty == []
  arguments: {
    "damage_items": [
      {"damage_type": "dent", "part": "front_bumper", "severity": "moderate", "confidence": 0.85}
    ]
  }

RECOMMENDED WORKFLOW:
  1. Look at the raw image → form initial hypothesis in "thought"
  2. Call run_damage_detection → read the annotated image + detection list
  3. For regions where confidence < 0.60 or severity is unclear:
       → call zoom_region or segment_damage
  4. If the affected part is uncertain → call detect_part
  5. If panel may be structurally deformed (not just surface) → call estimate_depth
  6. Call execute_cost_computation with ALL confirmed damage items (use the template)
  7. Call Terminate with the final verified damage_items list

Valid damage_type: dent | scratch | crack | glass_shatter | lamp_broken | tire_flat
Valid part:        front_bumper | rear_bumper | hood | windshield | rear_windshield |
                   front_left_door | front_right_door | rear_left_door | rear_right_door |
                   left_fender | right_fender | trunk_lid | roof_panel |
                   headlight | taillight | tire
Valid severity:    minor | moderate | severe

OUTPUT ONLY THE JSON OBJECT. NO MARKDOWN. NO TEXT BEFORE OR AFTER IT.\
"""

# ── Sub-task prompts ──────────────────────────────────────────────────────────

DAMAGE_DETECTION_PROMPT = """\
Inspect this vehicle image carefully.
Identify ALL visible damage regions — be thorough, do not miss any.

Respond with ONLY this JSON (no markdown, no preamble):
{
  "detections": [
    {
      "class":       "dent",
      "confidence":  0.85,
      "bbox_pct":    [10, 20, 40, 60],
      "part":        "front_bumper",
      "severity":    "moderate",
      "description": "crumple deformation on lower bumper"
    }
  ]
}

Field rules:
  class      → one of: dent | scratch | crack | glass_shatter | lamp_broken | tire_flat
  confidence → your visual certainty 0.0–1.0
  bbox_pct   → [x1, y1, x2, y2] as image PERCENTAGE coordinates 0–100
  part       → one of: front_bumper | rear_bumper | hood | windshield | rear_windshield |
               front_left_door | front_right_door | rear_left_door | rear_right_door |
               left_fender | right_fender | trunk_lid | roof_panel |
               headlight | taillight | tire
  severity   → minor (surface only) | moderate (panel deformation, repair needed) |
               severe (structural damage, replacement needed)
  description → one-sentence visual description of what you see

If no damage is visible: {"detections": []}"""

_PART_DETECTION_PROMPT = """\
Inspect this vehicle image. Locate: "{part_query}"

Find where "{part_query}" is in this image and output ONLY this JSON:
{{
  "part_label": "{part_query}",
  "found": true,
  "bbox_pct": [x1, y1, x2, y2],
  "condition": "undamaged | damaged | not_visible",
  "description": "brief description of the part and its visible condition"
}}

bbox_pct values are image PERCENTAGES (0–100).
If the part is not visible: {{"part_label": "{part_query}", "found": false, "condition": "not_visible", "description": "not clearly visible in this view"}}\
"""


# ── JSON parsing helpers ──────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by Qwen3.5 thinking mode."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json_objects(text: str) -> List[str]:
    """
    Bracket-matching extractor. Returns all top-level JSON objects found in text.
    Handles nested structures correctly (more robust than greedy regex).
    """
    candidates: List[str] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            start = i
            in_string = False
            escape_next = False
            for j in range(i, len(text)):
                c = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if c == "\\" and in_string:
                    escape_next = True
                    continue
                if c == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            candidates.append(text[start : j + 1])
                            i = j + 1
                            break
            else:
                i += 1
        else:
            i += 1
    return candidates


def _repair_json(text: str) -> Optional[str]:
    """Attempt common JSON repairs: trailing commas, single-quoted keys."""
    try:
        repaired = re.sub(r",\s*([}\]])", r"\1", text)
        repaired = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', repaired)
        return repaired
    except Exception:
        return None


def _parse_codeact_turn(raw: str) -> Tuple[Optional[CodeActTurn], str]:
    """
    Parse VLM raw output into a CodeActTurn using three recovery strategies:
      1. Bracket-matching on clean text
      2. JSON repair then bracket-matching
      3. Regex extraction of 'actions' array + 'thought' field

    Returns (turn, error_msg). error_msg is '' on success.
    """
    if not raw or not raw.strip():
        return None, "Empty response from VLM"

    # Strip thinking tokens, markdown fences
    clean = _strip_thinking(raw).strip()
    clean = re.sub(r"^```json\s*", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"^```\s*$", "", clean, flags=re.MULTILINE).strip()

    # Strategy 1: bracket-matching on clean text
    for candidate in _extract_json_objects(clean):
        try:
            data = json.loads(candidate)
            if not isinstance(data.get("actions"), list):
                continue
            return CodeActTurn(**data), ""
        except Exception:
            continue

    # Strategy 2: JSON repair then retry
    repaired = _repair_json(clean)
    if repaired:
        for candidate in _extract_json_objects(repaired):
            try:
                data = json.loads(candidate)
                if not isinstance(data.get("actions"), list):
                    continue
                return CodeActTurn(**data), ""
            except Exception:
                continue

    # Strategy 3: regex extraction of actions array + thought field
    actions_m = re.search(r'"actions"\s*:\s*(\[.*?\])', clean, re.DOTALL)
    thought_m = re.search(r'"thought"\s*:\s*"([^"]*)"', clean)
    if actions_m:
        try:
            actions = json.loads(actions_m.group(1))
            thought = thought_m.group(1) if thought_m else "recovered"
            turn = CodeActTurn(
                thought=thought,
                uncertainty=[],
                actions=[CodeActAction(**a) for a in actions],
            )
            return turn, ""
        except Exception:
            pass

    return None, (
        f"JSON parse failed after all recovery strategies. "
        f"Raw (first 300 chars): {raw[:300]}"
    )


def _canonicalize_action_names(turn: CodeActTurn) -> None:
    """Rewrite each action.name to its canonical form in-place (case/space-insensitive)."""
    for action in turn.actions:
        key = (action.name or "").strip().lower().replace(" ", "_")
        if key in _CANONICAL_TOOL_LOOKUP:
            action.name = _CANONICAL_TOOL_LOOKUP[key]


def _enforce_turn_policy(
    turn: CodeActTurn,
    iteration: int,
    tool_calls_made: int,
    cost_result: Optional[dict],
) -> Tuple[bool, str]:
    """
    Validate a parsed CodeActTurn before executing it.

    Returns (is_valid, rejection_reason).
    Rejection reason is injected back into the conversation as a corrective user message.
    """
    actions = turn.actions
    is_terminate = any(a.name == "Terminate" for a in actions)

    if is_terminate:
        conf = turn.confidence or 0.0

        # Allow high-confidence pure-visual assessment without prior tool call
        if tool_calls_made == 0 and iteration == 0 and conf < 0.70:
            return False, (
                "You terminated immediately with low confidence and no tool calls. "
                "Call run_damage_detection first to gather evidence, then Terminate."
            )
        if conf < 0.70:
            return False, (
                f"Confidence {conf:.2f} is below the required 0.70. "
                f"Resolve these uncertainties first: {turn.uncertainty}"
            )
        if turn.uncertainty:
            return False, (
                f"You have {len(turn.uncertainty)} unresolved uncertainties: "
                f"{turn.uncertainty}. "
                "Call zoom_region, detect_part, or segment_damage to resolve them."
            )
        if cost_result is None:
            return False, (
                "You must call execute_cost_computation before Terminate. "
                "Use the mandatory template from the system prompt."
            )

        term_action = next(a for a in actions if a.name == "Terminate")
        items = term_action.arguments.get("damage_items", [])

        if not items and conf < 0.90:
            return False, (
                "Terminate requires at least one damage_item, or confidence >= 0.90 "
                "if you are certain the vehicle has no damage."
            )
        for item in items:
            if item.get("damage_type") not in VALID_DAMAGE_CLASSES:
                return False, (
                    f"Invalid damage_type: '{item.get('damage_type')}'. "
                    f"Must be one of: {sorted(VALID_DAMAGE_CLASSES)}"
                )
            if item.get("part") not in VALID_PARTS:
                return False, (
                    f"Invalid part: '{item.get('part')}'. "
                    f"Must be one of: {sorted(VALID_PARTS)}"
                )
            if item.get("severity") not in VALID_SEVERITY:
                return False, (
                    f"Invalid severity: '{item.get('severity')}'. "
                    "Must be: minor | moderate | severe"
                )

    else:
        if not actions:
            return False, (
                "Your response has no actions. "
                "Call a tool or call Terminate with your findings."
            )
        for a in actions:
            if a.name not in _CANONICAL_TOOLS:
                return False, (
                    f"Unknown tool: '{a.name}'. "
                    f"Valid tools: {sorted(_CANONICAL_TOOLS)}"
                )

    return True, ""


# ── PIL vision helpers (pure, no external CV deps) ────────────────────────────

def _resize_for_vlm(image_path: str, max_dim: int = 640) -> str:
    """
    Resize image so its longest dimension <= max_dim before passing to the VLM.
    Saves resized copy to /tmp. Returns path (original if already small enough).
    Tool calls (zoom, segment, YOLO) always receive the ORIGINAL full-res image.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if max(w, h) <= max_dim:
                return image_path
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = img.resize((new_w, new_h), Image.LANCZOS)
            out = f"/tmp/vlm_r_{uuid.uuid4().hex[:8]}.jpg"
            resized.save(out, quality=90, optimize=True)
            logger.info(f"Resized for VLM: {w}x{h} → {new_w}x{new_h}")
            return out
    except Exception as e:
        logger.warning(f"Resize failed: {e} — using original image")
        return image_path


def _zoom_region(image_path: str, bbox: List[float], padding: float = 0.12) -> str:
    """
    PIL crop + upscale to min 320px, max 512px.
    Adds padding around the requested bbox.
    Returns path to the cropped image.
    """
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size

    x1, y1, x2, y2 = [float(v) for v in bbox]
    pw = (x2 - x1) * padding
    ph = (y2 - y1) * padding
    x1 = max(0.0, x1 - pw)
    y1 = max(0.0, y1 - ph)
    x2 = min(float(w), x2 + pw)
    y2 = min(float(h), y2 + ph)

    crop = img_rgb.crop((x1, y1, x2, y2))
    min_d = min(crop.size)
    if min_d < 320:
        scale = 320 / min_d
        crop = crop.resize(
            (int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS
        )
    max_d = max(crop.size)
    if max_d > 512:
        scale = 512 / max_d
        crop = crop.resize(
            (int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS
        )

    out = f"/tmp/zoom_{uuid.uuid4().hex[:8]}.jpg"
    crop.save(out, quality=95)
    logger.info(f"Zoomed region {[int(v) for v in (x1, y1, x2, y2)]} → {crop.size}")
    return out


def _estimate_depth_map(image_path: str) -> str:
    """
    Generate a pseudo-depth / panel-deformation heatmap using PIL edge detection.

    Algorithm:
      1. Convert to greyscale
      2. Apply FIND_EDGES filter (detects panel boundary gradients)
      3. Gaussian blur to smooth noise
      4. Enhance contrast
      5. Colorize: 0 (smooth) → blue, 128 (mid) → green, 255 (edges/deformed) → red
      6. Blend 60% heatmap over original image

    Bright red = high-gradient regions → likely structural deformation.
    Dark blue  = smooth flat surfaces → surface-only or undamaged.

    Returns path to the blended heatmap image.
    """
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size

    grey    = img_rgb.convert("L")
    edges   = grey.filter(ImageFilter.FIND_EDGES)
    blurred = edges.filter(ImageFilter.GaussianBlur(radius=4))
    enhanced = ImageEnhance.Contrast(blurred).enhance(2.5)

    # False-colour mapping: low → blue, mid → green, high → red
    heatmap_pix = []
    for v in enhanced.getdata():
        if v < 128:
            ratio = v / 128.0
            heatmap_pix.append((0, int(200 * ratio), int(200 * (1.0 - ratio))))
        else:
            ratio = (v - 128) / 128.0
            heatmap_pix.append((int(220 * ratio), int(200 * (1.0 - ratio)), 0))

    heatmap = Image.new("RGB", (w, h))
    heatmap.putdata(heatmap_pix)

    # Blend 60% heatmap over original
    blended = Image.blend(img_rgb, heatmap, alpha=0.6)

    # Header label
    draw = ImageDraw.Draw(blended)
    draw.rectangle([(0, 0), (w, 22)], fill=(20, 20, 20))
    draw.text(
        (6, 4),
        "DEFORMATION MAP  ■ RED = pushed-in / deformed   ■ BLUE = flat / smooth",
        fill=(255, 255, 255),
    )

    out = f"/tmp/depth_{uuid.uuid4().hex[:8]}.jpg"
    blended.save(out, quality=90)
    logger.info(f"Depth map saved: {out}")
    return out


def _draw_detections_on_image(
    image_path: str,
    detections: List[dict],
    img_w: int,
    img_h: int,
) -> str:
    """
    Draw VLM detection boxes on the image using PIL.

    Accepts detections with either:
      bbox_pct [x1,y1,x2,y2] in 0–100 percentage coordinates, or
      bbox     [x1,y1,x2,y2] in pixel coordinates.

    Saves annotated image to data/uploads/vlm_annotated/.
    """
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")

    draw = ImageDraw.Draw(img_rgb)

    for i, det in enumerate(detections):
        cls   = det.get("class", "dent")
        conf  = float(det.get("confidence", 0.5))
        color = CLASS_COLORS_RGB.get(cls, _DEFAULT_COLOR_RGB)

        # Resolve bbox to pixel coordinates
        if "bbox_pct" in det:
            bpct = det["bbox_pct"]
            x1 = int(bpct[0] / 100 * img_w)
            y1 = int(bpct[1] / 100 * img_h)
            x2 = int(bpct[2] / 100 * img_w)
            y2 = int(bpct[3] / 100 * img_h)
        elif "bbox" in det:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        else:
            continue

        # Clamp to image bounds
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))

        # Solid box for confident detections; corner markers for low-confidence
        if conf >= 0.40:
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        else:
            c = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            for cx, cy, dx, dy in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1),
            ]:
                draw.line([(cx, cy), (cx + dx * c, cy)], fill=color, width=2)
                draw.line([(cx, cy), (cx, cy + dy * c)], fill=color, width=2)

        # Numbered badge circle
        badge_r = 12
        bx = min(x1 + badge_r + 2, img_w - badge_r - 2)
        by = max(y1 - badge_r - 2, badge_r + 2)
        draw.ellipse(
            [(bx - badge_r, by - badge_r), (bx + badge_r, by + badge_r)],
            fill=color, outline=(255, 255, 255), width=1,
        )
        num_str = str(i + 1)
        draw.text((bx - len(num_str) * 3, by - 6), num_str, fill=(255, 255, 255))

        # Label background + text
        label = f"{cls} {conf:.0%}"
        lx, ly = x1, max(y1 - 18, 0)
        draw.rectangle(
            [(lx, ly), (lx + len(label) * 6 + 4, ly + 14)], fill=color
        )
        draw.text((lx + 2, ly + 2), label, fill=(255, 255, 255))

    out_dir = Path("data/uploads/vlm_annotated")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = str(out_dir / f"vlm_{uuid.uuid4().hex[:8]}.jpg")
    img_rgb.save(out, quality=90)
    return out


def _segment_damage(image_path: str, bbox: List[float], config: dict) -> str:
    """
    Generate a segmentation mask over a damage region.
    Attempts SAM2 (if weights are present); falls back to PIL bbox outline.
    """
    try:
        from shared.sam_mask import generate_masked_image
        from pipeline.schema import DetectionWithBBox

        det = DetectionWithBBox(
            index=1, bbox=[float(v) for v in bbox],
            damage="damage", part="unknown",
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
        logger.warning(f"SAM2 segment failed: {e} — falling back to PIL bbox outline")
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")

        draw = ImageDraw.Draw(img_rgb)
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            draw.rectangle([(x1, y1), (x2, y2)], outline=(186, 117, 23), width=3)
            draw.text((x1 + 4, y1 + 4), "DAMAGE REGION", fill=(255, 255, 255))

        out = f"/tmp/seg_fallback_{uuid.uuid4().hex[:8]}.jpg"
        img_rgb.save(out, quality=90)
        return out


# ── PiAgent ───────────────────────────────────────────────────────────────────

class PiAgent:
    """
    Recursive Pi-style agentic loop using Ollama qwen3.5:9b as the VLM brain.

    The agent iteratively:
      1. Sends the current message history (with images) to the VLM
      2. Parses the JSON turn response
      3. Executes the requested tool actions
      4. Feeds observation images / JSON text back to the VLM
      5. Repeats until the VLM calls Terminate or max_iterations is reached

    All VLM calls go through Ollama. All cost computation goes through
    the Monty sandbox (execute_cost_computation). No YOLO, no GroundingDINO,
    no SAM2 in the reasoning path (SAM2 is optionally used for segment_damage).
    """

    def __init__(self, config: dict) -> None:
        self.config      = config
        self.vlm_cfg     = config.get("vlm", {})
        self.model       = self.vlm_cfg.get("model_id", "qwen3.5:9b")
        self.base_url    = self.vlm_cfg.get("ollama_base_url", "http://localhost:11434")
        self.temperature = float(self.vlm_cfg.get("temperature", 0.1))
        self.max_iter    = int(self.vlm_cfg.get("max_iterations", 6))
        self.max_retry   = int(self.vlm_cfg.get("codeact_max_retries", 2))
        self.max_dim     = int(self.vlm_cfg.get("image_max_dim", 640))
        self.engine      = self.vlm_cfg.get("sandbox_engine", "monty")
        self.max_tok_tool  = int(self.vlm_cfg.get("max_new_tokens_tool", 512))
        self.max_tok_final = int(self.vlm_cfg.get("max_new_tokens_final", 1024))

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, image_path: str, trajectory_steps: list) -> dict:
        """
        Execute the recursive CodeAct loop on the given image.

        Returns a dict matching the contract used by pipeline/orchestrator.py::run():
        {
          "damage_items":         list,          # from Terminate action
          "cost_result":          dict | None,   # from execute_cost_computation
          "yolo_detections":      list,          # populated by run_damage_detection
          "annotated_image_path": str | None,    # annotated image from VLM detection
          "tool_calls":           int,
          "warnings":             list[str],
          "raw_vlm_response":     str | None,
        }
        """
        warnings: List[str] = []
        tool_calls          = 0
        cost_result: Optional[dict] = None
        yolo_detections: List[dict] = []   # renamed for compat; contains VLM detections
        annotated_path: Optional[str] = None
        last_raw: Optional[str] = None

        # Resize for VLM; tool helpers use the ORIGINAL full-res image path
        vlm_img_path = _resize_for_vlm(image_path, self.max_dim)
        initial_b64  = encode_image(vlm_img_path)

        logger.info(
            f"PiAgent START | model={self.model} | max_iter={self.max_iter} "
            f"| image={Path(image_path).name} | vlm_img={Path(vlm_img_path).name}"
        )

        # Initial conversation: system prompt + raw vehicle image
        messages: List[dict] = [
            {"role": "system", "content": CODEACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Assess all visible damage on this vehicle. "
                    "Call your tools to inspect the damage, compute repair cost with "
                    "execute_cost_computation, then call Terminate. "
                    "Output ONLY the JSON object."
                ),
                "images": [initial_b64],
            },
        ]

        t_loop = time.time()

        for iteration in range(self.max_iter):

            # Hard wall-clock cap (prevents runaway on very slow machines)
            if time.time() - t_loop > 480:
                warnings.append(f"PiAgent: wall-clock timeout after {iteration} iterations")
                break

            logger.info(
                f"--- PiAgent iter {iteration + 1}/{self.max_iter} "
                f"| tool_calls={tool_calls} "
                f"| elapsed={time.time() - t_loop:.1f}s ---"
            )

            turn: Optional[CodeActTurn] = None

            # Inner retry loop: JSON parse / policy violation recovery
            for attempt in range(self.max_retry + 1):
                try:
                    raw = self._call_model(messages, max_tokens=self.max_tok_tool)
                except Exception as e:
                    warnings.append(f"Ollama call error iter={iteration} attempt={attempt}: {e}")
                    raw = None
                    break

                last_raw = raw
                turn, parse_err = _parse_codeact_turn(raw)

                if parse_err:
                    if attempt < self.max_retry:
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Your output was not valid JSON. Error: {parse_err}. "
                                "Respond with ONLY the JSON object — no other text."
                            ),
                        })
                        turn = None
                        continue
                    warnings.append(
                        f"JSON parse failed after {self.max_retry} retries: {parse_err}"
                    )
                    break

                _canonicalize_action_names(turn)
                valid, reject_reason = _enforce_turn_policy(
                    turn, iteration, tool_calls, cost_result
                )

                if not valid:
                    if attempt < self.max_retry:
                        messages.append({
                            "role": "user",
                            "content": f"Policy violation: {reject_reason} Try again.",
                        })
                        turn = None
                        continue
                    warnings.append(
                        f"Policy failed after {self.max_retry} retries: {reject_reason}"
                    )
                    break

                break  # valid turn parsed successfully

            if turn is None:
                break

            logger.info(f"[iter {iteration}] thought: {turn.thought[:150]}")
            if turn.uncertainty:
                logger.info(f"[iter {iteration}] uncertainty: {turn.uncertainty}")

            # Append assistant reasoning to conversation history
            messages.append({"role": "assistant", "content": last_raw})

            # Execute all actions in this turn
            for action in turn.actions:
                t_action = time.time()

                if action.name == "Terminate":
                    damage_items = action.arguments.get("damage_items", [])
                    trajectory_steps.append(TrajectoryStep(
                        turn_index=iteration,
                        action=action,
                        observation_type="json",
                        observation_summary=f"Terminated with {len(damage_items)} item(s)",
                        observation_data={"damage_items": damage_items},
                        elapsed_s=round(time.time() - t_action, 3),
                    ))
                    logger.info(
                        f"PiAgent: Terminate called — {len(damage_items)} damage item(s)"
                    )
                    return {
                        "damage_items":         damage_items,
                        "cost_result":          cost_result,
                        "yolo_detections":      yolo_detections,
                        "annotated_image_path": annotated_path,
                        "tool_calls":           tool_calls,
                        "warnings":             warnings,
                        "raw_vlm_response":     last_raw,
                    }

                logger.info(
                    f"[iter {iteration}] tool={action.name} "
                    f"args={str(action.arguments)[:150]}"
                )
                result  = self._dispatch_action(action, image_path)
                tool_calls += 1
                elapsed = round(time.time() - t_action, 3)

                logger.info(
                    f"[iter {iteration}] {action.name} → type={result.get('type')} "
                    f"in {elapsed}s | {result.get('summary', '')[:120]}"
                )

                trajectory_steps.append(TrajectoryStep(
                    turn_index=iteration,
                    action=action,
                    observation_type=result["type"],
                    observation_summary=result.get("summary", ""),
                    observation_image_path=result.get("image_path"),
                    observation_data=result.get("data"),
                    elapsed_s=elapsed,
                ))

                # Side-channel capture (for dashboard annotation UI)
                if action.name == "run_damage_detection" and result["type"] == "image":
                    yolo_detections = result.get("detections", yolo_detections)
                    annotated_path  = result.get("image_path", annotated_path)
                if action.name == "execute_cost_computation" and result["type"] == "json":
                    cost_result = result.get("data", cost_result)

                # Append observation to conversation (image or text)
                self._append_observation(messages, action, result)

        warnings.append(
            f"PiAgent: loop ended without Terminate after {self.max_iter} iterations"
        )
        return {
            "damage_items":         [],
            "cost_result":          cost_result,
            "yolo_detections":      yolo_detections,
            "annotated_image_path": annotated_path,
            "tool_calls":           tool_calls,
            "warnings":             warnings,
            "raw_vlm_response":     last_raw,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_model(self, messages: List[dict], max_tokens: int = 512) -> str:
        """Forward messages to Ollama and return the raw content string."""
        return ollama_chat(
            messages=messages,
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=max_tokens,
        )

    def _append_observation(
        self, messages: List[dict], action: CodeActAction, result: dict
    ) -> None:
        """
        Add the tool result back to the conversation as a user message.
        Image results are base64-encoded; JSON results are formatted as text.
        """
        if result["type"] == "image" and result.get("image_path"):
            obs_path = _resize_for_vlm(result["image_path"], self.max_dim)
            obs_b64  = encode_image(obs_path)

            if action.name == "run_damage_detection":
                dets = result.get("detections", [])
                if dets:
                    det_lines = "\n".join(
                        f"  Box {i+1}: {d.get('class','?')} "
                        f"({int(d.get('confidence', 0)*100)}%) "
                        f"→ part={d.get('part','?')} severity={d.get('severity','?')}"
                        for i, d in enumerate(dets)
                    )
                    obs_text = (
                        f"run_damage_detection result: {result.get('summary','')}\n"
                        f"{det_lines}\n\n"
                        "The annotated image shows numbered detection boxes. "
                        "Inspect uncertain areas with zoom_region or segment_damage. "
                        "Then call execute_cost_computation and Terminate."
                    )
                else:
                    obs_text = (
                        f"run_damage_detection result: {result.get('summary','')}\n"
                        "No damage detected above threshold. "
                        "Visually inspect the image. If you see damage, include it in Terminate. "
                        "Still call execute_cost_computation before Terminate."
                    )
            else:
                obs_text = (
                    f"{action.name} result: {result.get('summary', '')}. "
                    "The image above shows the tool output. Continue your assessment."
                )

            messages.append({
                "role": "user",
                "content": obs_text,
                "images": [obs_b64],
            })

        elif result["type"] == "json":
            messages.append({
                "role": "user",
                "content": (
                    f"{action.name} result: {result.get('summary', '')}.\n"
                    f"Data: {json.dumps(result.get('data', {}), default=str)}\n\n"
                    "Verify the cost data above. If correct, call Terminate with "
                    "the matching damage_items list."
                ),
            })

        elif result["type"] == "error":
            messages.append({
                "role": "user",
                "content": (
                    f"Tool {action.name} failed: {result['error']}. "
                    "Continue with available information or try a different tool."
                ),
            })

    def _dispatch_action(self, action: CodeActAction, image_path: str) -> dict:
        """
        Route an action to its tool implementation.
        Never raises — returns {"type": "error", ...} on any failure.
        """
        name = action.name
        args = action.arguments

        try:
            if name == "run_damage_detection":
                return self._vlm_damage_detection(image_path)

            elif name == "zoom_region":
                bbox = args.get("bbox", [])
                if len(bbox) != 4:
                    return {
                        "type": "error",
                        "error": "zoom_region requires bbox [x1,y1,x2,y2]",
                        "summary": "missing bbox argument",
                    }
                out = _zoom_region(image_path, bbox)
                return {
                    "type": "image",
                    "image_path": out,
                    "summary": f"Zoomed into region {[int(v) for v in bbox]}",
                }

            elif name == "detect_part":
                query = str(args.get("part_query", "")).strip()
                if not query:
                    return {
                        "type": "error",
                        "error": "detect_part requires part_query argument",
                        "summary": "missing part_query",
                    }
                out = self._vlm_detect_part(image_path, query)
                return {
                    "type": "image",
                    "image_path": out,
                    "summary": f"Part detection: '{query}'",
                }

            elif name == "segment_damage":
                bbox = args.get("bbox", [])
                if len(bbox) != 4:
                    return {
                        "type": "error",
                        "error": "segment_damage requires bbox [x1,y1,x2,y2]",
                        "summary": "missing bbox argument",
                    }
                out = _segment_damage(image_path, bbox, self.config)
                return {
                    "type": "image",
                    "image_path": out,
                    "summary": f"Segmented damage region {[int(v) for v in bbox]}",
                }

            elif name == "estimate_depth":
                out = _estimate_depth_map(image_path)
                return {
                    "type": "image",
                    "image_path": out,
                    "summary": "Deformation heatmap: red=pushed-in, blue=flat",
                }

            elif name == "execute_cost_computation":
                code = str(args.get("code", "")).strip()
                if not code:
                    return {
                        "type": "error",
                        "error": "execute_cost_computation requires non-empty 'code' argument",
                        "summary": "no code provided",
                    }
                sandbox_out = execute_sandboxed(code, engine=self.engine)
                if "error" in sandbox_out:
                    return {
                        "type": "error",
                        "error": sandbox_out["error"],
                        "summary": f"Sandbox rejected code: {sandbox_out['error']}",
                    }
                cr = sandbox_out["result"]
                n  = len(cr.get("damage_part_map", []))
                return {
                    "type": "json",
                    "data": cr,
                    "summary": (
                        f"Cost computed: {n} item(s), "
                        f"total INR {cr.get('total_min',0):,}–{cr.get('total_max',0):,}"
                    ),
                }

            else:
                return {
                    "type": "error",
                    "error": f"Unknown tool: '{name}'",
                    "summary": "unknown tool name",
                }

        except Exception as exc:
            logger.error(f"_dispatch_action({name}) raised: {exc}", exc_info=True)
            return {
                "type": "error",
                "error": str(exc),
                "summary": f"{name} raised an unexpected exception: {exc}",
            }

    # ── VLM-powered tool implementations ─────────────────────────────────────

    def _vlm_damage_detection(self, image_path: str) -> dict:
        """
        VLM-only damage detection forward pass.

        Encodes the vehicle image, asks Ollama to locate and classify all damage,
        parses the JSON response, converts bbox_pct to pixel coords, draws
        coloured numbered boxes, and returns the annotated image + structured list.
        """
        vlm_path = _resize_for_vlm(image_path, self.max_dim)
        b64      = encode_image(vlm_path)

        messages = [
            {
                "role": "user",
                "content": DAMAGE_DETECTION_PROMPT,
                "images": [b64],
            }
        ]

        try:
            raw = self._call_model(messages, max_tokens=900)
        except Exception as e:
            logger.error(f"VLM damage detection call failed: {e}")
            return {
                "type": "error",
                "error": str(e),
                "summary": f"VLM damage detection failed: {e}",
            }

        # Parse thinking + JSON
        clean = _strip_thinking(raw).strip()
        clean = re.sub(r"^```json\s*", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"^```\s*$", "", clean, flags=re.MULTILINE).strip()

        detections: List[dict] = []
        for candidate in _extract_json_objects(clean):
            try:
                parsed = json.loads(candidate)
                if "detections" in parsed and isinstance(parsed["detections"], list):
                    detections = parsed["detections"]
                    break
            except Exception:
                continue

        # Filter to valid damage classes; warn on unknowns
        valid_dets = []
        for d in detections:
            if d.get("class") not in VALID_DAMAGE_CLASSES:
                logger.warning(
                    f"VLM returned invalid class '{d.get('class')}' — skipping"
                )
                continue
            if d.get("part") not in VALID_PARTS:
                logger.warning(
                    f"VLM returned invalid part '{d.get('part')}' — defaulting to front_bumper"
                )
                d["part"] = "front_bumper"
            if d.get("severity") not in VALID_SEVERITY:
                d["severity"] = "minor"
            valid_dets.append(d)

        # Get original image dimensions for bbox_pct → pixel conversion
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except Exception:
            img_w, img_h = 640, 480

        # Build structured detections with pixel bboxes
        structured: List[dict] = []
        for d in valid_dets:
            bpct = d.get("bbox_pct", [5, 5, 95, 95])
            bbox_px = [
                int(bpct[0] / 100.0 * img_w),
                int(bpct[1] / 100.0 * img_h),
                int(bpct[2] / 100.0 * img_w),
                int(bpct[3] / 100.0 * img_h),
            ]
            structured.append({
                "class":       d.get("class", "dent"),
                "confidence":  float(d.get("confidence", 0.5)),
                "bbox":        bbox_px,
                "bbox_pct":    bpct,
                "part":        d.get("part", "front_bumper"),
                "severity":    d.get("severity", "minor"),
                "description": d.get("description", ""),
            })

        # Draw annotated image with PIL
        try:
            annotated = _draw_detections_on_image(image_path, structured, img_w, img_h)
        except Exception as e:
            logger.warning(f"Draw detections failed: {e} — using original")
            annotated = image_path

        if structured:
            counts  = Counter(d["class"] for d in structured)
            summary = (
                f"VLM detected {len(structured)} damage region(s): "
                + ", ".join(f"{v}× {k}" for k, v in sorted(counts.items()))
            )
        else:
            summary = "VLM found no damage (visual reasoning continues)"

        logger.info(f"VLM damage detection: {summary}")
        return {
            "type":             "image",
            "image_path":       annotated,
            "detections":       structured,
            "total_detections": len(structured),
            "summary":          summary,
        }

    def _vlm_detect_part(self, image_path: str, part_query: str) -> str:
        """
        VLM-based part localization.

        Asks Ollama to identify the specified part's location in the image,
        draws a blue bounding box + label if found, and returns the path to
        the annotated image.
        """
        vlm_path = _resize_for_vlm(image_path, self.max_dim)
        b64      = encode_image(vlm_path)

        prompt   = _PART_DETECTION_PROMPT.format(part_query=part_query)
        messages = [{"role": "user", "content": prompt, "images": [b64]}]

        try:
            raw = self._call_model(messages, max_tokens=400)
        except Exception as e:
            logger.warning(f"VLM detect_part call failed: {e}")
            return image_path  # fall back to original image

        clean = _strip_thinking(raw).strip()
        clean = re.sub(r"^```json\s*", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"^```\s*$", "", clean, flags=re.MULTILINE).strip()

        # Parse the part location JSON
        part_info: Optional[dict] = None
        for candidate in _extract_json_objects(clean):
            try:
                parsed = json.loads(candidate)
                if "part_label" in parsed:
                    part_info = parsed
                    break
            except Exception:
                continue

        # Draw annotation on the ORIGINAL (full-res) image
        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                img_w, img_h = img_rgb.size
        except Exception as e:
            logger.warning(f"Cannot open image for part detection annotation: {e}")
            return image_path

        draw = ImageDraw.Draw(img_rgb)

        if part_info and part_info.get("found", True) and "bbox_pct" in part_info:
            bpct = part_info["bbox_pct"]
            x1 = int(bpct[0] / 100 * img_w)
            y1 = int(bpct[1] / 100 * img_h)
            x2 = int(bpct[2] / 100 * img_w)
            y2 = int(bpct[3] / 100 * img_h)
            color = (55, 138, 221)  # Blue
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
            label = f"✓ {part_query}"
            lw = len(label) * 7 + 6
            draw.rectangle([(x1, max(y1 - 20, 0)), (x1 + lw, max(y1 - 2, 0))], fill=color)
            draw.text((x1 + 3, max(y1 - 18, 2)), label, fill=(255, 255, 255))
            cond = part_info.get("condition", "")
            if cond:
                draw.text((x1 + 3, y1 + 3), cond, fill=(255, 255, 255))
        else:
            # Part not found — header overlay
            draw.rectangle([(0, 0), (img_w, 26)], fill=(40, 40, 40))
            draw.text(
                (6, 5),
                f"Part '{part_query}' not clearly visible in this image view",
                fill=(255, 220, 80),
            )

        out = f"/tmp/part_{uuid.uuid4().hex[:8]}.jpg"
        img_rgb.save(out, quality=90)
        logger.info(f"Part detection '{part_query}': {part_info}")
        return out

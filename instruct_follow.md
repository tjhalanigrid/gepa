# instruct_follow.md
# Claude Code — Build Instruction: Thinking with Images (Qwen VLM + Tool Calling)

---

## Read Before Writing a Single Line

1. Read `CLAUDE.md` at the repo root in full. It is the authoritative context document.
2. Run `find . -type f -name "*.py" | head -60` to audit what already exists.
3. Run `cat pipeline/schema.py` if it exists — append to it, never overwrite.
4. Run `cat models/damage_detection/__init__.py` to confirm the existing `run()` signature.
5. Do not generate any file that already exists unless explicitly told to rewrite it.
   Check first, then act.

---

## Your Mission

Build the complete "Thinking with Images" VLM orchestration layer on top of the
existing CV models. The existing `damage_detection` and `part_segmentation` `run()`
functions are already written and frozen. You are wiring them into a tool-calling
loop driven by Qwen2-VL-7B-Instruct.

When you are done, a single call to `pipeline/orchestrator.py::run(image_path, config)`
must:
- Load and invoke Qwen2-VL-7B-Instruct
- Have the VLM call CV model tools as needed
- Execute sandboxed cost computation code the VLM generates
- Return a `FinalDamageReport` dict

---

## Phase 0 — Environment Verification

Run these commands first. Fix any failure before proceeding to Phase 1.

```bash
python --version
# Must output Python 3.10.x — if not, stop and report the version mismatch

pip show torch | grep Version
# Must be installed. If not: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip show transformers | grep Version
# If missing: pip install transformers==4.45.0

pip install qwen-vl-utils accelerate einops
# Required for Qwen2-VL. Install unconditionally.

pip install pydantic pyyaml python-dotenv
# Required for schema + config. Install unconditionally.

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
# Log the output. If CUDA is False, all config device values must be set to "cpu".
```

After running, report:
- Python version
- CUDA available: yes/no
- GPU name if available
- Which packages were newly installed

---

## Phase 1 — Config Setup

### 1.1 Create or update `configs/global_config.yaml`

If the file already exists, read it first, then add any missing top-level keys.
Do not overwrite existing keys with different values — flag the conflict instead.

Write the following structure, preserving any existing content:

```yaml
vlm:
  model_id: "Qwen/Qwen2-VL-7B-Instruct"
  device: "cuda"                        # change to "cpu" if Phase 0 reported no CUDA
  max_new_tokens_tool: 512
  max_new_tokens_final: 1024
  temperature: 0.1
  do_sample: false
  max_iterations: 6

approval:
  auto_approve_threshold_inr: 50000

damage_detection:
  weights_path: "models/damage_detection/best.pt"
  confidence_threshold: 0.25
  device: "cuda"                        # same CUDA rule applies

part_segmentation:
  grounding_dino:
    config_path: "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path: "weights/groundingdino_swint_ogc.pth"
    box_threshold: 0.30
    text_threshold: 0.25
    text_prompt: "front bumper . rear bumper . hood . car door . windshield . rear windshield . fender . trunk lid . roof panel . headlight . taillight"
  sam2:
    config: "sam2_hiera_base_plus.yaml"
    weights_path: "weights/sam2.1_hiera_base_plus.pt"
    device: "cuda"
  postprocess:
    min_mask_area_px: 500
    allow_duplicate_labels: false
  output:
    save_annotated_images: true
    annotated_output_dir: "outputs/part_segmentation/"

plate_rc_detection:
  weights_path: "models/plate_rc_detection/plate.pt"
  device: "cuda"
  enabled: false                        # set to true only when model is ready

database:
  url: "postgresql://localhost:5432/veh_dmg_db"

storage:
  image_upload_dir: "data/uploads/"
  output_dir: "outputs/"

logging:
  level: "INFO"
  format: "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
```

---

## Phase 2 — Schema Updates

### 2.1 Append to `pipeline/schema.py`

Read the file first. Append exactly these classes at the bottom.
If any class name already exists in the file, skip that class and log a note.
Never import * — use explicit imports.

```python
# ── Appended by instruct_follow.md ──────────────────────────────────────────

from typing import Optional, List, Dict


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
    total_min: int
    total_max: int
    currency: str = "INR"
    approval_decision: str             # AUTO_APPROVED | ESCALATE_TO_HUMAN | UNKNOWN
    tool_call_log: List[ToolCallRecord]
    total_inference_s: float
    warnings: List[str]
    raw_vlm_response: Optional[str]    # last assistant message — keep for MVP debugging
```

---

## Phase 3 — VLM Reasoning Module

Create the directory `models/vlm_reasoning/` if it does not exist.
Create `models/vlm_reasoning/__init__.py` as empty.

### 3.1 Create `models/vlm_reasoning/tool_registry.py`

Full implementation. No stubs.

```python
"""
Tool definitions and dispatcher for the Qwen2-VL tool-calling loop.

Each tool wraps an existing model run() function or a utility (sandbox).
The VLM sees TOOL_DEFINITIONS and decides which to call.
The dispatcher routes the call to the real implementation.

Do not add tools here for models that are not yet implemented.
An undefined tool return is worse than a missing tool.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# ── Tool Schemas (sent to Qwen2-VL as function definitions) ─────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "run_damage_detection",
            "description": (
                "Runs YOLOv8 damage detection on the vehicle image. "
                "Returns bounding boxes, damage class labels "
                "(dent, scratch, crack, glass_shatter, lamp_broken, tire_flat), "
                "and confidence scores. "
                "Call this first on every image. "
                "Confidence is lower on dents, scratches, and cracks — "
                "do not treat low-confidence detections as definitive."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to the vehicle image file."
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_part_segmentation",
            "description": (
                "Runs Grounding DINO + SAM2 part segmentation on the vehicle image. "
                "Returns part labels (front_bumper, hood, car door, windshield, etc.) "
                "with bounding boxes and mask quality scores. "
                "Call this when you need to map a damage location to a specific part. "
                "If this returns empty results, use your visual assessment of part "
                "locations relative to the car's orientation in the image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to the vehicle image file."
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_cost_computation",
            "description": (
                "Executes a Python code snippet in a sandboxed environment to compute "
                "repair cost estimates. "
                "Your code has access to COST_DB: a dict of "
                "COST_DB[damage_class][part_label] = (cost_min_inr, cost_max_inr). "
                "Your code MUST assign a dict to a variable named 'result' with keys: "
                "damage_part_map (list of dicts), total_min (int), total_max (int), "
                "currency (str, use 'INR'). "
                "Only math, json, statistics, decimal are importable. "
                "No file I/O, no network calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Valid Python 3.10 code. Must assign final output to "
                            "'result' as a dict. Has access to COST_DB."
                        )
                    }
                },
                "required": ["code"]
            }
        }
    }
]


def _get_damage_detection_runner(config: dict) -> Callable:
    """Import and partially apply damage_detection.run with its config slice."""
    try:
        from models.damage_detection import run as _run
        damage_cfg = config.get("damage_detection", {})

        def _call(image_path: str) -> dict:
            return _run(image_path, damage_cfg)

        return _call
    except ImportError as e:
        raise ImportError(
            f"models/damage_detection/__init__.py must expose run(). Got: {e}"
        )


def _get_part_segmentation_runner(config: dict) -> Callable:
    """Import and partially apply part_segmentation.run with its config slice."""
    try:
        from models.part_segmentation import run as _run
        seg_cfg = config.get("part_segmentation", {})

        def _call(image_path: str) -> dict:
            return _run(image_path, seg_cfg)

        return _call
    except ImportError as e:
        raise ImportError(
            f"models/part_segmentation/__init__.py must expose run(). Got: {e}"
        )


def _get_plate_runner(config: dict) -> Callable | None:
    """
    Import plate_rc_detection runner only if enabled in config.
    Returns None if disabled — dispatcher will handle gracefully.
    """
    if not config.get("plate_rc_detection", {}).get("enabled", False):
        return None
    try:
        from models.plate_rc_detection import run as _run
        plate_cfg = config.get("plate_rc_detection", {})

        def _call(image_path: str) -> dict:
            return _run(image_path, plate_cfg)

        return _call
    except ImportError:
        logger.warning(
            "plate_rc_detection enabled in config but module not importable. "
            "Tool will return error response."
        )
        return None


def get_tool_executor(config: dict) -> Callable:
    """
    Returns a dispatcher function that routes tool_name → real implementation.

    Args:
        config: Full pipeline config dict (loaded from global_config.yaml)

    Returns:
        dispatch(tool_name: str, tool_args: dict) -> dict
    """
    from models.vlm_reasoning.sandbox import execute_sandboxed

    damage_runner = _get_damage_detection_runner(config)
    seg_runner = _get_part_segmentation_runner(config)
    plate_runner = _get_plate_runner(config)

    def dispatch(tool_name: str, tool_args: dict) -> dict:
        """
        Routes a tool call to its implementation.
        Never raises — returns {"error": str} on any failure.
        """
        t0 = time.time()
        try:
            if tool_name == "run_damage_detection":
                image_path = tool_args.get("image_path")
                if not image_path:
                    return {"error": "run_damage_detection requires image_path argument"}
                if not Path(image_path).exists():
                    return {"error": f"Image not found: {image_path}"}
                result = damage_runner(image_path)

            elif tool_name == "run_part_segmentation":
                image_path = tool_args.get("image_path")
                if not image_path:
                    return {"error": "run_part_segmentation requires image_path argument"}
                if not Path(image_path).exists():
                    return {"error": f"Image not found: {image_path}"}
                result = seg_runner(image_path)

            elif tool_name == "run_plate_detection":
                if plate_runner is None:
                    return {"error": "Plate detection is not enabled. Set plate_rc_detection.enabled: true in config."}
                image_path = tool_args.get("image_path")
                if not image_path:
                    return {"error": "run_plate_detection requires image_path argument"}
                result = plate_runner(image_path)

            elif tool_name == "execute_cost_computation":
                code = tool_args.get("code")
                if not code or not code.strip():
                    return {"error": "execute_cost_computation requires non-empty code argument"}
                result = execute_sandboxed(code)

            else:
                return {"error": f"Unknown tool: '{tool_name}'. Valid tools: run_damage_detection, run_part_segmentation, run_plate_detection, execute_cost_computation"}

            elapsed = round(time.time() - t0, 3)
            logger.info(f"Tool '{tool_name}' completed in {elapsed}s")
            return result

        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            logger.error(f"Tool '{tool_name}' raised unexpected exception after {elapsed}s: {e}", exc_info=True)
            return {"error": f"Tool execution failed: {str(e)}"}

    return dispatch
```

---

### 3.2 Create `models/vlm_reasoning/sandbox.py`

Full implementation. No stubs.

```python
"""
Sandboxed CodeAct execution environment.

The VLM (Qwen2-VL) generates Python code to compute repair costs.
This module executes that code in a restricted namespace with:
  - AST validation before exec (blocks unsafe nodes)
  - Whitelisted builtins only
  - Whitelisted imports only (math, json, statistics, decimal)
  - Hard 10-second execution timeout via signal.alarm()
  - Required output: code must set 'result' as a dict

COST_DB is injected into the execution namespace.
The VLM's generated code reads from it directly.
Update COST_DB as domain knowledge improves — it is the only pricing source.
"""

import ast
import logging
import signal
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Pricing database — all values in INR ────────────────────────────────────
# Structure: COST_DB[damage_class][part_label] = (min_cost, max_cost)
# Part labels must match those returned by part_segmentation tool exactly.

COST_DB: dict[str, dict[str, tuple[int, int]]] = {
    "dent": {
        "front bumper":      (8000,  25000),
        "rear bumper":       (8000,  25000),
        "hood":              (12000, 35000),
        "front_left_door":   (10000, 30000),
        "front_right_door":  (10000, 30000),
        "rear_left_door":    (10000, 30000),
        "rear_right_door":   (10000, 30000),
        "left fender":       (8000,  20000),
        "right fender":      (8000,  20000),
        "roof panel":        (15000, 45000),
        "trunk lid":         (10000, 28000),
    },
    "scratch": {
        "front bumper":      (3000,  8000),
        "rear bumper":       (3000,  8000),
        "hood":              (4000,  10000),
        "front_left_door":   (3500,  9000),
        "front_right_door":  (3500,  9000),
        "rear_left_door":    (3500,  9000),
        "rear_right_door":   (3500,  9000),
        "left fender":       (3000,  7000),
        "right fender":      (3000,  7000),
        "roof panel":        (5000,  12000),
        "trunk lid":         (3500,  9000),
    },
    "crack": {
        "windshield":        (15000, 40000),
        "rear windshield":   (12000, 35000),
        "front bumper":      (5000,  15000),
        "rear bumper":       (5000,  15000),
    },
    "glass_shatter": {
        "windshield":        (20000, 55000),
        "rear windshield":   (15000, 45000),
        "headlight":         (8000,  20000),
        "taillight":         (5000,  15000),
        "front_left_door":   (10000, 25000),
        "front_right_door":  (10000, 25000),
        "rear_left_door":    (10000, 25000),
        "rear_right_door":   (10000, 25000),
    },
    "lamp_broken": {
        "headlight":         (10000, 28000),
        "taillight":         (6000,  18000),
    },
    "tire_flat": {
        "tire":              (4000,  12000),
    },
}

# ── Allowed identifiers in generated code ────────────────────────────────────

_ALLOWED_IMPORTS = frozenset({"math", "json", "statistics", "decimal"})

_ALLOWED_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum,
    "round": round, "len": len, "range": range,
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "zip": zip, "enumerate": enumerate, "sorted": sorted,
    "isinstance": isinstance, "type": type,
    "True": True, "False": False, "None": None,
    "print": lambda *a, **kw: None,   # silenced — no stdout in sandbox
}

_BLOCKED_NAMES = frozenset({
    "open", "exec", "eval", "__import__", "compile",
    "getattr", "setattr", "delattr", "hasattr",
    "globals", "locals", "vars", "dir",
    "input", "breakpoint", "exit", "quit",
    "__builtins__", "__spec__", "__loader__",
})


class SandboxViolation(Exception):
    """Raised when generated code fails AST safety check."""
    pass


def _validate_ast(code: str) -> None:
    """
    Walk the AST of generated code and reject unsafe constructs.

    Raises:
        SyntaxError: if code cannot be parsed
        SandboxViolation: if code contains disallowed nodes
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise SyntaxError(f"Generated code has syntax error: {e}") from e

    for node in ast.walk(tree):
        # Block all imports except whitelist
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _ALLOWED_IMPORTS:
                    raise SandboxViolation(f"Import not allowed: '{alias.name}'")

        if isinstance(node, ast.ImportFrom):
            if node.module not in _ALLOWED_IMPORTS:
                raise SandboxViolation(f"Import not allowed: 'from {node.module}'")

        # Block calls to dangerous builtins by name
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in _BLOCKED_NAMES:
                    raise SandboxViolation(f"Blocked builtin: '{node.func.id}'")
            # Block attribute access that could escape sandbox (e.g. ().__class__.__bases__)
            if isinstance(node.func, ast.Attribute):
                if node.func.attr.startswith("__"):
                    raise SandboxViolation(f"Dunder attribute access not allowed: '{node.func.attr}'")

        # Block dunder attribute access via ast.Attribute nodes (not just in calls)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise SandboxViolation(f"Dunder attribute access not allowed: '{node.attr}'")


def execute_sandboxed(code: str, timeout_seconds: int = 10) -> dict:
    """
    Execute VLM-generated Python in a restricted namespace.

    The code must assign its output to a variable named 'result' as a dict.
    COST_DB is available in the namespace for pricing lookups.

    Args:
        code: Python 3.10 code string generated by the VLM
        timeout_seconds: Hard execution timeout (default 10s)

    Returns:
        {"result": dict} on success
        {"error": str} on any failure (validation, timeout, runtime)
    """
    if not code or not code.strip():
        return {"error": "Empty code string received"}

    logger.debug(f"Sandbox received code ({len(code)} chars):\n{code[:300]}...")

    # AST validation before any execution
    try:
        _validate_ast(code)
    except (SyntaxError, SandboxViolation) as e:
        logger.warning(f"Sandbox AST check failed: {e}")
        return {"error": f"Code safety check failed: {str(e)}"}

    # Build restricted execution namespace
    namespace: dict[str, Any] = {
        "__builtins__": _ALLOWED_BUILTINS,
        "COST_DB": COST_DB,
        "result": None,
    }

    # Inject allowed standard library modules
    import math, json, statistics, decimal
    namespace.update({
        "math": math,
        "json": json,
        "statistics": statistics,
        "decimal": decimal,
    })

    # Timeout handler
    def _timeout_handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Sandboxed execution exceeded {timeout_seconds}s limit")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    t0 = time.time()
    try:
        compiled = compile(code, "<vlm_generated_cost_code>", "exec")
        exec(compiled, namespace)
        signal.alarm(0)

        elapsed = round(time.time() - t0, 3)
        result = namespace.get("result")

        if result is None:
            logger.warning("Sandbox: code ran but 'result' was never set")
            return {"error": "Code executed but did not assign to 'result'"}

        if not isinstance(result, dict):
            logger.warning(f"Sandbox: 'result' is {type(result).__name__}, expected dict")
            return {"error": f"'result' must be a dict, got {type(result).__name__}"}

        # Validate required keys
        required_keys = {"damage_part_map", "total_min", "total_max", "currency"}
        missing = required_keys - set(result.keys())
        if missing:
            return {"error": f"'result' dict is missing required keys: {missing}"}

        logger.info(f"Sandbox execution succeeded in {elapsed}s. Result keys: {list(result.keys())}")
        return {"result": result}

    except TimeoutError as e:
        logger.warning(f"Sandbox timeout: {e}")
        return {"error": str(e)}
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        logger.warning(f"Sandbox runtime error after {elapsed}s: {e}")
        return {"error": f"Runtime error in generated code: {str(e)}"}
    finally:
        signal.alarm(0)
```

---

## Phase 4 — Pipeline Orchestrator (Full Rewrite)

**Read the existing `pipeline/orchestrator.py` first.**
If it exists and contains a sequential pipeline call chain, replace it entirely.
If it does not exist, create it.

```python
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

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Optional

import torch
import yaml
from PIL import Image

from models.vlm_reasoning.tool_registry import TOOL_DEFINITIONS, get_tool_executor
from pipeline.schema import FinalDamageReport, DamagePartEntry, ToolCallRecord

logger = logging.getLogger(__name__)

# ── Module-level singletons ──────────────────────────────────────────────────
_model = None
_processor = None
_tool_executor = None
_model_lock = threading.Lock()

# ── System prompt ────────────────────────────────────────────────────────────
# Update in CLAUDE.md when this changes so changes are tracked.

SYSTEM_PROMPT = """You are an expert vehicle damage assessment AI with computer vision tools.

STRICT WORKFLOW — follow in order:
1. Examine the uploaded image using your vision. Identify visible damage and affected areas.
2. Call run_damage_detection to get precise bounding boxes and damage class labels.
3. Call run_part_segmentation to get vehicle part bounding boxes and labels.
4. Cross-reference: for each damage bbox, find which part bbox it overlaps with spatially.
   Use the bbox coordinates directly. Do not guess from visual appearance alone.
5. If part segmentation returns no results, use your visual assessment of part locations
   based on the car's visible geometry and orientation.
6. Call execute_cost_computation with Python code that reads from COST_DB to compute
   cost ranges. Your code must set: result = {
     "damage_part_map": [{"damage": str, "part": str, "severity": str,
                          "cost_min": int, "cost_max": int}],
     "total_min": int, "total_max": int, "currency": "INR"
   }
7. Return your final assessment as a JSON object with exactly these keys:
   damage_part_map, total_min, total_max, currency, warnings.

SEVERITY RULES:
- minor: surface damage, no structural impact, paint or glass only
- moderate: panel deformation, part requires repair
- severe: structural damage, part requires full replacement

IMPORTANT:
- Be conservative with severity — bias toward minor/moderate unless clearly severe.
- If damage_detection confidence is below 0.4 on a region you visually see as damaged,
  include it with a warning note.
- Include an empty warnings list if there are no issues.
- Your final response must contain a valid JSON block wrapped in ```json ... ```."""


def _load_models(config: dict) -> None:
    """
    Lazy-load Qwen2-VL-7B-Instruct. Thread-safe via _model_lock.
    Called on first request and on /health startup warmup.
    """
    global _model, _processor, _tool_executor

    with _model_lock:
        if _model is not None:
            return  # already loaded

        vlm_cfg = config.get("vlm", {})
        model_id = vlm_cfg.get("model_id", "Qwen/Qwen2-VL-7B-Instruct")
        device = vlm_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

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
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else device,
                trust_remote_code=True
            )
            _model.eval()

        except Exception as e:
            logger.error(f"Failed to load VLM {model_id}: {e}")
            raise RuntimeError(f"VLM load failed: {e}") from e

        _tool_executor = get_tool_executor(config)

        elapsed = round(time.time() - t0, 1)
        logger.info(f"VLM loaded in {elapsed}s. Device: {device}")


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


def _build_initial_messages(image_path: str, claim_metadata: Optional[dict]) -> list:
    """Construct the first user message with image and claim context."""
    meta_text = (
        f"Claim metadata: {json.dumps(claim_metadata, indent=2)}"
        if claim_metadata
        else "No claim metadata provided."
    )

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
                        f"Assess all visible damage on this vehicle.\n"
                        f"{meta_text}\n\n"
                        f"Use your tools to produce a complete damage and cost report. "
                        f"Follow the workflow in your instructions exactly."
                    )
                }
            ]
        }
    ]


def _call_vlm(messages: list, config: dict, is_final_turn: bool = False) -> str:
    """
    Run a single VLM forward pass and return the decoded response string.

    Args:
        messages: Full conversation history
        config: Pipeline config dict
        is_final_turn: If True, use larger max_new_tokens for synthesis
    """
    from qwen_vl_utils import process_vision_info

    vlm_cfg = config.get("vlm", {})
    max_tokens = (
        vlm_cfg.get("max_new_tokens_final", 1024)
        if is_final_turn
        else vlm_cfg.get("max_new_tokens_tool", 512)
    )

    text_input = _processor.apply_chat_template(
        messages,
        tools=TOOL_DEFINITIONS,
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

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=vlm_cfg.get("temperature", 0.1),
            do_sample=vlm_cfg.get("do_sample", False),
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _processor.decode(new_tokens, skip_special_tokens=False)


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


def _run_tool_loop(
    messages: list,
    config: dict,
) -> tuple[list, list[ToolCallRecord], list[str]]:
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
    tool_call_log: list[ToolCallRecord] = []
    warnings: list[str] = []

    for iteration in range(max_iterations):
        logger.info(f"Orchestrator loop — iteration {iteration + 1}/{max_iterations}")

        is_final = (iteration == max_iterations - 1)
        t0 = time.time()

        response_text = _call_vlm(messages, config, is_final_turn=is_final)
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
            args_summary=f"image_path={Path(image_arg).name}" if image_arg else str(list(tool_args.keys())),
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


def _build_final_report(
    raw_report: dict,
    image_path: str,
    tool_call_log: list[ToolCallRecord],
    warnings: list[str],
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


def run(
    image_path: str,
    config: dict,
    claim_metadata: Optional[dict] = None
) -> dict:
    """
    Main orchestrator entry point. Called by FastAPI backend.

    Args:
        image_path: Absolute path to vehicle image
        config: Full pipeline config (loaded from global_config.yaml)
        claim_metadata: Optional dict with vehicle ID, claim number, etc.

    Returns:
        FinalDamageReport as dict (via .model_dump())

    Raises:
        ValueError: if image_path does not exist or image is invalid
        RuntimeError: if VLM fails to load
    """
    t_start = time.time()

    # Validate and normalise image
    image_path = _preprocess_image(image_path)

    # Load VLM (no-op if already loaded)
    _load_models(config)

    # Build initial conversation
    messages = _build_initial_messages(image_path, claim_metadata)

    # Run agentic tool-calling loop
    messages, tool_call_log, warnings = _run_tool_loop(messages, config)

    total_elapsed = round(time.time() - t_start, 2)

    # Extract final report from last assistant message
    raw_report = _extract_final_report(messages)

    # Get last assistant message text for raw_vlm_response
    last_assistant = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"),
        ""
    )

    approval_threshold = config.get("approval", {}).get("auto_approve_threshold_inr", 50000)

    report = _build_final_report(
        raw_report=raw_report,
        image_path=image_path,
        tool_call_log=tool_call_log,
        warnings=warnings,
        total_elapsed=total_elapsed,
        approval_threshold=approval_threshold,
        last_assistant_message=last_assistant if isinstance(last_assistant, str) else "",
        config=config
    )

    return report.model_dump()
```

---

## Phase 5 — Context Manager

Create `pipeline/context_manager.py`. This file is new — check it does not exist first.

```python
"""
pipeline/context_manager.py

Sliding window context manager for multi-turn adjuster follow-up interactions.
Implements the three-tier memory model from the architecture diagram:
  PINNED   → always in context (system prompt, vehicle ID, damage summary)
  RETAINED → last max_active_turns message pairs + tool results
  DROPPED  → compressed to one-line summaries

Usage:
  ctx = ClaimContext()
  ctx.pin("vehicle_id", "MH12AB1234")
  ctx.pin("damage_summary", report["damage_part_map"])
  messages = ctx.build_messages("Can you check if the subframe is affected?")
  # pass messages to orchestrator._run_tool_loop()
  ctx.add_turn(user_msg, assistant_msg, tool_results)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ClaimContext:
    """
    Manages conversation context across multiple adjuster turns on one claim.

    Do NOT share a ClaimContext instance across different claims.
    Create one per claim submission and persist it server-side keyed by claim_id.
    """
    max_active_turns: int = 3
    pinned: dict = field(default_factory=dict)
    active_window: list = field(default_factory=list)
    compressed_history: list = field(default_factory=list)

    def pin(self, key: str, value: Any) -> None:
        """
        Pin a value that will always appear in context.
        Use for: vehicle_id, claim_number, initial damage summary.

        Args:
            key: Label for the pinned item
            value: Any JSON-serialisable value
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Pinned key must be a non-empty string")
        self.pinned[key] = value
        logger.debug(f"Pinned context key: '{key}'")

    def add_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        tool_results: dict | None = None
    ) -> None:
        """
        Add a completed interaction turn to the active window.
        If window is full, the oldest turn is compressed and moved to history.

        Args:
            user_msg: The user's message text
            assistant_msg: The assistant's response text
            tool_results: Optional dict of {tool_name: result} from this turn
        """
        turn = {
            "user": user_msg,
            "assistant": assistant_msg,
            "tool_results": tool_results or {}
        }
        self.active_window.append(turn)

        if len(self.active_window) > self.max_active_turns:
            dropped = self.active_window.pop(0)
            tools_used = list(dropped["tool_results"].keys())
            summary = (
                f"[Earlier turn] User: '{dropped['user'][:80]}' | "
                f"Tools used: {tools_used} | "
                f"Assistant summary: '{dropped['assistant'][:120]}'"
            )
            self.compressed_history.append(summary)
            logger.debug(f"Compressed turn into history. Active turns: {len(self.active_window)}")

    def build_messages(self, new_user_message: str) -> list:
        """
        Assemble the complete message list for the next VLM call.
        Order: pinned context → compressed history → active window → new message.

        Args:
            new_user_message: The latest user input

        Returns:
            List of message dicts in Qwen2-VL conversation format
        """
        if not new_user_message or not new_user_message.strip():
            raise ValueError("new_user_message cannot be empty")

        messages = []

        # Inject pinned context as a system-level addendum
        if self.pinned:
            pinned_lines = "\n".join(
                f"  {k}: {json.dumps(v, default=str)}"
                for k, v in self.pinned.items()
            )
            messages.append({
                "role": "system",
                "content": f"PINNED CLAIM CONTEXT (always available):\n{pinned_lines}"
            })

        # Compressed history as a single assistant-narrated block
        if self.compressed_history:
            history_block = "\n".join(self.compressed_history)
            messages.append({
                "role": "assistant",
                "content": f"[Summary of earlier conversation turns]\n{history_block}"
            })

        # Active window turns
        for turn in self.active_window:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        # New user message
        messages.append({"role": "user", "content": new_user_message})

        logger.debug(
            f"Built message list: {len(messages)} messages "
            f"({len(self.compressed_history)} compressed, "
            f"{len(self.active_window)} active)"
        )
        return messages

    @property
    def turn_count(self) -> int:
        """Total turns processed (active + compressed)."""
        return len(self.active_window) + len(self.compressed_history)
```

---

## Phase 6 — Backend Update

Read `backend/app.py` in full before modifying.

Find and replace the existing pipeline call chain with the following pattern.
Do not restructure the entire file — only update the relevant endpoint and add
the startup warmup event.

### 6.1 Add startup warmup

Find the FastAPI `app = FastAPI(...)` instantiation and add immediately after it:

```python
import logging
import yaml
from pipeline.orchestrator import _load_models

logger = logging.getLogger(__name__)

@app.on_event("startup")
async def warmup_vlm():
    """Pre-load VLM on server start so the first request is not penalised."""
    try:
        with open("configs/global_config.yaml") as f:
            config = yaml.safe_load(f)
        _load_models(config)
        logger.info("VLM warmup complete — server ready")
    except Exception as e:
        logger.error(f"VLM warmup failed: {e}. First request will trigger load.")
```

### 6.2 Add `/assess` endpoint

If an `/assess` endpoint already exists, replace its body.
If it does not exist, add it.

```python
from fastapi import UploadFile, File, Form, HTTPException
from pathlib import Path
import shutil
import uuid
import yaml
from pipeline.orchestrator import run as orchestrator_run

@app.post("/assess")
async def assess_damage(
    image: UploadFile = File(...),
    claim_id: str = Form(default=None),
    vehicle_id: str = Form(default=None),
):
    """
    Accept a vehicle image upload and return a FinalDamageReport.

    Returns:
        FinalDamageReport JSON with damage_part_map, cost range,
        approval_decision, and tool_call_log.
    """
    # Validate file type
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {image.content_type}. Use JPEG, PNG, or WebP."
        )

    # Save uploaded image to disk
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}_{image.filename}"
    save_path = upload_dir / unique_name

    with save_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)

    logger.info(f"Image saved: {save_path} | claim_id={claim_id}")

    # Load config
    with open("configs/global_config.yaml") as f:
        config = yaml.safe_load(f)

    # Build claim metadata
    claim_metadata = {}
    if claim_id:
        claim_metadata["claim_id"] = claim_id
    if vehicle_id:
        claim_metadata["vehicle_id"] = vehicle_id

    # Run orchestrator
    try:
        report = orchestrator_run(
            image_path=str(save_path.resolve()),
            config=config,
            claim_metadata=claim_metadata if claim_metadata else None
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Model error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during assessment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal assessment error")

    return report


@app.get("/health")
async def health():
    """Health check. Returns 200 when VLM is loaded and ready."""
    global _model
    from pipeline.orchestrator import _model as vlm_model
    return {
        "status": "ready" if vlm_model is not None else "warming_up",
        "vlm_loaded": vlm_model is not None
    }
```

---

## Phase 7 — Smoke Tests

After all files are created, run these in order.
Fix any failure before declaring the phase done.

### 7.1 Import test — no model loading

```bash
python -c "
from pipeline.schema import FinalDamageReport, DamagePartEntry, ToolCallRecord
from models.vlm_reasoning.tool_registry import TOOL_DEFINITIONS, get_tool_executor
from models.vlm_reasoning.sandbox import execute_sandboxed, COST_DB
from pipeline.context_manager import ClaimContext
print('All imports OK')
print(f'Tools defined: {[t[\"function\"][\"name\"] for t in TOOL_DEFINITIONS]}')
print(f'COST_DB damage classes: {list(COST_DB.keys())}')
"
```

Expected: prints `All imports OK`, tool names, damage classes. No exceptions.

### 7.2 Sandbox unit test — no VLM needed

```bash
python -c "
from models.vlm_reasoning.sandbox import execute_sandboxed

# Valid code test
code = '''
entries = [('dent', 'hood', 'moderate', *COST_DB['dent']['hood'])]
result = {
    'damage_part_map': [
        {'damage': 'dent', 'part': 'hood', 'severity': 'moderate',
         'cost_min': COST_DB['dent']['hood'][0],
         'cost_max': COST_DB['dent']['hood'][1]}
    ],
    'total_min': COST_DB['dent']['hood'][0],
    'total_max': COST_DB['dent']['hood'][1],
    'currency': 'INR'
}
'''
out = execute_sandboxed(code)
assert 'result' in out, f'Expected result key, got: {out}'
assert out['result']['total_min'] == 12000
print('Sandbox valid code test PASSED')

# Blocked import test
bad_code = 'import os; result = {\"x\": os.listdir(\".\"), \"total_min\": 0, \"total_max\": 0, \"currency\": \"INR\", \"damage_part_map\": []}'
out2 = execute_sandboxed(bad_code)
assert 'error' in out2
print('Sandbox blocked import test PASSED')

# Missing result test
no_result_code = 'x = 1 + 1'
out3 = execute_sandboxed(no_result_code)
assert 'error' in out3
print('Sandbox missing result test PASSED')
"
```

Expected: all three `PASSED` lines. No exceptions.

### 7.3 Context manager test

```bash
python -c "
from pipeline.context_manager import ClaimContext
ctx = ClaimContext(max_active_turns=2)
ctx.pin('vehicle_id', 'MH12AB1234')
ctx.pin('claim_id', 'CLM_001')
ctx.add_turn('What is damaged?', 'Front bumper is dented.', {'run_damage_detection': {'damages': []}})
ctx.add_turn('How much will it cost?', 'Approximately INR 12000–25000.', {})
ctx.add_turn('Is the hood affected?', 'No hood damage detected.', {})
# At this point the first turn should be compressed
assert len(ctx.active_window) == 2
assert len(ctx.compressed_history) == 1
msgs = ctx.build_messages('What about the windshield?')
assert any(m['role'] == 'user' and 'windshield' in m['content'] for m in msgs)
print(f'Context manager test PASSED. Messages built: {len(msgs)}')
"
```

Expected: `Context manager test PASSED`.

### 7.4 Full pipeline smoke test — requires model weights

Only run this after confirming weights exist. Replace the image path.

```bash
python -c "
import yaml
from pathlib import Path
from pipeline.orchestrator import run

# Check a test image exists
test_images = list(Path('data/examples').glob('*.jpg')) + list(Path('data/examples').glob('*.png'))
if not test_images:
    print('No test images found in data/examples/. Add one and re-run.')
    exit(1)

image_path = str(test_images[0].resolve())
print(f'Testing with: {image_path}')

with open('configs/global_config.yaml') as f:
    config = yaml.safe_load(f)

report = run(image_path, config, claim_metadata={'claim_id': 'SMOKE_TEST_001'})

print(f'Pipeline completed in {report[\"total_inference_s\"]}s')
print(f'Damage entries: {len(report[\"damage_part_map\"])}')
print(f'Cost range: INR {report[\"total_min\"]} - {report[\"total_max\"]}')
print(f'Decision: {report[\"approval_decision\"]}')
print(f'Tools called: {[t[\"tool\"] for t in report[\"tool_call_log\"]]}')
print(f'Warnings: {report[\"warnings\"]}')
" 2>&1 | tail -20
```

Expected: Pipeline completes, at least one damage entry if the test image shows damage,
a valid decision string, no Python tracebacks.

---

## Phase 8 — File Checklist

Before finishing, verify every file exists and is non-empty:

```bash
python -c "
import os
required = [
    'configs/global_config.yaml',
    'pipeline/schema.py',
    'pipeline/orchestrator.py',
    'pipeline/context_manager.py',
    'models/vlm_reasoning/__init__.py',
    'models/vlm_reasoning/tool_registry.py',
    'models/vlm_reasoning/sandbox.py',
    'backend/app.py',
]
all_ok = True
for f in required:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    status = 'OK' if exists and size > 0 else 'MISSING OR EMPTY'
    print(f'  {status:<20} {f}')
    if status != 'OK':
        all_ok = False
print()
print('All files present.' if all_ok else 'SOME FILES MISSING — do not proceed.')
"
```

---

## Constraints — Read Before Starting Any Phase

- Python 3.10 throughout. No 3.11+ syntax.
- No paid APIs. No cloud services. No managed inference endpoints.
- No microservices. Single FastAPI app.
- No modification to `models/damage_detection/`. It is frozen.
- Do not add new dependencies without listing them here and explaining why.
- `pipeline/schema.py` is append-only. Never delete existing schema classes.
- All paths come from `global_config.yaml`. No hardcoded paths in Python.
- Logging via `logging.getLogger(__name__)`. No `print()` statements.
- Every function that can fail must handle failure and return a degraded result
  rather than raising uncaught exceptions to the caller, except for init-time
  failures (model load, weight file missing) which should raise and halt startup.
- If a phase fails, stop, report the exact error and file, and wait for instruction.
  Do not skip ahead.
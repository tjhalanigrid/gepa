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
import time
from pathlib import Path
from typing import Callable, Optional

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


def _get_plate_runner(config: dict) -> Optional[Callable]:
    """
    Import plate_rc_detection runner only if enabled in config.
    Returns None if disabled — dispatcher handles gracefully.
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
                return {
                    "error": (
                        f"Unknown tool: '{tool_name}'. "
                        "Valid tools: run_damage_detection, run_part_segmentation, "
                        "run_plate_detection, execute_cost_computation"
                    )
                }

            elapsed = round(time.time() - t0, 3)
            logger.info(f"Tool '{tool_name}' completed in {elapsed}s")
            return result

        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            logger.error(
                f"Tool '{tool_name}' raised unexpected exception after {elapsed}s: {e}",
                exc_info=True
            )
            return {"error": f"Tool execution failed: {str(e)}"}

    return dispatch

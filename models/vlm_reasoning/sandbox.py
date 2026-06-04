"""
Sandboxed CodeAct execution environment.

The VLM (Qwen2-VL) generates Python code to compute repair costs.
This module executes that code in a restricted namespace with:
  - AST validation before exec (blocks unsafe nodes)
  - Whitelisted builtins only
  - Whitelisted imports only (math, json, statistics, decimal)
  - Hard 10-second execution timeout via daemon thread join()
  - Required output: code must set 'result' as a dict

COST_DB is injected into the execution namespace.
The VLM's generated code reads from it directly.
Update COST_DB as domain knowledge improves — it is the only pricing source.
"""

import ast
import logging
import threading
import time
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

# ── Pricing database — all values in INR ────────────────────────────────────
# Structure: COST_DB[damage_class][part_label] = (min_cost, max_cost)
# Part labels must match those returned by part_segmentation tool exactly.

COST_DB: Dict[str, Dict[str, Tuple[int, int]]] = {
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
    namespace: Dict[str, Any] = {
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

    # Execute in a daemon thread so join(timeout) acts as a wall-clock cap.
    # signal.alarm cannot be used here — execute_sandboxed runs in a threadpool thread.
    exec_exc: list = []

    def _exec_target() -> None:
        try:
            compiled = compile(code, "<vlm_generated_cost_code>", "exec")
            exec(compiled, namespace)
        except Exception as e:
            exec_exc.append(e)

    t0 = time.time()
    exec_thread = threading.Thread(target=_exec_target, daemon=True)
    exec_thread.start()
    exec_thread.join(timeout_seconds)

    if exec_thread.is_alive():
        logger.warning(f"Sandbox timeout after {timeout_seconds}s")
        return {"error": f"Sandboxed execution exceeded {timeout_seconds}s limit"}

    elapsed = round(time.time() - t0, 3)

    if exec_exc:
        logger.warning(f"Sandbox runtime error after {elapsed}s: {exec_exc[0]}")
        return {"error": f"Runtime error in generated code: {str(exec_exc[0])}"}

    result = namespace.get("result")

    if result is None:
        logger.warning("Sandbox: code ran but 'result' was never set")
        return {"error": "Code executed but did not assign to 'result'"}

    if not isinstance(result, dict):
        logger.warning(f"Sandbox: 'result' is {type(result).__name__}, expected dict")
        return {"error": f"'result' must be a dict, got {type(result).__name__}"}

    required_keys = {"damage_part_map", "total_min", "total_max", "currency"}
    missing = required_keys - set(result.keys())
    if missing:
        return {"error": f"'result' dict is missing required keys: {missing}"}

    logger.info(f"Sandbox execution succeeded in {elapsed}s. Result keys: {list(result.keys())}")
    return {"result": result}

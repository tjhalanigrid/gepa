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

# Optional Monty engine (pydantic/monty): a minimal, secure Python interpreter
# written in Rust. Used as the default CodeAct execution engine when available;
# we fall back to the restricted exec() engine if it is not installed.
try:
    import pydantic_monty as _monty
    _HAS_MONTY = True
except Exception:  # pragma: no cover - import guard
    _monty = None
    _HAS_MONTY = False

# ── Pricing database — all values in INR ────────────────────────────────────
# Single source of truth lives in cost_db.py (UNDERSCORE part keys). The sandbox
# injects this exact dict, so the model's underscore part names (front_bumper,
# right_fender, …) match by exact key. Do NOT redefine a second copy here — a
# divergent duplicate is what caused every bumper/fender lookup to fall back.
from models.vlm_reasoning.cost_db import COST_DB  # noqa: E402

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


_REQUIRED_RESULT_KEYS = {"damage_part_map", "total_min", "total_max", "currency"}


def _validate_result(result: Any) -> dict:
    """
    Shared validation for the value produced by any sandbox engine.

    Anchors correctness regardless of engine: the value MUST be a dict with the
    required cost-report keys. A bare price table or any other shape is rejected,
    so the model cannot bypass COST_DB by returning arbitrary data.

    Returns {"result": dict} on success, else {"error": str}.
    """
    if result is None:
        return {"error": "Code executed but did not assign to 'result'"}
    if not isinstance(result, dict):
        return {"error": f"'result' must be a dict, got {type(result).__name__}"}
    missing = _REQUIRED_RESULT_KEYS - set(result.keys())
    if missing:
        return {"error": f"'result' dict is missing required keys: {missing}"}
    return {"result": result}


def _assigns_result(code: str) -> bool:
    """True if the code has a top-level (or nested) assignment to `result`."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "result":
                    return True
        if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            tgt = getattr(node, "target", None)
            if isinstance(tgt, ast.Name) and tgt.id == "result":
                return True
    return False


def execute_with_monty(code: str, timeout_seconds: int = 10) -> dict:
    """
    Execute VLM-generated cost code with the Monty interpreter.

    Monty returns the value of the final expression. We support both styles the
    model may produce:
      • assigns `result = {...}`  → we append a trailing `result` expression so
        that value is returned.
      • ends with a bare dict expression → returned directly.

    COST_DB is injected as a Monty input (read-only canonical pricing). Monty
    blocks filesystem / network / env access by default, so this is strictly
    more sandboxed than exec().

    Returns {"result": dict} on success, else {"error": str}.
    """
    if not _HAS_MONTY:
        return {"error": "Monty engine not available"}
    if not code or not code.strip():
        return {"error": "Empty code string received"}

    code_to_run = code + "\nresult" if _assigns_result(code) else code

    try:
        limits = _monty.ResourceLimits(max_duration_secs=float(timeout_seconds))
        m = _monty.Monty(code_to_run, inputs=["COST_DB"])
        t0 = time.time()
        output = m.run(inputs={"COST_DB": COST_DB}, limits=limits)
        elapsed = round(time.time() - t0, 3)
    except _monty.MontySyntaxError as e:
        return {"error": f"Code safety check failed: {e}"}
    except _monty.MontyRuntimeError as e:
        return {"error": f"Runtime error in generated code: {e}"}
    except Exception as e:  # MontyError base / typing / limits exceeded
        return {"error": f"Monty execution failed: {e}"}

    validated = _validate_result(output)
    if "error" in validated:
        logger.warning(f"Monty result rejected: {validated['error']}")
        return validated
    logger.info(f"Monty execution succeeded in {elapsed}s. Result keys: {list(output.keys())}")
    return validated


def execute_sandboxed(code: str, timeout_seconds: int = 10, engine: str = "monty") -> dict:
    """
    Execute VLM-generated cost code in a sandbox and return the cost report.

    Args:
        code: Python code string generated by the VLM
        timeout_seconds: Hard execution timeout (default 10s)
        engine: "monty" (default, secure Rust interpreter) or "exec" (restricted
            CPython exec). Monty silently falls back to "exec" if not installed.

    Returns:
        {"result": dict} on success
        {"error": str} on any failure (validation, timeout, runtime)
    """
    if engine == "monty" and _HAS_MONTY:
        return execute_with_monty(code, timeout_seconds)
    if engine == "monty" and not _HAS_MONTY:
        logger.warning("Monty requested but not installed; falling back to exec sandbox")
    return _execute_with_exec(code, timeout_seconds)


def _execute_with_exec(code: str, timeout_seconds: int = 10) -> dict:
    """
    Execute VLM-generated Python in a restricted CPython exec namespace.

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

    validated = _validate_result(namespace.get("result"))
    if "error" in validated:
        logger.warning(f"Sandbox result rejected: {validated['error']}")
        return validated
    logger.info(f"Sandbox execution succeeded in {elapsed}s. Result keys: {list(validated['result'].keys())}")
    return validated

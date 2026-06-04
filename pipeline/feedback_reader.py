"""
Reads corrections_log.jsonl and produces few-shot examples
for injection into the VLM vision assessment prompt.

Called by pipeline/orchestrator.py at the start of each VLM pass.
Cached in memory with a 5-minute TTL to avoid re-reading on every request.
"""

import json
import logging
import time
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

_cache: dict = {"examples": [], "loaded_at": 0.0}
_CACHE_TTL_S = 300  # 5 minutes


def get_few_shot_examples(
    corrections_log_path: str = "data/feedback/corrections_log.jsonl",
    n: int = 5,
    min_quality_score: float = 0.7,
) -> str:
    """
    Returns a formatted string of few-shot correction examples for VLM prompt injection.

    Selects the N highest-quality corrections that had actual changes.
    Prioritises entries where missed damages were found or false positives were removed.
    Returns empty string if no qualifying corrections exist yet.
    """
    global _cache

    if time.time() - _cache["loaded_at"] < _CACHE_TTL_S and _cache["examples"]:
        examples = _cache["examples"]
    else:
        examples = _load_corrections(corrections_log_path, min_quality_score)
        _cache = {"examples": examples, "loaded_at": time.time()}

    if not examples:
        return ""

    priority = [e for e in examples if e.get("had_missed_damages") or e.get("had_false_positives")]
    rest = [e for e in examples if e not in priority]
    selected = (priority + rest)[:n]

    lines = [
        "\n── PAST CORRECTION EXAMPLES (learn from these mistakes) ──\n"
        "These show real cases where the pipeline was wrong and a human corrected it.\n"
        "Use them to avoid repeating the same errors.\n"
    ]

    example_count = 0
    for e in selected:
        original = e.get("original_damage_map", [])
        final = e.get("final_damage_map", [])
        actions = e.get("correction_actions", [])

        missed  = [a for a in actions if a.get("action") == "add"]
        removed = [a for a in actions if a.get("action") == "remove"]
        edited  = [a for a in actions if a.get("action") == "edit"]

        change_summary = []
        if missed:
            change_summary.append(
                f"  MISSED DAMAGES ADDED: "
                + str([a.get("corrected") for a in missed])
            )
        if removed:
            change_summary.append(
                f"  FALSE POSITIVES REMOVED: "
                + str([a.get("original") for a in removed])
            )
        if edited:
            change_summary.append(
                f"  SEVERITY/PART CORRECTED: "
                + str([(a.get("original"), "->", a.get("corrected")) for a in edited])
            )

        if not change_summary:
            continue

        example_count += 1
        lines.append(
            f"Example {example_count}:\n"
            f"  Pipeline output: {original}\n"
            f"  Human corrections:\n"
            + "\n".join(change_summary)
            + f"\n  Final correct output: {final}\n"
        )

    if example_count == 0:
        return ""

    lines.append("── END OF EXAMPLES ──\n")
    return "\n".join(lines)


def get_correction_stats() -> dict:
    """Returns quick stats from corrections_log.jsonl for logging and dashboard."""
    examples = _load_corrections("data/feedback/corrections_log.jsonl", 0.0)
    return {
        "total": len(examples),
        "with_missed": sum(1 for e in examples if e.get("had_missed_damages")),
        "with_false_pos": sum(1 for e in examples if e.get("had_false_positives")),
        "avg_quality": (
            sum(e.get("correction_quality_score", 0) for e in examples) / len(examples)
            if examples else 0.0
        ),
    }


def _load_corrections(path: str, min_quality: float) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    entries = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                if e.get("correction_quality_score", 0) >= min_quality:
                    entries.append(e)
            except Exception:
                continue
    entries.sort(key=lambda x: x.get("correction_quality_score", 0), reverse=True)
    return entries[:50]

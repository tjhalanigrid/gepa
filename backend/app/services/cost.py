"""
Repair-cost computation.

Pricing is deterministic and anchored to COST_DB — it is NOT produced by the LLM.
Unknown (damage, part) pairs fall back to a severity-weighted class average.
"""

from models.vlm_reasoning.sandbox import COST_DB
from pipeline.schema import DamagePartEntry

_SEVERITY_MULTIPLIERS = {"minor": 0.6, "moderate": 1.0, "severe": 1.6}
_DEFAULT_RANGE = (5000, 15000)  # INR — used when a damage class is unknown


def apply_cost_lookup(
    entries: list[DamagePartEntry],
) -> tuple[list[DamagePartEntry], int, int]:
    """
    Recalculate cost_min/cost_max for each entry using COST_DB.

    Returns (updated_entries, total_min, total_max).
    """
    updated: list[DamagePartEntry] = []
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
                avg_min, avg_max = _DEFAULT_RANGE
            m = _SEVERITY_MULTIPLIERS.get(e.severity, 1.0)
            cost_min = int(avg_min * m)
            cost_max = int(avg_max * m)
        updated.append(
            DamagePartEntry(
                damage=e.damage,
                part=e.part,
                severity=e.severity,
                cost_min=cost_min,
                cost_max=cost_max,
            )
        )
    total_min = sum(e.cost_min for e in updated)
    total_max = sum(e.cost_max for e in updated)
    return updated, total_min, total_max

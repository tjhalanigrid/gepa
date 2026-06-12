"""
Pricing database for vehicle repair cost estimation.

COST_DB is the single source of truth for all (damage, part) → INR cost ranges.
Update this dict as domain knowledge improves.
Structure: COST_DB[damage_class][part_label] = (cost_min_inr, cost_max_inr)

Part labels use UNDERSCORE format consistently, matching the part vocabulary in
the system prompt (front_bumper, right_fender, …). This matters because the cost
sandbox does an EXACT-key lookup `COST_DB.get(damage,{}).get(part, default)` — if
the keys here used spaces, the model's underscore part names would all miss and
fall back to the default. lookup_cost() additionally normalises at query time.
"""

from typing import Dict, Tuple

COST_DB: Dict[str, Dict[str, Tuple[int, int]]] = {
    "dent": {
        "front_bumper":      (8000,  25000),
        "rear_bumper":       (8000,  25000),
        "hood":              (12000, 35000),
        "front_left_door":   (10000, 30000),
        "front_right_door":  (10000, 30000),
        "rear_left_door":    (10000, 30000),
        "rear_right_door":   (10000, 30000),
        "left_fender":       (8000,  20000),
        "right_fender":      (8000,  20000),
        "roof_panel":        (15000, 45000),
        "trunk_lid":         (10000, 28000),
    },
    "scratch": {
        "front_bumper":      (3000,  8000),
        "rear_bumper":       (3000,  8000),
        "hood":              (4000,  10000),
        "front_left_door":   (3500,  9000),
        "front_right_door":  (3500,  9000),
        "rear_left_door":    (3500,  9000),
        "rear_right_door":   (3500,  9000),
        "left_fender":       (3000,  7000),
        "right_fender":      (3000,  7000),
        "roof_panel":        (5000,  12000),
        "trunk_lid":         (3500,  9000),
    },
    "crack": {
        "windshield":        (15000, 40000),
        "rear_windshield":   (12000, 35000),
        "front_bumper":      (5000,  15000),
        "rear_bumper":       (5000,  15000),
    },
    "glass_shatter": {
        "windshield":        (20000, 55000),
        "rear_windshield":   (15000, 45000),
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


def lookup_cost(damage_type: str, part: str) -> Tuple[int, int]:
    """
    Returns (cost_min, cost_max) in INR.
    Falls back to (3000, 8000) for unknown (damage_type, part) pairs.
    Matching is case-insensitive and normalises underscores to spaces.
    """
    def normalise(s: str) -> str:
        return s.lower().replace("_", " ").strip()

    norm_damage = normalise(damage_type)
    norm_part = normalise(part)

    for db_damage, parts in COST_DB.items():
        if normalise(db_damage) == norm_damage:
            for db_part, costs in parts.items():
                if normalise(db_part) == norm_part:
                    return costs
    return (3000, 8000)

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
        "headlight":         (12000, 32000),
        "taillight":         (8000,  22000),
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
        "windshield":        (2000,  6000),
        "rear_windshield":   (2000,  5500),
        "headlight":         (1500,  4000),
        "taillight":         (1500,  3500),
    },
    "crack": {
        "windshield":        (15000, 40000),
        "rear_windshield":   (12000, 35000),
        "front_bumper":      (5000,  15000),
        "rear_bumper":       (5000,  15000),
        "hood":              (8000,  22000),
        "front_left_door":   (6000,  18000),
        "front_right_door":  (6000,  18000),
        "rear_left_door":    (6000,  18000),
        "rear_right_door":   (6000,  18000),
        "left_fender":       (5000,  14000),
        "right_fender":      (5000,  14000),
        "roof_panel":        (10000, 28000),
        "trunk_lid":         (6000,  18000),
        "headlight":         (5000,  14000),
        "taillight":         (4000,  10000),
        "tire":              (3000,  8000),
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
        "fog_lamp":          (4000,  12000),
    },
    "tire_flat": {
        "tire":              (4000,  12000),
    },
    "mirror_broken": {
        "side_mirror":       (5000,  15000),
    },
    "paint_damage": {
        "front_bumper":      (4000,  10000),
        "rear_bumper":       (4000,  10000),
        "hood":              (5000,  12000),
        "front_left_door":   (4500,  11000),
        "front_right_door":  (4500,  11000),
        "rear_left_door":    (4500,  11000),
        "rear_right_door":   (4500,  11000),
        "left_fender":       (4000,  10000),
        "right_fender":      (4000,  10000),
        "roof_panel":        (6000,  14000),
        "trunk_lid":         (4500,  11000),
        "quarter_panel":     (4500,  11000),
        "rocker_panel":      (3500,  9000),
    },
    "scuff": {
        "front_bumper":      (2000,  6000),
        "rear_bumper":       (2000,  6000),
        "hood":              (2500,  7000),
        "front_left_door":   (2000,  6000),
        "front_right_door":  (2000,  6000),
        "rear_left_door":    (2000,  6000),
        "rear_right_door":   (2000,  6000),
        "left_fender":       (2000,  5500),
        "right_fender":      (2000,  5500),
        "rocker_panel":      (2000,  5000),
        "quarter_panel":     (2000,  6000),
        "side_mirror":       (1500,  4000),
    },
    "bent": {
        "front_bumper":      (6000,  18000),
        "rear_bumper":       (6000,  18000),
        "hood":              (10000, 28000),
        "left_fender":       (7000,  18000),
        "right_fender":      (7000,  18000),
        "trunk_lid":         (8000,  22000),
        "rocker_panel":      (6000,  16000),
        "quarter_panel":     (8000,  22000),
        "radiator_support":  (12000, 35000),
        "side_mirror":       (3000,  8000),
        "wheel":             (5000,  15000),
    },
    "crumpled": {
        "front_bumper":      (10000, 30000),
        "rear_bumper":       (10000, 30000),
        "hood":              (15000, 45000),
        "left_fender":       (12000, 32000),
        "right_fender":      (12000, 32000),
        "trunk_lid":         (12000, 35000),
        "roof_panel":        (20000, 55000),
        "quarter_panel":     (15000, 40000),
        "radiator_support":  (18000, 50000),
        "front_left_door":   (14000, 38000),
        "front_right_door":  (14000, 38000),
        "rear_left_door":    (14000, 38000),
        "rear_right_door":   (14000, 38000),
    },
    "missing_part": {
        "side_mirror":       (6000,  18000),
        "fog_lamp":          (4000,  12000),
        "headlight":         (10000, 28000),
        "taillight":         (6000,  18000),
        "grill":             (5000,  15000),
        "front_bumper":      (12000, 35000),
        "rear_bumper":       (12000, 35000),
        "wheel":             (8000,  25000),
        "tire":              (4000,  12000),
    },
    "detached_part": {
        "side_mirror":       (4000,  12000),
        "fog_lamp":          (3000,  9000),
        "grill":             (4000,  12000),
        "front_bumper":      (8000,  22000),
        "rear_bumper":       (8000,  22000),
        "rocker_panel":      (6000,  16000),
        "headlight":         (8000,  22000),
        "taillight":         (5000,  14000),
    },
    "wheel_damage": {
        "wheel":             (6000,  20000),
        "tire":              (4000,  12000),
    },
    "structural_damage": {
        "hood":              (25000, 70000),
        "roof_panel":        (30000, 80000),
        "front_bumper":      (20000, 55000),
        "rear_bumper":       (20000, 55000),
        "left_fender":       (18000, 50000),
        "right_fender":      (18000, 50000),
        "trunk_lid":         (20000, 55000),
        "quarter_panel":     (22000, 60000),
        "radiator_support":  (25000, 65000),
        "rocker_panel":      (18000, 48000),
        "front_left_door":   (20000, 55000),
        "front_right_door":  (20000, 55000),
        "rear_left_door":    (20000, 55000),
        "rear_right_door":   (20000, 55000),
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

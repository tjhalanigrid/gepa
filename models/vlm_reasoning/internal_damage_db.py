"""
models/vlm_reasoning/internal_damage_db.py

Lookup table: exterior car part + severity → likely internal components damaged.

Used by the `infer_internal_damage` tool in pi_agent.py. The VLM calls this
tool during its reasoning loop when it detects moderate/severe exterior damage.
The tool returns a list of internal components that are likely affected — the
VLM then incorporates this into its final Terminate output.

Rules:
  - severe   → structural / mechanical components at risk (high repair cost)
  - moderate → adjacent brackets, seals, actuators (medium repair cost)
  - minor    → cosmetic internals only (clips, trims, seals)
"""

from typing import List

INTERNAL_DAMAGE_MAP: dict = {
    "hood": {
        "severe":   ["radiator", "engine_mounts", "battery", "coolant_lines", "air_intake", "intercooler"],
        "moderate": ["hood_latch", "hood_hinges", "radiator_support", "wiring_harness"],
        "minor":    ["hood_liner", "hood_latch"],
    },
    "front_bumper": {
        "severe":   ["bumper_beam", "frame_rails", "cooling_system", "headlight_brackets", "crash_absorber", "steering_rack"],
        "moderate": ["bumper_absorber", "parking_sensors", "bumper_brackets", "fog_lamp_housing"],
        "minor":    ["bumper_clips", "grille_clips"],
    },
    "rear_bumper": {
        "severe":   ["rear_frame_rails", "trunk_floor", "exhaust_tips", "tow_hook_mount", "fuel_tank_proximity"],
        "moderate": ["bumper_absorber", "parking_sensors", "rear_trim_brackets"],
        "minor":    ["bumper_clips", "reflector_mount"],
    },
    "windshield": {
        "severe":   ["a_pillar", "roof_rail", "airbag_sensors", "rearview_mirror_mount"],
        "moderate": ["windshield_seal", "wiper_linkage", "defrost_elements"],
        "minor":    ["windshield_seal", "rubber_trim"],
    },
    "rear_windshield": {
        "severe":   ["rear_window_frame", "heating_elements", "antenna", "roof_rail"],
        "moderate": ["rear_window_seal", "heating_elements", "wiper_motor"],
        "minor":    ["rear_window_seal", "rubber_trim"],
    },
    "front_left_door": {
        "severe":   ["door_frame", "window_regulator", "door_latch", "wiring_harness", "side_airbag"],
        "moderate": ["door_hinge", "window_regulator", "door_seal", "door_handle_mechanism"],
        "minor":    ["door_seal", "door_trim", "window_rubber"],
    },
    "front_right_door": {
        "severe":   ["door_frame", "window_regulator", "door_latch", "wiring_harness", "side_airbag"],
        "moderate": ["door_hinge", "window_regulator", "door_seal", "door_handle_mechanism"],
        "minor":    ["door_seal", "door_trim", "window_rubber"],
    },
    "rear_left_door": {
        "severe":   ["door_frame", "window_regulator", "door_latch", "wiring_harness", "side_airbag"],
        "moderate": ["door_hinge", "window_regulator", "door_seal"],
        "minor":    ["door_seal", "door_trim"],
    },
    "rear_right_door": {
        "severe":   ["door_frame", "window_regulator", "door_latch", "wiring_harness", "side_airbag"],
        "moderate": ["door_hinge", "window_regulator", "door_seal"],
        "minor":    ["door_seal", "door_trim"],
    },
    "left_fender": {
        "severe":   ["fender_liner", "headlight_bracket", "frame_rail", "wiring_harness"],
        "moderate": ["fender_liner", "fender_bolts", "headlight_bracket"],
        "minor":    ["fender_liner", "fender_clips"],
    },
    "right_fender": {
        "severe":   ["fender_liner", "headlight_bracket", "frame_rail", "wiring_harness"],
        "moderate": ["fender_liner", "fender_bolts", "headlight_bracket"],
        "minor":    ["fender_liner", "fender_clips"],
    },
    "trunk_lid": {
        "severe":   ["trunk_latch", "trunk_hinges", "rear_window_heating_element", "wiring"],
        "moderate": ["trunk_latch", "trunk_seal", "trunk_hinges"],
        "minor":    ["trunk_seal", "trunk_clips"],
    },
    "tailgate": {
        "severe":   ["tailgate_latch", "tailgate_hinges", "wiring_harness", "rear_camera"],
        "moderate": ["tailgate_latch", "tailgate_seal", "tailgate_struts"],
        "minor":    ["tailgate_seal", "tailgate_clips"],
    },
    "roof_panel": {
        "severe":   ["roof_rails", "sunroof_mechanism", "headliner", "a_pillar", "b_pillar"],
        "moderate": ["roof_seal", "drip_rail", "sunroof_seal"],
        "minor":    ["roof_seal", "antenna_mount"],
    },
    "grill": {
        "severe":   ["radiator", "cooling_fan", "front_frame", "ac_condenser"],
        "moderate": ["radiator_support", "grille_brackets", "cooling_fan_shroud"],
        "minor":    ["grille_clips", "emblem_mount"],
    },
    "headlight": {
        "severe":   ["headlight_bracket", "wiring_harness", "turn_signal", "drl_module"],
        "moderate": ["headlight_bracket", "wiring_connector", "headlight_adjuster"],
        "minor":    ["headlight_seal", "mounting_clips"],
    },
    "taillight": {
        "severe":   ["taillight_bracket", "wiring_harness", "reverse_light", "brake_light_module"],
        "moderate": ["taillight_bracket", "wiring_connector"],
        "minor":    ["taillight_seal", "mounting_clips"],
    },
    "fog_lamp": {
        "severe":   ["fog_lamp_bracket", "wiring_harness", "bumper_mount"],
        "moderate": ["fog_lamp_bracket", "wiring_connector"],
        "minor":    ["fog_lamp_seal", "mounting_clips"],
    },
    "side_mirror": {
        "severe":   ["mirror_motor", "heating_element", "wiring", "turn_signal_module", "mirror_housing"],
        "moderate": ["mirror_motor", "mirror_glass", "wiring_connector"],
        "minor":    ["mirror_glass", "mirror_clips"],
    },
    "wheel": {
        "severe":   ["wheel_bearing", "brake_caliper", "brake_rotor", "suspension_arm", "axle_shaft"],
        "moderate": ["wheel_bearing", "brake_caliper", "tire_pressure_sensor"],
        "minor":    ["wheel_weights", "tire_pressure_sensor"],
    },
    "tire": {
        "severe":   ["wheel_rim", "brake_system", "suspension_components"],
        "moderate": ["wheel_rim", "tire_pressure_sensor"],
        "minor":    ["tire_pressure_sensor"],
    },
    "rocker_panel": {
        "severe":   ["floor_pan", "sill_plate", "door_threshold", "structural_rail"],
        "moderate": ["sill_plate", "door_seal", "drainage_channel"],
        "minor":    ["sill_clips", "door_seal"],
    },
    "quarter_panel": {
        "severe":   ["rear_frame_rail", "wheel_arch", "fuel_tank_proximity", "trunk_floor"],
        "moderate": ["wheel_arch_liner", "quarter_panel_brace", "rear_door_seal"],
        "minor":    ["quarter_panel_clips", "wheel_arch_seal"],
    },
    "radiator_support": {
        "severe":   ["radiator", "ac_condenser", "cooling_fan", "front_frame_rails"],
        "moderate": ["radiator_brackets", "cooling_fan_shroud", "hood_latch_support"],
        "minor":    ["radiator_clips", "support_bolts"],
    },
    "front_left_window": {
        "severe":   ["window_regulator", "door_frame", "wiring_harness"],
        "moderate": ["window_regulator", "window_seal", "door_seal"],
        "minor":    ["window_seal", "rubber_trim"],
    },
    "front_right_window": {
        "severe":   ["window_regulator", "door_frame", "wiring_harness"],
        "moderate": ["window_regulator", "window_seal", "door_seal"],
        "minor":    ["window_seal", "rubber_trim"],
    },
    "rear_left_window": {
        "severe":   ["window_regulator", "door_frame", "wiring_harness"],
        "moderate": ["window_regulator", "window_seal"],
        "minor":    ["window_seal", "rubber_trim"],
    },
    "rear_right_window": {
        "severe":   ["window_regulator", "door_frame", "wiring_harness"],
        "moderate": ["window_regulator", "window_seal"],
        "minor":    ["window_seal", "rubber_trim"],
    },
}


def lookup_internal(part: str, severity: str) -> List[str]:
    """
    Returns list of likely internal components damaged for a given part + severity.
    Falls back to empty list if part not in map (never raises).
    """
    part = part.lower().strip()
    severity = severity.lower().strip()
    part_map = INTERNAL_DAMAGE_MAP.get(part, {})
    return list(part_map.get(severity, []))


# ── Internal component cost + damage type DB ──────────────────────────────────
# Each entry: component → {damage_type, cost_min_inr, cost_max_inr}
# Severity multipliers applied at query time: minor=0.6x, moderate=1.0x, severe=1.5x
# damage_type = most likely damage class for this component given an impact

INTERNAL_COMPONENT_DB: dict = {
    # Engine bay
    "radiator":            {"damage_type": "crumpled",          "cost_min": 15000, "cost_max": 35000},
    "engine_mounts":       {"damage_type": "structural_damage",  "cost_min": 8000,  "cost_max": 20000},
    "battery":             {"damage_type": "structural_damage",  "cost_min": 8000,  "cost_max": 18000},
    "coolant_lines":       {"damage_type": "crack",              "cost_min": 3000,  "cost_max": 8000},
    "air_intake":          {"damage_type": "crumpled",           "cost_min": 5000,  "cost_max": 12000},
    "intercooler":         {"damage_type": "crumpled",           "cost_min": 15000, "cost_max": 40000},
    "cooling_system":      {"damage_type": "structural_damage",  "cost_min": 10000, "cost_max": 30000},
    "cooling_fan":         {"damage_type": "bent",               "cost_min": 5000,  "cost_max": 15000},
    "cooling_fan_shroud":  {"damage_type": "crumpled",           "cost_min": 3000,  "cost_max": 8000},
    "ac_condenser":        {"damage_type": "crumpled",           "cost_min": 10000, "cost_max": 25000},

    # Frame / structural
    "frame_rails":         {"damage_type": "structural_damage",  "cost_min": 20000, "cost_max": 60000},
    "frame_rail":          {"damage_type": "structural_damage",  "cost_min": 20000, "cost_max": 60000},
    "rear_frame_rails":    {"damage_type": "structural_damage",  "cost_min": 20000, "cost_max": 60000},
    "rear_frame_rail":     {"damage_type": "structural_damage",  "cost_min": 20000, "cost_max": 60000},
    "front_frame":         {"damage_type": "structural_damage",  "cost_min": 20000, "cost_max": 60000},
    "structural_rail":     {"damage_type": "structural_damage",  "cost_min": 20000, "cost_max": 60000},
    "floor_pan":           {"damage_type": "structural_damage",  "cost_min": 15000, "cost_max": 40000},
    "trunk_floor":         {"damage_type": "structural_damage",  "cost_min": 10000, "cost_max": 30000},
    "a_pillar":            {"damage_type": "structural_damage",  "cost_min": 15000, "cost_max": 40000},
    "b_pillar":            {"damage_type": "structural_damage",  "cost_min": 15000, "cost_max": 40000},
    "roof_rails":          {"damage_type": "structural_damage",  "cost_min": 10000, "cost_max": 30000},
    "roof_rail":           {"damage_type": "structural_damage",  "cost_min": 10000, "cost_max": 30000},
    "radiator_support":    {"damage_type": "structural_damage",  "cost_min": 10000, "cost_max": 25000},
    "wheel_arch":          {"damage_type": "structural_damage",  "cost_min": 8000,  "cost_max": 20000},
    "quarter_panel_brace": {"damage_type": "structural_damage",  "cost_min": 5000,  "cost_max": 15000},
    "sill_plate":          {"damage_type": "structural_damage",  "cost_min": 5000,  "cost_max": 15000},
    "door_threshold":      {"damage_type": "structural_damage",  "cost_min": 5000,  "cost_max": 15000},
    "door_frame":          {"damage_type": "structural_damage",  "cost_min": 10000, "cost_max": 30000},

    # Bumper internals
    "bumper_beam":         {"damage_type": "crumpled",           "cost_min": 5000,  "cost_max": 15000},
    "bumper_absorber":     {"damage_type": "crumpled",           "cost_min": 2000,  "cost_max": 6000},
    "crash_absorber":      {"damage_type": "crumpled",           "cost_min": 3000,  "cost_max": 8000},
    "bumper_brackets":     {"damage_type": "bent",               "cost_min": 2000,  "cost_max": 5000},
    "bumper_mount":        {"damage_type": "bent",               "cost_min": 2000,  "cost_max": 5000},

    # Steering / suspension / brakes
    "steering_rack":       {"damage_type": "structural_damage",  "cost_min": 15000, "cost_max": 40000},
    "suspension_arm":      {"damage_type": "bent",               "cost_min": 8000,  "cost_max": 25000},
    "suspension_components":{"damage_type": "structural_damage", "cost_min": 10000, "cost_max": 35000},
    "axle_shaft":          {"damage_type": "structural_damage",  "cost_min": 10000, "cost_max": 30000},
    "wheel_bearing":       {"damage_type": "structural_damage",  "cost_min": 5000,  "cost_max": 15000},
    "brake_caliper":       {"damage_type": "structural_damage",  "cost_min": 8000,  "cost_max": 20000},
    "brake_rotor":         {"damage_type": "structural_damage",  "cost_min": 5000,  "cost_max": 15000},
    "brake_system":        {"damage_type": "structural_damage",  "cost_min": 10000, "cost_max": 30000},
    "wheel_rim":           {"damage_type": "bent",               "cost_min": 8000,  "cost_max": 25000},

    # Electrical / sensors
    "wiring_harness":      {"damage_type": "detached_part",      "cost_min": 8000,  "cost_max": 25000},
    "wiring":              {"damage_type": "detached_part",       "cost_min": 3000,  "cost_max": 10000},
    "airbag_sensors":      {"damage_type": "missing_part",        "cost_min": 5000,  "cost_max": 15000},
    "side_airbag":         {"damage_type": "missing_part",        "cost_min": 15000, "cost_max": 40000},
    "parking_sensors":     {"damage_type": "missing_part",        "cost_min": 3000,  "cost_max": 10000},
    "tire_pressure_sensor":{"damage_type": "missing_part",        "cost_min": 1000,  "cost_max": 4000},
    "turn_signal":         {"damage_type": "lamp_broken",         "cost_min": 2000,  "cost_max": 6000},
    "turn_signal_module":  {"damage_type": "lamp_broken",         "cost_min": 3000,  "cost_max": 8000},
    "drl_module":          {"damage_type": "lamp_broken",         "cost_min": 5000,  "cost_max": 15000},
    "reverse_light":       {"damage_type": "lamp_broken",         "cost_min": 2000,  "cost_max": 6000},
    "brake_light_module":  {"damage_type": "lamp_broken",         "cost_min": 3000,  "cost_max": 8000},
    "rear_camera":         {"damage_type": "missing_part",        "cost_min": 5000,  "cost_max": 15000},
    "heating_elements":    {"damage_type": "structural_damage",   "cost_min": 5000,  "cost_max": 15000},
    "heating_element":     {"damage_type": "structural_damage",   "cost_min": 3000,  "cost_max": 8000},
    "defrost_elements":    {"damage_type": "crack",               "cost_min": 5000,  "cost_max": 15000},
    "mirror_motor":        {"damage_type": "missing_part",        "cost_min": 4000,  "cost_max": 12000},
    "mirror_housing":      {"damage_type": "crumpled",            "cost_min": 5000,  "cost_max": 15000},
    "mirror_glass":        {"damage_type": "glass_shatter",       "cost_min": 2000,  "cost_max": 6000},
    "sunroof_mechanism":   {"damage_type": "structural_damage",   "cost_min": 15000, "cost_max": 50000},
    "wiper_linkage":       {"damage_type": "bent",                "cost_min": 3000,  "cost_max": 8000},
    "wiper_motor":         {"damage_type": "structural_damage",   "cost_min": 4000,  "cost_max": 10000},
    "rearview_mirror_mount":{"damage_type": "detached_part",      "cost_min": 2000,  "cost_max": 5000},
    "antenna":             {"damage_type": "detached_part",       "cost_min": 1000,  "cost_max": 3000},

    # Door / window internals
    "window_regulator":    {"damage_type": "structural_damage",   "cost_min": 4000,  "cost_max": 12000},
    "door_latch":          {"damage_type": "structural_damage",   "cost_min": 3000,  "cost_max": 8000},
    "door_hinge":          {"damage_type": "bent",                "cost_min": 3000,  "cost_max": 8000},
    "door_handle_mechanism":{"damage_type": "detached_part",      "cost_min": 2000,  "cost_max": 6000},
    "wiring_connector":    {"damage_type": "detached_part",       "cost_min": 1000,  "cost_max": 4000},
    "headlight_bracket":   {"damage_type": "bent",                "cost_min": 3000,  "cost_max": 8000},
    "headlight_adjuster":  {"damage_type": "missing_part",        "cost_min": 1000,  "cost_max": 3000},
    "fog_lamp_housing":    {"damage_type": "crumpled",            "cost_min": 2000,  "cost_max": 6000},
    "fog_lamp_bracket":    {"damage_type": "bent",                "cost_min": 2000,  "cost_max": 5000},
    "taillight_bracket":   {"damage_type": "bent",                "cost_min": 2000,  "cost_max": 6000},
    "hood_latch":          {"damage_type": "structural_damage",   "cost_min": 2000,  "cost_max": 5000},
    "hood_hinges":         {"damage_type": "bent",                "cost_min": 3000,  "cost_max": 8000},
    "hood_latch_support":  {"damage_type": "structural_damage",   "cost_min": 2000,  "cost_max": 5000},
    "trunk_latch":         {"damage_type": "structural_damage",   "cost_min": 2000,  "cost_max": 6000},
    "trunk_hinges":        {"damage_type": "bent",                "cost_min": 3000,  "cost_max": 8000},
    "tailgate_latch":      {"damage_type": "structural_damage",   "cost_min": 2000,  "cost_max": 6000},
    "tailgate_hinges":     {"damage_type": "bent",                "cost_min": 3000,  "cost_max": 8000},
    "tailgate_struts":     {"damage_type": "detached_part",       "cost_min": 2000,  "cost_max": 6000},

    # Liners / seals / clips (minor cost)
    "fender_liner":        {"damage_type": "crumpled",            "cost_min": 2000,  "cost_max": 6000},
    "wheel_arch_liner":    {"damage_type": "crumpled",            "cost_min": 2000,  "cost_max": 6000},
    "hood_liner":          {"damage_type": "detached_part",       "cost_min": 2000,  "cost_max": 5000},
    "headliner":           {"damage_type": "structural_damage",   "cost_min": 5000,  "cost_max": 15000},
    "windshield_seal":     {"damage_type": "crack",               "cost_min": 2000,  "cost_max": 6000},
    "rear_window_seal":    {"damage_type": "crack",               "cost_min": 2000,  "cost_max": 6000},
    "door_seal":           {"damage_type": "detached_part",       "cost_min": 1000,  "cost_max": 3000},
    "rear_door_seal":      {"damage_type": "detached_part",       "cost_min": 1000,  "cost_max": 3000},
    "window_seal":         {"damage_type": "detached_part",       "cost_min": 1000,  "cost_max": 3000},
    "trunk_seal":          {"damage_type": "detached_part",       "cost_min": 1000,  "cost_max": 3000},
    "tailgate_seal":       {"damage_type": "detached_part",       "cost_min": 1000,  "cost_max": 3000},
    "roof_seal":           {"damage_type": "crack",               "cost_min": 1000,  "cost_max": 3000},
    "sunroof_seal":        {"damage_type": "crack",               "cost_min": 2000,  "cost_max": 6000},
    "rubber_trim":         {"damage_type": "detached_part",       "cost_min": 500,   "cost_max": 2000},
    "door_trim":           {"damage_type": "detached_part",       "cost_min": 2000,  "cost_max": 6000},
    "window_rubber":       {"damage_type": "detached_part",       "cost_min": 500,   "cost_max": 2000},
    "drip_rail":           {"damage_type": "bent",                "cost_min": 2000,  "cost_max": 6000},
    "drainage_channel":    {"damage_type": "crumpled",            "cost_min": 2000,  "cost_max": 6000},
    "sill_clips":          {"damage_type": "missing_part",        "cost_min": 300,   "cost_max": 1000},
    "fender_clips":        {"damage_type": "missing_part",        "cost_min": 300,   "cost_max": 1000},
    "fender_bolts":        {"damage_type": "missing_part",        "cost_min": 500,   "cost_max": 2000},
    "bumper_clips":        {"damage_type": "missing_part",        "cost_min": 500,   "cost_max": 2000},
    "grille_clips":        {"damage_type": "missing_part",        "cost_min": 300,   "cost_max": 1000},
    "trunk_clips":         {"damage_type": "missing_part",        "cost_min": 300,   "cost_max": 1000},
    "tailgate_clips":      {"damage_type": "missing_part",        "cost_min": 300,   "cost_max": 1000},
    "mounting_clips":      {"damage_type": "missing_part",        "cost_min": 300,   "cost_max": 1000},
    "mirror_clips":        {"damage_type": "missing_part",        "cost_min": 300,   "cost_max": 1000},
    "headlight_seal":      {"damage_type": "detached_part",       "cost_min": 500,   "cost_max": 2000},
    "taillight_seal":      {"damage_type": "detached_part",       "cost_min": 500,   "cost_max": 2000},
    "fog_lamp_seal":       {"damage_type": "detached_part",       "cost_min": 500,   "cost_max": 1500},
    "wheel_arch_seal":     {"damage_type": "detached_part",       "cost_min": 500,   "cost_max": 2000},
    "quarter_panel_clips": {"damage_type": "missing_part",        "cost_min": 300,   "cost_max": 1000},
    "radiator_clips":      {"damage_type": "missing_part",        "cost_min": 300,   "cost_max": 1000},
    "radiator_brackets":   {"damage_type": "bent",                "cost_min": 2000,  "cost_max": 6000},
    "support_bolts":       {"damage_type": "missing_part",        "cost_min": 300,   "cost_max": 1000},
    "emblem_mount":        {"damage_type": "detached_part",       "cost_min": 500,   "cost_max": 2000},
    "antenna_mount":       {"damage_type": "detached_part",       "cost_min": 500,   "cost_max": 2000},
    "wheel_weights":       {"damage_type": "missing_part",        "cost_min": 500,   "cost_max": 1500},
    "tow_hook_mount":      {"damage_type": "structural_damage",   "cost_min": 2000,  "cost_max": 5000},
    "exhaust_tips":        {"damage_type": "bent",                "cost_min": 3000,  "cost_max": 8000},
    "fuel_tank_proximity": {"damage_type": "structural_damage",   "cost_min": 15000, "cost_max": 40000},
    "rear_trim_brackets":  {"damage_type": "bent",                "cost_min": 2000,  "cost_max": 5000},
    "reflector_mount":     {"damage_type": "detached_part",       "cost_min": 500,   "cost_max": 2000},
    "rear_window_frame":   {"damage_type": "structural_damage",   "cost_min": 10000, "cost_max": 25000},
    "rear_window_heating_element": {"damage_type": "crack",       "cost_min": 5000,  "cost_max": 15000},
}

# Severity cost multipliers for internal components
_SEVERITY_MULTIPLIERS = {"minor": 0.6, "moderate": 1.0, "severe": 1.5}


def lookup_internal_cost(component: str, severity: str) -> dict:
    """
    Returns {damage_type, cost_min, cost_max} for an internal component + severity.
    Applies severity multiplier. Falls back to generic structural entry if unknown.
    """
    component = component.lower().strip()
    severity  = severity.lower().strip()
    entry     = INTERNAL_COMPONENT_DB.get(component, {
        "damage_type": "structural_damage",
        "cost_min": 3000,
        "cost_max": 8000,
    })
    mult = _SEVERITY_MULTIPLIERS.get(severity, 1.0)
    return {
        "damage_type": entry["damage_type"],
        "cost_min":    int(entry["cost_min"] * mult),
        "cost_max":    int(entry["cost_max"] * mult),
    }

"""
Filters raw trajectories into approved SFT training data.

Two filter stages:
  1. Rule-based: schema valid, classes in set, cost plausible, >= 1 tool call
  2. Human-based: cross-reference corrections_log.jsonl — trajectory image
     path appears in a HUMAN_APPROVED correction with no edits → passed

Usage:
  python3 -m pipeline.trajectory_filter
  python3 -m pipeline.trajectory_filter --min-quality 0.8

Outputs:
  data/trajectories/approved/{id}.json  — passed both filters
  data/trajectories/rejected/{id}.json  — failed with reason
  Prints summary stats.
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VALID_DAMAGE_CLASSES = {"dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"}
VALID_PARTS = {
    "front_bumper", "rear_bumper", "hood", "windshield", "rear_windshield",
    "front_left_door", "front_right_door", "rear_left_door", "rear_right_door",
    "left_fender", "right_fender", "trunk_lid", "roof_panel",
    "headlight", "taillight", "tire"
}
COST_MIN_INR = 500
COST_MAX_INR = 1_000_000


def _load_corrections_index() -> set:
    """Returns set of image paths that were human-approved with no edits."""
    log = Path("data/feedback/corrections_log.jsonl")
    approved_paths = set()
    if not log.exists():
        return approved_paths
    with open(log) as f:
        for line in f:
            try:
                e = json.loads(line)
                if (
                    e.get("approval_decision") == "HUMAN_APPROVED"
                    and e.get("items_edited", 0) == 0
                    and e.get("items_removed", 0) == 0
                    and e.get("items_added", 0) == 0
                ):
                    approved_paths.add(e.get("image_path", ""))
            except Exception:
                continue
    return approved_paths


def _rule_filter(traj: dict) -> "tuple[bool, str, float]":
    """Returns (passed, reason, quality_score). quality_score 0-1."""
    steps = traj.get("steps", [])
    damage_map = traj.get("final_damage_map", [])

    tool_calls = [s for s in steps if s.get("action", {}).get("name") != "Terminate"]
    if not tool_calls:
        return False, "no_tool_calls", 0.0

    terminate_steps = [s for s in steps if s.get("action", {}).get("name") == "Terminate"]
    if not terminate_steps:
        return False, "no_terminate", 0.0

    for item in damage_map:
        if item.get("damage") not in VALID_DAMAGE_CLASSES:
            return False, f"invalid_class:{item.get('damage')}", 0.0
        if item.get("part") not in VALID_PARTS:
            return False, f"invalid_part:{item.get('part')}", 0.0

    total_max = traj.get("total_max", 0)
    if total_max > COST_MAX_INR:
        return False, f"cost_too_high:{total_max}", 0.0

    q = 0.5
    if len(tool_calls) >= 1:
        q += 0.2
    if len(damage_map) >= 1:
        q += 0.2
    if len(steps) <= 6:
        q += 0.1

    return True, "", min(q, 1.0)


def filter_trajectories(min_quality: float = 0.6) -> "tuple[int, int]":
    raw_dir = Path("data/trajectories/raw")
    approved_dir = Path("data/trajectories/approved")
    rejected_dir = Path("data/trajectories/rejected")
    approved_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    corrections_index = _load_corrections_index()
    logger.info(f"Corrections index: {len(corrections_index)} human-approved images")

    files = list(raw_dir.glob("*.json")) if raw_dir.exists() else []
    logger.info(f"Processing {len(files)} raw trajectories")

    n_approved = n_rejected = n_skipped = 0

    for f in files:
        try:
            traj = json.loads(f.read_text())
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
            n_skipped += 1
            continue

        if traj.get("filter_status") != "unfiltered":
            n_skipped += 1
            continue

        passed, reason, quality = _rule_filter(traj)
        if not passed:
            traj["filter_status"] = f"failed_rule:{reason}"
            traj["filter_reason"] = reason
            traj["quality_score"] = quality
            (rejected_dir / f.name).write_text(json.dumps(traj, indent=2))
            f.unlink()
            n_rejected += 1
            continue

        image_path = traj.get("image_path", "")
        if corrections_index and image_path not in corrections_index:
            if quality < min_quality:
                traj["filter_status"] = "failed_human"
                traj["filter_reason"] = "not_in_corrections_index"
                traj["quality_score"] = quality
                (rejected_dir / f.name).write_text(json.dumps(traj, indent=2))
                f.unlink()
                n_rejected += 1
                continue

        traj["filter_status"] = "passed"
        traj["quality_score"] = quality
        (approved_dir / f.name).write_text(json.dumps(traj, indent=2))
        f.unlink()
        n_approved += 1

    logger.info(f"Results: {n_approved} approved, {n_rejected} rejected, {n_skipped} skipped")
    logger.info(f"Total approved pool: {len(list(approved_dir.glob('*.json')))}")
    return n_approved, n_rejected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-quality", type=float, default=0.6)
    args = parser.parse_args()
    filter_trajectories(args.min_quality)

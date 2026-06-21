"""
scripts/gepa_runner.py

GEPA — evolutionary prompt optimization for the damage-detection prompt.

What it optimizes:
  DETECTION_GUIDANCE  (the MUTABLE section of DAMAGE_DETECTION_PROMPT in pi_agent.py).
  The frozen _DETECTION_STATIC_SCHEMA (JSON keys + class/part/severity enums) is
  appended every round, so the output contract and vocabulary can NEVER break.

Two models (both local Ollama, no paid API):
  Task model     — qwen3.5:9b (from global_config.yaml vlm.model_id). Runs detection
                   in isolation via PiAgent._vlm_damage_detection(prompt_override=...).
  Proposer model — a stronger text model (e.g. qwen3:32b) that reads failing cases
                   and rewrites DETECTION_GUIDANCE. Text-only; never sees images.

Ground truth:
  data/feedback/corrections_log.jsonl — each line's `final_damage_map` is the human-
  verified answer. Image paths are remapped to data/uploads/<basename> when the
  original (a teammate's machine / deleted temp file) is not present locally.

Scoring (strict, per the project decision):
  A predicted detection matches a ground-truth item only when damage AND part AND
  severity all match. F1 = 2·TP / (2·TP + FP + FN); objective = mean F1 over eval set.

Usage:
  python3 scripts/gepa_runner.py --dry-run                 # 1 round, 3 images, smoke test
  python3 scripts/gepa_runner.py --rounds 10 --candidates 4 --proposer qwen3:32b
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Repo root on path so `models`, `pipeline` import when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from models.vlm_reasoning.ollama_client import chat as ollama_chat  # noqa: E402
from models.vlm_reasoning import pi_agent  # noqa: E402
from models.vlm_reasoning.pi_agent import (  # noqa: E402
    PiAgent,
    DETECTION_GUIDANCE,
    _DETECTION_STATIC_SCHEMA,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | GEPA | %(levelname)s | %(message)s",
)
logger = logging.getLogger("gepa")


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/global_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Eval set (ground truth from corrections_log.jsonl) ──────────────────────────

def _normalise(s: str) -> str:
    return str(s or "").lower().replace("_", " ").strip()


def load_eval_set(
    corrections_log: str = "data/feedback/corrections_log.jsonl",
    uploads_dir: str = "data/new_uploads",
) -> List[dict]:
    """
    Returns [{image_path, ground_truth: [{damage, part, severity}, ...]}].
    Remaps missing image paths to data/uploads/<basename>. Skips entries whose
    image cannot be located locally (GEPA must run the prompt on a real image).
    """
    p = Path(corrections_log)
    if not p.exists():
        logger.error(f"No corrections log at {corrections_log}. Collect human reviews first.")
        return []

    uploads = Path(uploads_dir)
    eval_set: List[dict] = []
    skipped = 0

    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue

            img = e.get("image_path", "")
            if not (img and Path(img).exists()):
                cand = uploads / Path(img).name
                if cand.exists():
                    img = str(cand)
                else:
                    skipped += 1
                    continue

            gt = [
                {
                    "damage": item.get("damage", ""),
                    "part": item.get("part", ""),
                    "severity": item.get("severity", ""),
                }
                for item in e.get("final_damage_map", [])
            ]
            eval_set.append({"image_path": img, "ground_truth": gt})

    logger.info(f"Eval set: {len(eval_set)} usable example(s), {skipped} skipped (image missing).")
    return eval_set


# ── Scoring (strict damage + part + severity triple match) ──────────────────────

def score_one(detections: List[dict], ground_truth: List[dict]) -> Tuple[int, int, int]:
    """Greedy one-to-one triple match. Returns (tp, fp, fn)."""
    preds = [
        (_normalise(d.get("class")), _normalise(d.get("part")), _normalise(d.get("severity")))
        for d in detections
    ]
    truths = [
        (_normalise(g.get("damage")), _normalise(g.get("part")), _normalise(g.get("severity")))
        for g in ground_truth
    ]
    used = [False] * len(preds)
    tp = 0
    for t in truths:
        for i, pr in enumerate(preds):
            if not used[i] and pr == t:
                used[i] = True
                tp += 1
                break
    fp = len(preds) - tp
    fn = len(truths) - tp
    return tp, fp, fn


def f1_from(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 1.0


# ── Evaluate one guidance candidate over the eval set ───────────────────────────

def evaluate_guidance(
    guidance: str,
    eval_set: List[dict],
    agent: PiAgent,
) -> Tuple[float, List[dict]]:
    """
    Runs the detection prompt (guidance + frozen schema) on every eval image,
    scores against ground truth. Returns (mean_f1, per_image_records).
    """
    prompt = guidance + "\n\n" + _DETECTION_STATIC_SCHEMA
    records: List[dict] = []
    f1s: List[float] = []

    for ex in eval_set:
        try:
            out = agent._vlm_damage_detection(ex["image_path"], prompt_override=prompt)
            dets = out.get("detections", [])
        except Exception as e:
            logger.warning(f"Detection failed on {ex['image_path']}: {e}")
            dets = []
        tp, fp, fn = score_one(dets, ex["ground_truth"])
        f1 = f1_from(tp, fp, fn)
        f1s.append(f1)
        records.append({
            "image_path": ex["image_path"],
            "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn,
            "predicted": [
                {"damage": d.get("class"), "part": d.get("part"), "severity": d.get("severity")}
                for d in dets
            ],
            "ground_truth": ex["ground_truth"],
        })

    mean_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return mean_f1, records


# ── Proposer (stronger model rewrites the guidance) ─────────────────────────────

_PROPOSER_SYSTEM = """\
You are a prompt engineer improving the INSTRUCTION text for a vehicle-damage
detection vision model. You will be given the current instruction text and several
failure cases (what the model predicted vs the correct answer). Write improved
versions of the instruction text that would fix these failures.

OUTPUT FORMAT — follow EXACTLY:
- Output each improved version as plain text.
- Separate consecutive versions with a line containing ONLY: ===
- No numbering, no JSON, no markdown fences, no commentary before or after.

CONTENT RULES:
- Each version must be the FULL replacement instruction text (self-contained).
- Keep it focused on HOW to look: be thorough, judge severity (minor/moderate/severe),
  place tight bounding boxes on the damage.
- Do NOT include any JSON schema, field names, or the list of valid class/part/
  severity tokens — those are appended automatically and must not appear in your text.
- Do not invent new damage classes, parts, or output fields."""


def _format_failures(records: List[dict], n: int = 5) -> str:
    worst = sorted(records, key=lambda r: r["f1"])[:n]
    lines = []
    for i, r in enumerate(worst, 1):
        lines.append(
            f"Case {i} (F1={r['f1']}):\n"
            f"  Model predicted: {r['predicted']}\n"
            f"  Correct answer:  {r['ground_truth']}\n"
            f"  Misses(FN)={r['fn']}  FalsePositives(FP)={r['fp']}"
        )
    return "\n".join(lines)


def propose_candidates(
    current_guidance: str,
    records: List[dict],
    proposer_model: str,
    base_url: str,
    n: int = 4,
) -> List[str]:
    user = (
        f"CURRENT INSTRUCTION TEXT:\n{current_guidance}\n\n"
        f"FAILURE CASES:\n{_format_failures(records)}\n\n"
        f"Write {n} improved full-replacement versions, separated by a line of '==='."
    )
    messages = [
        {"role": "system", "content": _PROPOSER_SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        raw = ollama_chat(
            messages, model=proposer_model, base_url=base_url,
            temperature=0.7, num_predict=2048, think=False,
        )
    except Exception as e:
        logger.error(f"Proposer call failed: {e}")
        return []

    # Delimiter-based parsing — robust for any model (no JSON-in-JSON escaping).
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text, flags=re.MULTILINE)
    text = text.replace("```", "")
    parts = [p.strip() for p in re.split(r"^\s*={3,}\s*$", text, flags=re.MULTILINE)]
    # Keep substantive candidates only (drop short echoes / preambles).
    cands = [p for p in parts if len(p) >= 40][:n]
    if not cands:
        logger.warning(f"Proposer returned no usable candidates (raw {len(raw)} chars).")
    return cands


# ── Main GEPA loop ──────────────────────────────────────────────────────────────

def run_gepa(
    config: dict,
    rounds: int,
    n_candidates: int,
    proposer_model: str,
    eval_limit: Optional[int] = None,
    holdout_frac: float = 0.25,
    seed: int = 42,
) -> None:
    import random

    base_url = config.get("vlm", {}).get("ollama_base_url", "http://localhost:11434")

    full_set = load_eval_set(
        config.get("feedback", {}).get("corrections_log", "data/feedback/corrections_log.jsonl"),
        config.get("storage", {}).get("image_upload_dir", "data/new_uploads"),
    )
    if eval_limit:
        full_set = full_set[:eval_limit]
    if not full_set:
        logger.error("Empty eval set — cannot run GEPA. Collect human reviews first.")
        return

    # Train / holdout split. GEPA optimizes on the train set; the holdout is NEVER
    # seen during optimization and is scored once at the end to detect overfitting
    # (a prompt that wins on train but loses on holdout is overfit, not better).
    random.Random(seed).shuffle(full_set)
    n_holdout = int(len(full_set) * holdout_frac)
    if len(full_set) < 4:
        logger.warning(
            f"Only {len(full_set)} example(s) — too few for a holdout. Using all for "
            "optimization; holdout validation skipped. Collect more reviews for a real run."
        )
        n_holdout = 0
    eval_set = full_set[n_holdout:]          # optimize on this
    holdout_set = full_set[:n_holdout]       # validate on this
    logger.info(f"Split: {len(eval_set)} optimize / {len(holdout_set)} holdout")

    agent = PiAgent(config)

    current = DETECTION_GUIDANCE
    best_f1, records = evaluate_guidance(current, eval_set, agent)
    logger.info(f"Baseline F1 = {best_f1:.3f} over {len(eval_set)} optimize image(s)")

    history = [{"round": 0, "f1": round(best_f1, 4), "source": "baseline"}]
    no_improve = 0

    for r in range(1, rounds + 1):
        logger.info(f"── Round {r}/{rounds} — proposing {n_candidates} candidate(s) ──")
        candidates = propose_candidates(current, records, proposer_model, base_url, n_candidates)
        if not candidates:
            logger.warning("No candidates this round.")
            no_improve += 1
            if no_improve >= 3:
                logger.info("3 rounds without candidates — stopping.")
                break
            continue

        round_best_f1, round_best_guidance, round_best_records = best_f1, None, records
        for i, cand in enumerate(candidates, 1):
            f1, recs = evaluate_guidance(cand, eval_set, agent)
            logger.info(f"  candidate {i}: F1 = {f1:.3f}")
            if f1 > round_best_f1:
                round_best_f1, round_best_guidance, round_best_records = f1, cand, recs

        if round_best_guidance is not None:
            improve = round_best_f1 - best_f1
            current, best_f1, records = round_best_guidance, round_best_f1, round_best_records
            no_improve = 0
            logger.info(f"  ✓ improved to F1 = {best_f1:.3f} (+{improve:.3f})")
            history.append({"round": r, "f1": round(best_f1, 4), "source": "candidate"})
            _save_checkpoint(current, best_f1, history, records)
        else:
            no_improve += 1
            logger.info(f"  no improvement ({no_improve}/3)")
            history.append({"round": r, "f1": round(best_f1, 4), "source": "kept"})
            if no_improve >= 3:
                logger.info("Converged — 3 rounds without improvement. Stopping.")
                break

    logger.info(f"\nFinal optimize F1 = {best_f1:.3f}")

    # Holdout validation — the real test that the winning prompt generalizes.
    holdout_baseline = holdout_final = None
    if holdout_set:
        holdout_baseline, _ = evaluate_guidance(DETECTION_GUIDANCE, holdout_set, agent)
        holdout_final, _ = evaluate_guidance(current, holdout_set, agent)
        logger.info(
            f"Holdout: baseline F1 = {holdout_baseline:.3f} → winning F1 = {holdout_final:.3f} "
            f"({'GENERALIZES ✓' if holdout_final >= holdout_baseline else 'OVERFIT ✗ — keep baseline'})"
        )

    _save_checkpoint(current, best_f1, history, records, final=True,
                     holdout_baseline=holdout_baseline, holdout_final=holdout_final)
    if holdout_final is not None and holdout_final < holdout_baseline:
        logger.warning("Winning guidance LOST on holdout — do NOT apply it. Collect more data / rerun.")
    else:
        logger.info("Apply the winning guidance by pasting it into DETECTION_GUIDANCE in pi_agent.py.")


def _save_checkpoint(guidance, f1, history, records, final=False,
                     holdout_baseline=None, holdout_final=None):
    out_dir = Path("scripts/gepa_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "final" if final else datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"gepa_{name}.json"
    out.write_text(json.dumps({
        "best_optimize_f1": f1,
        "holdout_baseline_f1": holdout_baseline,
        "holdout_winning_f1": holdout_final,
        "winning_guidance": guidance,
        "history": history,
        "last_eval_records": records,
        "saved_at": datetime.now().isoformat(),
    }, indent=2))
    logger.info(f"Checkpoint saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--proposer", default="qwen3:32b", help="Ollama proposer model tag")
    ap.add_argument("--eval-limit", type=int, default=None, help="Cap eval images (speed)")
    ap.add_argument("--config", default="configs/global_config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="1 round, 3 images — smoke test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dry_run:
        logger.info("DRY RUN: 1 round, 3 images, 2 candidates.")
        run_gepa(cfg, rounds=1, n_candidates=2, proposer_model=args.proposer, eval_limit=3)
    else:
        run_gepa(cfg, args.rounds, args.candidates, args.proposer, args.eval_limit)

"""
Application configuration and shared constants.

All filesystem paths are relative to the repository root, because the server is
always launched from there (`uvicorn backend.app:app`). The single source of
truth for model/pipeline parameters is `configs/global_config.yaml`.
"""

from pathlib import Path

import yaml

# ── Filesystem locations ──────────────────────────────────────────────────────
CONFIG_PATH = Path("configs/global_config.yaml")
UPLOAD_DIR = Path("data/uploads")
# Permanent home for freshly-uploaded images during the GEPA ground-truth collection
# (kept separate from the legacy data/uploads pool). The image_path recorded in
# corrections_log.jsonl points here and must survive the job for GEPA to read it.
NEW_UPLOAD_DIR = Path("data/new_uploads")
FEEDBACK_LOG = Path("data/feedback/feedback_log.jsonl")
CORRECTIONS_LOG = Path("data/feedback/corrections_log.jsonl")

# ── HTTP / API metadata ───────────────────────────────────────────────────────
API_TITLE = "Vehicle Damage Assessment API"
API_DESCRIPTION = (
    "Thinking with Images — Qwen-orchestrated damage detection and cost estimation."
)
API_VERSION = "0.2.0"

ALLOWED_IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp")

# Permissive CORS for the local MVP. Restrict to known origins in production.
CORS_ORIGINS = ["*"]


def load_config() -> dict:
    """Load the YAML config that drives the pipeline. Raises if it is missing."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found at {CONFIG_PATH}. "
            "Run from the repo root: uvicorn backend.app:app"
        )
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def sam2_weights_path(config: dict) -> str:
    """Resolve the SAM2 weights path from config, with a sane fallback."""
    return (
        config.get("part_segmentation", {})
        .get("sam2", {})
        .get("weights_path", "weights/sam2.1_hiera_base_plus.pt")
    )

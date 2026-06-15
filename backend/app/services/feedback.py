"""Feedback-log persistence (append-only JSONL)."""

import logging

from pipeline.schema import FeedbackEntry

from ..core import config

logger = logging.getLogger(__name__)


def write_feedback(entry: FeedbackEntry) -> None:
    """Append one feedback entry to the feedback log."""
    config.FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(config.FEEDBACK_LOG, "a") as f:
        f.write(entry.model_dump_json() + "\n")
    logger.info(f"Feedback written: session={entry.session_id}")

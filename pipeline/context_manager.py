"""
pipeline/context_manager.py

Sliding window context manager for multi-turn adjuster follow-up interactions.
Implements the three-tier memory model from the architecture diagram:
  PINNED   → always in context (system prompt, vehicle ID, damage summary)
  RETAINED → last max_active_turns message pairs + tool results
  DROPPED  → compressed to one-line summaries

Usage:
  ctx = ClaimContext()
  ctx.pin("vehicle_id", "MH12AB1234")
  ctx.pin("damage_summary", report["damage_part_map"])
  messages = ctx.build_messages("Can you check if the subframe is affected?")
  # pass messages to orchestrator._run_tool_loop()
  ctx.add_turn(user_msg, assistant_msg, tool_results)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClaimContext:
    """
    Manages conversation context across multiple adjuster turns on one claim.

    Do NOT share a ClaimContext instance across different claims.
    Create one per claim submission and persist it server-side keyed by claim_id.
    """
    max_active_turns: int = 3
    pinned: Dict[str, Any] = field(default_factory=dict)
    active_window: List[dict] = field(default_factory=list)
    compressed_history: List[str] = field(default_factory=list)

    def pin(self, key: str, value: Any) -> None:
        """
        Pin a value that will always appear in context.
        Use for: vehicle_id, claim_number, initial damage summary.

        Args:
            key: Label for the pinned item
            value: Any JSON-serialisable value
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Pinned key must be a non-empty string")
        self.pinned[key] = value
        logger.debug(f"Pinned context key: '{key}'")

    def add_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        tool_results: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a completed interaction turn to the active window.
        If window is full, the oldest turn is compressed and moved to history.

        Args:
            user_msg: The user's message text
            assistant_msg: The assistant's response text
            tool_results: Optional dict of {tool_name: result} from this turn
        """
        turn = {
            "user": user_msg,
            "assistant": assistant_msg,
            "tool_results": tool_results or {}
        }
        self.active_window.append(turn)

        if len(self.active_window) > self.max_active_turns:
            dropped = self.active_window.pop(0)
            tools_used = list(dropped["tool_results"].keys())
            summary = (
                f"[Earlier turn] User: '{dropped['user'][:80]}' | "
                f"Tools used: {tools_used} | "
                f"Assistant summary: '{dropped['assistant'][:120]}'"
            )
            self.compressed_history.append(summary)
            logger.debug(f"Compressed turn into history. Active turns: {len(self.active_window)}")

    def build_messages(self, new_user_message: str) -> list:
        """
        Assemble the complete message list for the next VLM call.
        Order: pinned context → compressed history → active window → new message.

        Args:
            new_user_message: The latest user input

        Returns:
            List of message dicts in Qwen2-VL conversation format
        """
        if not new_user_message or not new_user_message.strip():
            raise ValueError("new_user_message cannot be empty")

        messages = []

        # Inject pinned context as a system-level addendum
        if self.pinned:
            pinned_lines = "\n".join(
                f"  {k}: {json.dumps(v, default=str)}"
                for k, v in self.pinned.items()
            )
            messages.append({
                "role": "system",
                "content": f"PINNED CLAIM CONTEXT (always available):\n{pinned_lines}"
            })

        # Compressed history as a single assistant-narrated block
        if self.compressed_history:
            history_block = "\n".join(self.compressed_history)
            messages.append({
                "role": "assistant",
                "content": f"[Summary of earlier conversation turns]\n{history_block}"
            })

        # Active window turns
        for turn in self.active_window:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        # New user message
        messages.append({"role": "user", "content": new_user_message})

        logger.debug(
            f"Built message list: {len(messages)} messages "
            f"({len(self.compressed_history)} compressed, "
            f"{len(self.active_window)} active)"
        )
        return messages

    @property
    def turn_count(self) -> int:
        """Total turns processed (active + compressed)."""
        return len(self.active_window) + len(self.compressed_history)

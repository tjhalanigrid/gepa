"""
Converts approved trajectories to SFT conversation format.

Output: data/sft_dataset/train.jsonl and val.jsonl
Format: HuggingFace messages format compatible with TRL SFTTrainer

Usage:
  python3 scripts/prepare_sft_dataset.py
  python3 scripts/prepare_sft_dataset.py --val-split 0.2 --min-quality 0.7
"""

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT_PLACEHOLDER = "CODEACT_SYSTEM_PROMPT"  # replaced at runtime


def trajectory_to_conversations(traj: dict) -> list:
    """
    Converts a trajectory into a list of conversation messages.
    Format: [{role, content}] where content can be list for multimodal.

    Each (thought+action, observation) pair becomes one assistant + user turn.
    The final Terminate is the last assistant turn.
    """
    messages = []

    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT_PLACEHOLDER
    })

    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{traj['image_path']}"},
            {"type": "text", "text": "Assess all vehicle damage visible in this image. Follow the output format exactly."}
        ]
    })

    steps = traj.get("steps", [])
    i = 0
    while i < len(steps):
        step = steps[i]
        action = step.get("action", {})
        action_name = action.get("name", "")

        codeact_output = {
            "thought": f"Calling {action_name}: {step.get('observation_summary', '')}",
            "uncertainty": [],
            "actions": [action]
        }

        if action_name == "Terminate":
            term_args = action.get("arguments", {})
            items = term_args.get("damage_items", [])
            avg_conf = sum(x.get("confidence", 0.8) for x in items) / max(len(items), 1)
            codeact_output["confidence"] = round(avg_conf, 2)
            messages.append({
                "role": "assistant",
                "content": json.dumps(codeact_output)
            })
            break

        messages.append({
            "role": "assistant",
            "content": json.dumps(codeact_output)
        })

        obs_type = step.get("observation_type", "json")
        obs_img = step.get("observation_image_path")
        obs_summary = step.get("observation_summary", "")

        if obs_type == "image" and obs_img and Path(obs_img).exists():
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{obs_img}"},
                    {"type": "text", "text": f"Tool result for {action_name}: {obs_summary}. Continue your assessment based on what you now see."}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Tool result for {action_name}: {obs_summary}. Continue your assessment."
            })

        i += 1

    return messages


def prepare_dataset(val_split: float = 0.2, min_quality: float = 0.6):
    approved_dir = Path("data/trajectories/approved")
    out_dir = Path("data/sft_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not approved_dir.exists():
        print(f"Approved directory not found: {approved_dir}")
        print("Run: python3 -m pipeline.trajectory_filter first.")
        return

    files = list(approved_dir.glob("*.json"))
    print(f"Found {len(files)} approved trajectories")

    trajs = []
    for f in files:
        try:
            t = json.loads(f.read_text())
            if t.get("quality_score", 0) >= min_quality:
                trajs.append(t)
        except Exception as e:
            print(f"Skip {f}: {e}")

    if len(trajs) < 10:
        print(f"Only {len(trajs)} trajectories meet quality threshold {min_quality}.")
        print("Need at least 10. Collect more approved trajectories first.")
        return

    random.shuffle(trajs)
    split = int(len(trajs) * (1 - val_split))
    train_trajs = trajs[:split]
    val_trajs = trajs[split:]

    for name, subset in [("train", train_trajs), ("val", val_trajs)]:
        out_path = out_dir / f"{name}.jsonl"
        with open(out_path, "w") as f:
            for traj in subset:
                convs = trajectory_to_conversations(traj)
                f.write(json.dumps({"messages": convs}) + "\n")
        print(f"Written {len(subset)} examples to {out_path}")

    print(f"\nDataset ready: {split} train / {len(trajs) - split} val")
    print("Next step: run scripts/sft_train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--min-quality", type=float, default=0.6)
    args = parser.parse_args()
    prepare_dataset(args.val_split, args.min_quality)

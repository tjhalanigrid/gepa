"""
Cold-start SFT fine-tune of Qwen2-VL-2B on vehicle damage trajectories.

Prerequisites:
  pip install trl peft --break-system-packages
  200+ approved trajectories in data/trajectories/approved/
  SFT dataset prepared: python3 scripts/prepare_sft_dataset.py

Usage:
  python3 scripts/sft_train.py
  python3 scripts/sft_train.py --model-id Qwen/Qwen2-VL-2B-Instruct --epochs 3

After training:
  New checkpoint saved to models/fine_tuned/veh_dmg_sft_{timestamp}/
  Update configs/global_config.yaml: model_id to the checkpoint path
  Restart uvicorn.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def train(model_id: str, epochs: int, batch_size: int, grad_accum: int):
    try:
        from trl import SFTTrainer, SFTConfig
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install trl peft --break-system-packages")
        return

    train_path = Path("data/sft_dataset/train.jsonl")
    val_path = Path("data/sft_dataset/val.jsonl")

    if not train_path.exists():
        print("SFT dataset not found. Run: python3 scripts/prepare_sft_dataset.py")
        return

    n_train = sum(1 for _ in open(train_path))
    n_val = sum(1 for _ in open(val_path)) if val_path.exists() else 0
    print(f"Training on {n_train} examples, validating on {n_val}")

    if n_train < 10:
        print(f"Only {n_train} training examples. Need at least 10. Collect more trajectories.")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: Training on CPU will be very slow (days not hours).")
        print("Consider renting an A100 on Vast.ai (~$1.50/hr) for overnight runs.")

    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    dtype = torch.float32 if device in ("mps", "cpu") else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Import the actual system prompt to replace the placeholder
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pipeline.orchestrator import CODEACT_SYSTEM_PROMPT

    def load_dataset(path):
        from datasets import Dataset
        examples = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                for msg in ex["messages"]:
                    if msg.get("role") == "system" and msg.get("content") == "CODEACT_SYSTEM_PROMPT":
                        msg["content"] = CODEACT_SYSTEM_PROMPT
                examples.append(ex)
        return Dataset.from_list(examples)

    train_dataset = load_dataset(train_path)
    eval_dataset = load_dataset(val_path) if val_path.exists() else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"models/fine_tuned/veh_dmg_sft_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_steps=50 if eval_dataset else None,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        report_to="none",
        max_seq_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.tokenizer,
    )

    print(f"Starting training... output: {output_dir}")
    print(f"Estimated time on MPS: {n_train * epochs * 0.8 / 60:.0f} minutes")
    trainer.train()

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    print(f"\nTraining complete. Checkpoint: {output_dir}")
    print("\nTo deploy:")
    print(f"  Edit configs/global_config.yaml:")
    print(f"    model_id: \"{output_dir}\"")
    print("  Restart uvicorn.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()
    train(args.model_id, args.epochs, args.batch_size, args.grad_accum)

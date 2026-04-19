#!/usr/bin/env python3
"""Train LoRA on Qwen3.6-35B-A3B using Unsloth (4-bit, CUDA).

Runs on kxkm-ai RTX 4090 24GB. Uses 4-bit quantization to fit in VRAM.
"""
import argparse
import json
import sys
import time
from pathlib import Path

CURRICULUM = [
    "chat-fr", "reasoning", "python", "typescript", "cpp", "rust",
    "html-css", "shell", "sql", "yaml-json", "docker", "kicad-dsl", "spice", "lua-upy",
    "embedded", "stm32", "iot", "freecad", "platformio", "power", "emc", "dsp",
    "spice-sim", "electronics", "kicad-pcb",
    "web-frontend", "web-backend", "music-audio", "devops", "llm-orch",
    "math", "security",
    "components", "llm-ops", "ml-training",
]


def train_domain(model, tokenizer, domain, data_dir, output_dir, max_steps=300):
    """Train one domain adapter."""
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    train_file = data_dir / domain / "train.jsonl"
    if not train_file.exists():
        print(f"  SKIP {domain}: no data")
        return None

    # Load data
    items = []
    for line in open(train_file):
        try:
            d = json.loads(line.strip())
            msgs = d.get("messages", [])
            if len(msgs) >= 2:
                clean = [{"role": m["role"], "content": str(m["content"])} for m in msgs if isinstance(m, dict)]
                if len(clean) >= 2:
                    items.append({"messages": clean})
        except:
            pass

    if not items:
        print(f"  SKIP {domain}: empty data")
        return None

    n = len(items)
    iters = min(max_steps, max(50, n // 30))
    print(f"  {domain}: {n} examples, {iters} steps")

    train_ds = Dataset.from_list(items)
    stack_dir = output_dir / domain
    stack_dir.mkdir(parents=True, exist_ok=True)

    def formatting_func(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return texts

    config = SFTConfig(
        output_dir=str(stack_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        max_steps=iters,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=25,
        save_steps=iters,  # save at end
        bf16=True,
        seed=42,
        max_seq_length=1024,
        dataset_text_field="",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        formatting_func=formatting_func,
        args=config,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Save adapter
    model.save_pretrained(str(stack_dir))
    tokenizer.save_pretrained(str(stack_dir))

    # Eval
    eval_result = trainer.evaluate()
    val_loss = eval_result.get("eval_loss", 0)

    meta = {"domain": domain, "val_loss": val_loss, "steps": iters,
            "examples": n, "elapsed_s": round(elapsed), "model": "Qwen3.6-35B-A3B-4bit"}
    with open(stack_dir / "stack_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  {domain}: val_loss={val_loss:.4f} in {elapsed:.0f}s")
    return val_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--data-dir", default="/home/kxkm/micro-kiki/data/v3")
    ap.add_argument("--output-dir", default="/home/kxkm/micro-kiki/output/lora-qwen36-35b-4bit")
    ap.add_argument("--resume-from", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Qwen3.6-35B LoRA (Unsloth 4-bit CUDA)")
    print(f"  Model: {args.model}")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing="unsloth",
    )

    print(f"Model loaded. {model.print_trainable_parameters()}")

    completed = 0
    for i, domain in enumerate(CURRICULUM, 1):
        if i < args.resume_from:
            continue
        if (output_dir / domain / "adapter_model.safetensors").exists():
            print(f"[{i}] SKIP {domain} (done)")
            continue

        print(f"\n[{i}/{len(CURRICULUM)}] {domain}")
        val = train_domain(model, tokenizer, domain, data_dir, output_dir, args.max_steps)
        if val is not None:
            completed += 1

    print(f"\nDone: {completed}/{len(CURRICULUM)}")


if __name__ == "__main__":
    main()

"""Stack trainer — standard LoRA on Qwen3.5-35B-A3B MoE base.

Post-pivot: MoE-LoRA replaced by standard LoRA. The base model is already
a 256-expert MoE — we only tune attention projections, not MoE FFN layers.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REQUIRED_KEYS = [
    "base_model", "lora_rank", "lora_alpha",
    "learning_rate", "batch_size", "grad_accum", "epochs", "seq_len", "dataset",
]

DEFAULT_BASE = "Qwen/Qwen3.5-35B-A3B"


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    dropout: float = 0.0


def load_training_config(config_path: Path | str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key in REQUIRED_KEYS:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
    return config


def lora_config_from_dict(config: dict) -> LoRAConfig:
    return LoRAConfig(
        rank=config["lora_rank"],
        alpha=config["lora_alpha"],
    )


class StackTrainer:
    """Trains a single LoRA adapter on Qwen3.5-35B-A3B for a domain."""

    def __init__(
        self,
        base_model_path: str = DEFAULT_BASE,
        lora_config: LoRAConfig | None = None,
        output_dir: str = "outputs/stacks/stack-01",
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        grad_accum: int = 8,
        epochs: int = 3,
        seq_len: int = 4096,
        init_lora_weights: str = "default",
    ) -> None:
        self.base_model_path = base_model_path
        self.lora_config = lora_config or LoRAConfig()
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.epochs = epochs
        self.seq_len = seq_len
        self.init_lora_weights = init_lora_weights

    @classmethod
    def from_config(cls, config_path: str | Path) -> StackTrainer:
        config = load_training_config(config_path)
        lc = lora_config_from_dict(config)
        return cls(
            base_model_path=config.get("base_model", DEFAULT_BASE),
            lora_config=lc,
            output_dir=f"outputs/stacks/{Path(config_path).stem}",
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            grad_accum=config["grad_accum"],
            epochs=config["epochs"],
            seq_len=config["seq_len"],
            init_lora_weights=config.get("init_lora_weights", "default"),
        )

    def train(self, dataset_path: str) -> dict:
        """Train the LoRA adapter. Requires torch + transformers + peft + trl.

        IMPORTANT: Set UNSLOTH_COMPILE_DISABLE=1 before import to avoid
        mixed-precision kernel errors with MoE layers.
        """
        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig as PeftLoraConfig, get_peft_model
        from trl import SFTTrainer
        from datasets import load_dataset

        logger.info("Loading base model: %s", self.base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        peft_config = PeftLoraConfig(
            r=self.lora_config.rank,
            lora_alpha=self.lora_config.alpha,
            target_modules=list(self.lora_config.target_modules),
            lora_dropout=self.lora_config.dropout,
            init_lora_weights=self.init_lora_weights,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        dataset = load_dataset("json", data_files=dataset_path, split="train")

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_path),
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            bf16=True,
            gradient_checkpointing=True,  # Required: 74 GB is tight
            logging_steps=10,
            save_strategy="epoch",
            dataloader_num_workers=0,  # MoE compat
            dataset_num_proc=1,  # MoE compat
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        result = trainer.train()
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))

        metrics = {
            "train_loss": result.training_loss,
            "output_dir": str(output_path),
            "epochs": self.epochs,
            "base_model": self.base_model_path,
            "lora_rank": self.lora_config.rank,
        }
        logger.info("Training complete: %s", metrics)
        return metrics

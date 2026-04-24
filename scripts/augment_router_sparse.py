#!/usr/bin/env python3
"""
Augment router-v4 dataset by extracting prompts from LoRA adapter training data
for sparse domains and missing domains.
"""

import json
import random
from collections import Counter
from pathlib import Path

ROUTER_DIR = Path("/Users/clems/Projets/micro-kiki/data/router-v4")
LORA_DIR = Path("/Users/clems/KIKI-Mac_tunner/data/micro-kiki")
IOT_PROMPTS = Path("/Users/clems/Projets/micro-kiki/data/prompts/iot.jsonl")

TRAIN_FILE = ROUTER_DIR / "train.jsonl"
VALID_FILE = ROUTER_DIR / "valid.jsonl"

# French function words for detection
FRENCH_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "est",
    "en", "dans", "que", "qui", "pour", "par", "sur", "avec", "ou",
    "mais", "donc", "car", "si", "je", "tu", "il", "elle", "nous",
    "vous", "ils", "elles", "ce", "se", "sa", "son", "ses", "mon",
    "ton", "ma", "ta", "mes", "tes", "au", "aux", "cet", "cette",
    "ces", "leur", "leurs", "quand", "comme", "plus", "très", "bien",
}

# Technical keywords per domain (at least one must appear for domain-specific content)
DOMAIN_TECH_KEYWORDS = {
    "freecad": ["freecad", "part design", "sketcher", "constraint", "fillet", "chamfer",
                "workbench", "solid", "cad", "3d model", "sketch", "extrude", "pocket",
                "body", "assembly", "spreadsheet", "macro", "python", "topological"],
    "stm32": ["stm32", "hal", "cubemx", "cortex", "uart", "spi", "i2c", "dma", "rtos",
              "freertos", "interrupt", "gpio", "timer", "adc", "dac", "flash", "clock",
              "mcu", "microcontroller", "arm", "register", "peripheral", "nucleo",
              "discovery", "bootloader", "cmsis"],
    "platformio": ["platformio", "pio", "arduino", "esp32", "esp8266", "board", "library",
                   "firmware", "upload", "serial", "embedded", "microcontroller", "ini",
                   "platformio.ini", "framework", "avr", "stm32", "nrf52", "rp2040",
                   "monitor", "build", "flash", "debug"],
    "dsp": ["dsp", "signal", "filter", "fft", "fourier", "convolution", "frequency",
            "sampling", "nyquist", "biquad", "iir", "fir", "windowing", "spectrum",
            "noise", "bandwidth", "aliasing", "decimation", "interpolation", "audio",
            "waveform", "magnitude", "phase", "amplitude", "hz", "khz"],
    "emc": ["emc", "electromagnetic", "emi", "conducted", "radiated", "shielding",
            "ferrite", "ground plane", "bypass capacitor", "decoupling", "pcb",
            "trace", "impedance", "crosstalk", "common mode", "differential",
            "fcc", "ce", "cispr", "regulatory", "compliance", "emission", "susceptibility"],
    "iot": ["iot", "mqtt", "sensor", "device", "gateway", "cloud", "wifi", "bluetooth",
            "zigbee", "lorawan", "protocol", "firmware", "esp32", "raspberry", "arduino",
            "node-red", "home assistant", "azure iot", "aws iot", "thingsboard"],
    "components": ["component", "resistor", "capacitor", "inductor", "transistor", "diode",
                   "mosfet", "bjt", "op-amp", "ic", "chip", "datasheet", "footprint",
                   "package", "smd", "through-hole", "bom", "schematic", "kicad"],
    "llm-ops": ["llm", "model", "inference", "prompt", "token", "context", "fine-tuning",
                "quantization", "gguf", "ollama", "vllm", "langchain", "llamaindex",
                "embedding", "vector", "rag", "gpu", "cuda", "deployment", "api"],
    "ml-training": ["training", "model", "dataset", "loss", "gradient", "optimizer",
                    "epoch", "batch", "learning rate", "overfitting", "validation",
                    "pytorch", "tensorflow", "keras", "neural network", "layer",
                    "backpropagation", "fine-tuning", "lora", "qlora", "unsloth"],
}

# Domains config: (source_type, cap)
# source_type: "lora" or "prompts_file"
DOMAINS = {
    "freecad":     ("lora", 2000),
    "stm32":       ("lora", 2000),
    "platformio":  ("lora", 694),   # only 694 available
    "dsp":         ("lora", 2000),
    "emc":         ("lora", 2000),
    "iot":         ("prompts_file", 200),
    "components":  ("lora", 2000),
    "llm-ops":     ("lora", 1722),  # all available
    "ml-training": ("lora", 1975),  # all available
}

VALID_RATIO = 0.20
random.seed(42)


def is_too_french(prompt: str, domain: str) -> bool:
    """Return True if the prompt is generic French text with no domain keywords."""
    words = prompt.lower().split()
    french_count = sum(1 for w in words if w.rstrip(".,;:!?") in FRENCH_WORDS)
    if french_count <= 8:
        return False  # Not enough French to flag

    # Check for any domain technical keyword
    prompt_lower = prompt.lower()
    keywords = DOMAIN_TECH_KEYWORDS.get(domain, [])
    has_tech = any(kw in prompt_lower for kw in keywords)
    return not has_tech  # French AND no tech keyword → skip


def extract_from_lora(domain: str, cap: int) -> list[dict]:
    """Extract user prompts from LoRA training data."""
    source_file = LORA_DIR / domain / "train.jsonl"
    if not source_file.exists():
        print(f"  WARNING: {source_file} not found, skipping.")
        return []

    candidates = []
    with open(source_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = entry.get("messages", [])
            if not messages:
                continue

            # Get first user message
            user_content = None
            for msg in messages:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "").strip()
                    break

            if not user_content:
                continue

            # Skip too short
            if len(user_content) < 20:
                continue

            # Skip French garbage
            if is_too_french(user_content, domain):
                continue

            candidates.append({"prompt": user_content, "domain": domain})

    # Shuffle and cap
    random.shuffle(candidates)
    result = candidates[:cap]
    print(f"  {domain}: {len(candidates)} candidates after filtering → {len(result)} selected (cap={cap})")
    return result


def extract_from_prompts_file(domain: str, cap: int) -> list[dict]:
    """Extract prompts from a curated prompts file."""
    result = []
    with open(IOT_PROMPTS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = entry.get("prompt", "").strip()
            if len(prompt) < 20:
                continue
            result.append({"prompt": prompt, "domain": domain})

    result = result[:cap]
    print(f"  {domain}: {len(result)} prompts loaded from curated file (cap={cap})")
    return result


def get_before_counts() -> Counter:
    counts = Counter()
    with open(TRAIN_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                counts[json.loads(line)["domain"]] += 1
    return counts


def main():
    print("=== Router Dataset Augmentation ===\n")

    # Count existing examples
    before_train = get_before_counts()
    before_valid = Counter()
    with open(VALID_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                before_valid[json.loads(line)["domain"]] += 1

    print("Before augmentation:")
    for d in sorted(DOMAINS.keys()):
        print(f"  {d:20s}  train={before_train.get(d, 0):5d}  valid={before_valid.get(d, 0):5d}")
    print(f"  Total train: {sum(before_train.values())}, valid: {sum(before_valid.values())}\n")

    # Extract new examples
    all_new = []
    for domain, (source_type, cap) in DOMAINS.items():
        print(f"Processing {domain} ({source_type}, cap={cap})...")
        if source_type == "lora":
            examples = extract_from_lora(domain, cap)
        elif source_type == "prompts_file":
            examples = extract_from_prompts_file(domain, cap)
        else:
            examples = []
        all_new.extend(examples)

    print(f"\nTotal new examples extracted: {len(all_new)}")

    # Split into train/valid (80/20 per domain)
    new_train = []
    new_valid = []
    by_domain: dict[str, list] = {}
    for ex in all_new:
        by_domain.setdefault(ex["domain"], []).append(ex)

    for domain, examples in by_domain.items():
        random.shuffle(examples)
        split = int(len(examples) * VALID_RATIO)
        new_valid.extend(examples[:split])
        new_train.extend(examples[split:])

    print(f"Split → train: {len(new_train)}, valid: {len(new_valid)}\n")

    # Append to files
    with open(TRAIN_FILE, "a") as f:
        for ex in new_train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(VALID_FILE, "a") as f:
        for ex in new_valid:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("Appended to train.jsonl and valid.jsonl\n")

    # After counts
    after_train = Counter()
    with open(TRAIN_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                after_train[json.loads(line)["domain"]] += 1

    after_valid = Counter()
    with open(VALID_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                after_valid[json.loads(line)["domain"]] += 1

    print("After augmentation (augmented domains):")
    for d in sorted(DOMAINS.keys()):
        bt = before_train.get(d, 0)
        at = after_train.get(d, 0)
        bv = before_valid.get(d, 0)
        av = after_valid.get(d, 0)
        print(f"  {d:20s}  train: {bt:5d} → {at:5d} (+{at-bt:4d})  valid: {bv:4d} → {av:4d} (+{av-bv:3d})")

    total_train_after = sum(after_train.values())
    total_valid_after = sum(after_valid.values())
    total_before_train = sum(before_train.values())
    total_before_valid = sum(before_valid.values())
    print(f"\nTotal train: {total_before_train} → {total_train_after} (+{total_train_after - total_before_train})")
    print(f"Total valid: {total_before_valid} → {total_valid_after} (+{total_valid_after - total_before_valid})")


if __name__ == "__main__":
    main()

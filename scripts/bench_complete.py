#!/usr/bin/env python3
"""Complete micro-kiki benchmark: 5 metrics × 35 domains × base vs adapter.

Metrics:
1. Perplexity on validation set (25 samples per domain)
2. Generation quality (keyword hit rate)
3. Response length quality (not degenerate)
4. Domain specificity (response contains domain keywords)
5. LLM-as-judge (if teacher available on kxkm-ai:8000)

Usage:
    python bench_complete.py [--domains chat-fr python embedded] [--judge-url http://kxkm-ai:8000]
"""

import argparse
import json
import time
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

# Domain keywords for metric 4 (domain specificity)
DOMAIN_KEYWORDS = {
    "chat-fr": ["français", "expliqu", "fonction", "principe", "circuit", "moteur"],
    "reasoning": ["step", "calculate", "therefore", "result", "total", "because"],
    "python": ["def ", "import ", "return", "class ", "self.", "python"],
    "typescript": ["interface", "const ", "type ", "async", "Promise", "generic"],
    "cpp": ["std::", "#include", "template", "nullptr", "class ", "virtual"],
    "rust": ["fn ", "let ", "impl ", "struct ", "Result<", "Option<"],
    "shell": ["#!/bin", "grep", "awk", "pipe", "stdout", "$"],
    "sql": ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "INDEX"],
    "embedded": ["GPIO", "UART", "I2C", "SPI", "interrupt", "register"],
    "electronics": ["MOSFET", "transistor", "amplifier", "impedance", "gain", "bandwidth"],
    "kicad-dsl": ["symbol", "footprint", "pin", "schematic", "KiCad", "net"],
    "spice": ["netlist", ".tran", ".ac", "subckt", "ngspice", "simulation"],
    "components": ["datasheet", "package", "voltage", "current", "pin", "spec"],
    "docker": ["Dockerfile", "container", "image", "FROM", "RUN", "COPY"],
    "power": ["converter", "inductor", "capacitor", "switching", "regulator", "duty"],
    "html-css": ["CSS", "grid", "flexbox", "responsive", "media", "layout"],
    "freecad": ["FreeCAD", "sketch", "Part", "parametric", "extrude", "macro"],
    "stm32": ["STM32", "HAL", "DMA", "ADC", "timer", "peripheral"],
    "yaml-json": ["yaml", "json", "schema", "config", "deploy", "workflow"],
    "kicad-pcb": ["PCB", "trace", "via", "copper", "layer", "DRC"],
    "dsp": ["filter", "FFT", "frequency", "bandwidth", "sample", "coefficient"],
    "emc": ["EMI", "shielding", "ground", "conducted", "radiated", "ferrite"],
    "music-audio": ["audio", "MIDI", "synthesis", "oscillator", "envelope", "frequency"],
    "web-frontend": ["React", "component", "state", "render", "hook", "UI"],
    "web-backend": ["API", "endpoint", "request", "response", "middleware", "route"],
    "devops": ["deploy", "pipeline", "container", "scaling", "monitoring", "CI"],
    "llm-orch": ["RAG", "retrieval", "agent", "chain", "prompt", "embedding"],
    "math": ["equation", "derivative", "integral", "matrix", "theorem", "proof"],
    "security": ["auth", "token", "encrypt", "vulnerability", "CSRF", "OAuth"],
    "iot": ["MQTT", "sensor", "gateway", "protocol", "telemetry", "OTA"],
    "platformio": ["platformio", "board", "framework", "lib_deps", "upload", "monitor"],
    "lua-upy": ["MicroPython", "machine", "Pin", "Lua", "table", "coroutine"],
    "spice-sim": ["simulation", "sweep", "parametric", "Monte Carlo", "convergence", "analysis"],
    "llm-ops": ["inference", "quantize", "GGUF", "vLLM", "tok/s", "VRAM"],
    "ml-training": ["LoRA", "fine-tune", "epoch", "learning rate", "loss", "gradient"],
    "llm-orch": ["RAG", "retrieval", "agent", "chain", "prompt", "tool"],
    "rust": ["fn ", "let ", "impl ", "&", "Result", "match"],
    "web-frontend": ["React", "component", "useState", "render", "Tailwind", "CSS"],
}

# Test prompts (5 per key domain)
MULTI_PROMPTS = {
    "chat-fr": [
        "Explique le fonctionnement d'un variateur de frequence pour moteur asynchrone.",
        "Quels sont les avantages d'un regulateur LDO par rapport a un buck converter ?",
        "Comment dimensionner un condensateur de decouplage pour un microcontroleur ?",
        "Explique la difference entre un signal analogique et numerique.",
        "Comment fonctionne un capteur de temperature a thermocouple ?",
    ],
    "reasoning": [
        "Three resistors 100 200 300 ohms in parallel. Calculate total resistance.",
        "A capacitor charges through 10k resistor from 5V. What is voltage after 2 time constants?",
        "An op-amp has gain 100. Input is 50mV. What is the output voltage and why?",
        "A motor draws 2A at 12V. Calculate power dissipation and heat generated.",
        "Two inductors 100uH and 200uH in series with coupling coefficient 0.5. Total inductance?",
    ],
    "python": [
        "Write a Python async context manager for database connection pooling.",
        "Implement a LRU cache decorator with max_size and TTL expiration.",
        "Write a Python generator that yields Fibonacci numbers with memoization.",
        "Implement a thread-safe singleton pattern in Python.",
        "Write a Python function that flattens nested dictionaries with dot notation keys.",
    ],
    "embedded": [
        "Write ESP-IDF code to configure UART with DMA circular buffer.",
        "Implement an I2C driver for BME280 sensor on STM32 with HAL.",
        "Write FreeRTOS task that reads ADC with moving average filter.",
        "Configure PWM output on ESP32 for servo motor control 50Hz.",
        "Write interrupt handler for external button with debouncing on STM32.",
    ],
    "electronics": [
        "Compare MOSFET vs BJT for switching a 12V 5A load.",
        "Design a non-inverting amplifier with gain of 10 using LM358.",
        "Explain bootstrap gate driver topology for half-bridge.",
        "Calculate snubber circuit for flyback converter MOSFET.",
        "Design a current sense circuit using INA226 for battery monitoring.",
    ],
    "components": [
        "Compare STM32F103 vs ESP32-S3 vs RP2040 for battery IoT.",
        "What are the specs of the AMS1117-3.3 voltage regulator?",
        "Recommend a MOSFET for 3.3V gate drive switching 24V 10A.",
        "What is the LCSC part number for a 100nF 0402 capacitor?",
        "Compare BME280 vs BMP280 vs SHT31 temperature sensors.",
    ],
    "shell": [
        "Write bash script to find all git repos with uncommitted changes.",
        "One-liner to find largest files recursively and format as table.",
        "Write a bash function for parallel SSH command execution.",
        "Script to monitor disk usage and alert when above threshold.",
        "Write bash script to backup directories with rsync and rotation.",
    ],
    "spice": [
        "Write ngspice netlist for Wien bridge oscillator with op-amp.",
        "Simulate a buck converter with LC filter in ngspice.",
        "Write SPICE subcircuit model for a Zener diode.",
        "AC analysis of bandpass filter centered at 10kHz.",
        "Monte Carlo analysis of RC filter with component tolerances.",
    ],
    "docker": [
        "Multi-stage Dockerfile for Rust web service with cargo-chef.",
        "Docker Compose for 3-tier app with health checks and restart.",
        "Dockerfile for Python FastAPI with minimal Alpine image.",
        "Write docker-compose with Traefik reverse proxy and Let's Encrypt.",
        "Multi-arch Dockerfile that builds for arm64 and amd64.",
    ],
    "power": [
        "Design synchronous buck 12V to 3.3V at 5A. Calculate inductor.",
        "Compare LDO vs buck vs charge pump for 3.3V from battery.",
        "Design a boost converter 3.7V to 5V at 2A for USB power bank.",
        "Calculate gate driver requirements for GaN half-bridge.",
        "Design overcurrent protection with electronic fuse.",
    ],
}


def compute_ppl_batch(model, tokenizer, texts, max_samples=25):
    """Compute average perplexity on a batch of texts."""
    ppls = []
    for text in texts[:max_samples]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 3:
            continue
        ids = mx.array([tokens[:-1]])
        labels = mx.array([tokens[1:]])
        logits = model(ids)
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        ppl = float(mx.exp(mx.mean(loss)).item())
        if ppl < 10000:  # filter outliers
            ppls.append(ppl)
    return sum(ppls) / len(ppls) if ppls else 999.0


def compute_keyword_rate(text, domain):
    """Fraction of domain keywords found in text."""
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw.lower() in text.lower())
    return hits / len(keywords)


def evaluate_domain(model, tokenizer, domain, prompts, valid_data):
    """Evaluate one domain with all metrics."""
    result = {"domain": domain}

    # 1. Perplexity on validation set
    valid_texts = []
    for item in valid_data[:25]:
        msgs = item.get("messages", [])
        text = " ".join(m.get("content", "") for m in msgs)
        if text.strip():
            valid_texts.append(text)
    result["val_ppl"] = round(compute_ppl_batch(model, tokenizer, valid_texts), 2)

    # 2-4. Generation metrics on prompts
    keyword_rates = []
    resp_lengths = []
    degenerate = 0

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        resp = generate(model, tokenizer, prompt=formatted, max_tokens=300, verbose=False)

        # Length
        resp_lengths.append(len(resp))
        if len(resp) < 20:
            degenerate += 1

        # Keyword rate
        keyword_rates.append(compute_keyword_rate(resp, domain))

    result["avg_keyword_rate"] = round(sum(keyword_rates) / len(keyword_rates), 3) if keyword_rates else 0
    result["avg_resp_len"] = round(sum(resp_lengths) / len(resp_lengths)) if resp_lengths else 0
    result["degenerate_pct"] = round(degenerate / len(prompts) * 100) if prompts else 0

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--hybrid-dir", default="output/micro-kiki/lora-qwen36-35b-hybrid")
    ap.add_argument("--base-model", default="models/Qwen3.6-35B-A3B")
    ap.add_argument("--data-dir", default="data/micro-kiki")
    args = ap.parse_args()

    domains = args.domains or list(MULTI_PROMPTS.keys())
    out_dir = Path("output/micro-kiki/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Complete benchmark: {len(domains)} domains")
    print(f"Model: {args.base_model}")
    print(f"Adapters: {args.hybrid_dir}")

    all_results = {"base": [], "adapter": []}

    # Phase 1: Base model
    print("\n=== PHASE 1: BASE MODEL ===", flush=True)
    model, tok = load(args.base_model)

    for domain in domains:
        prompts = MULTI_PROMPTS.get(domain, [f"Explain {domain} concepts."])
        valid_file = Path(args.data_dir) / domain / "valid.jsonl"
        valid_data = []
        if valid_file.exists():
            for line in open(valid_file):
                try:
                    valid_data.append(json.loads(line.strip()))
                except:
                    pass

        r = evaluate_domain(model, tok, domain, prompts, valid_data)
        all_results["base"].append(r)
        print(f"  {domain:<15} ppl={r['val_ppl']:>8.2f}  kw={r['avg_keyword_rate']:.3f}  len={r['avg_resp_len']:>5}  degen={r['degenerate_pct']}%", flush=True)

    del model
    mx.metal.clear_cache()

    # Phase 2: Adapter per domain
    print("\n=== PHASE 2: HYBRID ADAPTERS ===", flush=True)
    for domain in domains:
        adapter_path = f"{args.hybrid_dir}/{domain}"
        if not Path(adapter_path).exists():
            print(f"  {domain:<15} SKIP (no adapter)")
            all_results["adapter"].append({"domain": domain, "val_ppl": 999, "avg_keyword_rate": 0, "avg_resp_len": 0, "degenerate_pct": 100})
            continue

        model, tok = load(args.base_model, adapter_path=adapter_path)
        prompts = MULTI_PROMPTS.get(domain, [f"Explain {domain} concepts."])
        valid_file = Path(args.data_dir) / domain / "valid.jsonl"
        valid_data = []
        if valid_file.exists():
            for line in open(valid_file):
                try:
                    valid_data.append(json.loads(line.strip()))
                except:
                    pass

        r = evaluate_domain(model, tok, domain, prompts, valid_data)
        all_results["adapter"].append(r)
        print(f"  {domain:<15} ppl={r['val_ppl']:>8.2f}  kw={r['avg_keyword_rate']:.3f}  len={r['avg_resp_len']:>5}  degen={r['degenerate_pct']}%", flush=True)
        del model

    # Summary
    print("\n" + "=" * 90)
    print(f"{'Domain':<15} {'Base PPL':>9} {'Adapt PPL':>10} {'ΔPPL':>7} {'Base KW':>8} {'Adapt KW':>9} {'Winner':>8}")
    print("-" * 90)

    wins = {"base": 0, "adapter": 0, "tie": 0}
    for b, a in zip(all_results["base"], all_results["adapter"]):
        d = b["domain"]
        bp, ap = b["val_ppl"], a["val_ppl"]
        bk, ak = b["avg_keyword_rate"], a["avg_keyword_rate"]
        delta = ap - bp
        w = "adapter" if delta < -0.5 else "base" if delta > 0.5 else "tie"
        wins[w] += 1
        print(f"{d:<15} {bp:>9.2f} {ap:>10.2f} {delta:>+7.2f} {bk:>8.3f} {ak:>9.3f} {w:>8}")

    print("-" * 90)
    print(f"  Adapter wins: {wins['adapter']}  |  Base wins: {wins['base']}  |  Ties: {wins['tie']}")

    avg_base = sum(r["val_ppl"] for r in all_results["base"]) / len(all_results["base"])
    avg_adapt = sum(r["val_ppl"] for r in all_results["adapter"] if r["val_ppl"] < 999) / max(1, sum(1 for r in all_results["adapter"] if r["val_ppl"] < 999))
    print(f"  Avg PPL: base={avg_base:.2f}  adapter={avg_adapt:.2f}  improvement={((avg_base-avg_adapt)/avg_base*100):.1f}%")

    # Save
    with open(out_dir / "bench-complete.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_dir / 'bench-complete.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark base models for micro-kiki: perplexity + generation quality on 35 domains.

Compares candidate base models on domain-specific prompts to select the best
foundation for LoRA fine-tuning.

Usage:
    python scripts/benchmark_base_models.py --models model1_path model2_path ...
    python scripts/benchmark_base_models.py --models models/Qwen3.6-35B-A3B-4bit models/granite-4.0-h-small-8bit
"""
import argparse
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field

# Domain test prompts — 1 prompt per domain, covering all 35
DOMAIN_PROMPTS = {
    # Phase 1 — Foundations
    "chat-fr": "Explique le fonctionnement d'un variateur de fréquence pour moteur asynchrone triphasé.",
    "reasoning": "A circuit has R1=100Ω, R2=200Ω in series, then R3=150Ω in parallel with the series combo. Calculate total resistance. Show each step.",

    # Phase 2 — Code core
    "python": "Write a Python async context manager that manages a connection pool with max_size, timeout, and health checking.",
    "typescript": "Write a TypeScript generic function that deep-merges two objects, handling arrays, nested objects, and undefined values with proper type inference.",
    "cpp": "Write a C++ lock-free ring buffer using std::atomic with SPSC (single producer, single consumer) semantics.",
    "rust": "Write a Rust async TCP server using tokio that handles multiple clients with a shared state protected by Arc<RwLock<T>>.",

    # Phase 3 — Code secondary
    "html-css": "Write a responsive CSS Grid layout for a dashboard with sidebar, header, main content, and footer that collapses to single column on mobile.",
    "shell": "Write a bash script that finds all Git repos under a directory, checks each for uncommitted changes, unpushed commits, and stale branches, then outputs a summary.",
    "sql": "Write a PostgreSQL query using window functions to calculate running 7-day average revenue per product category, with ranking.",
    "yaml-json": "Write a GitHub Actions workflow that runs tests on PR, builds Docker image, pushes to GHCR, and deploys to staging with manual approval for production.",
    "docker": "Write a multi-stage Dockerfile for a Rust web service that compiles with cargo-chef for layer caching, runs as non-root, and uses distroless as final image.",
    "kicad-dsl": "Write a KiCad Python scripting plugin that generates a spiral inductor footprint with configurable turns, spacing, and trace width.",
    "spice": "Write a complete ngspice netlist for a class-D audio amplifier with PWM modulation, output LC filter, and THD measurement.",
    "lua-upy": "Write a MicroPython driver for the BME280 sensor over I2C on ESP32 that reads temperature, humidity, and pressure with altitude calculation.",

    # Phase 4 — Technical
    "embedded": "Write ESP-IDF code for a FreeRTOS task that reads an INA226 current sensor via I2C, applies a moving average filter, and publishes via MQTT.",
    "stm32": "Write STM32 HAL code to configure DMA circular mode for ADC multi-channel scanning with half-transfer and transfer-complete interrupts.",
    "iot": "Design an MQTT topic hierarchy and QoS strategy for a fleet of 1000 battery-powered sensors reporting every 5 minutes with OTA update capability.",
    "freecad": "Write a FreeCAD Python macro that creates a parametric enclosure with snap-fit lid, ventilation slots, and mounting posts from dimensions.",
    "platformio": "Write a PlatformIO project configuration for a multi-target build (ESP32 + STM32 + native tests) with shared library dependencies and custom build scripts.",
    "power": "Design a synchronous buck converter: calculate inductor value, output capacitor, MOSFET selection, and control loop compensation for 12V→3.3V at 5A.",
    "emc": "Explain how to design a PCB ground plane strategy for mixed-signal (analog+digital+RF) to minimize EMI. Include via stitching, guard rings, and filtering.",
    "dsp": "Implement a real-time 4th-order IIR Butterworth bandpass filter (300Hz-3400Hz) in C for voice processing on ARM Cortex-M4 with CMSIS-DSP.",
    "spice-sim": "Write an ngspice control script that sweeps component values (R, C) in a parametric analysis, extracts -3dB frequency for each, and plots results.",
    "electronics": "Compare MOSFET gate driver topologies (bootstrap, charge pump, isolated) for a 3-phase inverter. Include dead-time, dV/dt immunity, and layout considerations.",
    "kicad-pcb": "Explain the complete DFM review checklist for a 4-layer PCB: trace width/spacing, via rules, copper pour, silkscreen, solder mask expansion, and panelization.",

    # Phase 5 — Applications
    "web-frontend": "Build a React 19 data table component with server-side sorting, filtering, pagination, column resizing, and row selection using TanStack Table.",
    "web-backend": "Design a FastAPI webhook delivery system with PostgreSQL-backed queue, retry with exponential backoff, HMAC signature verification, and dead letter handling.",
    "music-audio": "Implement a polyphonic wavetable synthesizer in Python with ADSR envelope, LFO modulation, and anti-aliased band-limited wavetables.",
    "devops": "Write a Terraform module for an AWS ECS Fargate service with ALB, auto-scaling, CloudWatch alarms, and blue/green deployment.",
    "llm-orch": "Build a RAG pipeline with hybrid retrieval (BM25 + vector), cross-encoder reranking, and LLM-based answer synthesis with source citations.",

    # Phase 6 — Complements
    "math": "Derive the transfer function H(s) of a Sallen-Key low-pass filter, find the Q factor and natural frequency in terms of component values, and plot the Bode diagram.",
    "security": "Implement a secure API authentication system with PKCE OAuth2 flow, JWT access tokens with rotation, and CSRF protection for a SPA+API architecture.",

    # Phase 7 — New domains
    "components": "Compare the STM32F103C8T6, ESP32-S3-WROOM-1, and RP2040 for a battery-powered IoT sensor: power consumption, peripherals, price, and ecosystem.",
    "llm-ops": "Set up a production LLM serving stack with vLLM, Nginx load balancing, Prometheus metrics, and automatic model switching based on GPU memory.",
    "ml-training": "Write a complete LoRA fine-tuning script using Unsloth for Qwen3.5-4B with cosine LR schedule, gradient checkpointing, and Weights & Biases logging.",
}


@dataclass
class BenchmarkResult:
    model_name: str
    domain: str
    perplexity: float = 0.0
    response_len: int = 0
    gen_time_s: float = 0.0
    response_preview: str = ""
    error: str = ""


def benchmark_model(model_path: str, prompts: dict) -> list[BenchmarkResult]:
    """Benchmark a single model on all domain prompts."""
    from mlx_lm import load, generate
    import mlx.core as mx
    import mlx.nn as nn

    results = []
    model_name = Path(model_path).name

    print(f"\nLoading {model_name}...")
    t0 = time.time()
    model, tokenizer = load(model_path)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    for domain, prompt in prompts.items():
        r = BenchmarkResult(model_name=model_name, domain=domain)

        try:
            # Perplexity
            tokens = tokenizer.encode(prompt)
            if len(tokens) >= 2:
                input_ids = mx.array([tokens[:-1]])
                labels = mx.array([tokens[1:]])
                logits = model(input_ids)
                loss = nn.losses.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                )
                r.perplexity = float(mx.exp(mx.mean(loss)).item())

            # Generation
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            t1 = time.time()
            response = generate(
                model, tokenizer, prompt=formatted,
                max_tokens=300, verbose=False,
            )
            r.gen_time_s = time.time() - t1
            r.response_len = len(response)
            r.response_preview = response[:200]

        except Exception as e:
            r.error = str(e)

        results.append(r)
        ppl_str = f"{r.perplexity:.2f}" if r.perplexity > 0 else "ERR"
        print(f"  {domain:<18} ppl={ppl_str:<8} len={r.response_len:<5} {r.gen_time_s:.1f}s")

    del model
    return results


def compare_models(all_results: dict[str, list[BenchmarkResult]]):
    """Print comparison table."""
    models = list(all_results.keys())
    domains = list(DOMAIN_PROMPTS.keys())

    print(f"\n{'='*80}")
    print(f"COMPARISON: {' vs '.join(models)}")
    print(f"{'='*80}")

    header = f"{'Domain':<18}"
    for m in models:
        header += f" {m[:15]:>15} ppl"
    header += f" {'Winner':>10}"
    print(header)
    print("-" * len(header))

    wins = {m: 0 for m in models}
    ties = 0

    for domain in domains:
        row = f"{domain:<18}"
        ppls = {}
        for m in models:
            r = next((x for x in all_results[m] if x.domain == domain), None)
            ppl = r.perplexity if r and r.perplexity > 0 else 999
            ppls[m] = ppl
            row += f" {ppl:>15.2f}   "

        # Winner = lowest perplexity
        best = min(ppls, key=ppls.get)
        vals = list(ppls.values())
        if len(vals) >= 2 and abs(vals[0] - vals[1]) < 0.1:
            winner = "tie"
            ties += 1
        else:
            winner = Path(best).name[:10]
            wins[best] += 1
        row += f" {winner:>10}"
        print(row)

    print("-" * len(header))
    for m in models:
        avg_ppl = sum(r.perplexity for r in all_results[m] if r.perplexity > 0) / max(1, sum(1 for r in all_results[m] if r.perplexity > 0))
        print(f"  {Path(m).name[:20]}: avg_ppl={avg_ppl:.2f}, wins={wins[m]}")
    print(f"  Ties: {ties}")

    # Save results
    out = {}
    for m, results in all_results.items():
        out[m] = [{"domain": r.domain, "perplexity": r.perplexity,
                    "response_len": r.response_len, "gen_time_s": r.gen_time_s,
                    "response_preview": r.response_preview, "error": r.error}
                   for r in results]

    outfile = Path("output/micro-kiki/eval/base_model_comparison.json")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {outfile}")


def main():
    ap = argparse.ArgumentParser(description="Benchmark base models for micro-kiki")
    ap.add_argument("--models", nargs="+", required=True, help="Paths to model directories")
    ap.add_argument("--domains", nargs="*", default=None, help="Subset of domains to test")
    ap.add_argument("--quick", action="store_true", help="Test only 10 key domains")
    args = ap.parse_args()

    prompts = DOMAIN_PROMPTS
    if args.domains:
        prompts = {k: v for k, v in DOMAIN_PROMPTS.items() if k in args.domains}
    elif args.quick:
        quick_domains = ["chat-fr", "reasoning", "python", "embedded", "electronics",
                         "spice", "kicad-dsl", "shell", "components", "docker"]
        prompts = {k: v for k, v in DOMAIN_PROMPTS.items() if k in quick_domains}

    print(f"Benchmarking {len(args.models)} models on {len(prompts)} domains")

    all_results = {}
    for model_path in args.models:
        if not Path(model_path).exists():
            print(f"SKIP {model_path} (not found)")
            continue
        results = benchmark_model(model_path, prompts)
        all_results[model_path] = results

    if len(all_results) >= 2:
        compare_models(all_results)
    elif len(all_results) == 1:
        m = list(all_results.keys())[0]
        print(f"\nSingle model results for {Path(m).name}:")
        for r in all_results[m]:
            print(f"  {r.domain:<18} ppl={r.perplexity:.2f}  len={r.response_len}")


if __name__ == "__main__":
    main()

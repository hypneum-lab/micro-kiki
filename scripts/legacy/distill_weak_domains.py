#!/usr/bin/env python3
"""Distill training data for weak domains using Qwen3.5-35B teacher.

Generates high-quality Q&A pairs by sending domain-specific prompts
to the teacher model and collecting responses.
"""
import json
import hashlib
import time
import sys
from pathlib import Path

import httpx

TEACHER_URL = "http://localhost:8000/v1/chat/completions"
OUTPUT_DIR = Path("/home/kxkm/micro-kiki/data/distilled-v2")

# Weak domains with target counts and seed prompts
WEAK_DOMAINS = {
    "platformio": {
        "target": 500,
        "topics": [
            "Write a platformio.ini for ESP32 with WiFi + MQTT libraries",
            "How to set up PlatformIO CI with GitHub Actions for STM32",
            "Configure PlatformIO for multi-environment builds (debug + release)",
            "How to use PlatformIO library manager to install ArduinoJson",
            "Debug an ESP32 project with PlatformIO and JTAG",
            "Write a platformio.ini with custom build flags for power saving",
            "How to add a custom board definition in PlatformIO",
            "Set up PlatformIO unit testing for an embedded project",
            "PlatformIO monitor filters for colored serial output",
            "How to use PlatformIO with Zephyr RTOS on nRF52840",
        ],
    },
    "spice": {
        "target": 500,
        "topics": [
            "Write an ngspice netlist for a common-emitter BJT amplifier with biasing",
            "Simulate a buck converter with feedback loop in ngspice",
            "Write a SPICE subcircuit model for an op-amp (LM741 equivalent)",
            "AC analysis of a band-pass filter with center frequency 10kHz in ngspice",
            "Monte Carlo analysis of component tolerances in an RC filter",
            "Write a SPICE netlist for a MOSFET H-bridge motor driver",
            "Simulate thermal effects on a power MOSFET using ngspice",
            "Write a behavioral voltage source (B source) for a PLL VCO",
            "Transient analysis of an LC tank circuit with initial conditions",
            "Compare Butterworth vs Chebyshev filter responses in ngspice",
        ],
    },
    "music-audio": {
        "target": 500,
        "topics": [
            "Implement a biquad IIR filter for audio EQ in C",
            "Write a MIDI parser in Python that handles note on/off and CC",
            "Explain how wavetable synthesis works with code example",
            "Implement a simple reverb using a feedback delay network",
            "Write a real-time audio callback using PortAudio in C",
            "Design a compressor with attack/release envelope in Python",
            "Explain FM synthesis and implement a basic 2-operator FM synth",
            "Write a Web Audio API synth with oscillator + filter + envelope",
            "Implement pitch detection using autocorrelation in Python",
            "Design an audio effects chain with dry/wet mix control",
        ],
    },
    "web-frontend": {
        "target": 500,
        "topics": [
            "Build a React 19 component with useOptimistic for instant UI updates",
            "Configure Vite with path aliases, proxy, and environment variables",
            "Write a custom React hook for debounced search with AbortController",
            "Implement virtualized list rendering with react-window",
            "Set up Tailwind CSS dark mode with system preference detection",
            "Write a Playwright E2E test for a login flow with MFA",
            "Implement optimistic UI updates with server actions in React 19",
            "Build an accessible modal dialog with focus trap and keyboard nav",
            "Configure code splitting with React.lazy and Suspense boundaries",
            "Write a Zustand store with persistence and devtools middleware",
        ],
    },
    "web-backend": {
        "target": 500,
        "topics": [
            "Build a FastAPI CRUD API with Pydantic v2 models and SQLAlchemy",
            "Implement JWT auth with refresh tokens in FastAPI",
            "Write a Hono middleware for rate limiting with sliding window",
            "Design a WebSocket server with room management in Python",
            "Implement database connection pooling with asyncpg in FastAPI",
            "Write a background task queue with Redis and FastAPI",
            "Build a file upload endpoint with multipart handling and virus scan",
            "Implement API versioning with header-based routing in Hono",
            "Write a health check endpoint with dependency verification",
            "Design a webhook delivery system with retry and exponential backoff",
        ],
    },
    "yaml-json": {
        "target": 400,
        "topics": [
            "Write a GitHub Actions workflow for Python CI with matrix testing",
            "Create a Docker Compose file for a 3-tier app (frontend+API+DB)",
            "Write a JSON Schema for validating an API response with nested objects",
            "Design a Kubernetes deployment with HPA, PDB, and resource limits",
            "Write an OpenAPI 3.1 spec for a REST API with auth and pagination",
            "Create a Helm chart values.yaml for a microservice with configmaps",
            "Write a YAML config for Prometheus alerting rules",
            "Design a JSON Schema for a plugin manifest with versioning",
        ],
    },
    "llm-ops": {
        "target": 500,
        "topics": [
            "Set up Ollama with a custom Modelfile for a fine-tuned model",
            "Configure vLLM with tensor parallelism on 2 GPUs",
            "Quantize a model from HF safetensors to GGUF Q4_K_M with llama.cpp",
            "Write a Python script to benchmark LLM inference (tok/s, TTFT)",
            "Set up MLX serving with adapter hot-swap for multiple LoRA models",
            "Configure KV cache quantization in llama.cpp for memory savings",
            "Write a load balancer for multiple llama-server instances",
            "Compare GGUF quantization levels (Q4_K_M vs Q5_K_S vs Q8_0)",
            "Set up speculative decoding with a draft model in vLLM",
            "Calculate VRAM requirements for serving a 70B model in 4-bit",
        ],
    },
    "llm-orch": {
        "target": 500,
        "topics": [
            "Build a RAG pipeline with hybrid search (BM25 + vector) in Python",
            "Implement a ReAct agent with tool use using the OpenAI function calling API",
            "Design a multi-agent debate system for fact verification",
            "Write a LangChain chain that summarizes documents with map-reduce",
            "Implement semantic caching for LLM responses with Redis",
            "Build an MCP server with 3 tools (search, calculator, file reader)",
            "Design a routing system that sends queries to specialized LLMs by domain",
            "Implement conversation memory with sliding window + summary compression",
            "Write a judge LLM evaluation pipeline comparing 2 model outputs",
            "Build a retrieval-augmented generation system with reranking (Cohere)",
        ],
    },
    "ml-training": {
        "target": 500,
        "topics": [
            "Fine-tune Qwen3.5-4B with LoRA using Unsloth on a single GPU",
            "Write a training script with cosine LR schedule and warmup",
            "Implement DPO training with preference pairs in TRL",
            "Create a dataset preparation pipeline (dedup, filter, split, format)",
            "Set up Weights & Biases logging for a fine-tuning run",
            "Implement early stopping based on validation loss plateau",
            "Write a script to merge LoRA adapters into base model weights",
            "Compare LoRA vs QLoRA vs full fine-tuning for a 7B model",
            "Implement curriculum learning with domain-ordered training data",
            "Set up distributed training with FSDP on 4 GPUs",
        ],
    },
    "dsp": {
        "target": 500,
        "topics": [
            "Design a 4th-order Butterworth low-pass filter in C for ARM Cortex-M",
            "Implement an FFT-based spectrum analyzer with windowing in Python",
            "Write a fixed-point FIR filter using Q15 arithmetic for embedded",
            "Design a PLL frequency synthesizer with loop filter calculation",
            "Implement overlap-add convolution for real-time audio processing",
            "Write CMSIS-DSP code for a biquad cascade filter on STM32",
            "Design a CIC decimation filter for sigma-delta ADC output",
            "Implement a Goertzel algorithm for DTMF tone detection",
            "Write a resampling filter with polyphase decomposition",
            "Design an adaptive noise cancellation filter using LMS algorithm",
        ],
    },
    "math": {
        "target": 500,
        "topics": [
            "Derive the transfer function of a second-order RLC circuit",
            "Calculate Bode plot magnitude and phase for a PI controller",
            "Solve a system of linear equations using Gaussian elimination in Python",
            "Derive the DFT of a windowed sinusoid and explain spectral leakage",
            "Calculate the noise figure of a cascaded amplifier chain",
            "Implement Newton-Raphson root finding for a nonlinear circuit equation",
            "Derive the Z-transform of a digital IIR filter from its analog prototype",
            "Calculate eigenvalues of a state-space system and determine stability",
            "Implement least-squares curve fitting for sensor calibration data",
            "Derive the CMRR of a differential amplifier with resistor mismatch",
        ],
    },
}


def generate_prompt(domain: str, topic: str, variant: int) -> str:
    """Create a varied prompt from a topic seed."""
    styles = [
        f"{topic}",
        f"Explain in detail: {topic}",
        f"As an expert in {domain}, {topic.lower()}. Include code examples.",
        f"Write a tutorial on how to {topic.lower()}. Step by step with code.",
        f"I'm working on a project and need help: {topic}",
    ]
    return styles[variant % len(styles)]


def call_teacher(prompt: str, client: httpx.Client) -> str | None:
    """Call the teacher model and return the response."""
    try:
        resp = client.post(TEACHER_URL, json={
            "model": "qwen",
            "messages": [
                {"role": "system", "content": "You are a technical expert. Give detailed, accurate answers with code examples when relevant. Be concise but thorough."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "")
        if not content:
            content = data["choices"][0]["message"].get("reasoning_content", "")
        return content if content and len(content) > 50 else None
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Wait for server
    client = httpx.Client()
    for attempt in range(30):
        try:
            r = client.get("http://localhost:8000/health")
            if r.status_code == 200:
                h = r.json()
                if h.get("status") == "ok":
                    print("Teacher ready")
                    break
        except:
            pass
        print(f"Waiting for teacher... ({attempt+1}/30)")
        time.sleep(5)

    total = 0
    for domain, cfg in WEAK_DOMAINS.items():
        target = cfg["target"]
        topics = cfg["topics"]
        out_file = OUTPUT_DIR / f"{domain}.jsonl"

        # Load existing
        existing = set()
        if out_file.exists():
            for line in open(out_file):
                existing.add(line.strip()[:200])

        print(f"\n=== {domain} (target={target}, existing={len(existing)}) ===")
        added = 0

        for round_idx in range(target // len(topics) + 1):
            for topic_idx, topic in enumerate(topics):
                if added >= target:
                    break
                variant = round_idx * len(topics) + topic_idx
                prompt = generate_prompt(domain, topic, variant)

                response = call_teacher(prompt, client)
                if response is None:
                    continue

                pair = json.dumps({
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    "_source": f"distill-35b-{domain}",
                }, ensure_ascii=False)

                if pair[:200] not in existing:
                    with open(out_file, "a") as f:
                        f.write(pair + "\n")
                    existing.add(pair[:200])
                    added += 1

                if added % 10 == 0:
                    print(f"  {domain}: {added}/{target}", flush=True)

            if added >= target:
                break

        total += added
        print(f"  {domain}: {added} distilled")

    print(f"\nTotal: {total} examples across {len(WEAK_DOMAINS)} domains")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

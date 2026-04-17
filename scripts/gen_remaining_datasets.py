#!/usr/bin/env python3
"""Generate training data for remaining weak domains using templates."""
import json
import sys

def gen(domain, pairs):
    out = []
    for q, a in pairs:
        out.append(json.dumps({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]}, ensure_ascii=False))
    return out

def gen_web_backend():
    pairs = []
    # FastAPI
    for endpoint, method, desc in [
        ("/users", "GET", "list all users with pagination"),
        ("/users/{id}", "GET", "get user by ID"),
        ("/users", "POST", "create a new user"),
        ("/users/{id}", "PUT", "update user"),
        ("/users/{id}", "DELETE", "delete user"),
        ("/auth/login", "POST", "authenticate and return JWT"),
        ("/auth/refresh", "POST", "refresh access token"),
        ("/items", "GET", "list items with filtering"),
        ("/upload", "POST", "upload file with validation"),
        ("/ws", "WS", "WebSocket endpoint for real-time updates"),
    ]:
        pairs.append((
            f"Write a FastAPI {method} endpoint for {endpoint} that {desc}.",
            f'```python\nfrom fastapi import FastAPI, HTTPException, Depends\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\n@app.{method.lower()}("{endpoint}")\nasync def handle():\n    """Endpoint: {desc}"""\n    # Implementation for {endpoint}\n    pass\n```\n\nThis endpoint handles {desc}. Use Pydantic models for request/response validation, proper HTTP status codes, and dependency injection for auth.'
        ))
    # Middleware patterns
    for mw in ["CORS", "rate limiting", "request logging", "auth JWT verification", "error handling"]:
        pairs.append((
            f"Write FastAPI middleware for {mw}.",
            f"```python\nfrom fastapi import FastAPI, Request\nfrom starlette.middleware.base import BaseHTTPMiddleware\n\nclass {mw.replace(' ', '')}Middleware(BaseHTTPMiddleware):\n    async def dispatch(self, request: Request, call_next):\n        # {mw} logic\n        response = await call_next(request)\n        return response\n\napp = FastAPI()\napp.add_middleware({mw.replace(' ', '')}Middleware)\n```"
        ))
    # DB patterns
    for pattern in ["connection pooling with asyncpg", "SQLAlchemy async session", "Alembic migration", "Redis caching", "background task with Celery"]:
        pairs.append((
            f"How to implement {pattern} in a FastAPI application?",
            f"To implement {pattern} in FastAPI:\n\n1. Install dependencies\n2. Configure the connection\n3. Create dependency injection\n4. Use in endpoints\n\n```python\n# {pattern} implementation\n# See FastAPI docs for complete example\n```"
        ))
    return pairs

def gen_yaml_json():
    pairs = []
    # Docker Compose
    for stack in ["nginx+fastapi+postgres", "react+node+mongo", "grafana+prometheus+alertmanager", "redis+celery+flower"]:
        parts = stack.split("+")
        pairs.append((
            f"Write a Docker Compose file for a {stack} stack.",
            f"```yaml\nversion: '3.8'\nservices:\n" + "\n".join(f"  {p}:\n    image: {p}:latest\n    ports:\n      - \"{8080+i}:{80+i}\"\n    restart: unless-stopped" for i, p in enumerate(parts)) + "\n\nnetworks:\n  default:\n    driver: bridge\n```"
        ))
    # GitHub Actions
    for lang, cmd in [("Python", "uv run pytest"), ("Node", "npm test"), ("Rust", "cargo test"), ("Go", "go test ./..."), ("C++", "cmake --build build && ctest")]:
        pairs.append((
            f"Write a GitHub Actions CI workflow for a {lang} project.",
            f"```yaml\nname: CI\non: [push, pull_request]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - name: Run tests\n        run: {cmd}\n```"
        ))
    # JSON Schema
    for obj in ["user profile", "API error response", "configuration file", "webhook event", "sensor reading"]:
        pairs.append((
            f"Write a JSON Schema for a {obj}.",
            f'{{"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "title": "{obj}", "properties": {{}}, "required": []}}'
        ))
    # K8s
    for resource in ["Deployment", "Service", "Ingress", "ConfigMap", "HPA"]:
        pairs.append((
            f"Write a Kubernetes {resource} YAML for a web application.",
            f"```yaml\napiVersion: apps/v1\nkind: {resource}\nmetadata:\n  name: webapp\nspec:\n  # {resource} configuration\n```"
        ))
    return pairs

def gen_lua_upy():
    pairs = []
    # MicroPython
    for peripheral, code in [
        ("GPIO LED blink", "from machine import Pin\nimport time\nled = Pin(2, Pin.OUT)\nwhile True:\n    led.toggle()\n    time.sleep(0.5)"),
        ("I2C sensor read", "from machine import I2C, Pin\ni2c = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)\ndevices = i2c.scan()\ndata = i2c.readfrom(0x68, 14)"),
        ("PWM motor control", "from machine import PWM, Pin\npwm = PWM(Pin(5), freq=1000, duty=512)"),
        ("ADC voltage read", "from machine import ADC, Pin\nadc = ADC(Pin(34))\nadc.atten(ADC.ATTN_11DB)\nvoltage = adc.read() * 3.3 / 4095"),
        ("UART serial", "from machine import UART\nuart = UART(1, baudrate=9600, tx=17, rx=16)\nuart.write('Hello\\n')"),
        ("WiFi connect", "import network\nwlan = network.WLAN(network.STA_IF)\nwlan.active(True)\nwlan.connect('SSID', 'password')"),
        ("Timer interrupt", "from machine import Timer\ndef callback(t):\n    print('tick')\ntimer = Timer(0)\ntimer.init(period=1000, callback=callback)"),
        ("SPI display", "from machine import SPI, Pin\nspi = SPI(1, baudrate=40000000, sck=Pin(18), mosi=Pin(23))\ncs = Pin(5, Pin.OUT)"),
    ]:
        pairs.append((
            f"Write MicroPython code for {peripheral} on ESP32.",
            f"```python\n{code}\n```\n\nThis configures the ESP32 {peripheral}. Make sure to use the correct pin numbers for your board."
        ))
    # Lua
    for topic, code in [
        ("table manipulation", "local t = {1,2,3}\ntable.insert(t, 4)\nfor i, v in ipairs(t) do print(i, v) end"),
        ("metatables", "local mt = {__index = function(t, k) return 'default' end}\nlocal obj = setmetatable({}, mt)\nprint(obj.anything) -- 'default'"),
        ("coroutines", "local co = coroutine.create(function()\n  for i=1,3 do\n    coroutine.yield(i)\n  end\nend)\nprint(coroutine.resume(co)) -- true, 1"),
        ("string patterns", "local s = 'Hello 123 World'\nfor word in s:gmatch('%a+') do print(word) end"),
        ("file I/O", "local f = io.open('data.txt', 'r')\nlocal content = f:read('*all')\nf:close()"),
        ("OOP with metatables", "local Class = {}\nClass.__index = Class\nfunction Class.new(name)\n  return setmetatable({name=name}, Class)\nend"),
    ]:
        pairs.append((
            f"Write Lua code demonstrating {topic}.",
            f"```lua\n{code}\n```\n\nThis demonstrates {topic} in Lua."
        ))
    # CircuitPython vs MicroPython
    pairs.append(("What are the differences between MicroPython and CircuitPython?",
        "Key differences:\n- **CircuitPython** (by Adafruit): focus on education, `board` module, auto-reload on save, USB drive\n- **MicroPython**: more hardware support, `machine` module, lower-level control, REPL\n- CircuitPython has better library ecosystem for Adafruit boards\n- MicroPython supports more MCUs (ESP32, STM32, RP2040, etc.)"))
    return pairs

def gen_ml_training():
    pairs = []
    for topic, q, a in [
        ("LoRA", "Explain LoRA and how to choose rank and alpha.", "LoRA (Low-Rank Adaptation) adds trainable low-rank matrices A and B to frozen weights: W' = W + BA. Key parameters:\n- **Rank (r)**: 4-64, controls capacity. Higher = more expressive but slower\n- **Alpha (α)**: scaling factor, typically α = 2×r. Effective scaling = α/r\n- **Target modules**: usually q_proj, v_proj for attention. Adding k_proj, o_proj helps quality\n- Rule of thumb: r=16, α=32 is a good default. Increase r for larger datasets."),
        ("LR schedule", "What learning rate schedule should I use for fine-tuning?", "For LoRA fine-tuning:\n- **Cosine decay with warmup**: best general choice\n- Warmup: 5-10% of total steps\n- Peak LR: 1e-4 to 5e-5 for LoRA, 1e-5 to 5e-6 for full fine-tune\n- Use AdamW with weight_decay=0.01\n```python\nscheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)\n```"),
        ("batch size", "How does batch size affect fine-tuning quality?", "Batch size impacts:\n- **Larger batch**: smoother gradients, faster convergence, but needs more memory\n- **Gradient accumulation**: simulate large batches with small GPU memory\n- Effective batch = per_device_batch × num_devices × grad_accum_steps\n- Rule: effective batch 16-64 works well. Scale LR linearly with batch size."),
        ("DPO", "Explain DPO vs RLHF for alignment.", "**RLHF**: Train reward model → PPO optimization. Complex, unstable.\n**DPO**: Direct preference optimization without reward model. Loss: -log σ(β(log π(chosen) - log π(rejected)))\n- Simpler: no reward model, no RL loop\n- More stable training\n- Needs (prompt, chosen, rejected) triples\n- β controls how much to deviate from reference model"),
        ("eval", "How to evaluate a fine-tuned model properly?", "Multi-faceted evaluation:\n1. **Perplexity**: lower = better language modeling\n2. **Task-specific**: pass@k for code, ROUGE for summarization\n3. **LLM-as-judge**: GPT-4/Claude rates outputs 1-5\n4. **A/B win rate**: compare base vs fine-tuned\n5. **Forgetting**: test on original capabilities\n6. Hold out 10% as validation, never train on it"),
        ("data quality", "What makes a good fine-tuning dataset?", "Quality > Quantity:\n1. **Diverse**: cover all expected use cases\n2. **Consistent format**: same chat template throughout\n3. **Deduplicated**: MinHash LSH for near-duplicate removal\n4. **Balanced**: not 90% one category\n5. **Clean**: no truncated responses, no hallucinations\n6. **Length-appropriate**: match expected output length\n- 1K high-quality > 100K noisy examples"),
    ]:
        pairs.append((q, a))
    # More training recipes
    for framework, code in [
        ("Unsloth", "from unsloth import FastLanguageModel\nmodel, tokenizer = FastLanguageModel.from_pretrained('model', load_in_4bit=True)\nmodel = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32)"),
        ("TRL SFTTrainer", "from trl import SFTTrainer, SFTConfig\ntrainer = SFTTrainer(model=model, train_dataset=ds, args=SFTConfig(output_dir='out', max_steps=500))"),
        ("MLX LoRA", "python -m mlx_lm lora --model model --data data/ --train --iters 500 --batch-size 1 --adapter-path adapters/"),
    ]:
        pairs.append((
            f"Show a minimal {framework} fine-tuning example.",
            f"```python\n{code}\n```\n\nThis is the minimal setup for {framework}. Add data loading, evaluation, and saving as needed."
        ))
    return pairs

def gen_math():
    pairs = []
    # Circuit math
    for circuit, formula in [
        ("parallel resistors", "1/R_total = 1/R1 + 1/R2 + ... + 1/Rn"),
        ("RC time constant", "τ = R × C, V(t) = V0 × (1 - e^(-t/τ))"),
        ("RLC resonance", "f0 = 1 / (2π√(LC)), Q = (1/R)√(L/C)"),
        ("voltage divider", "Vout = Vin × R2 / (R1 + R2)"),
        ("op-amp gain", "inverting: G = -Rf/Rin, non-inverting: G = 1 + Rf/Rin"),
        ("power dissipation", "P = V²/R = I²R = VI"),
        ("impedance", "Z_C = 1/(jωC), Z_L = jωL, |Z| = √(R² + (X_L - X_C)²)"),
        ("decibels", "dB = 20 log10(Vout/Vin) for voltage, dB = 10 log10(Pout/Pin) for power"),
    ]:
        pairs.append((
            f"Derive and explain the {circuit} formula.",
            f"**{circuit}**: {formula}\n\nDerivation: Apply Kirchhoff's laws to the circuit. For {circuit}, we start from the basic relationships and combine them algebraically."
        ))
    # Numerical problems
    for problem, solution in [
        ("Three resistors 100Ω, 200Ω, 300Ω in parallel", "1/R = 1/100 + 1/200 + 1/300 = 6/600 + 3/600 + 2/600 = 11/600\nR = 600/11 ≈ 54.5Ω"),
        ("RC filter cutoff with R=10kΩ, C=100nF", "f_c = 1/(2πRC) = 1/(2π × 10000 × 100e-9) = 1/(6.283e-3) ≈ 159.2 Hz"),
        ("Voltage divider R1=10kΩ, R2=5kΩ, Vin=12V", "Vout = 12 × 5000/(10000+5000) = 12 × 1/3 = 4V"),
        ("Power in 47Ω resistor with 5V across it", "P = V²/R = 25/47 ≈ 0.532W → use at least 1W resistor"),
        ("LC resonance with L=100µH, C=10nF", "f0 = 1/(2π√(LC)) = 1/(2π√(100e-6 × 10e-9)) = 1/(2π × 31.6e-6) ≈ 159.2 kHz"),
    ]:
        pairs.append((f"Calculate: {problem}. Show your work.", f"**Solution:**\n{solution}"))
    return pairs

def gen_dsp():
    pairs = []
    for topic, q, a in [
        ("FIR filter", "Implement a simple FIR low-pass filter in C.",
         "```c\n#define N 16\nfloat coeffs[N] = { /* windowed sinc coefficients */ };\nfloat buffer[N] = {0};\nint idx = 0;\n\nfloat fir_filter(float input) {\n    buffer[idx] = input;\n    float output = 0;\n    for (int i = 0; i < N; i++) {\n        int j = (idx - i + N) % N;\n        output += coeffs[i] * buffer[j];\n    }\n    idx = (idx + 1) % N;\n    return output;\n}\n```"),
        ("FFT windowing", "Why do we use windowing in FFT analysis?",
         "Windowing reduces **spectral leakage** caused by analyzing a finite-length signal:\n- The FFT assumes the signal is periodic\n- Truncation = multiplication by a rectangular window\n- This causes sinc-shaped sidelobes in the spectrum\n\nCommon windows:\n- **Hann**: good frequency resolution, -31dB sidelobes\n- **Hamming**: slightly better sidelobe suppression\n- **Blackman-Harris**: -92dB sidelobes, wider main lobe\n- **Flat-top**: accurate amplitude measurement"),
        ("sample rate", "What is the Nyquist theorem and why does it matter?",
         "**Nyquist-Shannon theorem**: To accurately represent a signal, the sampling rate must be at least 2× the highest frequency component.\n\n- fs ≥ 2 × fmax\n- Audio: fmax=20kHz → fs=44.1kHz (CD) or 48kHz (pro)\n- If fs < 2×fmax → **aliasing**: high frequencies fold back as phantom low frequencies\n- Anti-aliasing filter: analog low-pass before ADC, cutoff at fs/2"),
        ("fixed-point", "Explain Q15 fixed-point format for embedded DSP.",
         "**Q15**: 16-bit signed fixed-point with 15 fractional bits\n- Range: [-1.0, 1.0 - 2^-15] ≈ [-1.0, 0.99997]\n- Resolution: 2^-15 ≈ 3.05e-5\n- Multiply: `int32_t result = ((int32_t)a * b) >> 15;`\n- Add: direct addition (watch overflow)\n- Convert from float: `int16_t q = (int16_t)(f * 32768.0f);`\n- Advantage: no FPU needed, deterministic timing on Cortex-M0/M3"),
    ]:
        pairs.append((q, a))
    # More DSP
    for algo in ["Goertzel (DTMF detection)", "CIC decimation", "polyphase resampler", "LMS adaptive filter", "overlap-add convolution"]:
        pairs.append((
            f"Explain the {algo} algorithm and when to use it.",
            f"The {algo} algorithm is used in digital signal processing for efficient computation. It is particularly useful in embedded systems where computational resources are limited."
        ))
    return pairs

def main():
    generators = {
        "web-backend": gen_web_backend,
        "yaml-json": gen_yaml_json,
        "lua-upy": gen_lua_upy,
        "ml-training": gen_ml_training,
        "math": gen_math,
        "dsp": gen_dsp,
    }

    for domain, gen_fn in generators.items():
        pairs = gen_fn()
        lines = gen(domain, pairs)
        outfile = f"/tmp/gen-{domain}.jsonl"
        with open(outfile, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  {domain}: {len(lines)} examples → {outfile}")

if __name__ == "__main__":
    main()

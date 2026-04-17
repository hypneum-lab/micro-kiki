#!/usr/bin/env python3
"""Generate LLM Ops training dataset covering inference engines, quantization, serving, and hardware.

Covers 8 sub-domains:
1. Ollama (CLI, Modelfile, API, GPU layers, custom models)
2. vLLM (server config, tensor parallelism, quantization, PagedAttention)
3. llama.cpp (build, quantize, server, GGUF, Metal/CUDA, KV cache)
4. GGUF quantization (Q4_K_M, Q5_K_S, Q8_0, IQ2_XS, perplexity impact)
5. Model serving (batching, KV cache, speculative decoding, prefix caching)
6. Hardware requirements (VRAM calculation, CPU vs GPU, Apple Silicon MLX)
7. Benchmarking (tokens/sec, TTFT, throughput, perplexity eval)
8. MLX framework (load, generate, quantize, lora, convert)

Output: JSONL to stdout — pipe to file.

Usage::

    uv run python scripts/gen_llm_ops_dataset.py > /tmp/gen-llm-ops.jsonl
    uv run python scripts/gen_llm_ops_dataset.py --stats
"""
from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stderr)
logger = logging.getLogger(__name__)


def msg(user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]}


# ---------------------------------------------------------------------------
# 1. Ollama
# ---------------------------------------------------------------------------
def gen_ollama() -> list[dict]:
    pairs = []

    # --- CLI basics ---
    pairs.append(msg(
        "How do I pull and run a model with Ollama?",
        "Pull a model with `ollama pull <model>` and run it interactively with `ollama run <model>`. "
        "For example:\n```bash\nollama pull llama3.1:8b\nollama run llama3.1:8b\n```\n"
        "The model is downloaded to `~/.ollama/models/` and cached for future use. "
        "Use `ollama list` to see downloaded models and `ollama rm <model>` to delete one."
    ))
    pairs.append(msg(
        "How do I list all models available in Ollama?",
        "Use `ollama list` to show locally downloaded models with their size and modification date. "
        "To browse the Ollama library online, visit https://ollama.com/library. "
        "You can also search with `ollama search <query>` (Ollama 0.5+)."
    ))
    pairs.append(msg(
        "How do I delete a model from Ollama?",
        "Run `ollama rm <model-name>` to remove a model and free disk space. For example:\n"
        "```bash\nollama rm llama3.1:8b\n```\n"
        "This removes the model blobs from `~/.ollama/models/blobs/`. "
        "If multiple tags share the same blob, the blob is only deleted when the last reference is removed."
    ))
    pairs.append(msg(
        "How do I update Ollama to the latest version?",
        "On macOS, Ollama auto-updates via the menu bar app. On Linux:\n"
        "```bash\ncurl -fsSL https://ollama.com/install.sh | sh\n```\n"
        "This overwrites the binary in `/usr/local/bin/ollama`. Check version with `ollama --version`. "
        "After updating, restart the server: `systemctl restart ollama` (systemd) or relaunch the app."
    ))

    # --- Modelfile ---
    pairs.append(msg(
        "Write an Ollama Modelfile that creates a coding assistant from CodeLlama with temperature 0.2 and a system prompt.",
        "```dockerfile\nFROM codellama:13b\n\nPARAMETER temperature 0.2\nPARAMETER top_p 0.9\nPARAMETER num_ctx 4096\n\n"
        "SYSTEM \"\"\"You are an expert coding assistant. Write clean, well-documented code.\n"
        "Always explain your reasoning step by step before providing code.\"\"\"\n```\n\n"
        "Build it with:\n```bash\nollama create my-coder -f Modelfile\nollama run my-coder\n```\n"
        "The `PARAMETER` instruction sets inference defaults. `SYSTEM` sets the system prompt."
    ))
    pairs.append(msg(
        "What parameters can I set in an Ollama Modelfile?",
        "Key Modelfile parameters:\n\n"
        "| Parameter | Default | Description |\n"
        "|-----------|---------|-------------|\n"
        "| `temperature` | 0.8 | Sampling temperature (0=deterministic, 2=max random) |\n"
        "| `top_p` | 0.9 | Nucleus sampling threshold |\n"
        "| `top_k` | 40 | Top-k token sampling |\n"
        "| `num_ctx` | 2048 | Context window size in tokens |\n"
        "| `num_predict` | -1 | Max tokens to generate (-1=unlimited) |\n"
        "| `repeat_penalty` | 1.1 | Repetition penalty |\n"
        "| `stop` | - | Stop sequence (can have multiple) |\n"
        "| `seed` | 0 | Random seed (0=random) |\n"
        "| `num_gpu` | auto | Number of GPU layers to offload |\n"
        "| `num_thread` | auto | CPU threads for inference |\n\n"
        "Set them with `PARAMETER <name> <value>` in the Modelfile."
    ))
    pairs.append(msg(
        "How do I import a GGUF model into Ollama?",
        "Create a Modelfile pointing to the GGUF file:\n"
        "```dockerfile\nFROM ./my-model-Q4_K_M.gguf\n\nTEMPLATE \"\"\"{{ .System }}\n{{ .Prompt }}\"\"\"\n"
        "PARAMETER num_ctx 4096\n```\n\n"
        "Then build:\n```bash\nollama create my-model -f Modelfile\nollama run my-model\n```\n"
        "Ollama copies the GGUF into its blob store. The `TEMPLATE` instruction defines the chat format — "
        "match the original model's template (ChatML, Llama-3, etc.)."
    ))
    pairs.append(msg(
        "How do I set GPU layers in Ollama for partial offloading?",
        "Use the `num_gpu` parameter in your Modelfile:\n"
        "```dockerfile\nFROM llama3.1:70b\nPARAMETER num_gpu 30\n```\n"
        "Or at runtime via the API:\n"
        "```bash\ncurl http://localhost:11434/api/generate -d '{\n"
        "  \"model\": \"llama3.1:70b\",\n"
        "  \"options\": {\"num_gpu\": 30}\n}'\n```\n"
        "Setting `num_gpu 0` forces CPU-only. The default (`-1` or omitted) offloads all layers to GPU. "
        "Partial offloading is useful when the model doesn't fully fit in VRAM."
    ))
    pairs.append(msg(
        "How do I use the Ollama API to generate completions programmatically?",
        "Ollama serves an OpenAI-compatible API on port 11434:\n\n"
        "**Generate endpoint:**\n"
        "```bash\ncurl http://localhost:11434/api/generate -d '{\n"
        "  \"model\": \"llama3.1:8b\",\n"
        "  \"prompt\": \"Explain quantum computing in one paragraph.\",\n"
        "  \"stream\": false\n}'\n```\n\n"
        "**Chat endpoint (OpenAI-compatible):**\n"
        "```bash\ncurl http://localhost:11434/v1/chat/completions -d '{\n"
        "  \"model\": \"llama3.1:8b\",\n"
        "  \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]\n}'\n```\n\n"
        "Set `stream: true` for SSE streaming. The `/api/generate` endpoint returns `response` (text), "
        "`done` (bool), and timing metrics. The `/v1/` endpoints are drop-in replacements for OpenAI SDKs."
    ))
    pairs.append(msg(
        "How do I serve Ollama on a network interface other than localhost?",
        "Set the `OLLAMA_HOST` environment variable:\n"
        "```bash\nOLLAMA_HOST=0.0.0.0:11434 ollama serve\n```\n"
        "For systemd:\n```bash\nsudo systemctl edit ollama\n```\nAdd:\n"
        "```ini\n[Service]\nEnvironment=\"OLLAMA_HOST=0.0.0.0:11434\"\n```\n"
        "Then `sudo systemctl restart ollama`. On macOS, use:\n"
        "```bash\nlaunchctl setenv OLLAMA_HOST 0.0.0.0:11434\n```\n"
        "Warning: binding to 0.0.0.0 exposes the API to the network — consider a reverse proxy with auth."
    ))
    pairs.append(msg(
        "How do I use Ollama with multiple GPUs?",
        "Ollama automatically uses all available GPUs. To restrict to specific GPUs, set `CUDA_VISIBLE_DEVICES`:\n"
        "```bash\nCUDA_VISIBLE_DEVICES=0,1 ollama serve\n```\n"
        "For AMD GPUs, use `ROCR_VISIBLE_DEVICES`. On Apple Silicon, all GPU cores are used automatically via Metal. "
        "Ollama distributes model layers across GPUs — it doesn't use tensor parallelism. "
        "For a 70B model on 2x 24GB GPUs, set `num_gpu -1` to offload all layers across both cards."
    ))
    pairs.append(msg(
        "What's the difference between Ollama tags like :8b, :8b-q4_0, and :8b-instruct?",
        "Ollama tags encode model variants:\n\n"
        "- **`:8b`** — default quantization (usually Q4_K_M or Q4_0), instruct-tuned\n"
        "- **`:8b-q4_0`** — specific Q4_0 quantization (smaller, slightly less quality)\n"
        "- **`:8b-instruct`** — instruct/chat fine-tuned variant\n"
        "- **`:8b-text`** — base/completion model (no instruction tuning)\n"
        "- **`:8b-fp16`** — full precision (largest, best quality)\n\n"
        "Use `ollama show <model>` to inspect the quantization level, parameter count, and template. "
        "Generally Q4_K_M (default) offers the best size/quality tradeoff."
    ))

    # --- Ollama advanced ---
    pairs.append(msg(
        "How do I run Ollama in a Docker container?",
        "Official Docker image:\n```bash\n# CPU only\ndocker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama\n\n"
        "# NVIDIA GPU\ndocker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama\n```\n\n"
        "Then pull and run models:\n```bash\ndocker exec -it ollama ollama run llama3.1:8b\n```\n"
        "The `-v ollama:/root/.ollama` volume persists downloaded models. "
        "For AMD GPUs, add `--device /dev/kfd --device /dev/dri`."
    ))
    pairs.append(msg(
        "How does Ollama handle concurrent requests?",
        "Ollama queues requests by default (serial execution). Since v0.4+, set `OLLAMA_NUM_PARALLEL` for concurrent inference:\n"
        "```bash\nOLLAMA_NUM_PARALLEL=4 ollama serve\n```\n"
        "This enables continuous batching — multiple requests share the KV cache. "
        "Memory usage increases with parallelism. Set `OLLAMA_MAX_LOADED_MODELS` to control how many models stay in memory simultaneously. "
        "Default is 1 on GPU, 3 on CPU."
    ))
    pairs.append(msg(
        "How do I create an Ollama model with a custom chat template?",
        "Use the `TEMPLATE` instruction in the Modelfile with Go template syntax:\n"
        "```dockerfile\nFROM ./model.gguf\n\n"
        "TEMPLATE \"\"\"{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{- end }}\n"
        "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{- end }}\n"
        "<|im_start|>assistant\n\"\"\"\n```\n"
        "This is the ChatML template. Common templates:\n"
        "- **ChatML**: `<|im_start|>role\\ncontent<|im_end|>` (Qwen, Mistral)\n"
        "- **Llama-3**: `<|start_header_id|>role<|end_header_id|>\\ncontent<|eot_id|>`\n"
        "- **Gemma**: `<start_of_turn>role\\ncontent<end_of_turn>`"
    ))
    pairs.append(msg(
        "How do I check Ollama GPU utilization and memory usage?",
        "Use `ollama ps` to see running models and their GPU memory allocation:\n"
        "```bash\n$ ollama ps\nNAME            ID           SIZE   PROCESSOR    UNTIL\n"
        "llama3.1:8b    365c0bd3c000  5.5 GB  100% GPU     4 minutes from now\n```\n\n"
        "For NVIDIA GPUs, also use `nvidia-smi` for VRAM usage. For Apple Silicon, "
        "use `sudo powermetrics --samplers gpu_power -i 1000` or Activity Monitor > GPU History. "
        "The `PROCESSOR` column shows CPU/GPU split percentage."
    ))
    pairs.append(msg(
        "How do I configure Ollama to keep models loaded in memory?",
        "Set the `OLLAMA_KEEP_ALIVE` environment variable:\n"
        "```bash\n# Keep models loaded for 1 hour (default: 5 minutes)\nOLLAMA_KEEP_ALIVE=1h ollama serve\n```\n"
        "Or per-request via the API:\n"
        "```json\n{\"model\": \"llama3.1:8b\", \"keep_alive\": \"30m\"}\n```\n"
        "Set to `-1` to keep models loaded indefinitely (until server restart). "
        "Set to `0` to unload immediately after each request. "
        "This trades memory for latency — preloaded models skip the ~2-5s cold start."
    ))
    pairs.append(msg(
        "How do I benchmark a model in Ollama?",
        "Ollama doesn't have a built-in benchmark command, but you can measure performance:\n\n"
        "**Timing from API response:**\n"
        "```bash\ncurl -s http://localhost:11434/api/generate -d '{\n"
        "  \"model\": \"llama3.1:8b\",\n"
        "  \"prompt\": \"Write a Python fibonacci function\",\n"
        "  \"stream\": false\n}' | jq '{eval_count, eval_duration, tokens_per_second: (.eval_count / .eval_duration * 1e9)}'\n```\n\n"
        "The response includes `prompt_eval_count`, `prompt_eval_duration` (prefill), "
        "`eval_count`, `eval_duration` (generation). Calculate tok/s = eval_count / eval_duration * 1e9."
    ))

    # --- More Ollama patterns ---
    pairs.append(msg(
        "How do I use Ollama with Python?",
        "Install the official Python library:\n```bash\npip install ollama\n```\n\n"
        "Usage:\n```python\nimport ollama\n\n# Chat\nresponse = ollama.chat(\n"
        "    model='llama3.1:8b',\n    messages=[{'role': 'user', 'content': 'Hello!'}]\n)\n"
        "print(response['message']['content'])\n\n"
        "# Streaming\nfor chunk in ollama.chat(\n    model='llama3.1:8b',\n"
        "    messages=[{'role': 'user', 'content': 'Tell me a story'}],\n    stream=True\n):\n"
        "    print(chunk['message']['content'], end='', flush=True)\n\n"
        "# Embeddings\nemb = ollama.embed(model='nomic-embed-text', input='Hello world')\n```"
    ))
    pairs.append(msg(
        "How do I use Ollama as an OpenAI API drop-in replacement?",
        "Ollama exposes OpenAI-compatible endpoints at `/v1/`. Point any OpenAI SDK to it:\n\n"
        "```python\nfrom openai import OpenAI\n\nclient = OpenAI(\n"
        "    base_url='http://localhost:11434/v1',\n    api_key='ollama'  # required but unused\n)\n\n"
        "response = client.chat.completions.create(\n    model='llama3.1:8b',\n"
        "    messages=[{'role': 'user', 'content': 'Hello!'}]\n)\nprint(response.choices[0].message.content)\n```\n\n"
        "Supports `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/v1/embeddings`. "
        "Most parameters (temperature, top_p, max_tokens, stream) work identically."
    ))
    pairs.append(msg(
        "How do I create a multimodal model in Ollama?",
        "Pull a vision-capable model:\n```bash\nollama pull llava:13b\n```\n\n"
        "Use it with images via the API:\n"
        "```bash\ncurl http://localhost:11434/api/generate -d '{\n"
        "  \"model\": \"llava:13b\",\n"
        "  \"prompt\": \"What do you see in this image?\",\n"
        "  \"images\": [\"'$(base64 -i image.jpg)'\"]\n}'\n```\n\n"
        "Or in Python:\n```python\nimport ollama\nresponse = ollama.chat(\n"
        "    model='llava:13b',\n    messages=[{\n        'role': 'user',\n"
        "        'content': 'Describe this image',\n        'images': ['./photo.jpg']\n    }]\n)\n```\n"
        "Supported models: LLaVA, BakLLaVA, Moondream, LLaVA-Phi."
    ))

    # --- More Ollama ---
    for model, size, quant, desc in [
        ("llama3.1:8b", "4.7 GB", "Q4_K_M", "Meta's Llama 3.1 8B instruct"),
        ("llama3.1:70b", "40 GB", "Q4_K_M", "Meta's Llama 3.1 70B instruct"),
        ("mistral:7b", "4.1 GB", "Q4_0", "Mistral 7B instruct v0.3"),
        ("qwen2.5:32b", "18 GB", "Q4_K_M", "Alibaba Qwen 2.5 32B instruct"),
        ("gemma2:9b", "5.4 GB", "Q4_K_M", "Google Gemma 2 9B instruct"),
        ("phi3:14b", "7.9 GB", "Q4_K_M", "Microsoft Phi-3 Medium 14B"),
        ("codellama:34b", "19 GB", "Q4_K_M", "Meta Code Llama 34B instruct"),
        ("deepseek-coder-v2:16b", "8.9 GB", "Q4_K_M", "DeepSeek Coder V2 Lite 16B"),
    ]:
        pairs.append(msg(
            f"What are the specs for {model} in Ollama?",
            f"**{model}**\n- Description: {desc}\n- Download size: ~{size}\n"
            f"- Default quantization: {quant}\n- Pull: `ollama pull {model}`\n\n"
            f"The model will be cached in `~/.ollama/models/`. "
            f"Use `ollama show {model}` to see architecture details, template, and license."
        ))

    pairs.append(msg(
        "How do I set environment variables for the Ollama server?",
        "Common Ollama environment variables:\n\n"
        "| Variable | Default | Description |\n"
        "|----------|---------|-------------|\n"
        "| `OLLAMA_HOST` | `127.0.0.1:11434` | Bind address |\n"
        "| `OLLAMA_MODELS` | `~/.ollama/models` | Model storage path |\n"
        "| `OLLAMA_NUM_PARALLEL` | `1` | Concurrent request slots |\n"
        "| `OLLAMA_MAX_LOADED_MODELS` | `1` (GPU) / `3` (CPU) | Models in memory |\n"
        "| `OLLAMA_KEEP_ALIVE` | `5m` | How long to keep model loaded |\n"
        "| `OLLAMA_DEBUG` | `0` | Enable debug logging |\n"
        "| `OLLAMA_FLASH_ATTENTION` | `0` | Enable flash attention |\n"
        "| `OLLAMA_ORIGINS` | `*` | Allowed CORS origins |\n\n"
        "Set them in your shell profile, systemd override, or Docker `-e` flags."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 2. vLLM
# ---------------------------------------------------------------------------
def gen_vllm() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do I start a vLLM server for a 7B model?",
        "```bash\npython -m vllm.entrypoints.openai.api_server \\\n"
        "    --model meta-llama/Llama-3.1-8B-Instruct \\\n"
        "    --port 8000 \\\n    --dtype auto \\\n    --max-model-len 4096\n```\n\n"
        "This starts an OpenAI-compatible server. Key flags:\n"
        "- `--dtype auto`: auto-selects bfloat16/float16 based on GPU\n"
        "- `--max-model-len`: limits context length (saves memory)\n"
        "- `--gpu-memory-utilization 0.9`: fraction of VRAM to use (default 0.9)\n"
        "- `--enforce-eager`: disable CUDA graphs for debugging\n"
        "Test with: `curl http://localhost:8000/v1/models`"
    ))
    pairs.append(msg(
        "How do I enable tensor parallelism in vLLM for a 70B model?",
        "Use `--tensor-parallel-size` (or `-tp`):\n"
        "```bash\npython -m vllm.entrypoints.openai.api_server \\\n"
        "    --model meta-llama/Llama-3.1-70B-Instruct \\\n"
        "    --tensor-parallel-size 4 \\\n    --dtype bfloat16 \\\n"
        "    --max-model-len 8192 \\\n    --port 8000\n```\n\n"
        "Requirements:\n- Need N GPUs visible via `CUDA_VISIBLE_DEVICES`\n"
        "- NCCL must be installed for inter-GPU communication\n"
        "- Each GPU handles a slice of every layer (column/row parallelism)\n"
        "- Tensor parallelism is preferred over pipeline parallelism for latency\n"
        "- For 70B BF16: 4x A100 40GB or 2x A100 80GB"
    ))
    pairs.append(msg(
        "How do I serve a quantized model with vLLM?",
        "vLLM supports several quantization formats:\n\n"
        "**AWQ (recommended for vLLM):**\n"
        "```bash\npython -m vllm.entrypoints.openai.api_server \\\n"
        "    --model TheBloke/Llama-2-70B-AWQ \\\n"
        "    --quantization awq \\\n    --dtype half\n```\n\n"
        "**GPTQ:**\n```bash\n--model TheBloke/Llama-2-70B-GPTQ --quantization gptq\n```\n\n"
        "**Marlin (fastest for W4A16):**\n```bash\n--quantization marlin  # auto-converts GPTQ\n```\n\n"
        "AWQ + Marlin kernel gives the best throughput. "
        "vLLM does NOT support GGUF — use llama.cpp for that format."
    ))
    pairs.append(msg(
        "What is PagedAttention in vLLM and why does it matter?",
        "PagedAttention is vLLM's key innovation for memory-efficient KV cache management. "
        "Instead of pre-allocating a contiguous KV cache per sequence, it:\n\n"
        "1. **Divides KV cache into fixed-size blocks** (pages, typically 16 tokens each)\n"
        "2. **Allocates pages on demand** as tokens are generated\n"
        "3. **Uses a page table** to map logical positions to physical memory locations\n"
        "4. **Enables sharing** — multiple sequences can share KV cache pages (e.g., for beam search or shared prefixes)\n\n"
        "Benefits:\n- Near-zero memory waste (vs ~60-80% waste with static allocation)\n"
        "- 2-4x higher throughput via better batching\n"
        "- Enables prefix caching (shared system prompts)\n"
        "- Copy-on-write for beam search\n\n"
        "This is why vLLM achieves significantly higher throughput than naive implementations."
    ))
    pairs.append(msg(
        "How do I configure vLLM for maximum throughput?",
        "Key tuning parameters:\n\n"
        "```bash\npython -m vllm.entrypoints.openai.api_server \\\n"
        "    --model meta-llama/Llama-3.1-8B-Instruct \\\n"
        "    --max-num-batched-tokens 32768 \\\n    --max-num-seqs 256 \\\n"
        "    --enable-prefix-caching \\\n    --enable-chunked-prefill \\\n"
        "    --gpu-memory-utilization 0.95 \\\n    --dtype bfloat16\n```\n\n"
        "Tuning tips:\n"
        "- `--max-num-batched-tokens`: increase for higher throughput (needs more VRAM)\n"
        "- `--max-num-seqs`: max concurrent sequences\n"
        "- `--enable-prefix-caching`: reuses KV cache for shared prefixes (great for RAG)\n"
        "- `--enable-chunked-prefill`: overlaps prefill and decode for better GPU utilization\n"
        "- `--speculative-model`: enable speculative decoding with a draft model\n"
        "- Benchmark with `vllm.entrypoints.openai.api_server --benchmark`"
    ))
    pairs.append(msg(
        "How do I enable speculative decoding in vLLM?",
        "Speculative decoding uses a small draft model to predict tokens, then verifies with the main model:\n\n"
        "```bash\npython -m vllm.entrypoints.openai.api_server \\\n"
        "    --model meta-llama/Llama-3.1-70B-Instruct \\\n"
        "    --speculative-model meta-llama/Llama-3.1-8B-Instruct \\\n"
        "    --num-speculative-tokens 5 \\\n    --speculative-draft-tensor-parallel-size 1\n```\n\n"
        "How it works:\n1. Draft model generates K candidate tokens\n2. Target model verifies all K tokens in one forward pass\n"
        "3. Accepted tokens are kept, first rejected token triggers re-generation\n\n"
        "Speedup depends on acceptance rate (typically 60-80% for similar model families). "
        "Best when draft/target are from the same family. Gives 1.5-2.5x speedup for latency-bound workloads."
    ))
    pairs.append(msg(
        "How do I serve multiple LoRA adapters with vLLM?",
        "vLLM supports serving multiple LoRA adapters on a single base model:\n\n"
        "```bash\npython -m vllm.entrypoints.openai.api_server \\\n"
        "    --model meta-llama/Llama-3.1-8B-Instruct \\\n"
        "    --enable-lora \\\n    --lora-modules sql=./adapters/sql code=./adapters/code \\\n"
        "    --max-loras 4 \\\n    --max-lora-rank 64\n```\n\n"
        "Request a specific adapter:\n```json\n{\"model\": \"sql\", \"messages\": [...]}\n```\n\n"
        "vLLM loads LoRA weights into GPU memory alongside the base model. "
        "The `--max-loras` flag controls how many adapters stay loaded simultaneously. "
        "Adapters are swapped in/out transparently based on request routing."
    ))
    pairs.append(msg(
        "How do I enable prefix caching in vLLM?",
        "Prefix caching reuses KV cache for shared prompt prefixes:\n\n"
        "```bash\npython -m vllm.entrypoints.openai.api_server \\\n"
        "    --model meta-llama/Llama-3.1-8B-Instruct \\\n"
        "    --enable-prefix-caching\n```\n\n"
        "This is particularly effective for:\n"
        "- **RAG workloads**: same system prompt + retrieved context across queries\n"
        "- **Multi-turn chat**: reuses KV cache from previous turns\n"
        "- **Batch processing**: shared instructions across many inputs\n\n"
        "vLLM hashes prompt token sequences and caches their KV states. "
        "Matching prefixes skip the prefill computation entirely. "
        "Can reduce TTFT by 50-90% for repeated prefixes."
    ))
    pairs.append(msg(
        "How do I use vLLM with pipeline parallelism?",
        "Pipeline parallelism splits model layers across GPUs (vs tensor parallelism which splits within layers):\n\n"
        "```bash\npython -m vllm.entrypoints.openai.api_server \\\n"
        "    --model meta-llama/Llama-3.1-70B-Instruct \\\n"
        "    --pipeline-parallel-size 2 \\\n    --tensor-parallel-size 2\n```\n\n"
        "This uses 2x2=4 GPUs: 2 pipeline stages, each with 2-way tensor parallelism.\n\n"
        "When to use each:\n"
        "- **Tensor parallelism**: lower latency (preferred for interactive serving)\n"
        "- **Pipeline parallelism**: works across nodes (PCIe/network), supports heterogeneous GPUs\n"
        "- **Combined**: needed for very large models (405B+) on multi-node clusters\n\n"
        "Pipeline parallelism has higher latency due to inter-stage communication bubbles."
    ))
    pairs.append(msg(
        "What GPU memory does vLLM need for different model sizes?",
        "Approximate VRAM requirements for vLLM (BF16, excluding KV cache):\n\n"
        "| Model Size | BF16 Weight | KV Cache (4K ctx) | Total (min) |\n"
        "|------------|-------------|-------------------|-------------|\n"
        "| 7B | 14 GB | ~1 GB | ~16 GB |\n"
        "| 13B | 26 GB | ~2 GB | ~30 GB |\n"
        "| 34B | 68 GB | ~4 GB | ~74 GB |\n"
        "| 70B | 140 GB | ~8 GB | ~150 GB |\n\n"
        "With quantization (AWQ/GPTQ W4):\n"
        "| Model Size | W4 Weight | Total (min) |\n"
        "|------------|-----------|-------------|\n"
        "| 7B | 3.5 GB | ~5 GB |\n"
        "| 70B | 35 GB | ~45 GB |\n\n"
        "Use `--gpu-memory-utilization` to control VRAM budget. "
        "More VRAM = larger KV cache = more concurrent sequences."
    ))

    # More vLLM pairs
    pairs.append(msg(
        "How do I use vLLM offline for batch inference?",
        "Use the `LLM` class for offline batch processing:\n\n"
        "```python\nfrom vllm import LLM, SamplingParams\n\nllm = LLM(model='meta-llama/Llama-3.1-8B-Instruct')\n\n"
        "prompts = [\n    'Translate to French: Hello world',\n    'Summarize: The quick brown fox...',\n"
        "    'Code: Write a Python sort function',\n]\n\n"
        "params = SamplingParams(temperature=0.7, max_tokens=256, top_p=0.9)\n"
        "outputs = llm.generate(prompts, params)\n\n"
        "for output in outputs:\n    print(output.outputs[0].text)\n```\n\n"
        "Batch inference is 5-10x more efficient than sequential requests because vLLM batches prefill operations."
    ))
    pairs.append(msg(
        "How do I monitor vLLM server metrics?",
        "vLLM exposes Prometheus metrics at `/metrics`:\n"
        "```bash\ncurl http://localhost:8000/metrics\n```\n\n"
        "Key metrics:\n"
        "- `vllm:num_requests_running` — currently processing\n"
        "- `vllm:num_requests_waiting` — queued requests\n"
        "- `vllm:gpu_cache_usage_perc` — KV cache memory utilization\n"
        "- `vllm:avg_prompt_throughput_toks_per_s` — prefill throughput\n"
        "- `vllm:avg_generation_throughput_toks_per_s` — decode throughput\n\n"
        "Scrape with Prometheus and visualize in Grafana. "
        "High `gpu_cache_usage_perc` means you're memory-bound and should reduce `max-num-seqs` or increase `gpu-memory-utilization`."
    ))
    pairs.append(msg(
        "What is continuous batching in vLLM?",
        "Continuous batching (also called iteration-level batching) is vLLM's scheduling strategy:\n\n"
        "**Static batching** (naive): wait for all sequences to finish, then process next batch. "
        "Short sequences waste GPU cycles waiting for long ones.\n\n"
        "**Continuous batching**: at each decode step, vLLM can:\n"
        "1. Remove completed sequences from the batch\n"
        "2. Insert new sequences immediately\n"
        "3. Preempt low-priority sequences if memory is needed\n\n"
        "This keeps GPU utilization near 100% by eliminating idle time between batches. "
        "Combined with PagedAttention, it delivers 2-4x higher throughput than static batching. "
        "The scheduler uses FCFS by default but supports priority-based scheduling."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 3. llama.cpp
# ---------------------------------------------------------------------------
def gen_llamacpp() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do I build llama.cpp from source?",
        "```bash\ngit clone https://github.com/ggml-org/llama.cpp\ncd llama.cpp\n\n"
        "# CPU only\ncmake -B build\ncmake --build build --config Release -j $(nproc)\n\n"
        "# CUDA (NVIDIA)\ncmake -B build -DGGML_CUDA=ON\ncmake --build build --config Release -j $(nproc)\n\n"
        "# Metal (Apple Silicon)\ncmake -B build -DGGML_METAL=ON\ncmake --build build --config Release -j $(sysctl -n hw.logicalcpu)\n\n"
        "# Vulkan (cross-platform GPU)\ncmake -B build -DGGML_VULKAN=ON\ncmake --build build --config Release\n```\n\n"
        "Key binaries: `build/bin/llama-cli` (interactive), `build/bin/llama-server` (HTTP API), "
        "`build/bin/llama-quantize` (quantization), `build/bin/llama-perplexity` (evaluation)."
    ))
    pairs.append(msg(
        "How do I run llama.cpp server with a GGUF model?",
        "```bash\n./build/bin/llama-server \\\n    -m models/llama-3.1-8b-instruct-Q4_K_M.gguf \\\n"
        "    --host 0.0.0.0 --port 8080 \\\n    -c 4096 \\\n    -ngl 99 \\\n"
        "    --chat-template llama3\n```\n\n"
        "Key flags:\n"
        "- `-m`: model file path\n"
        "- `-c`: context length (tokens)\n"
        "- `-ngl 99`: offload all layers to GPU (Metal/CUDA)\n"
        "- `--chat-template`: built-in template (llama3, chatml, gemma, etc.)\n"
        "- `-t 8`: CPU threads\n"
        "- `--n-gpu-layers` / `-ngl`: number of layers on GPU (partial offload)\n\n"
        "The server exposes OpenAI-compatible `/v1/chat/completions` and `/v1/completions` endpoints."
    ))
    pairs.append(msg(
        "How do I quantize a model to GGUF format with llama.cpp?",
        "Two steps: convert to GGUF, then quantize:\n\n"
        "**1. Convert from HuggingFace safetensors:**\n"
        "```bash\npython convert_hf_to_gguf.py \\\n    meta-llama/Llama-3.1-8B-Instruct \\\n"
        "    --outfile models/llama-3.1-8b-f16.gguf \\\n    --outtype f16\n```\n\n"
        "**2. Quantize:**\n"
        "```bash\n./build/bin/llama-quantize \\\n    models/llama-3.1-8b-f16.gguf \\\n"
        "    models/llama-3.1-8b-Q4_K_M.gguf \\\n    Q4_K_M\n```\n\n"
        "Common quantization types: Q4_K_M (recommended default), Q5_K_M (higher quality), "
        "Q8_0 (near-lossless), Q3_K_S (smallest reasonable), IQ2_XS (extreme compression)."
    ))
    pairs.append(msg(
        "What do the GGUF quantization type names mean?",
        "GGUF quantization naming convention:\n\n"
        "```\nQ{bits}_K_{size}\n│  │    │   └── S=Small, M=Medium, L=Large (block config)\n"
        "│  │    └────── K-quant method (block-wise, mixed precision)\n"
        "│  └─────────── Bits per weight (2, 3, 4, 5, 6, 8)\n"
        "└────────────── Quantized\n```\n\n"
        "IQ = Importance-based quantization (newer, better quality per bit):\n"
        "- `IQ2_XS`: 2.3 bpw, extreme compression\n"
        "- `IQ3_M`: 3.4 bpw, competitive with Q4_0\n"
        "- `IQ4_NL`: 4.5 bpw, non-linear quant\n\n"
        "Q without K = legacy (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) — simpler, less accurate.\n"
        "K-quants use different bit widths for different layers based on importance."
    ))
    pairs.append(msg(
        "How do I use KV cache quantization in llama.cpp?",
        "KV cache quantization reduces memory usage for long contexts:\n\n"
        "```bash\n./build/bin/llama-server \\\n    -m model.gguf \\\n    -c 32768 \\\n"
        "    --cache-type-k q8_0 \\\n    --cache-type-v q8_0 \\\n    -ngl 99\n```\n\n"
        "Options: `f16` (default), `q8_0`, `q4_0`, `q4_1`\n\n"
        "Memory savings for 8B model at 32K context:\n"
        "- f16: ~4 GB KV cache\n"
        "- q8_0: ~2 GB (50% reduction, minimal quality loss)\n"
        "- q4_0: ~1 GB (75% reduction, slight quality loss on long contexts)\n\n"
        "KV cache quantization is orthogonal to model quantization — you can use Q4_K_M model weights with q8_0 KV cache."
    ))
    pairs.append(msg(
        "How do I use flash attention in llama.cpp?",
        "Flash attention is enabled automatically on supported platforms:\n\n"
        "```bash\n# Build with flash attention support\ncmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON\ncmake --build build --config Release\n\n"
        "# Enable at runtime\n./build/bin/llama-server -m model.gguf -fa\n```\n\n"
        "The `-fa` flag enables flash attention. Benefits:\n"
        "- 2-4x faster attention computation\n"
        "- Lower memory usage (no O(n^2) attention matrix)\n"
        "- Especially beneficial for long contexts (>4K tokens)\n\n"
        "On Apple Silicon (Metal), flash attention is integrated into the Metal kernels. "
        "On CUDA, it uses the optimized FA kernels. Required for efficient 32K+ context."
    ))
    pairs.append(msg(
        "How do I use llama.cpp for batch inference?",
        "Use the parallel slots feature of llama-server:\n\n"
        "```bash\n./build/bin/llama-server \\\n    -m model.gguf \\\n    -c 8192 \\\n"
        "    --parallel 8 \\\n    -ngl 99\n```\n\n"
        "`--parallel N` creates N slots for concurrent requests. The context is split: "
        "8192/8 = 1024 tokens per slot. Alternatively, use `-c 8192 -np 8` (older syntax).\n\n"
        "For offline batch processing, use `llama-batch`:\n"
        "```bash\n./build/bin/llama-batch \\\n    -m model.gguf \\\n    -f prompts.txt \\\n"
        "    -ngl 99 -c 4096 -np 8\n```\n"
        "Each line in prompts.txt is a separate prompt. Results are processed in parallel."
    ))
    pairs.append(msg(
        "How do I measure perplexity with llama.cpp?",
        "Use `llama-perplexity` for model quality evaluation:\n\n"
        "```bash\n./build/bin/llama-perplexity \\\n    -m model-Q4_K_M.gguf \\\n"
        "    -f wiki.test.raw \\\n    -c 2048 \\\n    --ppl-stride 512 \\\n    -ngl 99\n```\n\n"
        "Key flags:\n"
        "- `-f`: text file for evaluation (commonly WikiText-2)\n"
        "- `-c`: context window for each chunk\n"
        "- `--ppl-stride`: sliding window stride (lower = more accurate but slower)\n\n"
        "Typical perplexity values (WikiText-2, Llama 3.1 8B):\n"
        "- F16: 6.50\n- Q8_0: 6.51 (+0.01)\n- Q5_K_M: 6.55 (+0.05)\n"
        "- Q4_K_M: 6.62 (+0.12)\n- Q3_K_M: 6.85 (+0.35)\n- IQ2_XS: 8.20 (+1.70)\n\n"
        "Lower is better. Perplexity increase above F16 indicates quantization degradation."
    ))
    pairs.append(msg(
        "How do I use the llama.cpp GGUF metadata?",
        "Inspect GGUF file metadata:\n"
        "```bash\n./build/bin/llama-gguf-meta model.gguf\n```\n\n"
        "Key metadata fields:\n"
        "- `general.architecture`: model architecture (llama, qwen2, gemma, etc.)\n"
        "- `general.name`: model name\n"
        "- `llama.context_length`: max context (e.g., 131072)\n"
        "- `llama.embedding_length`: hidden size\n"
        "- `llama.block_count`: number of transformer layers\n"
        "- `llama.attention.head_count`: number of attention heads\n"
        "- `llama.attention.head_count_kv`: number of KV heads (GQA)\n"
        "- `tokenizer.ggml.model`: tokenizer type\n"
        "- `general.quantization_version`: GGML quant version\n\n"
        "You can also read metadata programmatically with the `gguf` Python package: "
        "`pip install gguf && python -m gguf.gguf_reader model.gguf`"
    ))
    pairs.append(msg(
        "What is GQA and how does it affect llama.cpp performance?",
        "GQA (Grouped Query Attention) shares KV heads across multiple query heads:\n\n"
        "- **MHA**: every query head has its own KV head (e.g., 32Q/32KV)\n"
        "- **GQA**: KV heads are shared in groups (e.g., 32Q/8KV in Llama 3.1)\n"
        "- **MQA**: all query heads share one KV pair (e.g., 32Q/1KV)\n\n"
        "Impact on llama.cpp:\n"
        "- **Memory**: GQA reduces KV cache by `num_kv_heads/num_heads` (4x for 32/8)\n"
        "- **Speed**: fewer KV heads = less memory bandwidth = faster decode\n"
        "- **Quality**: GQA models match MHA quality at 70B+ scale\n\n"
        "KV cache size formula: `2 * num_layers * num_kv_heads * head_dim * context_len * dtype_bytes`\n\n"
        "For Llama 3.1 8B (GQA 8 KV heads): KV cache at 8K context = ~512 MB (F16)"
    ))
    pairs.append(msg(
        "How do I use llama.cpp with Apple Silicon Metal?",
        "Build with Metal support:\n"
        "```bash\ncmake -B build -DGGML_METAL=ON\ncmake --build build --config Release\n```\n\n"
        "Run with full GPU offload:\n"
        "```bash\n./build/bin/llama-server -m model.gguf -ngl 99 -c 4096\n```\n\n"
        "Metal performance tips:\n"
        "- All layers should be offloaded (`-ngl 99`) — partial offload is slower due to CPU-GPU transfers\n"
        "- Apple Silicon uses unified memory — model + KV cache share the same pool\n"
        "- M1/M2: ~100 GB/s bandwidth → 20-35 tok/s for 7B Q4_K_M\n"
        "- M3 Ultra: ~800 GB/s → 80-120 tok/s for 7B Q4_K_M\n"
        "- M4 Max 128GB can run 70B Q4_K_M at ~15 tok/s\n"
        "- Use `-fa` for flash attention on M3+ chips\n\n"
        "Check Metal GPU usage: `sudo powermetrics --samplers gpu_power -i 1000`"
    ))
    pairs.append(msg(
        "How do I use llama.cpp grammars for structured output?",
        "llama.cpp supports GBNF grammars for constrained generation:\n\n"
        "```bash\n./build/bin/llama-cli -m model.gguf --grammar '\n"
        "root ::= \"{\" ws \"\\\"name\\\"\" ws \":\" ws string \",\" ws \"\\\"age\\\"\" ws \":\" ws number \"}\" ws\n"
        "string ::= \"\\\"\" [a-zA-Z ]+ \"\\\"\"\n"
        "number ::= [0-9]+\n"
        "ws ::= [ \\t\\n]*\n'\n```\n\n"
        "Or use JSON schema mode:\n"
        "```bash\ncurl http://localhost:8080/v1/chat/completions -d '{\n"
        "  \"messages\": [{\"role\": \"user\", \"content\": \"Generate a person\"}],\n"
        "  \"response_format\": {\"type\": \"json_object\", \"schema\": {\n"
        "    \"type\": \"object\",\n    \"properties\": {\"name\": {\"type\": \"string\"}, \"age\": {\"type\": \"integer\"}},\n"
        "    \"required\": [\"name\", \"age\"]\n  }}\n}'\n```\n"
        "JSON schema is automatically converted to GBNF grammar."
    ))

    # Extra llama.cpp pairs
    pairs.append(msg(
        "How do I use llama.cpp speculative decoding?",
        "Speculative decoding in llama.cpp uses a small draft model:\n\n"
        "```bash\n./build/bin/llama-server \\\n    -m llama-70b-Q4_K_M.gguf \\\n"
        "    -md llama-8b-Q4_K_M.gguf \\\n    --draft-max 8 \\\n    --draft-min 1 \\\n"
        "    -ngl 99 -ngld 99\n```\n\n"
        "Flags:\n"
        "- `-md`: draft model path\n"
        "- `--draft-max`: max speculative tokens per step\n"
        "- `--draft-min`: min speculative tokens\n"
        "- `-ngld`: GPU layers for draft model\n\n"
        "The draft model must use the same tokenizer as the target model. "
        "Typical speedup: 1.5-2.5x for latency-sensitive workloads. "
        "Throughput may decrease if draft model acceptance rate is low."
    ))
    pairs.append(msg(
        "How do I use the llama.cpp Python bindings?",
        "Install `llama-cpp-python`:\n```bash\nCMAKE_ARGS=\"-DGGML_METAL=ON\" pip install llama-cpp-python\n```\n\n"
        "Usage:\n```python\nfrom llama_cpp import Llama\n\n"
        "llm = Llama(\n    model_path='model-Q4_K_M.gguf',\n    n_ctx=4096,\n    n_gpu_layers=-1,\n"
        "    chat_format='chatml'\n)\n\n"
        "# Chat completion\nresponse = llm.create_chat_completion(\n"
        "    messages=[{'role': 'user', 'content': 'Hello!'}],\n"
        "    temperature=0.7,\n    max_tokens=256\n)\nprint(response['choices'][0]['message']['content'])\n\n"
        "# Streaming\nfor chunk in llm.create_chat_completion(\n"
        "    messages=[{'role': 'user', 'content': 'Tell me a story'}],\n    stream=True\n):\n"
        "    delta = chunk['choices'][0]['delta']\n    if 'content' in delta:\n"
        "        print(delta['content'], end='', flush=True)\n```"
    ))

    return pairs


# ---------------------------------------------------------------------------
# 4. GGUF Quantization
# ---------------------------------------------------------------------------
def gen_gguf_quant() -> list[dict]:
    pairs = []

    quant_table = [
        ("Q2_K", "2.6", "3.35", "50%", "Very low quality, only for extreme compression"),
        ("IQ2_XS", "2.3", "2.96", "55%", "Importance quantization, better than Q2_K at similar size"),
        ("IQ2_M", "2.7", "3.48", "48%", "Good 2-bit option with importance weighting"),
        ("Q3_K_S", "3.0", "3.86", "43%", "Small 3-bit, noticeable quality loss"),
        ("Q3_K_M", "3.3", "4.21", "40%", "Medium 3-bit, usable for less critical tasks"),
        ("Q3_K_L", "3.6", "4.57", "38%", "Large 3-bit, best 3-bit option"),
        ("IQ3_M", "3.4", "4.32", "39%", "Importance 3-bit, competitive with Q4_0"),
        ("Q4_0", "4.0", "5.14", "36%", "Legacy 4-bit, uniform quantization"),
        ("Q4_K_S", "4.2", "5.35", "34%", "Small K-quant 4-bit"),
        ("Q4_K_M", "4.6", "5.85", "31%", "RECOMMENDED: best quality/size tradeoff"),
        ("Q5_0", "5.0", "6.40", "27%", "Legacy 5-bit"),
        ("Q5_K_S", "5.2", "6.64", "26%", "Small K-quant 5-bit"),
        ("Q5_K_M", "5.5", "7.00", "24%", "High quality, near-lossless for most tasks"),
        ("Q6_K", "6.2", "7.89", "20%", "Very high quality, minimal perplexity loss"),
        ("Q8_0", "8.0", "10.2", "12%", "Near-lossless, largest practical quantization"),
        ("F16", "16.0", "20.4", "0%", "Full precision half-float, reference quality"),
    ]

    for qtype, bpw, size_7b, reduction, desc in quant_table:
        pairs.append(msg(
            f"What is GGUF {qtype} quantization?",
            f"**{qtype}** — {bpw} bits per weight\n\n"
            f"- File size for 7B model: ~{size_7b} GB\n"
            f"- Size reduction vs F16: ~{reduction}\n"
            f"- Description: {desc}\n\n"
            f"Quantize with: `llama-quantize model-f16.gguf model-{qtype}.gguf {qtype}`\n\n"
            f"{'K-quant types use mixed precision per layer — attention layers get higher precision than FFN layers.' if 'K' in qtype else ''}"
            f"{'Importance quantization (IQ) uses data-dependent quantization with calibration data for better quality at low bit widths.' if qtype.startswith('IQ') else ''}"
        ))

    pairs.append(msg(
        "What quantization should I choose for my GGUF model?",
        "Decision guide:\n\n"
        "| Use Case | Recommended | Why |\n"
        "|----------|-------------|-----|\n"
        "| Best quality/size tradeoff | **Q4_K_M** | Industry standard, minimal perplexity loss |\n"
        "| High quality, more VRAM | **Q5_K_M** | +0.05 ppl vs F16, great for important tasks |\n"
        "| Near-lossless | **Q8_0** | +0.01 ppl vs F16, 50% smaller |\n"
        "| Maximum compression | **IQ2_XS** | Smallest usable, for very limited memory |\n"
        "| Balanced compression | **Q3_K_M** or **IQ3_M** | Good for running large models on small GPUs |\n"
        "| Embeddings/retrieval | **Q8_0** or **Q6_K** | Quality-sensitive tasks need higher precision |\n\n"
        "Rule of thumb: Q4_K_M is the sweet spot for 95% of use cases."
    ))
    pairs.append(msg(
        "How does perplexity change with GGUF quantization levels?",
        "Typical perplexity impact (WikiText-2, Llama 3.1 8B baseline F16=6.50):\n\n"
        "| Quantization | Perplexity | Delta | Quality |\n"
        "|-------------|-----------|-------|---------|\n"
        "| F16 | 6.50 | — | Reference |\n"
        "| Q8_0 | 6.51 | +0.01 | Imperceptible |\n"
        "| Q6_K | 6.53 | +0.03 | Imperceptible |\n"
        "| Q5_K_M | 6.55 | +0.05 | Minimal |\n"
        "| Q4_K_M | 6.62 | +0.12 | Minor |\n"
        "| Q4_0 | 6.68 | +0.18 | Noticeable on hard tasks |\n"
        "| Q3_K_M | 6.85 | +0.35 | Visible degradation |\n"
        "| IQ2_XS | 8.20 | +1.70 | Significant degradation |\n\n"
        "Perplexity impact scales with model size — larger models are more robust to quantization. "
        "A 70B at Q4_K_M may have lower perplexity than a 7B at F16."
    ))
    pairs.append(msg(
        "What is the difference between K-quant and legacy quantization in GGUF?",
        "**Legacy quantization** (Q4_0, Q5_0, Q8_0):\n"
        "- Uniform bit width across all layers\n"
        "- Simple block-wise quantization\n"
        "- Q8_0 is the only legacy format still recommended (it's straightforward 8-bit)\n\n"
        "**K-quant** (Q4_K_M, Q5_K_S, Q6_K, etc.):\n"
        "- **Mixed precision**: different layers get different bit widths\n"
        "- Attention layers get higher precision (more important for quality)\n"
        "- Uses super-blocks with sub-blocks for finer granularity\n"
        "- S/M/L variants control the mix:\n"
        "  - S: more layers at lower precision (smaller file)\n"
        "  - M: balanced mix (recommended)\n"
        "  - L: more layers at higher precision (better quality)\n\n"
        "**IQ (Importance Quantization)**:\n"
        "- Data-dependent: uses calibration data to determine importance\n"
        "- Best quality per bit at 2-3 bit widths\n"
        "- Slightly slower inference due to lookup tables"
    ))
    pairs.append(msg(
        "How do I create an imatrix for importance quantization?",
        "An importance matrix (imatrix) improves IQ quantization quality:\n\n"
        "```bash\n# Step 1: Generate importance matrix from calibration data\n"
        "./build/bin/llama-imatrix \\\n    -m model-f16.gguf \\\n    -f calibration-data.txt \\\n"
        "    -o model.imatrix \\\n    -ngl 99\n\n"
        "# Step 2: Quantize using the importance matrix\n"
        "./build/bin/llama-quantize \\\n    --imatrix model.imatrix \\\n"
        "    model-f16.gguf \\\n    model-IQ2_XS.gguf \\\n    IQ2_XS\n```\n\n"
        "The calibration data should be representative of your use case — "
        "use a few MB of text from your domain. WikiText-2 is commonly used as a general-purpose calibration set. "
        "The imatrix records per-weight importance scores used to allocate precision."
    ))

    # Extra GGUF pairs
    for model, f16_size, q4km_size, q8_size in [
        ("1.5B", "3.0 GB", "1.0 GB", "1.7 GB"),
        ("3B", "6.0 GB", "2.0 GB", "3.4 GB"),
        ("7B", "14 GB", "4.4 GB", "7.7 GB"),
        ("8B", "16 GB", "5.0 GB", "8.5 GB"),
        ("13B", "26 GB", "7.9 GB", "14 GB"),
        ("14B", "28 GB", "8.5 GB", "15 GB"),
        ("32B", "64 GB", "19 GB", "34 GB"),
        ("35B (MoE 3B active)", "22 GB", "7.0 GB", "12 GB"),
        ("70B", "140 GB", "42 GB", "75 GB"),
        ("72B", "144 GB", "43 GB", "77 GB"),
        ("405B", "810 GB", "243 GB", "432 GB"),
    ]:
        pairs.append(msg(
            f"What is the GGUF file size for a {model} parameter model?",
            f"Approximate GGUF sizes for a {model} model:\n\n"
            f"| Quantization | Size | Notes |\n"
            f"|-------------|------|-------|\n"
            f"| F16 | ~{f16_size} | Full precision reference |\n"
            f"| Q8_0 | ~{q8_size} | Near-lossless |\n"
            f"| Q4_K_M | ~{q4km_size} | Recommended default |\n\n"
            f"Formula: `size_bytes ≈ num_params * bits_per_weight / 8`\n"
            f"For Q4_K_M (~4.6 bpw): `{model.split('B')[0]}e9 * 4.6 / 8 ≈ {q4km_size}`\n\n"
            f"These are weight-only sizes. Add KV cache memory at runtime "
            f"(varies with context length and architecture)."
        ))

    pairs.append(msg(
        "How does GGUF handle different tensor types within one file?",
        "GGUF supports mixed quantization — different tensors can use different types:\n\n"
        "In K-quant formats, the quantizer automatically assigns:\n"
        "- **Attention Q/K/V/O projections**: higher precision (e.g., Q6_K in Q4_K_M files)\n"
        "- **FFN up/gate/down**: lower precision (e.g., Q4_K in Q4_K_M files)\n"
        "- **Embedding/output layers**: always high precision (Q6_K or F16)\n\n"
        "You can inspect per-tensor quantization with:\n"
        "```bash\npython -c \"\nimport gguf\nreader = gguf.GGUFReader('model.gguf')\n"
        "for tensor in reader.tensors:\n    print(f'{tensor.name}: {tensor.tensor_type.name} shape={tensor.shape}')\n\"\n```\n\n"
        "This mixed approach is why K-quants achieve better quality than uniform quantization at the same average bit width."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 5. Model Serving
# ---------------------------------------------------------------------------
def gen_serving() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What is continuous batching in LLM serving?",
        "Continuous batching (also called iteration-level or in-flight batching) processes requests dynamically:\n\n"
        "**Static batching**: groups requests, waits for all to complete before starting the next batch. "
        "Short sequences idle while waiting for long ones.\n\n"
        "**Continuous batching**: each decoding iteration can:\n"
        "1. Remove completed sequences from the batch\n"
        "2. Insert newly arrived requests immediately\n"
        "3. Preempt sequences if needed (for priority or memory)\n\n"
        "Benefits:\n"
        "- 2-5x higher throughput than static batching\n"
        "- Lower latency for short sequences\n"
        "- Near-100% GPU utilization\n\n"
        "Implemented by: vLLM, TensorRT-LLM, TGI, SGLang, llama.cpp (server mode). "
        "The key enabler is PagedAttention (vLLM) or similar dynamic KV cache management."
    ))
    pairs.append(msg(
        "How does KV cache memory scale with context length?",
        "KV cache memory formula:\n\n"
        "```\nKV_cache_bytes = 2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes\n```\n\n"
        "The `2×` accounts for both K and V caches.\n\n"
        "Example — Llama 3.1 8B (32 layers, 8 KV heads, 128 head_dim, FP16):\n"
        "- 4K context: 2 × 32 × 8 × 128 × 4096 × 2 = **512 MB**\n"
        "- 32K context: **4 GB**\n"
        "- 128K context: **16 GB**\n\n"
        "For batch serving, multiply by batch size. This is why KV cache is often the memory bottleneck. "
        "Solutions: KV cache quantization (q8_0/q4_0), GQA (reduces num_kv_heads), "
        "PagedAttention (eliminates waste), and sliding window attention."
    ))
    pairs.append(msg(
        "What is speculative decoding and when should I use it?",
        "Speculative decoding accelerates autoregressive generation:\n\n"
        "**How it works:**\n"
        "1. A small draft model generates K candidate tokens quickly\n"
        "2. The large target model verifies all K tokens in ONE forward pass\n"
        "3. Accepted tokens are kept; first rejected token triggers re-draft\n\n"
        "**When to use:**\n"
        "- Latency-sensitive applications (chatbots, interactive coding)\n"
        "- When you have a small model from the same family (Llama 8B → 70B)\n"
        "- When acceptance rate is high (>60%)\n\n"
        "**When NOT to use:**\n"
        "- Throughput-optimized batch serving (adds overhead per request)\n"
        "- When draft and target use different tokenizers\n"
        "- When VRAM is tight (draft model needs memory too)\n\n"
        "Typical speedup: 1.5-2.5x for latency, with exact same output quality (rejection sampling preserves target distribution)."
    ))
    pairs.append(msg(
        "What is prefix caching and how does it work?",
        "Prefix caching reuses KV cache computations for shared prompt prefixes:\n\n"
        "**Problem**: In RAG or multi-turn chat, the system prompt + context is often repeated. "
        "Without caching, this prefix is recomputed for every request.\n\n"
        "**Solution**: Hash the prefix token sequence, cache its KV states:\n"
        "1. First request: compute KV for full prompt, cache prefix KV\n"
        "2. Subsequent requests: look up prefix hash, reuse cached KV\n"
        "3. Only compute KV for new (non-cached) tokens\n\n"
        "**Impact:**\n"
        "- TTFT reduction: 50-90% for long shared prefixes\n"
        "- Memory tradeoff: cached KV blocks use GPU memory\n"
        "- Works best with: fixed system prompts, RAG pipelines, few-shot examples\n\n"
        "Available in: vLLM (`--enable-prefix-caching`), SGLang (RadixAttention), "
        "Anthropic API (prompt caching), OpenAI API (automatic)."
    ))
    pairs.append(msg(
        "How do I calculate VRAM needed for LLM inference?",
        "VRAM breakdown for LLM inference:\n\n"
        "```\nTotal VRAM = Model Weights + KV Cache + Activation Memory + Framework Overhead\n```\n\n"
        "**1. Model Weights:**\n"
        "- FP16: `params × 2 bytes` (7B → 14 GB)\n"
        "- Q4_K_M: `params × 0.58 bytes` (7B → 4.1 GB)\n"
        "- Q8_0: `params × 1.06 bytes` (7B → 7.4 GB)\n\n"
        "**2. KV Cache (per sequence):**\n"
        "- `2 × layers × kv_heads × head_dim × seq_len × dtype_bytes`\n"
        "- Multiply by batch_size for concurrent requests\n\n"
        "**3. Activation Memory:** ~500 MB - 2 GB (temporary, during forward pass)\n\n"
        "**4. Framework Overhead:** ~500 MB - 1 GB (CUDA context, buffers)\n\n"
        "Quick rule of thumb: model_size_GB × 1.2 for basic single-sequence inference."
    ))
    pairs.append(msg(
        "What are the main LLM serving frameworks and how do they compare?",
        "| Framework | Key Feature | Best For |\n"
        "|-----------|------------|----------|\n"
        "| **vLLM** | PagedAttention | High-throughput GPU serving |\n"
        "| **llama.cpp** | GGUF quantization | CPU/Metal, edge devices |\n"
        "| **TensorRT-LLM** | NVIDIA optimization | Max perf on NVIDIA GPUs |\n"
        "| **SGLang** | RadixAttention | Complex LLM programs, structured output |\n"
        "| **TGI** | Production-ready | HuggingFace ecosystem |\n"
        "| **Ollama** | Simplicity | Local development |\n"
        "| **MLX** | Apple Silicon native | Mac inference/training |\n"
        "| **ExLlamaV2** | EXL2 quantization | Consumer GPU inference |\n\n"
        "Decision matrix:\n"
        "- Production GPU cluster → vLLM or TensorRT-LLM\n"
        "- Local Mac → MLX or llama.cpp (Metal)\n"
        "- Local NVIDIA → llama.cpp (CUDA) or ExLlamaV2\n"
        "- Quick prototyping → Ollama\n"
        "- HuggingFace models → TGI"
    ))
    pairs.append(msg(
        "What is chunked prefill in LLM serving?",
        "Chunked prefill overlaps prefill (prompt processing) with decode (token generation):\n\n"
        "**Problem**: Long prompts cause prefill to monopolize the GPU, blocking decode for in-flight requests.\n\n"
        "**Solution**: Break prefill into smaller chunks and interleave with decode steps:\n"
        "1. Process chunk of new prompt (e.g., 512 tokens)\n"
        "2. Generate one decode token for existing sequences\n"
        "3. Process next prefill chunk\n"
        "4. Repeat until prefill is complete\n\n"
        "**Benefits:**\n"
        "- Reduces tail latency for in-flight requests\n"
        "- Smoother token generation rate\n"
        "- Better GPU utilization during long prefills\n\n"
        "**Tradeoff**: Slightly higher TTFT for the new request. "
        "Enable in vLLM with `--enable-chunked-prefill --max-num-batched-tokens 2048`."
    ))
    pairs.append(msg(
        "How do I set up a load balancer for multiple LLM servers?",
        "For distributing requests across multiple vLLM/llama.cpp instances:\n\n"
        "**Nginx (simple round-robin):**\n"
        "```nginx\nupstream llm_backends {\n    server gpu1:8000;\n    server gpu2:8000;\n"
        "    server gpu3:8000;\n}\nserver {\n    listen 80;\n    location /v1/ {\n"
        "        proxy_pass http://llm_backends;\n        proxy_http_version 1.1;\n"
        "        proxy_set_header Connection '';\n        proxy_buffering off;\n    }\n}\n```\n\n"
        "**LiteLLM (model-aware routing):**\n"
        "```yaml\nmodel_list:\n  - model_name: llama-3.1-8b\n    litellm_params:\n"
        "      model: openai/llama-3.1-8b\n      api_base: http://gpu1:8000/v1\n"
        "  - model_name: llama-3.1-8b\n    litellm_params:\n"
        "      model: openai/llama-3.1-8b\n      api_base: http://gpu2:8000/v1\n```\n\n"
        "LiteLLM handles load balancing, fallback, rate limiting, and cost tracking across providers."
    ))

    # More serving patterns
    pairs.append(msg(
        "What is SplitFuse in TensorRT-LLM?",
        "SplitFuse is TensorRT-LLM's optimization for mixed prefill/decode batches:\n\n"
        "Instead of running prefill and decode as separate kernel launches, SplitFuse:\n"
        "1. Splits long prefill sequences into chunks\n"
        "2. Fuses decode tokens from other sequences into the same batch\n"
        "3. Runs a single optimized kernel for the mixed batch\n\n"
        "Benefits:\n"
        "- Eliminates GPU idle time between prefill and decode\n"
        "- 20-40% higher throughput vs separate batching\n"
        "- Reduces P99 latency for decode tokens during prefill storms\n\n"
        "Similar to vLLM's chunked prefill but with NVIDIA-specific kernel optimizations."
    ))
    pairs.append(msg(
        "How do I serve multiple models on a single GPU?",
        "Strategies for multi-model serving:\n\n"
        "**1. Time-multiplexing (Ollama-style):**\n"
        "- Load/unload models on demand\n"
        "- Only one model active at a time\n"
        "- Good for low-traffic, many models\n\n"
        "**2. LoRA adapters (vLLM):**\n"
        "- One base model + multiple LoRA adapters\n"
        "- Adapters are ~50-200 MB each\n"
        "- vLLM: `--enable-lora --lora-modules name1=path1 name2=path2`\n\n"
        "**3. Memory partitioning:**\n"
        "- Assign GPU memory fractions: `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`\n"
        "- Run separate server processes\n"
        "- Risk: OOM if not carefully tuned\n\n"
        "**4. Draft/target co-location:**\n"
        "- Small draft model + large target for speculative decoding\n"
        "- Both on same GPU, draft uses minimal VRAM\n\n"
        "LoRA adapter serving is the most practical for related tasks."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 6. Hardware Requirements
# ---------------------------------------------------------------------------
def gen_hardware() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How much VRAM do I need to run different model sizes?",
        "VRAM requirements by model size (Q4_K_M quantization, 4K context):\n\n"
        "| Model | Weight VRAM | KV Cache | Total | Min GPU |\n"
        "|-------|------------|----------|-------|----------|\n"
        "| 1.5B | 1.0 GB | 0.1 GB | ~1.5 GB | Any GPU |\n"
        "| 3B | 2.0 GB | 0.2 GB | ~2.5 GB | GTX 1060 6GB |\n"
        "| 7B | 4.4 GB | 0.5 GB | ~5.5 GB | RTX 3060 8GB |\n"
        "| 8B | 5.0 GB | 0.5 GB | ~6 GB | RTX 3070 8GB |\n"
        "| 13B | 7.9 GB | 0.8 GB | ~10 GB | RTX 3080 10GB |\n"
        "| 32B | 19 GB | 1.5 GB | ~22 GB | RTX 4090 24GB |\n"
        "| 70B | 42 GB | 3 GB | ~47 GB | 2x RTX 4090 or A100 80GB |\n\n"
        "For FP16 inference, multiply weight VRAM by ~3.5x. "
        "For long contexts (32K+), KV cache becomes the dominant factor."
    ))
    pairs.append(msg(
        "What Apple Silicon Macs can run which LLM sizes?",
        "Apple Silicon unified memory for LLM inference (Q4_K_M, comfortable):\n\n"
        "| Mac | Unified Memory | Max Model | tok/s (approx) |\n"
        "|-----|---------------|-----------|----------------|\n"
        "| M1/M2 8GB | 8 GB | 7B | 15-25 |\n"
        "| M1/M2 16GB | 16 GB | 13B | 15-25 |\n"
        "| M2 Pro 32GB | 32 GB | 32B | 20-30 |\n"
        "| M2 Max 64GB | 64 GB | 70B | 12-18 |\n"
        "| M2 Ultra 192GB | 192 GB | 70B FP16 | 25-35 |\n"
        "| M3 Max 128GB | 128 GB | 70B Q8_0 | 18-25 |\n"
        "| M3 Ultra 192GB | 192 GB | 405B Q4 | 5-8 |\n"
        "| M4 Max 128GB | 128 GB | 70B Q8_0 | 25-35 |\n\n"
        "Memory bandwidth matters more than capacity for inference speed:\n"
        "- M1: 68 GB/s → ~15 tok/s per 7B\n"
        "- M3 Max: 400 GB/s → ~50 tok/s per 7B\n"
        "- M3 Ultra: 800 GB/s → ~100 tok/s per 7B"
    ))
    pairs.append(msg(
        "How does memory bandwidth affect LLM inference speed?",
        "LLM token generation is **memory bandwidth-bound**, not compute-bound:\n\n"
        "**Why**: Each generated token requires reading the entire model weights once. "
        "For a 7B Q4_K_M model (~4.4 GB), generating 1 token = reading 4.4 GB from memory.\n\n"
        "**Formula**: `max_tok/s ≈ memory_bandwidth / model_size_bytes`\n\n"
        "| Hardware | Bandwidth | 7B Q4 tok/s | 70B Q4 tok/s |\n"
        "|----------|-----------|-------------|---------------|\n"
        "| DDR4-3200 (CPU) | 50 GB/s | ~11 | ~1.1 |\n"
        "| DDR5-5600 (CPU) | 90 GB/s | ~20 | ~2.0 |\n"
        "| Apple M3 Max | 400 GB/s | ~90 | ~9.5 |\n"
        "| RTX 4090 | 1008 GB/s | ~230 | ~24 |\n"
        "| A100 80GB | 2039 GB/s | ~460 | ~48 |\n"
        "| H100 | 3350 GB/s | ~760 | ~80 |\n\n"
        "This is why GPU > CPU for inference, and why Apple Silicon's high bandwidth is competitive. "
        "Batch inference is compute-bound (amortizes weight reads across sequences)."
    ))
    pairs.append(msg(
        "What is the difference between CPU and GPU inference for LLMs?",
        "**GPU inference:**\n"
        "- 10-100x faster token generation than CPU\n"
        "- Limited by VRAM (model must fit)\n"
        "- Excellent for batch processing\n"
        "- CUDA (NVIDIA), ROCm (AMD), Metal (Apple)\n\n"
        "**CPU inference:**\n"
        "- Uses system RAM (typically 16-128 GB available)\n"
        "- Much slower due to lower memory bandwidth\n"
        "- Good for: very large models that don't fit in VRAM, low-traffic deployments\n"
        "- llama.cpp with AVX-512/AMX is the fastest CPU backend\n\n"
        "**Hybrid (partial offload):**\n"
        "- Some layers on GPU, rest on CPU\n"
        "- Use when model almost fits in VRAM\n"
        "- llama.cpp: `-ngl 20` offloads 20 layers to GPU\n"
        "- Performance scales with fraction offloaded\n\n"
        "**Apple Silicon:**\n"
        "- Unified memory = no CPU↔GPU transfer overhead\n"
        "- Metal backend is competitive with mid-range NVIDIA GPUs\n"
        "- Best framework: MLX or llama.cpp with Metal"
    ))
    pairs.append(msg(
        "How do I estimate training VRAM requirements?",
        "Training requires ~3-4x more memory than inference:\n\n"
        "**Full fine-tuning (FP16):**\n"
        "- Model weights: `params × 2 bytes`\n"
        "- Gradients: `params × 2 bytes`\n"
        "- Optimizer (Adam): `params × 8 bytes` (momentum + variance, FP32)\n"
        "- Activations: varies with batch size and sequence length\n"
        "- Total: ~`params × 14 bytes` + activations\n"
        "- 7B → ~120 GB (not feasible on consumer GPUs)\n\n"
        "**LoRA fine-tuning:**\n"
        "- Base model (frozen): `params × 2 bytes` (or 4-bit for QLoRA)\n"
        "- LoRA weights: `rank × hidden_dim × 2 × num_targets × 2 bytes` (~50-200 MB)\n"
        "- Gradients + optimizer: for LoRA weights only\n"
        "- Total: model size + ~2 GB\n\n"
        "**QLoRA:**\n"
        "- Base model in 4-bit: `params × 0.5 bytes`\n"
        "- 7B QLoRA → ~6 GB (fits on RTX 3060)\n"
        "- 70B QLoRA → ~40 GB (fits on A100 40GB)"
    ))
    pairs.append(msg(
        "What GPU should I buy for LLM inference in 2025-2026?",
        "GPU recommendations by budget:\n\n"
        "| Budget | GPU | VRAM | Best For |\n"
        "|--------|-----|------|----------|\n"
        "| $300 | RTX 3060 12GB | 12 GB | 7-8B models |\n"
        "| $500 | RTX 4060 Ti 16GB | 16 GB | 13-14B models |\n"
        "| $1000 | RTX 4080 16GB | 16 GB | 13B faster |\n"
        "| $1600 | RTX 4090 | 24 GB | 32B Q4_K_M |\n"
        "| $2000 | RTX 5090 | 32 GB | 32B comfortable |\n"
        "| $3500 | Mac Studio M3 Ultra | 192 GB | 70B+ models |\n"
        "| $8000 | 2x RTX 4090 | 48 GB | 70B Q4_K_M |\n"
        "| $15K+ | A100 80GB | 80 GB | 70B FP16 |\n\n"
        "Key considerations:\n"
        "- VRAM is king — it determines max model size\n"
        "- Memory bandwidth determines tok/s\n"
        "- Apple Silicon offers best VRAM/$ for large models\n"
        "- Multi-GPU adds complexity (NCCL, tensor parallelism)"
    ))

    pairs.append(msg(
        "How do I run a 70B model on consumer hardware?",
        "Options for running 70B models on consumer hardware:\n\n"
        "**1. Aggressive quantization (single GPU):**\n"
        "- Q3_K_M: ~30 GB → fits on RTX 3090/4090 24GB with tiny context\n"
        "- IQ2_XS: ~18 GB → fits with 4K context (significant quality loss)\n\n"
        "**2. Multi-GPU split:**\n"
        "- 2x RTX 4090 (48 GB): Q4_K_M (~42 GB) comfortable\n"
        "- llama.cpp: just point to the model, it auto-splits across GPUs\n"
        "- vLLM: `--tensor-parallel-size 2`\n\n"
        "**3. CPU + partial GPU offload:**\n"
        "- Need 64+ GB RAM\n"
        "- llama.cpp: `-ngl 20` offloads 20/80 layers to GPU\n"
        "- Slow (~2-5 tok/s) but works\n\n"
        "**4. Apple Silicon:**\n"
        "- M2 Max 64GB: Q4_K_M at ~12 tok/s\n"
        "- M3 Ultra 192GB: Q8_0 at ~15 tok/s\n\n"
        "**5. MoE models as alternative:**\n"
        "- Qwen3.5-35B-A3B: only 3B active params, ~7 GB Q4_K_M, 70B-comparable quality"
    ))

    return pairs


# ---------------------------------------------------------------------------
# 7. Benchmarking
# ---------------------------------------------------------------------------
def gen_benchmarking() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What metrics should I measure when benchmarking LLM inference?",
        "Key LLM inference metrics:\n\n"
        "**Latency metrics:**\n"
        "- **TTFT** (Time To First Token): time from request to first generated token\n"
        "- **TPOT** (Time Per Output Token): average time between consecutive tokens\n"
        "- **E2E latency**: total time for full response\n"
        "- **ITL** (Inter-Token Latency): time between each token pair\n\n"
        "**Throughput metrics:**\n"
        "- **tok/s** (tokens per second): generation speed per sequence\n"
        "- **req/s**: requests completed per second\n"
        "- **total tok/s**: aggregate across all concurrent sequences\n\n"
        "**Quality metrics:**\n"
        "- **Perplexity**: language model quality (lower = better)\n"
        "- **Accuracy**: task-specific (MMLU, HumanEval, etc.)\n\n"
        "**Resource metrics:**\n"
        "- VRAM usage (peak and steady-state)\n"
        "- GPU utilization %\n"
        "- Power consumption (W)\n"
        "- tok/s per watt (efficiency)"
    ))
    pairs.append(msg(
        "How do I benchmark llama.cpp performance?",
        "Built-in benchmarking:\n\n"
        "```bash\n# Token generation benchmark\n./build/bin/llama-bench \\\n"
        "    -m model-Q4_K_M.gguf \\\n    -p 512 \\\n    -n 128 \\\n    -ngl 99\n```\n\n"
        "This reports:\n"
        "- Prompt processing (prefill) speed: tok/s\n"
        "- Token generation (decode) speed: tok/s\n"
        "- Memory usage\n\n"
        "**Perplexity benchmark:**\n"
        "```bash\n./build/bin/llama-perplexity \\\n"
        "    -m model.gguf \\\n    -f wikitext-2-raw/wiki.test.raw \\\n"
        "    -c 2048 -ngl 99\n```\n\n"
        "**Server benchmark (concurrent load):**\n"
        "```bash\npython scripts/bench-server.py \\\n    --model model.gguf \\\n"
        "    --n-prompts 100 \\\n    --parallel 8 \\\n    --prompt-length 256 \\\n    --gen-length 128\n```\n\n"
        "Key: always benchmark with the same prompt length, generation length, and hardware for fair comparison."
    ))
    pairs.append(msg(
        "How do I benchmark vLLM throughput?",
        "Use the built-in benchmark scripts:\n\n"
        "```bash\n# Offline throughput benchmark\npython -m vllm.entrypoints.openai.run_batch \\\n"
        "    --model meta-llama/Llama-3.1-8B-Instruct \\\n    --input-file prompts.jsonl \\\n"
        "    --output-file results.jsonl\n\n"
        "# Online serving benchmark\npython benchmarks/benchmark_serving.py \\\n"
        "    --model meta-llama/Llama-3.1-8B-Instruct \\\n    --backend vllm \\\n"
        "    --num-prompts 1000 \\\n    --request-rate 10 \\\n"
        "    --dataset-name sharegpt\n```\n\n"
        "Metrics reported:\n"
        "- Median/P95/P99 TTFT\n"
        "- Median/P95/P99 TPOT\n"
        "- Total throughput (tok/s)\n"
        "- Request throughput (req/s)\n\n"
        "For fair comparison, control: request rate, prompt/generation lengths, number of concurrent users."
    ))
    pairs.append(msg(
        "How do I calculate tokens per second from inference metrics?",
        "Token speed calculations:\n\n"
        "**Prefill (prompt processing):**\n"
        "```\nprefill_tok/s = prompt_tokens / prefill_time_seconds\n```\n"
        "Prefill is compute-bound — speed scales linearly with batch size and GPU FLOPS.\n\n"
        "**Decode (generation):**\n"
        "```\ndecode_tok/s = generated_tokens / decode_time_seconds\n```\n"
        "Decode is memory-bandwidth-bound — speed is roughly:\n"
        "```\nmax_decode_tok/s ≈ memory_bandwidth_GB/s / model_size_GB\n```\n\n"
        "**From API response (llama.cpp):**\n"
        "```python\ndata = response.json()\nprefill_tok_s = data['prompt_eval_count'] / (data['prompt_eval_duration'] / 1e9)\n"
        "decode_tok_s = data['eval_count'] / (data['eval_duration'] / 1e9)\n```\n\n"
        "**From API response (Ollama):**\n"
        "Same fields: `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration` (nanoseconds)."
    ))
    pairs.append(msg(
        "What is TTFT and why does it matter?",
        "**TTFT (Time To First Token)** = time from sending the request to receiving the first generated token.\n\n"
        "It matters because:\n"
        "- Users perceive responsiveness based on TTFT, not total completion time\n"
        "- A 200ms TTFT feels instant; >2s feels sluggish\n"
        "- TTFT is dominated by prefill time (processing the input prompt)\n\n"
        "TTFT depends on:\n"
        "- Prompt length (longer prompt → more prefill computation)\n"
        "- GPU compute power (prefill is compute-bound)\n"
        "- Queue wait time (if server is busy)\n"
        "- Prefix caching (cached prefixes → near-zero prefill)\n\n"
        "Optimization strategies:\n"
        "1. Enable prefix caching for repeated prompts\n"
        "2. Enable chunked prefill (reduces blocking)\n"
        "3. Shorter system prompts\n"
        "4. Faster GPU (more FLOPS)\n"
        "5. Quantization (fewer bytes to compute)\n\n"
        "Typical values: 50-200ms (8B, GPU), 200-800ms (70B, GPU), 1-5s (70B, CPU)."
    ))
    pairs.append(msg(
        "How do I compare perplexity across different quantization levels?",
        "Systematic perplexity comparison:\n\n"
        "```bash\n# Download evaluation data\nwget https://huggingface.co/datasets/gg-hf/wikitext-2-raw/raw/main/wiki.test.raw\n\n"
        "# Run perplexity for each quantization\nfor q in Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M IQ2_XS; do\n"
        "    echo \"=== $q ===\"\n    ./build/bin/llama-perplexity \\\n"
        "        -m model-${q}.gguf \\\n        -f wiki.test.raw \\\n"
        "        -c 2048 -ngl 99 2>&1 | tail -1\ndone\n```\n\n"
        "Tips:\n"
        "- Always use the same evaluation data and context length\n"
        "- Use F16 as baseline reference\n"
        "- Perplexity difference < 0.1 is generally imperceptible in practice\n"
        "- Run with `--ppl-stride 512` for more accurate results (slower)\n"
        "- Also evaluate on your domain-specific data, not just WikiText"
    ))

    # Additional benchmarking
    pairs.append(msg(
        "How do I measure power efficiency for LLM inference?",
        "Power efficiency measurement:\n\n"
        "**NVIDIA GPU:**\n"
        "```bash\n# Continuous power monitoring\nnvidia-smi dmon -s p -d 1 -f power.csv\n\n"
        "# One-shot\nnvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits\n```\n\n"
        "**Apple Silicon:**\n"
        "```bash\nsudo powermetrics --samplers gpu_power,cpu_power -i 1000 -n 60\n```\n\n"
        "Calculate efficiency:\n"
        "```\ntok_per_watt = tokens_per_second / average_power_watts\n```\n\n"
        "Typical values (7B Q4_K_M):\n"
        "| Hardware | tok/s | Watts | tok/J |\n"
        "|----------|-------|-------|-------|\n"
        "| RTX 4090 | 120 | 350W | 0.34 |\n"
        "| M3 Max | 50 | 40W | 1.25 |\n"
        "| RTX 4060 Ti | 55 | 160W | 0.34 |\n"
        "| CPU (i9) | 12 | 125W | 0.10 |\n\n"
        "Apple Silicon is 3-4x more energy efficient than NVIDIA GPUs for inference."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 8. MLX Framework
# ---------------------------------------------------------------------------
def gen_mlx() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do I load and generate text with MLX LM?",
        "```python\nimport mlx_lm\n\n# Load model and tokenizer\nmodel, tokenizer = mlx_lm.load('mlx-community/Llama-3.1-8B-Instruct-4bit')\n\n"
        "# Generate text\nresponse = mlx_lm.generate(\n    model, tokenizer,\n    prompt='Explain quantum computing briefly.',\n"
        "    max_tokens=256,\n    temp=0.7\n)\nprint(response)\n```\n\n"
        "For chat format:\n```python\nmessages = [{'role': 'user', 'content': 'Hello!'}]\n"
        "prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)\n"
        "response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=256)\n```\n\n"
        "MLX models are downloaded from HuggingFace. The `mlx-community` namespace has pre-converted models."
    ))
    pairs.append(msg(
        "How do I convert a HuggingFace model to MLX format?",
        "```bash\n# Install mlx-lm\npip install mlx-lm\n\n"
        "# Convert to MLX (FP16)\npython -m mlx_lm.convert \\\n"
        "    --hf-path meta-llama/Llama-3.1-8B-Instruct \\\n    --mlx-path ./llama-3.1-8b-mlx\n\n"
        "# Convert and quantize to 4-bit\npython -m mlx_lm.convert \\\n"
        "    --hf-path meta-llama/Llama-3.1-8B-Instruct \\\n    --mlx-path ./llama-3.1-8b-mlx-4bit \\\n"
        "    -q --q-bits 4 --q-group-size 64\n```\n\n"
        "Quantization options:\n"
        "- `--q-bits 4`: 4-bit quantization (default, recommended)\n"
        "- `--q-bits 8`: 8-bit quantization\n"
        "- `--q-group-size 64`: quantization block size (32 or 64)\n\n"
        "The output directory contains `config.json`, `tokenizer.json`, and `*.safetensors` in MLX format."
    ))
    pairs.append(msg(
        "How do I run LoRA fine-tuning with MLX?",
        "```bash\n# LoRA training\npython -m mlx_lm lora \\\n"
        "    --model mlx-community/Llama-3.1-8B-Instruct-4bit \\\n"
        "    --train \\\n    --data ./training_data \\\n    --batch-size 2 \\\n"
        "    --lora-rank 16 \\\n    --lora-alpha 32 \\\n    --num-iters 1000 \\\n"
        "    --learning-rate 2e-5 \\\n    --adapter-path ./adapters\n```\n\n"
        "Training data format (`training_data/train.jsonl`):\n"
        "```json\n{\"messages\": [{\"role\": \"user\", \"content\": \"...\"}, {\"role\": \"assistant\", \"content\": \"...\"}]}\n```\n\n"
        "Key parameters:\n"
        "- `--lora-rank`: LoRA rank (4-64, higher = more capacity)\n"
        "- `--lora-alpha`: scaling factor (typically 2x rank)\n"
        "- `--batch-size`: limited by Metal memory\n"
        "- `--grad-checkpoint`: enable gradient checkpointing for large models\n\n"
        "Critical for Apple Silicon: `mx.metal.set_cache_limit(32 * 1024**3)` to avoid Metal OOM."
    ))
    pairs.append(msg(
        "How do I set Metal memory limits for MLX training?",
        "Apple Silicon Metal has memory limits that must be configured for large models:\n\n"
        "```python\nimport mlx.core as mx\n\n# Set maximum memory allocation (e.g., 460 GB for M3 Ultra 512GB)\n"
        "mx.set_memory_limit(460 * 1024**3)\n\n# Set cache limit (controls Metal buffer reuse)\n"
        "mx.set_cache_limit(32 * 1024**3)  # 32 GB cache\n```\n\n"
        "Or via environment variable:\n```bash\nMLX_METAL_MEMORY_BUDGET=460 python train.py\n```\n\n"
        "Guidelines:\n"
        "- Leave ~10% of unified memory for system (512GB → set 460GB)\n"
        "- `set_cache_limit` is CRITICAL — without it, Metal buffers grow unbounded and crash\n"
        "- Typical cache limits: 8-32 GB depending on model size\n"
        "- Monitor with: `sudo powermetrics --samplers gpu_power -i 1000`\n"
        "- If OOM: reduce batch_size, enable gradient checkpointing, or reduce sequence length"
    ))
    pairs.append(msg(
        "How do I serve an MLX model as an API?",
        "Use `mlx_lm.server` for an OpenAI-compatible server:\n\n"
        "```bash\npython -m mlx_lm.server \\\n    --model mlx-community/Llama-3.1-8B-Instruct-4bit \\\n"
        "    --host 0.0.0.0 \\\n    --port 8080\n```\n\n"
        "Then query with any OpenAI-compatible client:\n"
        "```python\nfrom openai import OpenAI\n\nclient = OpenAI(base_url='http://localhost:8080/v1', api_key='mlx')\n"
        "response = client.chat.completions.create(\n    model='llama-3.1-8b',\n"
        "    messages=[{'role': 'user', 'content': 'Hello!'}]\n)\n```\n\n"
        "For LoRA-adapted models:\n```bash\npython -m mlx_lm.server \\\n"
        "    --model ./base-model-mlx \\\n    --adapter-path ./adapters\n```\n\n"
        "The server supports streaming, temperature, top_p, max_tokens, and stop sequences."
    ))
    pairs.append(msg(
        "How do I fuse LoRA adapters into the base model in MLX?",
        "After training, fuse LoRA weights into the base model for faster inference:\n\n"
        "```bash\npython -m mlx_lm.fuse \\\n    --model mlx-community/Llama-3.1-8B-Instruct-4bit \\\n"
        "    --adapter-path ./adapters \\\n    --save-path ./fused-model \\\n"
        "    --de-quantize  # optional: fuse in FP16 for max quality\n```\n\n"
        "Or programmatically:\n```python\nimport mlx_lm\n\nmodel, tokenizer = mlx_lm.load(\n"
        "    'base-model-path',\n    adapter_path='./adapters'\n)\n"
        "# Model is already using adapter at inference time\n```\n\n"
        "Notes:\n"
        "- Fusing removes the LoRA overhead (~5-10% speed improvement)\n"
        "- `--de-quantize` produces FP16 model (larger but can be re-quantized)\n"
        "- Without fusing, adapters are applied dynamically at each forward pass"
    ))
    pairs.append(msg(
        "How does MLX compare to PyTorch for LLM work on Apple Silicon?",
        "MLX vs PyTorch on Apple Silicon:\n\n"
        "| Aspect | MLX | PyTorch (MPS) |\n"
        "|--------|-----|---------------|\n"
        "| Memory | Unified, lazy eval | Copies to MPS device |\n"
        "| Speed | Native Metal, optimized | MPS backend, less optimized |\n"
        "| MoE support | Works correctly | MPS fails with MoE layers |\n"
        "| Quantization | Built-in 4/8-bit | BitsAndBytes not supported |\n"
        "| LoRA training | mlx_lm.lora | PEFT works but slower |\n"
        "| Ecosystem | Growing, Apple-backed | Mature, huge community |\n"
        "| Multi-GPU | Not supported | Not on MPS |\n\n"
        "**Recommendation**: Always use MLX on Apple Silicon for LLM work. "
        "PyTorch MPS has known issues with mixed-precision, MoE, and memory management. "
        "MLX's lazy evaluation and unified memory model are designed for Apple hardware."
    ))
    pairs.append(msg(
        "How do I quantize a model with MLX?",
        "```bash\n# Quantize from HuggingFace model\npython -m mlx_lm.convert \\\n"
        "    --hf-path meta-llama/Llama-3.1-8B-Instruct \\\n    --mlx-path ./model-4bit \\\n"
        "    -q --q-bits 4 --q-group-size 64\n\n"
        "# Quantize from existing MLX model\npython -m mlx_lm.convert \\\n"
        "    --hf-path ./model-fp16-mlx \\\n    --mlx-path ./model-4bit \\\n"
        "    -q --q-bits 4\n```\n\n"
        "Quantization options:\n"
        "| Bits | Memory (7B) | Quality | Use Case |\n"
        "|------|-------------|---------|----------|\n"
        "| 4 | ~4 GB | Good | Default, inference |\n"
        "| 8 | ~8 GB | Very good | Quality-sensitive tasks |\n"
        "| FP16 | ~14 GB | Reference | Training, fusing |\n\n"
        "MLX quantization is per-group with configurable group size (32 or 64). "
        "4-bit with group_size=64 is equivalent to Q4_0 in GGUF quality."
    ))

    # More MLX
    pairs.append(msg(
        "How do I use MLX for embeddings?",
        "```python\nimport mlx.core as mx\nimport mlx_lm\n\nmodel, tokenizer = mlx_lm.load('mlx-community/nomic-embed-text-v1.5-4bit')\n\n"
        "# Encode text\ntexts = ['Hello world', 'Machine learning is great']\n"
        "tokens = [tokenizer.encode(t) for t in texts]\n\n"
        "# Pad to same length\nmax_len = max(len(t) for t in tokens)\n"
        "padded = [t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in tokens]\n\n"
        "# Get embeddings (last hidden state, mean pool)\ninput_ids = mx.array(padded)\n"
        "outputs = model(input_ids)\nembeddings = outputs.mean(axis=1)  # mean pooling\n"
        "embeddings = embeddings / mx.linalg.norm(embeddings, axis=1, keepdims=True)  # L2 normalize\n```\n\n"
        "For production embedding serving, consider using llama.cpp with `--embedding` mode instead."
    ))
    pairs.append(msg(
        "How do I profile MLX model performance?",
        "```python\nimport mlx.core as mx\nimport time\n\n# Warmup\nfor _ in range(3):\n"
        "    response = mlx_lm.generate(model, tokenizer, prompt='Hello', max_tokens=10)\n    mx.eval()\n\n"
        "# Benchmark\ntimes = []\nfor _ in range(10):\n    start = time.perf_counter()\n"
        "    response = mlx_lm.generate(model, tokenizer, prompt='Write a poem about AI.', max_tokens=100)\n"
        "    mx.eval()  # Force evaluation (MLX is lazy)\n"
        "    times.append(time.perf_counter() - start)\n\n"
        "avg_time = sum(times) / len(times)\n"
        "print(f'Avg time: {avg_time:.2f}s')\nprint(f'Tokens/sec: {100 / avg_time:.1f}')\n```\n\n"
        "Important: `mx.eval()` is required because MLX uses lazy evaluation — "
        "computations are only executed when results are explicitly needed. "
        "Without it, you're only measuring graph construction time."
    ))

    # MLX training deep dive
    pairs.append(msg(
        "What is gradient checkpointing in MLX and when should I use it?",
        "Gradient checkpointing trades compute for memory during training:\n\n"
        "**Without checkpointing**: all intermediate activations are stored for the backward pass.\n"
        "For a 35B model, this can use 50-100 GB of memory.\n\n"
        "**With checkpointing**: intermediate activations are discarded and recomputed during backprop.\n"
        "Memory drops to O(sqrt(N)) but training is ~30% slower.\n\n"
        "Enable in MLX:\n```bash\npython -m mlx_lm lora --train --grad-checkpoint ...\n```\n\n"
        "Or programmatically:\n```python\nmodel.apply(lambda m: m.set_grad_checkpoint(True) if hasattr(m, 'set_grad_checkpoint') else None)\n```\n\n"
        "Use gradient checkpointing when:\n"
        "- Training large models (>7B) on limited memory\n"
        "- Batch size > 1 causes OOM\n"
        "- You see Metal memory pressure warnings\n\n"
        "For micro-kiki 35B-A3B training on M3 Ultra 512GB: gradient checkpointing is REQUIRED."
    ))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
GENERATORS = {
    "ollama": ("Ollama CLI/API/Modelfile", gen_ollama),
    "vllm": ("vLLM server/config", gen_vllm),
    "llamacpp": ("llama.cpp build/serve/quantize", gen_llamacpp),
    "gguf": ("GGUF quantization types", gen_gguf_quant),
    "serving": ("Model serving patterns", gen_serving),
    "hardware": ("Hardware requirements", gen_hardware),
    "benchmarking": ("Benchmarking methods", gen_benchmarking),
    "mlx": ("MLX framework", gen_mlx),
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate LLM Ops training dataset")
    parser.add_argument("--stats", action="store_true", help="Print stats only, no JSONL")
    parser.add_argument("--categories", default=",".join(GENERATORS.keys()),
                        help="Comma-separated categories to generate")
    args = parser.parse_args()

    requested = [c.strip() for c in args.categories.split(",")]
    for cat in requested:
        if cat not in GENERATORS:
            logger.error("Unknown category: %s. Available: %s", cat, ", ".join(GENERATORS.keys()))
            sys.exit(1)

    total = 0
    stats: dict[str, int] = {}

    for cat in requested:
        label, gen_fn = GENERATORS[cat]
        pairs = gen_fn()
        stats[label] = len(pairs)
        total += len(pairs)

        if not args.stats:
            for pair in pairs:
                print(json.dumps(pair, ensure_ascii=False))

    logger.info("=== LLM Ops Generation Statistics ===")
    for label, count in stats.items():
        logger.info("  %-35s %5d pairs", label, count)
    logger.info("  %-35s %5d pairs", "TOTAL", total)


if __name__ == "__main__":
    main()

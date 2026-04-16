"""Neuron probing for bias detection (KnowBias, arxiv 2601.21864)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BiasNeuron:
    layer_idx: int
    neuron_idx: int
    delta: float


def probe_bias_neurons(model, tokenizer, bias_pairs: list[dict], top_n: int = 200) -> list[BiasNeuron]:
    """Probe model activations to find neurons most sensitive to bias.

    Requires torch. Runs hooks on model layers.
    """
    import torch

    activations: dict[int, list] = {}

    def hook_fn(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations.setdefault(layer_idx, []).append(output.detach().cpu())
        return hook

    hooks = []
    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(hook_fn(i))
        hooks.append(h)

    deltas: list[BiasNeuron] = []

    for pair in bias_pairs:
        activations.clear()

        tokens_biased = tokenizer(pair["biased_prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**tokens_biased)
        act_biased = {k: v[0].mean(dim=(0, 1)) for k, v in activations.items()}

        activations.clear()

        tokens_fair = tokenizer(pair["fair_prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**tokens_fair)
        act_fair = {k: v[0].mean(dim=(0, 1)) for k, v in activations.items()}

        for layer_idx in act_biased:
            diff = (act_biased[layer_idx] - act_fair[layer_idx]).abs()
            top_neurons = diff.topk(min(5, diff.shape[0]))
            for ni, d in zip(top_neurons.indices.tolist(), top_neurons.values.tolist()):
                deltas.append(BiasNeuron(layer_idx=layer_idx, neuron_idx=ni, delta=d))

    for h in hooks:
        h.remove()

    deltas.sort(key=lambda x: x.delta, reverse=True)
    return deltas[:top_n]


def save_bias_neurons(neurons: list[BiasNeuron], output_path: str = "results/bias_neurons_base.json") -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"layer": n.layer_idx, "neuron": n.neuron_idx, "delta": n.delta} for n in neurons]
    path.write_text(json.dumps(data, indent=2))
    return path

"""MLX port of the BSK spiking LM and its dense twin (NH-81).

Architecture of hypneum-lab/bsk ``training/remote_train_lm.py`` (commit
8cc7087), arms ``ann1`` (dense) and ``spkt`` (spiking):

* ANN causal multi-head attention, LayerNorm, learned positions, untied
  head; D=384, L=6, 6 heads, SEQ=128 (BSK defaults);
* FFN ``down(f(silu(gate(h)) * up(h)))`` with width 4D; ``f = identity``
  for ``ann1``; for ``spkt`` ``f`` is the signed rate-LIF over ``T`` ticks
  (binary spike at threshold 0.5, reset by 0.5, global scale
  ``s = mean|d|``, straight-through surrogate sigma(4x)).

Two exact-equivalent inference paths exist for the spiking FFN:

* ``fused`` (BSK reference): ``down`` is applied once to the spike rate;
* ``event``: ``down`` is applied to the signed spike vector of every tick
  and the T results are averaged (the per-event formulation of an SNN
  executed with dense kernels).

The same module serves training and energy measurement, so no weight
conversion sits between the trained model and the measured one.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def spike_ste(x: mx.array) -> mx.array:
    """Heaviside(x) forward, d/dx sigma(4x) backward (BSK surrogate)."""
    soft = mx.sigmoid(4.0 * x)
    return soft + mx.stop_gradient((x > 0).astype(x.dtype) - soft)


def lif_emissions(x: mx.array, ticks: int, train: bool) -> list[mx.array]:
    """Per-tick emissions (0/1) of one rate-LIF population driven by ``x``."""
    v = mx.zeros_like(x)
    out = []
    for _ in range(ticks):
        v = v + x
        a = spike_ste(v - 0.5) if train else (v > 0.5).astype(x.dtype)
        out.append(a)
        v = v - a * 0.5
    return out


class Block(nn.Module):
    def __init__(self, d: int, nhead: int, arm: str, ticks: int) -> None:
        super().__init__()
        self.nhead, self.arm, self.ticks = nhead, arm, ticks
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.gate = nn.Linear(d, 4 * d)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def attention(self, h: mx.array) -> mx.array:
        b, n, d = h.shape
        q, k, v = mx.split(self.qkv(h), 3, axis=-1)
        sh = lambda t: t.reshape(b, n, self.nhead, d // self.nhead).transpose(0, 2, 1, 3)  # noqa: E731
        o = mx.fast.scaled_dot_product_attention(
            sh(q), sh(k), sh(v), scale=(d // self.nhead) ** -0.5, mask="causal")
        return self.proj(o.transpose(0, 2, 1, 3).reshape(b, n, d))

    def ffn(self, h: mx.array, mode: str, train: bool, stats: list | None) -> mx.array:
        d = nn.silu(self.gate(h)) * self.up(h)
        if self.arm == "ann1":
            return self.down(d)
        s = mx.abs(d).mean() + 1e-6
        ep = lif_emissions(mx.maximum(d, 0) / s, self.ticks, train)
        en = lif_emissions(mx.maximum(-d, 0) / s, self.ticks, train)
        if stats is not None:
            stats.append(sum(p + n for p, n in zip(ep, en)).mean() / self.ticks)
        if mode == "event":
            acc = None
            for p, n in zip(ep, en):
                y = (p - n) @ self.down.weight.T
                acc = y if acc is None else acc + y
            return acc * (s / self.ticks) + self.down.bias
        rate = (sum(ep) - sum(en)) / self.ticks
        return self.down(rate * s)

    def __call__(self, x, mode="fused", train=False, stats=None):
        x = x + self.attention(self.n1(x))
        return x + self.ffn(self.n2(x), mode, train, stats)


class LM(nn.Module):
    def __init__(self, vocab: int = 16000, d: int = 384, n_layer: int = 6, seq: int = 128,
                 arm: str = "ann1", ticks: int = 6) -> None:
        super().__init__()
        assert arm in ("ann1", "spkt"), arm
        self.cfg = {"vocab": vocab, "D": d, "L": n_layer, "SEQ": seq, "arm": arm, "T": ticks}
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = [Block(d, max(1, d // 64), arm, ticks) for _ in range(n_layer)]
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def __call__(self, idx: mx.array, mode: str = "fused", train: bool = False,
                 stats: list | None = None) -> mx.array:
        x = self.emb(idx) + self.pos(mx.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x, mode, train, stats)
        return self.head(self.norm(x))


def n_params(model: LM) -> int:
    from mlx.utils import tree_flatten

    return sum(v.size for _, v in tree_flatten(model.parameters()))

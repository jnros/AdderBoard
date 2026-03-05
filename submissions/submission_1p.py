"""
1-parameter Qwen-style adder (PyTorch).

Single learnable parameter:
  c  (1) embedding scale

Everything else is derived or hardcoded, including carry threshold:
  g       = alpha * c
  v       = -22·c/√2
  norm[0] = 0.1·c/√2
  norm[1] = -c/(50·√2)
  gate[1] = 128·c
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIM = 2
HEAD_DIM = 2
INTERMEDIATE_SIZE = 2
VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1

CONST_NORM = math.sqrt(MODEL_DIM)

ROPE_PERIOD = 19.0
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PEAK_EPS = 0.3
PHI = OMEGA * (10.0 + PEAK_EPS)

TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS))
)
QK_NORM_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))

V_FACTOR = -22.0 / CONST_NORM
S_CONST = 100.0 / 256.0
ALPHA = -12.032


def _unit_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def _apply_rope(x: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[2]
    pos = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    theta = pos * OMEGA
    cos_t = torch.cos(theta).view(1, 1, -1, 1)
    sin_t = torch.sin(theta).view(1, 1, -1, 1)
    x0, x1 = x[..., 0:1], x[..., 1:2]
    return torch.cat([x0 * cos_t - x1 * sin_t, x0 * sin_t + x1 * cos_t], dim=-1)


class AdderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.zeros(1))
        self.attn_scale = (HEAD_DIM ** -0.5) * QK_NORM_SCALE ** 2

    def _embed_table(self):
        d = torch.arange(VOCAB_SIZE, device=self.c.device, dtype=torch.float32)
        c = self.c[0]
        return torch.stack([c - (d * d) / c, -d], dim=-1)

    def _v_weight(self):
        return V_FACTOR * self.c[0]

    def _norm_weight(self):
        c = self.c[0]
        return torch.stack([0.1 * c / CONST_NORM, -c / (50.0 * CONST_NORM)])

    def _gate_weight(self):
        c = self.c[0]
        return torch.stack([ALPHA * c, 128.0 * c])

    def _q_proj(self, x):
        return torch.stack([x[..., 0] * math.cos(PHI), x[..., 0] * (-math.sin(PHI))], dim=-1)

    def _k_proj(self, x):
        return torch.stack([x[..., 0], torch.zeros_like(x[..., 0])], dim=-1)

    def _v_proj(self, x):
        return torch.stack([x[..., 1] * self._v_weight(), torch.zeros_like(x[..., 0])], dim=-1)

    def _o_proj(self, x):
        return torch.stack([torch.zeros_like(x[..., 0]), x[..., 0]], dim=-1)

    def _attention(self, x, mask):
        bsz, seq_len, _ = x.shape
        q = self._q_proj(x).reshape(bsz, seq_len, 1, HEAD_DIM).transpose(1, 2)
        k = self._k_proj(x).reshape(bsz, seq_len, 1, HEAD_DIM).transpose(1, 2)
        v = self._v_proj(x).reshape(bsz, seq_len, 1, HEAD_DIM).transpose(1, 2)

        q = _apply_rope(_unit_rms_norm(q))
        k = _apply_rope(_unit_rms_norm(k))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self._o_proj(out.transpose(1, 2).reshape(bsz, seq_len, -1))

    def _mlp(self, x):
        gw = self._gate_weight()
        a, gc = gw[0], gw[1]
        g0 = x[..., 0] * a + x[..., 1] * gc
        g1 = x[..., 0] * (a - gc / self.c[0]) + x[..., 1] * gc
        gate = torch.stack([g0, g1], dim=-1)

        base = x[..., 0]
        up = base.unsqueeze(-1).expand(*base.shape, INTERMEDIATE_SIZE)
        mix = F.silu(gate) * up
        y1 = S_CONST * (mix[..., 1] - mix[..., 0])
        return torch.stack([torch.zeros_like(y1), y1], dim=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tab = self._embed_table()
        h = tab[tokens]
        seq_len = h.shape[1]
        mask = torch.triu(
            torch.full((seq_len, seq_len), -1e9, device=h.device, dtype=h.dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        hn = _unit_rms_norm(h)
        h = h + self._attention(hn, mask)

        hn = _unit_rms_norm(h)
        h = h + self._mlp(hn)

        nw = self._norm_weight()
        out = _unit_rms_norm(h) * nw
        return out @ tab.T


def _init_weights(model: AdderModel) -> None:
    with torch.no_grad():
        model.c.copy_(torch.tensor([1000.0]))


def _encode_prompt(a: int, b: int) -> list[int]:
    ad = [int(ch) for ch in f"{a:010d}"][::-1]
    bd = [int(ch) for ch in f"{b:010d}"][::-1]
    return [0] + ad + [0] * 9 + bd + [0]


@torch.no_grad()
def generate(model: AdderModel, a: int, b: int) -> str:
    model.eval()
    dev = next(model.parameters()).device
    seq = _encode_prompt(a, b)
    for _ in range(OUTPUT_DIGITS):
        x = torch.tensor([seq], dtype=torch.long, device=dev)
        logits = model(x)
        seq.append(int(logits[0, -1].argmax().item()))
    return "".join(str(d) for d in seq[-OUTPUT_DIGITS:])


def add(model, a: int, b: int) -> int:
    if not (isinstance(a, int) and isinstance(b, int)):
        raise ValueError("a and b must be ints")
    if not (0 <= a <= MAX_ADDEND and 0 <= b <= MAX_ADDEND):
        raise ValueError(f"a and b must be in [0, {MAX_ADDEND}]")
    return int(generate(model, a, b)[::-1])


def build_model():
    model = AdderModel()
    _init_weights(model)
    metadata = {
        "name": "adder_1p",
        "author": "jnros",
        "params": sum(p.numel() for p in model.parameters()),
        "architecture": "1 parameter (g tied to c)",
        "tricks": [
            "RoPE period-19 positional encoding (hardcoded)",
            "tied embedding (single scalar c -> full vocab table)",
            "gate threshold tied to embedding scale: g = alpha*c",
            "derived norm and value strengths from c",
            "hardcoded Q angle phi (positional constant)",
            "hardcoded carry pathway amplitude",
        ],
    }
    return model, metadata

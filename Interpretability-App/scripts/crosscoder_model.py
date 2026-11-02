"""
crosscoder_model.py

Inference-only CrossCoder v5 definition shared across all pipeline stages
that load a trained checkpoint (steps 03 and 04).

The training script (02_train_crosscoder/final_scripts/train_crosscoder_v5.py)
keeps its own full implementation because it requires BatchTopK, weight init,
and training-specific forward logic.
"""

import torch
import torch.nn as nn


class CrossCoderV5(nn.Module):
    """Inference-only CrossCoder v5 — separate W_enc_A / W_enc_B, BatchTopK threshold as buffer."""

    def __init__(self, d_model: int, n_features: int, k: int):
        super().__init__()
        self.d_model    = d_model
        self.n_features = n_features
        self.k          = k

        self.W_enc_A = nn.Parameter(torch.empty(d_model, n_features))
        self.W_enc_B = nn.Parameter(torch.empty(d_model, n_features))
        self.b_enc   = nn.Parameter(torch.zeros(n_features))
        self.W_dec_A = nn.Parameter(torch.empty(n_features, d_model))
        self.W_dec_B = nn.Parameter(torch.empty(n_features, d_model))
        self.b_dec_A = nn.Parameter(torch.zeros(d_model))
        self.b_dec_B = nn.Parameter(torch.zeros(d_model))

        # Populated from checkpoint via state_dict load
        self.register_buffer("rms_A",     torch.tensor(1.0))
        self.register_buffer("rms_B",     torch.tensor(1.0))
        self.register_buffer("threshold", torch.tensor(1.0))

    def _code_norm_scale(self) -> torch.Tensor:
        n_A = self.W_dec_A.norm(dim=1)
        n_B = self.W_dec_B.norm(dim=1)
        return n_A + n_B  # (n_features,)

    def encode(self, x_A: torch.Tensor, x_B: torch.Tensor) -> torch.Tensor:
        x_A = x_A / (self.rms_A + 1e-8)
        x_B = x_B / (self.rms_B + 1e-8)
        pre_acts = x_A @ self.W_enc_A + x_B @ self.W_enc_B + self.b_enc
        post_relu = torch.relu(pre_acts)
        # Use threshold-based selection (consistent with BatchTopK training)
        # Falls back to per-token topk if threshold was never set (legacy checkpoints)
        if self.threshold.item() > 0:
            code_scale = self._code_norm_scale()
            mask = (post_relu * code_scale) > self.threshold
            out = post_relu * mask
        else:
            topk_vals, topk_idx = post_relu.topk(self.k, dim=-1)
            out = torch.zeros_like(pre_acts)
            out.scatter_(-1, topk_idx, topk_vals)
        return out

    def decode(self, f: torch.Tensor):
        x_hat_A = f @ self.W_dec_A + self.b_dec_A
        x_hat_B = f @ self.W_dec_B + self.b_dec_B
        x_hat_A = x_hat_A * (self.rms_A + 1e-8)
        x_hat_B = x_hat_B * (self.rms_B + 1e-8)
        return x_hat_A, x_hat_B

    def forward(self, x_A: torch.Tensor, x_B: torch.Tensor):
        f = self.encode(x_A, x_B)
        x_hat_A, x_hat_B = self.decode(f)
        return f, x_hat_A, x_hat_B

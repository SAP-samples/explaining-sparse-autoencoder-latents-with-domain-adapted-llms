"""
train_crosscoder_v5.py

BatchTopK CrossCoder — closest to jkminder/Lindsey et al. 2025 paper implementation.

Key changes vs v4 (TopK):
  - BatchTopK sparsity: top-k*batch_size features across the whole batch
    Threshold stored and used at inference time
  - target_rms normalization: inputs scaled to RMS=1 (NormalizableMixin in jkminder)
    Encoder sees normalized inputs; decoder output denormalized for loss
    This is distinct from Gao unit-norm: just RMS scaling, no mean subtraction
  - Decoder initialized as transpose of encoder (init_with_transpose=True)
  - norm_init_scale=1.0 applied after init
  - No b_pre_A/b_pre_B per-model pre-bias (matches jkminder encoder)
  - Separate W_enc_A/W_enc_B remains (key for Δnorm)

Ref: Lindsey et al. 2025 arxiv:2504.02922
     https://github.com/jkminder/dictionary_learning
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

LA_TZ = timezone(timedelta(hours=-7))
DEAD_THRESHOLD = 1e-4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairedActivationsDataset(Dataset):
    def __init__(self, meta_base_path: str, meta_swallow_path: str):
        meta_base    = torch.load(meta_base_path,    weights_only=False)
        meta_swallow = torch.load(meta_swallow_path, weights_only=False)
        n_base, d_base       = meta_base["shape"]
        n_swallow, d_swallow = meta_swallow["shape"]
        if n_base != n_swallow:
            raise ValueError(f"Token count mismatch: base={n_base}, swallow={n_swallow}")
        if d_base != d_swallow:
            raise ValueError(f"d_model mismatch: base={d_base}, swallow={d_swallow}")
        self.n_tokens = n_base
        self.d_model  = d_base
        self.mmap_base    = np.memmap(meta_base["mmap_path"],    dtype="float32", mode="r", shape=(n_base,    d_base))
        self.mmap_swallow = np.memmap(meta_swallow["mmap_path"], dtype="float32", mode="r", shape=(n_swallow, d_swallow))
        print(f"  base mmap:    {meta_base['mmap_path']}  ({n_base:,} x {d_base})")
        print(f"  swallow mmap: {meta_swallow['mmap_path']}  ({n_swallow:,} x {d_swallow})")

    def __len__(self):  return self.n_tokens
    def __getitem__(self, idx):
        return (torch.from_numpy(self.mmap_base[idx].copy()),
                torch.from_numpy(self.mmap_swallow[idx].copy()))


# ---------------------------------------------------------------------------
# RMS normalization helper
# ---------------------------------------------------------------------------

def compute_rms_scale(dataset: PairedActivationsDataset, n_samples: int = 65536, device="cpu"):
    """Compute mean RMS of activations for target_rms normalization."""
    indices = torch.randperm(len(dataset))[:n_samples]
    rms_A_sum = rms_B_sum = 0.0
    n = 0
    for i in indices:
        xA, xB = dataset[i]
        rms_A_sum += xA.norm().item()
        rms_B_sum += xB.norm().item()
        n += 1
    d = dataset.d_model
    rms_A = rms_A_sum / (n * math.sqrt(d))  # normalize by sqrt(d) to get per-dim RMS
    rms_B = rms_B_sum / (n * math.sqrt(d))
    return rms_A, rms_B


# ---------------------------------------------------------------------------
# CrossCoder v5 — BatchTopK + target_rms normalization
# ---------------------------------------------------------------------------

class CrossCoderV5(nn.Module):
    """
    BatchTopK CrossCoder matching jkminder architecture.

    Architecture changes vs v4:
    - BatchTopK: topk selection on norm-scaled activations, threshold stored for inference
    - target_rms: inputs divided by their mean RMS before encoding
    - No b_pre_A/B (no per-model pre-bias; encoder bias b_enc is shared)
    - Decoder initialized as transpose of encoder
    """
    def __init__(self, d_model: int, n_features: int, k: int, aux_k: int,
                 rms_A: float = 1.0, rms_B: float = 1.0, norm_init_scale: float = 1.0):
        super().__init__()
        self.d_model    = d_model
        self.n_features = n_features
        self.k          = k
        self.aux_k      = aux_k

        # RMS scale factors for input normalization (fixed, not learned)
        self.register_buffer("rms_A", torch.tensor(rms_A, dtype=torch.float32))
        self.register_buffer("rms_B", torch.tensor(rms_B, dtype=torch.float32))

        # Shared encoder bias
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        # Separate encoder weight matrices per model — key for Δnorm
        self.W_enc_A = nn.Parameter(torch.empty(d_model, n_features))
        self.W_enc_B = nn.Parameter(torch.empty(d_model, n_features))

        # Decoders
        self.W_dec_A = nn.Parameter(torch.empty(n_features, d_model))
        self.W_dec_B = nn.Parameter(torch.empty(n_features, d_model))
        self.b_dec_A = nn.Parameter(torch.zeros(d_model))
        self.b_dec_B = nn.Parameter(torch.zeros(d_model))

        # BatchTopK threshold (updated during training, used for inference)
        self.register_buffer("threshold", torch.tensor(-1.0))

        self._init_weights(norm_init_scale)

    def _init_weights(self, norm_init_scale: float):
        # Random kaiming init for encoder A
        nn.init.kaiming_uniform_(self.W_enc_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_enc_B, a=math.sqrt(5))
        # Decoder initialized as transpose of average encoder
        W_dec_init = (self.W_enc_A.data + self.W_enc_B.data).T.clone() / 2
        self.W_dec_A.data = W_dec_init.clone()
        self.W_dec_B.data = W_dec_init.clone()
        if norm_init_scale != 1.0:
            with torch.no_grad():
                for p in [self.W_enc_A, self.W_enc_B, self.W_dec_A, self.W_dec_B]:
                    p.data *= norm_init_scale

    def _normalize_inputs(self, x_A, x_B):
        return x_A / (self.rms_A + 1e-8), x_B / (self.rms_B + 1e-8)

    def _code_norm_scale(self) -> torch.Tensor:
        """CodeNorm: sum of per-feature decoder norms across both models."""
        n_A = self.W_dec_A.norm(dim=1)
        n_B = self.W_dec_B.norm(dim=1)
        return n_A + n_B  # (n_features,)

    def encode_pre_relu(self, x_A_norm, x_B_norm) -> torch.Tensor:
        return x_A_norm @ self.W_enc_A + x_B_norm @ self.W_enc_B + self.b_enc

    def _batch_topk(self, pre_acts: torch.Tensor, code_scale: torch.Tensor) -> torch.Tensor:
        """
        Select top k*batch_size features across the batch, ranked by code-norm-scaled activations.
        Returns unscaled features (raw ReLU activations for selected positions).
        """
        batch_size = pre_acts.size(0)
        post_relu = torch.relu(pre_acts)

        # Scale by code norms for ranking (like jkminder)
        post_relu_scaled = post_relu * code_scale.unsqueeze(0)

        # Flatten, find top-k indices
        flat_scaled = post_relu_scaled.flatten()
        topk_vals, topk_indices = flat_scaled.topk(self.k * batch_size, sorted=False)

        # Store threshold for inference
        self.threshold.fill_(topk_vals.min().detach())

        # Return UNSCALED activations at selected positions
        flat_unscaled = post_relu.flatten()
        out = torch.zeros_like(flat_unscaled)
        out.scatter_(0, topk_indices, flat_unscaled[topk_indices])
        return out.reshape(post_relu.shape)

    def _threshold_topk(self, pre_acts: torch.Tensor, code_scale: torch.Tensor) -> torch.Tensor:
        """Use stored threshold for inference."""
        post_relu = torch.relu(pre_acts)
        post_relu_scaled = post_relu * code_scale.unsqueeze(0)
        mask = post_relu_scaled > self.threshold
        return post_relu * mask

    def encode(self, x_A: torch.Tensor, x_B: torch.Tensor,
               use_threshold: bool = False, normalize: bool = True) -> torch.Tensor:
        if normalize:
            x_A, x_B = self._normalize_inputs(x_A, x_B)
        pre_acts = self.encode_pre_relu(x_A, x_B)
        code_scale = self._code_norm_scale().detach()
        if use_threshold:
            return self._threshold_topk(pre_acts, code_scale)
        else:
            return self._batch_topk(pre_acts, code_scale)

    def decode(self, features: torch.Tensor):
        return (features @ self.W_dec_A + self.b_dec_A,
                features @ self.W_dec_B + self.b_dec_B)

    def forward(self, x_A, x_B, use_threshold: bool = False):
        x_A_norm, x_B_norm = self._normalize_inputs(x_A, x_B)
        f = self.encode(x_A_norm, x_B_norm, use_threshold=use_threshold, normalize=False)
        x_hat_A_norm, x_hat_B_norm = self.decode(f)
        # Denormalize
        x_hat_A = x_hat_A_norm * (self.rms_A + 1e-8)
        x_hat_B = x_hat_B_norm * (self.rms_B + 1e-8)
        return f, x_hat_A, x_hat_B

    def aux_loss(self, x_A: torch.Tensor, x_B: torch.Tensor,
                 dead_mask: torch.Tensor) -> torch.Tensor:
        x_A_norm, x_B_norm = self._normalize_inputs(x_A, x_B)
        with torch.no_grad():
            f_main, x_hat_A, x_hat_B = self.forward(x_A, x_B, use_threshold=True)
        err_A = x_A - x_hat_A
        err_B = x_B - x_hat_B

        pre_acts = self.encode_pre_relu(x_A_norm, x_B_norm)
        dead_pre = pre_acts * dead_mask.unsqueeze(0).float()
        n_dead   = dead_mask.sum().item()
        k_aux    = min(self.aux_k, max(1, int(n_dead)))
        topk_dead_vals, topk_dead_idx = torch.relu(dead_pre).topk(k_aux, dim=-1)
        f_dead = torch.zeros_like(dead_pre).scatter_(-1, topk_dead_idx, topk_dead_vals)

        err_hat_A_norm, err_hat_B_norm = self.decode(f_dead)
        err_hat_A = err_hat_A_norm * (self.rms_A + 1e-8)
        err_hat_B = err_hat_B_norm * (self.rms_B + 1e-8)
        return ((err_A - err_hat_A) ** 2).mean() + ((err_B - err_hat_B) ** 2).mean()

    def delta_norm(self) -> torch.Tensor:
        n_A = self.W_dec_A.norm(dim=1)
        n_B = self.W_dec_B.norm(dim=1)
        return (n_A - n_B) / (n_A + n_B + 1e-8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def variance_explained(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    residual = x - x_hat
    var_x = x.var().item()
    if var_x < 1e-10:
        return 0.0
    return 1.0 - residual.var().item() / var_x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nBuilding dataset ...")
    dataset  = PairedActivationsDataset(args.acts_base, args.acts_swallow)
    d_model  = dataset.d_model
    n_tokens = len(dataset)
    print(f"  Tokens:  {n_tokens:,}")
    print(f"  d_model: {d_model}")

    print("\nComputing RMS scale for input normalization ...")
    rms_A, rms_B = compute_rms_scale(dataset, n_samples=min(65536, n_tokens))
    print(f"  rms_A={rms_A:.4f}  rms_B={rms_B:.4f}")

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=4, pin_memory=True)

    print(f"\nBuilding CrossCoderV5 (d={d_model}, f={args.n_features}, k={args.topk}) ...")
    model = CrossCoderV5(
        d_model, args.n_features, args.topk, args.aux_k,
        rms_A=rms_A, rms_B=rms_B, norm_init_scale=args.norm_init_scale
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  W_enc_A: {model.W_enc_A.shape}  W_enc_B: {model.W_enc_B.shape}")
    print(f"  rms_A buffer: {model.rms_A.item():.4f}  rms_B buffer: {model.rms_B.item():.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def lr_lambda(step):
        if step < args.warmup:
            return 0.1 + 0.9 * step / args.warmup
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    date_str  = datetime.now(LA_TZ).strftime("%Y%m%d_%H%M")
    run_name  = f"cc_v5_btopk_f{args.n_features}_k{args.topk}_lr{args.lr:.0e}_{date_str}"
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path  = out_dir / f"log_{run_name}.jsonl"
    ckpt_path = out_dir / f"{run_name}.pt"

    print(f"\nTraining {args.epochs} epochs, {len(loader)} steps/epoch ...")
    print(f"  Checkpoint: {ckpt_path.name}")

    fire_ema    = torch.zeros(args.n_features, device=device)
    ema_decay   = 0.99
    global_step = 0
    t_start     = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_loss = ep_mse_A = ep_mse_B = ep_aux = ep_ve_A = ep_ve_B = 0.0
        n_batches = 0

        for x_A, x_B in loader:
            x_A = x_A.to(device)
            x_B = x_B.to(device)

            features, x_hat_A, x_hat_B = model(x_A, x_B)

            mse_A = ((x_A - x_hat_A) ** 2).mean()
            mse_B = ((x_B - x_hat_B) ** 2).mean()

            # CodeNorm scaling (jkminder 2025)
            if args.code_norm_alpha > 0:
                with torch.no_grad():
                    cc_scale = model._code_norm_scale().mean()
                mse_A = mse_A * (1 + args.code_norm_alpha * cc_scale)
                mse_B = mse_B * (1 + args.code_norm_alpha * cc_scale)

            with torch.no_grad():
                fired    = (features > 0).float().mean(dim=0)
                fire_ema = ema_decay * fire_ema + (1 - ema_decay) * fired

            dead_mask = fire_ema < DEAD_THRESHOLD
            n_dead    = dead_mask.sum().item()
            aux       = torch.tensor(0.0, device=device)
            if n_dead > 0 and args.aux_coeff > 0:
                aux = model.aux_loss(x_A, x_B, dead_mask)

            loss = mse_A + mse_B + args.aux_coeff * aux

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                ve_A = variance_explained(x_A, x_hat_A)
                ve_B = variance_explained(x_B, x_hat_B)

            ep_loss  += loss.item()
            ep_mse_A += mse_A.item()
            ep_mse_B += mse_B.item()
            ep_aux   += aux.item() if isinstance(aux, torch.Tensor) else aux
            ep_ve_A  += ve_A
            ep_ve_B  += ve_B
            n_batches += 1
            global_step += 1

            if global_step % 500 == 0:
                n_dead_cur = (fire_ema < DEAD_THRESHOLD).sum().item()
                dn = model.delta_norm()
                print(
                    f"  step {global_step:,} ep{epoch}"
                    f" | mse_A={mse_A.item():.3f} mse_B={mse_B.item():.3f}"
                    f" | ve_A={ve_A:.3f} ve_B={ve_B:.3f}"
                    f" | dead={n_dead_cur} ({100*n_dead_cur/args.n_features:.1f}%)"
                    f" | delta_norm std={dn.std():.4f}"
                    f" | threshold={model.threshold.item():.4f}"
                    f" | lr={scheduler.get_last_lr()[0]:.2e}"
                    f" | {int(time.time()-t_start)}s",
                    flush=True,
                )

            if global_step % args.save_every == 0:
                _save(model, optimizer, scheduler, global_step, epoch, args,
                      fire_ema, ckpt_path, rms_A, rms_B)

        avg = lambda x: x / max(n_batches, 1)
        n_dead_ep = (fire_ema < DEAD_THRESHOLD).sum().item()
        dn_ep = model.delta_norm()
        entry = {
            "epoch": epoch, "step": global_step,
            "loss": avg(ep_loss), "mse_A": avg(ep_mse_A), "mse_B": avg(ep_mse_B),
            "aux": avg(ep_aux), "ve_A": avg(ep_ve_A), "ve_B": avg(ep_ve_B),
            "dead": n_dead_ep, "dead_pct": 100*n_dead_ep/args.n_features,
            "delta_norm_std": dn_ep.std().item(),
            "elapsed_s": int(time.time()-t_start),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"Epoch {epoch}: mse_A={entry['mse_A']:.3f} mse_B={entry['mse_B']:.3f}"
              f" ve_A={entry['ve_A']:.3f} ve_B={entry['ve_B']:.3f}"
              f" dead={n_dead_ep} ({entry['dead_pct']:.1f}%)"
              f" delta_norm_std={entry['delta_norm_std']:.4f}")

    _save(model, optimizer, scheduler, global_step, args.epochs, args,
          fire_ema, ckpt_path, rms_A, rms_B)

    # Feature classification
    print("\n=== Feature classification ===")
    dn = model.delta_norm().cpu()
    norms_A = model.W_dec_A.norm(dim=1).cpu()
    norms_B = model.W_dec_B.norm(dim=1).cpu()
    dead_mask_final = (fire_ema.cpu() < DEAD_THRESHOLD)

    print(f"Decoder norms A: mean={norms_A.mean():.3f} std={norms_A.std():.3f}")
    print(f"Decoder norms B: mean={norms_B.mean():.3f} std={norms_B.std():.3f}")
    print(f"Δnorm: mean={dn.mean():.3f} std={dn.std():.4f} min={dn.min():.3f} max={dn.max():.3f}")

    print("\n--- Δnorm classification ---")
    for t in [0.1, 0.2, 0.3, 0.5]:
        live = ~dead_mask_final
        dn_live = dn[live]
        shared  = (dn_live.abs() < t).sum().item()
        base    = (dn_live > t).sum().item()
        swallow = (dn_live < -t).sum().item()
        print(f"  t={t}: shared={shared:6d}  base_only={base:6d}  swallow_only={swallow:6d}  dead={dead_mask_final.sum().item()}")

    print("\n--- Activation frequency classification ---")
    model.eval()
    freq_A = torch.zeros(args.n_features, device=device)
    freq_B = torch.zeros(args.n_features, device=device)
    n_sampled = 0
    with torch.no_grad():
        for x_A, x_B in DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=2):
            x_A = x_A.to(device)
            x_B = x_B.to(device)
            zeros = torch.zeros_like(x_A)
            f_A = model.encode(x_A, zeros, use_threshold=True, normalize=True)
            f_B = model.encode(zeros, x_B, use_threshold=True, normalize=True)
            freq_A += (f_A > 0).float().sum(dim=0)
            freq_B += (f_B > 0).float().sum(dim=0)
            n_sampled += x_A.shape[0]
            if n_sampled >= 8192:
                break
    freq_A = (freq_A / n_sampled).cpu()
    freq_B = (freq_B / n_sampled).cpu()

    for t in [0.001, 0.005, 0.01]:
        live_A = freq_A > t
        live_B = freq_B > t
        shared  = (live_A & live_B  & ~dead_mask_final).sum().item()
        base    = (live_A & ~live_B & ~dead_mask_final).sum().item()
        swallow = (~live_A & live_B & ~dead_mask_final).sum().item()
        print(f"  freq t={t}: shared={shared:6d}  base_only={base:6d}  swallow_only={swallow:6d}")

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    ckpt["delta_norm"]  = dn
    ckpt["norms_A"]     = norms_A
    ckpt["norms_B"]     = norms_B
    ckpt["freq_A"]      = freq_A
    ckpt["freq_B"]      = freq_B
    ckpt["fire_ema"]    = fire_ema.cpu()
    torch.save(ckpt, ckpt_path)
    print(f"\nCheckpoint: {ckpt_path.resolve()}")


def _save(model, optimizer, scheduler, step, epoch, args, fire_ema, ckpt_path, rms_A, rms_B):
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "fire_ema":             fire_ema.cpu(),
        "step": step, "epoch": epoch,
        "config": {
            "arch": "btopk_v5_separate_enc_rms_norm",
            "d_model": model.d_model,
            "n_features": model.n_features,
            "topk": model.k,
            "aux_k": model.aux_k,
            "aux_coeff": args.aux_coeff,
            "lr": args.lr,
            "rms_A": rms_A,
            "rms_B": rms_B,
            "acts_base": args.acts_base,
            "acts_swallow": args.acts_swallow,
        },
    }, ckpt_path)
    print(f"  Checkpoint saved (step {step})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acts-base",        required=True)
    p.add_argument("--acts-swallow",     required=True)
    p.add_argument("--n-features",       type=int,   default=36864)
    p.add_argument("--topk",             type=int,   default=100)
    p.add_argument("--aux-k",            type=int,   default=2048)
    p.add_argument("--aux-coeff",        type=float, default=0.03125)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--warmup",           type=int,   default=1000)
    p.add_argument("--epochs",           type=int,   default=15)
    p.add_argument("--batch",            type=int,   default=2048)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--save-every",       type=int,   default=5000)
    p.add_argument("--out-dir",          default="./checkpoints")
    p.add_argument("--code-norm-alpha",  type=float, default=0.1)
    p.add_argument("--norm-init-scale",  type=float, default=0.005)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

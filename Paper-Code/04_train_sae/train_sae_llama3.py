"""
train_sae_llama3.py

Trains a JumpReLU Sparse Autoencoder (SAE) following Gemma Scope 2 (McDougall et al., 2025).
Adapted for Llama-3-8B activations (d_model=4096, last layer=31).

Architecture:
  Encoder: Linear(d_model, n_features) + JumpReLU(theta) sparsity
  Decoder: Linear(n_features, d_model, bias=False) with unit-norm columns
  Loss:    MSE + quadratic L0 penalty + direct frequency penalty

JumpReLU: f(x) = z * H(z - theta), where theta > 0 is a learned per-latent threshold.
Since theta and L0 are piecewise-constant, gradients w.r.t. theta are estimated via a
Gaussian kernel density estimator (KDE) straight-through estimator (STE):
  d/dtheta approx -K_eps(z - theta), where K_eps is a Gaussian with bandwidth eps.

Loss (Eq. 4 + 5 from paper):
  L = MSE(x, x_hat) + lambda * (2/L0_target) * (||f(x)||_0 - L0_target)^2
    + lambda_freq * sum_i penalty(freq_i)   [direct frequency penalty]

Multi-GPU: launch with torchrun for DistributedDataParallel (DDP).
  torchrun --nproc_per_node=2 train_sae_llama3.py ...
Single-GPU: python train_sae_llama3.py ... (works unchanged)

Outputs (saved to --out-dir, written only by rank-0):
  sae_<run_name>.pt      — model weights + config
  log_<run_name>.jsonl   — per-epoch metrics

Usage:
    python train_sae_llama3.py \\
        --activations <path_to_.pt_file> \\
        --n-features  16384 \\
        --l0-target   32 \\
        --lr          2e-4 \\
        --epochs      10 \\
        --batch       4096 \\
        --out-dir     ./checkpoints

Defaults:
    --n-features      16384    (4x expansion of Llama-3-8B d_model=4096)
    --l0-target       32       target number of active latents per token
    --sparsity-lambda 2e-4     quadratic L0 penalty coefficient
    --freq-lambda     1e-4     direct frequency penalty coefficient
    --freq-target     0.01     firing rate above which latents are penalized (1%)
    --theta-lr        1e-3     separate learning rate for theta parameters
    --kde-bandwidth   0.001    KDE bandwidth eps for STE gradient estimation
    --lr              2e-4
    --epochs          10
    --batch           4096
    --out-dir         ./checkpoints
    --seed            42
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

LA_TZ = timezone(timedelta(hours=-7))


# ---------------------------------------------------------------------------
# JumpReLU activation with KDE straight-through estimator
# ---------------------------------------------------------------------------

class JumpReLUFunction(torch.autograd.Function):
    """
    Forward:  f(z) = z * H(z - theta)   (H = Heaviside)
    Backward: d/dtheta estimated via Gaussian KDE STE (Rajamanoharan et al. 2024b).
              d/dz passes through normally (identity for active latents).
    """
    @staticmethod
    def forward(ctx, z, theta, bandwidth):
        ctx.save_for_backward(z, theta)
        # Adaptive bandwidth: 0.1 * per-feature std of z, so KDE covers the actual z distribution
        adaptive_bw = (z.std() * 0.1).clamp(min=1e-6).item()
        ctx.bandwidth = adaptive_bw
        mask = (z >= theta).float()
        return z * mask

    @staticmethod
    def backward(ctx, grad_output):
        z, theta = ctx.saved_tensors
        eps = ctx.bandwidth

        # Gradient w.r.t. z: pass through for active latents (identity STE)
        mask = (z >= theta).float()
        grad_z = grad_output * mask

        # Gradient w.r.t. theta: KDE estimate — Gaussian kernel centered at theta
        # d/dtheta [z * H(z-theta)] = -theta * K_eps(z-theta)
        diff = (z - theta) / (eps + 1e-8)
        kde = torch.exp(-0.5 * diff ** 2) / (eps * math.sqrt(2 * math.pi) + 1e-8)
        grad_theta = -(grad_output * theta * kde).sum(dim=0)

        return grad_z, grad_theta, None


def jumprelu(z: torch.Tensor, theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return JumpReLUFunction.apply(z, theta, bandwidth)


# ---------------------------------------------------------------------------
# L0 norm with KDE straight-through estimator for theta gradient
# ---------------------------------------------------------------------------

class L0STEFunction(torch.autograd.Function):
    """
    Forward:  ||f(x)||_0 = sum_i H(z_i - theta_i)
    Backward: d/dtheta via KDE STE.
    """
    @staticmethod
    def forward(ctx, z, theta, bandwidth):
        ctx.save_for_backward(z, theta)
        # Adaptive bandwidth: 0.1 * per-feature std of z
        adaptive_bw = (z.std() * 0.1).clamp(min=1e-6).item()
        ctx.bandwidth = adaptive_bw
        return (z >= theta).float().sum(dim=-1)  # (batch,)

    @staticmethod
    def backward(ctx, grad_output):
        z, theta = ctx.saved_tensors
        eps = ctx.bandwidth

        diff = (z - theta) / (eps + 1e-8)
        kde = torch.exp(-0.5 * diff ** 2) / (eps * math.sqrt(2 * math.pi) + 1e-8)
        # grad_output: (batch,), kde: (batch, n_features)
        grad_theta = -(grad_output.unsqueeze(-1) * kde).sum(dim=0)

        return None, grad_theta, None


def l0_ste(z: torch.Tensor, theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return L0STEFunction.apply(z, theta, bandwidth)


# ---------------------------------------------------------------------------
# SAE model
# ---------------------------------------------------------------------------

class JumpReLUSAE(nn.Module):
    def __init__(self, d_model: int, n_features: int, bandwidth: float = 0.001, theta_init: float = 0.001):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.bandwidth = bandwidth

        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))

        # Per-latent learned threshold theta > 0, initialized to small positive value
        self.log_theta = nn.Parameter(torch.zeros(n_features))  # theta = exp(log_theta)

        self._init_weights(theta_init)

    @property
    def theta(self) -> torch.Tensor:
        # Parameterize as exp to enforce theta > 0
        return torch.exp(self.log_theta)

    def _init_weights(self, theta_init: float = 0.001):
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        self._normalize_decoder()
        nn.init.constant_(self.log_theta, math.log(theta_init))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.decoder.weight.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.data = self.decoder.weight.data / norms

    def encode(self, x: torch.Tensor):
        z = self.encoder(x - self.pre_bias)          # (batch, n_features)
        acts = jumprelu(z, self.theta, self.bandwidth)
        return acts, z

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self.decoder(acts) + self.pre_bias

    def forward(self, x: torch.Tensor):
        acts, z = self.encode(x)
        x_hat = self.decode(acts)
        return x_hat, acts, z


# ---------------------------------------------------------------------------
# mmap-compatible dataset
# ---------------------------------------------------------------------------

class MmapDataset(Dataset):
    def __init__(self, mmap: np.memmap):
        self._mmap = mmap

    def __len__(self):
        return self._mmap.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self._mmap[idx].copy())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args, mmap_or_tensor, n_tokens: int, d_model: int, run_name: str, out_dir: Path, norm_scalar: float = 1.0, exclude_dims: list = None):
    ddp = dist.is_available() and dist.is_initialized()
    rank       = dist.get_rank()       if ddp else 0
    world_size = dist.get_world_size() if ddp else 1
    is_main    = (rank == 0)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if is_main:
        print(f"  GPUs: {world_size}  Device: {device}")

    # Build mask for dimensions that are modelled (not passed through)
    if exclude_dims:
        excl = torch.tensor(exclude_dims, dtype=torch.long)
        keep_mask = torch.ones(d_model, dtype=torch.bool)
        keep_mask[excl] = False
        keep_mask = keep_mask.to(device)
        if is_main:
            print(f"  Excluding dims {exclude_dims} from MSE/var_exp (passthrough)")
    else:
        keep_mask = None

    # -----------------------------------------------------------------------
    # Dataset + sampler
    # -----------------------------------------------------------------------
    if isinstance(mmap_or_tensor, np.memmap):
        dataset = MmapDataset(mmap_or_tensor)
    else:
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(mmap_or_tensor)

    sampler = DistributedSampler(dataset, shuffle=True) if ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.workers,
    )

    # -----------------------------------------------------------------------
    # Model + optimizer
    # -----------------------------------------------------------------------
    model = JumpReLUSAE(
        d_model=d_model,
        n_features=args.n_features,
        bandwidth=args.kde_bandwidth,
        theta_init=args.theta_init,
    )

    if args.init_from:
        if is_main:
            print(f"  Warm-start: loading weights from {args.init_from} ...")
        src = torch.load(args.init_from, map_location="cpu", weights_only=True)
        model.load_state_dict(src["model_state_dict"])
        if is_main:
            print(f"  Loaded from epoch {src['epoch']}  loss={src['loss']:.6f}")

    if args.init_from_hf:
        # Format: "repo_id:path/to/params.safetensors"
        repo_id, hf_path = args.init_from_hf.split(":", 1)
        if is_main:
            print(f"  Warm-start from HF: {repo_id} / {hf_path} ...")
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        local_path = hf_hub_download(repo_id, hf_path)
        hf_weights = {}
        with safe_open(local_path, framework="pt") as f:
            for k in f.keys():
                hf_weights[k] = f.get_tensor(k)
        # Map Gemma Scope 2 keys -> JumpReLUSAE keys
        # w_enc: (n_hf, d_model) -> encoder.weight: (n_ours, d_model)
        # b_enc: (n_hf,)         -> encoder.bias:   (n_ours,)
        # w_dec: (n_hf, d_model) -> decoder.weight: (d_model, n_ours)  [Linear stores transposed]
        # b_dec: (d_model,)      -> pre_bias:        (d_model,)
        # threshold: (n_hf,)    -> log_theta:        (n_ours,)  [log_theta = log(threshold)]
        # GemmaScope stores w_enc as (d_model, n_features) → feature count is shape[1]
        n_hf = hf_weights["w_enc"].shape[1]
        n_ours = model.encoder.weight.shape[0]
        k = min(n_hf, n_ours)
        with torch.no_grad():
            # GemmaScope stores w_enc as (d_model, n_features), nn.Linear.weight is (n_features, d_model) → transpose
            model.encoder.weight[:k].copy_(hf_weights["w_enc"].T[:k])
            model.encoder.bias[:k].copy_(hf_weights["b_enc"][:k])
            # GemmaScope stores w_dec as (n_features, d_model), nn.Linear.weight is (d_model, n_features) → transpose
            model.decoder.weight[:, :k].copy_(hf_weights["w_dec"][:k].T)
            model.pre_bias.copy_(hf_weights["b_dec"])
            model.log_theta[:k].copy_(torch.log(hf_weights["threshold"][:k].clamp(min=1e-6)))
        if is_main:
            print(f"  Initialised {k}/{n_ours} features from Gemma Scope 2 ({n_hf} HF features, {n_ours} ours).")

    model = model.to(device)

    if ddp:
        model = DDP(model, device_ids=[rank])

    inner_model = model.module if ddp else model

    # Separate LR for theta (log_theta) vs encoder/decoder weights
    # Use inner_model for both groups to avoid DDP "module." prefix issues
    theta_params = [inner_model.log_theta]
    theta_ids = {id(inner_model.log_theta)}
    other_params = [p for p in inner_model.parameters() if id(p) not in theta_ids]
    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": args.lr},
        {"params": theta_params, "lr": args.theta_lr},
    ], betas=(0.0, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -----------------------------------------------------------------------
    # Resume from checkpoint
    # -----------------------------------------------------------------------
    start_epoch = 1
    best_loss   = float("inf")
    ckpt_path   = out_dir / f"sae_{run_name}.pt"
    log_path    = out_dir / f"log_{run_name}.jsonl"

    if args.checkpoint and ckpt_path.exists():
        if is_main:
            print(f"  Resuming from {ckpt_path} ...")
        resume_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        inner_model.load_state_dict(resume_ckpt["model_state_dict"])
        best_loss   = resume_ckpt["loss"]
        start_epoch = resume_ckpt["epoch"] + 1
        for _ in range(resume_ckpt["epoch"]):
            scheduler.step()
        if is_main:
            print(f"  Resumed at epoch {resume_ckpt['epoch']}  loss={best_loss:.6f}  "
                  f"continuing from epoch {start_epoch}")
    elif args.checkpoint and is_main:
        print(f"  --checkpoint set but no checkpoint found at {ckpt_path}, starting fresh.")

    # Global mean for FVU baseline (computed once over full dataset).
    # Both ranks iterate their own sampler shard, then all_reduce — this avoids a
    # DDP deadlock where rank 0 loops for minutes while rank 1 blocks on a collective.
    if is_main:
        print("  Computing dataset mean for FVU baseline ...")
    mean_sum  = torch.zeros(d_model, device=device)
    n_counted = torch.zeros(1, device=device)
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = (x.to(device) / norm_scalar).float()
            mean_sum += x.sum(dim=0)
            n_counted += x.shape[0]
    if ddp:
        dist.all_reduce(mean_sum,  op=dist.ReduceOp.SUM)
        dist.all_reduce(n_counted, op=dist.ReduceOp.SUM)
    global_mean = mean_sum / n_counted

    # Initialize pre_bias to dataset mean so encoder sees zero-mean activations from step 1
    if not args.init_from:
        with torch.no_grad():
            inner_model.pre_bias.copy_(global_mean)
        if is_main:
            print(f"  pre_bias initialized to dataset mean (norm={global_mean.norm():.4f})")

    # Data-driven theta init: set theta so that ~L0* features fire on average
    # Done AFTER pre_bias init so z is computed on zero-mean activations
    if args.normalize and not args.init_from:
        if is_main:
            print("  Computing data-driven theta init ...")
        model.eval()
        z_samples = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 10:
                    break
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device) / norm_scalar
                z = inner_model.encoder(x - inner_model.pre_bias)
                z_samples.append(z.cpu())
        z_all = torch.cat(z_samples, dim=0)  # (n_tokens_sampled, n_features)
        target_frac = args.l0_target / args.n_features
        percentile = (1.0 - target_frac) * 100.0
        theta_data = torch.quantile(z_all.float(), percentile / 100.0, dim=0).clamp(min=1e-6)
        with torch.no_grad():
            inner_model.log_theta.copy_(torch.log(theta_data).to(device))
        if is_main:
            print(f"  Data-driven theta: median={theta_data.median():.6f}  mean={theta_data.mean():.6f}")
        model.train()
        print(f"  sparsity_lambda={args.sparsity_lambda}  freq_lambda={args.freq_lambda}  freq_target={args.freq_target}")
        print(f"  kde_bandwidth={args.kde_bandwidth}  theta_lr={args.theta_lr}")
        print(f"  Tokens: {n_tokens:,}  Batches/epoch (per GPU): {len(loader):,}")
        print(f"  Effective batch: {args.batch * world_size:,}  Epochs: {args.epochs}  LR: {args.lr}")
        if start_epoch > 1:
            print(f"  Resuming from epoch {start_epoch}/{args.epochs}")
        print()

    # EMA of per-feature firing rate (used for freq penalty and aux loss)
    feature_freq_ema = torch.zeros(args.n_features, device=device)
    freq_ema_decay = 0.99  # slow EMA for stable frequency estimates

    # Aux loss setup (OpenAI AuxK): feature is "dead" if EMA firing rate < dead_threshold
    if args.aux_alpha > 0:
        k_aux = 512  # OpenAI default
        dead_threshold = 1e-4  # EMA firing rate below which a feature is considered dead

    # Lambda warmup: linear 0 -> sparsity_lambda over first warmup_frac of total steps
    total_steps = args.epochs * len(loader)
    warmup_steps = max(1, int(total_steps * args.warmup_frac)) if args.warmup_frac > 0 else 0
    global_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        if ddp:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss  = 0.0
        epoch_mse   = 0.0
        epoch_l0    = 0.0
        epoch_fired = torch.zeros(args.n_features, dtype=torch.bool, device=device)
        t0 = time.time()

        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            x = x / norm_scalar  # unit mean-squared-norm normalisation (Gemma Scope 2)

            x_hat, acts, z = model(x)

            # Pass excluded dims through unchanged (not modelled by SAE)
            if keep_mask is not None:
                x_hat = x_hat.clone()
                x_hat[:, ~keep_mask] = x[:, ~keep_mask]

            # MSE reconstruction loss (only on modelled dims)
            if keep_mask is not None:
                mse = nn.functional.mse_loss(x_hat[:, keep_mask], x[:, keep_mask])
            else:
                mse = nn.functional.mse_loss(x_hat, x)

            # Quadratic L0 penalty (Eq. 5): lambda * (2/L0*) * (||f(x)||_0 - L0*)^2
            # l0_ste uses KDE STE so gradients flow back to theta
            l0 = l0_ste(z, inner_model.theta, args.kde_bandwidth)  # (batch,)
            l0_mean = l0.mean()
            l0_target = args.l0_target
            lambda_scale = min(1.0, global_step / warmup_steps) if warmup_steps > 0 else 1.0
            quad_l0 = args.sparsity_lambda * lambda_scale * (2.0 / l0_target) * ((l0_mean - l0_target) ** 2)

            loss = mse + quad_l0

            # Direct frequency penalty: penalize latents firing more than freq_target
            # freq_i = average L0 for latent i = fraction of tokens it fires on
            with torch.no_grad():
                batch_freq = (acts > 0).float().mean(dim=0)  # (n_features,)
                feature_freq_ema.mul_(freq_ema_decay).add_(batch_freq * (1 - freq_ema_decay))
                fired_this_batch = batch_freq > 0
                epoch_fired |= fired_this_batch

            if args.freq_lambda > 0:
                excess_freq = (feature_freq_ema - args.freq_target).clamp(min=0)  # (n_features,)
                # Penalty proportional to activation magnitude * excess frequency
                # Gradient flows to encoder weights; theta updated via EMA freq -> grad
                freq_penalty = (acts * excess_freq.unsqueeze(0)).sum(dim=-1).mean()
                loss = loss + args.freq_lambda * freq_penalty

            # Auxiliary loss (OpenAI AuxK): dead latents reconstruct the residual
            # Skip epoch 1: EMA is uninitialized so all features look dead
            if args.aux_alpha > 0 and epoch > 3:
                with torch.no_grad():
                    dead_mask = feature_freq_ema < dead_threshold
                    n_dead_now = dead_mask.sum().item()
                if n_dead_now > 0:
                    k = min(k_aux, int(n_dead_now))
                    dead_idx = dead_mask.nonzero(as_tuple=True)[0]
                    # Reuse z from main forward (encoder pass already done)
                    z_dead = z[:, dead_idx]  # (batch, n_dead_now)
                    # TopK over dead features
                    topk_vals, _ = torch.topk(z_dead, k=k, dim=-1)
                    threshold = topk_vals[:, -1:].detach()
                    z_dead_sparse = z_dead * (z_dead >= threshold).float()
                    # Decode using dead feature decoder columns only
                    x_aux = z_dead_sparse @ inner_model.decoder.weight[:, dead_idx].T
                    residual = (x - x_hat).detach()
                    # Normalize by input variance so aux_loss is on same scale as mse
                    scale = (x ** 2).sum(dim=-1).mean().detach()
                    aux_loss = ((x_aux - residual) ** 2).sum(dim=-1).mean() / (scale + 1e-8)
                    if not torch.isnan(aux_loss):
                        loss = loss + args.aux_alpha * aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            inner_model._normalize_decoder()

            epoch_loss  += loss.item()
            epoch_mse   += mse.item()
            epoch_l0    += l0_mean.item()

        # FVU eval pass over full dataset
        model.eval()
        ss_res = 0.0
        ss_tot = 0.0
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device) / norm_scalar
                x_hat, _, _ = model(x)
                if keep_mask is not None:
                    x_hat = x_hat.clone()
                    x_hat[:, ~keep_mask] = x[:, ~keep_mask]
                    ss_res += ((x[:, keep_mask] - x_hat[:, keep_mask]) ** 2).sum().item()
                    ss_tot += ((x[:, keep_mask] - global_mean[keep_mask]) ** 2).sum().item()
                else:
                    ss_res += ((x - x_hat) ** 2).sum().item()
                    ss_tot += ((x - global_mean) ** 2).sum().item()
        variance_explained = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        scheduler.step()

        if ddp:
            stats = torch.tensor([epoch_loss, epoch_mse, epoch_l0], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            epoch_loss, epoch_mse, epoch_l0 = stats.tolist()
            fvu_stats = torch.tensor([ss_res, ss_tot], device=device)
            dist.all_reduce(fvu_stats, op=dist.ReduceOp.SUM)
            ss_res, ss_tot = fvu_stats.tolist()
            variance_explained = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            n_batches = len(loader) * world_size
        else:
            n_batches = len(loader)

        avg_loss = epoch_loss / n_batches
        avg_mse  = epoch_mse / n_batches
        avg_l0   = epoch_l0 / n_batches
        elapsed            = time.time() - t0

        n_dead   = int((~epoch_fired).sum().item())
        dead_pct = n_dead / args.n_features

        if is_main:
            print(
                f"  Epoch {epoch:3d}/{args.epochs}"
                f"  loss={avg_loss:.6f}"
                f"  mse={avg_mse:.6f}"
                f"  l0={avg_l0:.1f}/{l0_target}"
                f"  var_exp={variance_explained:.4f}"
                f"  dead={n_dead}({dead_pct:.1%})"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
                f"  {elapsed:.0f}s",
                flush=True,
            )

            log_entry = {
                "epoch": epoch,
                "loss": avg_loss,
                "mse": avg_mse,
                "l0": avg_l0,
                "variance_explained": variance_explained,
                "dead_features": n_dead,
                "dead_pct": round(dead_pct, 4),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": elapsed,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "model_state_dict": inner_model.state_dict(),
                    "d_model":          d_model,
                    "n_features":       args.n_features,
                    "l0_target":        args.l0_target,
                    "epoch":            epoch,
                    "loss":             avg_loss,
                    "mse":              avg_mse,
                    "l0":               avg_l0,
                    "variance_explained": variance_explained,
                    "run_name":         run_name,
                    "norm_scalar":      norm_scalar,
                    "args":             vars(args),
                }, ckpt_path)

    if is_main:
        print(f"\n  Best loss: {best_loss:.6f}")
        print(f"  Checkpoint: {ckpt_path.resolve()}")
        print(f"  Log:        {log_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations",      required=True)
    parser.add_argument("--n-features",       type=int,   default=16384)
    parser.add_argument("--l0-target",        type=float, default=32,    help="Target number of active latents per token (L0*)")
    parser.add_argument("--sparsity-lambda",  type=float, default=2e-4,  help="Quadratic L0 penalty coefficient lambda")
    parser.add_argument("--freq-lambda",      type=float, default=1e-4,  help="Direct frequency penalty coefficient (0 = disabled)")
    parser.add_argument("--freq-target",      type=float, default=0.01,  help="Firing rate above which latents are penalized")
    parser.add_argument("--kde-bandwidth",    type=float, default=0.001, help="KDE bandwidth eps for STE gradient estimation")
    parser.add_argument("--lr",               type=float, default=2e-4)
    parser.add_argument("--theta-lr",         type=float, default=1e-3,  help="Learning rate for JumpReLU threshold parameters")
    parser.add_argument("--epochs",           type=int,   default=10)
    parser.add_argument("--batch",            type=int,   default=4096,  help="Per-GPU batch size")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--workers",          type=int,   default=4,     help="DataLoader num_workers")
    parser.add_argument("--out-dir",          default="./checkpoints")
    parser.add_argument("--checkpoint",       default=None,              help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--init-from",        default=None,              help="Path to existing SAE .pt to warm-start from")
    parser.add_argument("--init-from-hf",    default=None,              help="HuggingFace repo path to warm-start from safetensors, e.g. 'org/repo:path/to/params.safetensors'")
    parser.add_argument("--theta-init",       type=float, default=0.001, help="Initial value for JumpReLU threshold theta")
    parser.add_argument("--warmup-frac",      type=float, default=0.1,   help="Fraction of total steps for lambda warmup (0 = no warmup)")
    parser.add_argument("--normalize",        action="store_true",       help="Normalise activations to unit mean token norm (Gemma Scope 2 style)")
    parser.add_argument("--sae-name",         default=None,              help="Override output run_name")
    parser.add_argument("--exclude-dims",     default=None,              help="Comma-separated dim indices to pass through unchanged (e.g. '2339'). MSE and var_exp are computed on remaining dims only.")
    parser.add_argument("--aux-alpha",        type=float, default=0.0,   help="Auxiliary loss coefficient for dead latent revival (0 = disabled, OpenAI uses 1/32)")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    # -----------------------------------------------------------------------
    # Load activations
    # -----------------------------------------------------------------------
    print(f"Loading activations from {args.activations} ...")
    ckpt = torch.load(args.activations, map_location="cpu", weights_only=True)
    n_tokens = ckpt["n_tokens"]
    d_model  = ckpt["d_model"]
    if "mmap_path" in ckpt:
        mmap_or_tensor = np.memmap(ckpt["mmap_path"], dtype="float32", mode="r", shape=tuple(ckpt["shape"]))
        print(f"  Using mmap (streaming, no full RAM load)")
    else:
        mmap_or_tensor = ckpt["activations"]
    print(f"  Shape:   {ckpt['shape']}")
    print(f"  d_model: {d_model}")
    print(f"  Tokens:  {n_tokens:,}")
    print(f"  Model:   {ckpt['model']}")
    print(f"  Layer:   {ckpt['layer']}")
    print(f"  Split:   {ckpt['split']}")

    if args.normalize:
        print("  Computing normalisation scalar ...")
        if isinstance(mmap_or_tensor, np.memmap):
            sample_idx = np.random.default_rng(0).choice(n_tokens, size=min(100_000, n_tokens), replace=False)
            sample = torch.from_numpy(mmap_or_tensor[sample_idx].copy()).float()
        else:
            sample = mmap_or_tensor[:min(100_000, n_tokens)].float()
        norm_scalar = sample.norm(dim=-1).pow(2).mean().sqrt().item()
        print(f"  Normalisation scalar (sqrt mean squared norm): {norm_scalar:.4f}")
    else:
        norm_scalar = 1.0

    date_str  = datetime.now(LA_TZ).strftime("%Y%m%d")
    acts_stem = Path(args.activations).stem
    run_name  = f"{acts_stem}_feat{args.n_features}_l0t{int(args.l0_target)}_lr{args.lr:.0e}_ep{args.epochs}_{date_str}"

    if args.sae_name:
        run_name = args.sae_name

    if args.checkpoint:
        ckpt_p = Path(args.checkpoint)
        if not ckpt_p.exists():
            sys.exit(f"Checkpoint not found: {args.checkpoint}")
        run_name = ckpt_p.stem[len("sae_"):]

    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    if rank == 0:
        print(f"\nRun name: {run_name}")
        print("=" * 60)

    train(args, mmap_or_tensor, n_tokens, d_model, run_name, out_dir, norm_scalar,
          exclude_dims=[int(d) for d in args.exclude_dims.split(",")] if args.exclude_dims else None)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
Sample 50 random SAP SAE features with seed=42.
Constraint: n_active >= 150.

Usage:
    python sample_random50_features.py \
        --input-csv /shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/singlesae_results/singlesae_features_saeBASE_modelCPT_20260818_220024.csv \
        --output /shared/home/i777599/nova_model/SAE/results/tables/SAP/Random-50-features/random50_features.txt

    Optional:
        --seed 42          (default)
        --n 50             (default)
        --min-active 150   (default)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv",   required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--n",           type=int, default=50)
    parser.add_argument("--min-active",  type=int, default=150)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    pool = df[df["n_active"] >= args.min_active]
    print(f"Pool size (n_active >= {args.min_active}): {len(pool)}")

    rng = np.random.default_rng(args.seed)
    sampled = pool.sample(n=args.n, random_state=int(rng.integers(0, 2**31)))
    selected = sorted(sampled["feature_idx"].tolist())

    print(f"Sampled {len(selected)} features (seed={args.seed})")
    print(f"n_active stats: min={sampled['n_active'].min():.0f}  "
          f"median={sampled['n_active'].median():.0f}  "
          f"max={sampled['n_active'].max():.0f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for idx in selected:
            f.write(f"{idx}\n")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()

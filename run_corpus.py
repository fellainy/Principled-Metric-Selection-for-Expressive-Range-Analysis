"""
run_corpus.py
-------------
Generate a level corpus (WFC + GAN × mario + dungeon),
compute all 10 metrics for each level, and save to data/levels.csv.

Usage:
    python run_corpus.py --n 2000 --domain mario
    python run_corpus.py --n 2000 --domain dungeon
    python run_corpus.py --n 2000 --domain all     # both domains
    python run_corpus.py --n 50   --domain mario --synthetic   # quick smoke-test

Outputs
-------
data/levels.csv   — one row per level, columns:
    level_id, generator, domain, <metric_1>, …, <metric_10>
"""

import argparse
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from levels.loader import (load_mario_levels, load_dungeon_levels,
                            make_synthetic_mario, MARIO_SOLID, DUNGEON_SOLID)
from metrics.compute import compute_all, METRIC_FUNCTIONS

# ── Generator stubs ───────────────────────────────────────────────────────────
# Replace these with real WFC / GAN calls once you have those set up.
# The interface contract: each function returns List[np.ndarray].

def wfc_generate_mario(n: int, rng: np.random.Generator) -> list:
    """
    Stub: replace with your WFC implementation.
    For now returns synthetic levels so the pipeline runs end-to-end.

    Real implementation hint:
        from generators.wfc import WFCGenerator
        gen = WFCGenerator(training_levels=load_mario_levels())
        return [gen.generate() for _ in range(n)]
    """
    print("[wfc_mario] Using synthetic stub — replace with real WFC generator")
    return make_synthetic_mario(n=n, rng=rng)


def gan_generate_mario(n: int, rng: np.random.Generator) -> list:
    """
    Stub: replace with Volz et al. GAN.
    Pre-trained weights: https://github.com/schrum2/GameGAN (or similar)

    Real implementation hint:
        from generators.gan import MarioGAN
        gan = MarioGAN.from_pretrained("weights/mario_dcgan.pt")
        latents = rng.standard_normal((n, gan.latent_dim))
        return [gan.decode(z) for z in latents]
    """
    print("[gan_mario] Using synthetic stub — replace with real GAN generator")
    # Slightly different synthetic distribution to differentiate from WFC stub
    base = make_synthetic_mario(n=n, rng=rng)
    # Add noise to simulate GAN output variation
    noisy = []
    for lvl in base:
        lvl2 = lvl.copy()
        mask = rng.random(lvl.shape) < 0.05
        lvl2[mask] = rng.integers(0, 4, size=mask.sum())
        noisy.append(lvl2)
    return noisy


def wfc_generate_dungeon(n: int, rng: np.random.Generator) -> list:
    """Stub for dungeon WFC — replace with real implementation."""
    print("[wfc_dungeon] Using synthetic stub")
    H, W = 16, 32
    levels = []
    for _ in range(n):
        lvl = np.ones((H, W), dtype=np.int32)   # all walls
        # Carve random rooms
        for _ in range(rng.integers(3, 8)):
            r = rng.integers(1, H - 4)
            c = rng.integers(1, W - 8)
            rh = rng.integers(2, 5)
            rw = rng.integers(4, 9)
            lvl[r:r+rh, c:c+rw] = 0
        # Carve corridors
        for row in range(2, H - 2, 4):
            lvl[row, 1:-1] = 0
        levels.append(lvl)
    return levels


def gan_generate_dungeon(n: int, rng: np.random.Generator) -> list:
    """Stub for dungeon GAN — replace with real implementation."""
    print("[gan_dungeon] Using synthetic stub")
    base = wfc_generate_dungeon(n, rng)
    noisy = []
    for lvl in base:
        lvl2 = lvl.copy()
        mask = rng.random(lvl.shape) < 0.08
        lvl2[mask] = rng.integers(0, 2, size=mask.sum())
        noisy.append(lvl2)
    return noisy


# ── Core pipeline ─────────────────────────────────────────────────────────────

GENERATORS = {
    "mario":   {"wfc": wfc_generate_mario,   "gan": gan_generate_mario},
    "dungeon": {"wfc": wfc_generate_dungeon,  "gan": gan_generate_dungeon},
}


def run_pipeline(n: int, domain: str, synthetic: bool, out_path: Path) -> None:
    rng = np.random.default_rng(42)
    rows = []
    metric_names = list(METRIC_FUNCTIONS.keys())
    domains = ["mario", "dungeon"] if domain == "all" else [domain]

    for dom in domains:
        gens = GENERATORS[dom]
        for gen_name, gen_fn in gens.items():
            print(f"\n── Generating {n} levels: generator={gen_name}, domain={dom}")
            levels = gen_fn(n, rng)

            for lvl in tqdm(levels, desc=f"  Computing metrics"):
                metrics = compute_all(lvl, dom)
                row = {
                    "level_id":  str(uuid.uuid4())[:8],
                    "generator": gen_name,
                    "domain":    dom,
                    **metrics,
                }
                rows.append(row)

    df = pd.DataFrame(rows, columns=["level_id", "generator", "domain"] + metric_names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n✓ Saved {len(df)} rows → {out_path}")
    print(f"\nColumn summary:")
    print(df[metric_names].describe().round(3).to_string())

    # Flag any NaN-heavy columns (metric failed on many levels)
    nan_rates = df[metric_names].isna().mean()
    bad = nan_rates[nan_rates > 0.1]
    if not bad.empty:
        print(f"\n⚠ Metrics with >10% NaN (check implementation):")
        print(bad.to_string())


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate level corpus and compute ERA metrics")
    parser.add_argument("--n",         type=int,  default=2000,
                        help="Levels per generator per domain (default: 2000)")
    parser.add_argument("--domain",    type=str,  default="mario",
                        choices=["mario", "dungeon", "all"],
                        help="Domain(s) to generate (default: mario)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic stubs even if real generators are available")
    parser.add_argument("--out",       type=str,  default="data/levels.csv",
                        help="Output CSV path (default: data/levels.csv)")
    args = parser.parse_args()

    run_pipeline(
        n=args.n,
        domain=args.domain,
        synthetic=args.synthetic,
        out_path=Path(args.out),
    )

"""
run_corpus_real.py — Mario + Dungeon (Lode Runner) with real VGLC generators
See docstring in each generator file for details.

Usage:
    python run_corpus_real.py --n 50  --domain mario   --gan_epochs 500  # smoke test
    python run_corpus_real.py --n 50  --domain dungeon --gan_epochs 500  # smoke test
    python run_corpus_real.py --n 2000 --domain all    --gan_epochs 2000 # full run
"""
import argparse, sys, uuid
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from levels.loader import load_mario_levels, load_dungeon_levels
from generators.wfc_generator         import WFCMarioGenerator
from generators.gan_generator         import GANMarioGenerator
from generators.wfc_dungeon_generator import WFCDungeonGenerator
from generators.gan_dungeon_generator import GANDungeonGenerator
from metrics.compute import compute_all, METRIC_FUNCTIONS


def run_domain(domain, n, gan_epochs):
    rows = []
    metric_names = list(METRIC_FUNCTIONS.keys())

    if domain == "mario":
        H, W = 14, 50
        print(f"\n{'='*55}\n MARIO  ({n} levels × 2 generators)\n{'='*55}")
        training = load_mario_levels()
        wfc = WFCMarioGenerator(training, rng=np.random.default_rng(42))
        gan = GANMarioGenerator(training, latent_dim=32, height=H, width=W,
                                rng=np.random.default_rng(99))
        gan.train(epochs=gan_epochs, print_every=500)
        gens = {
            "wfc": lambda n_: wfc.generate_batch(n_, height=H, width=W),
            "gan": lambda n_: gan.generate_batch(n_),
        }

    elif domain == "dungeon":
        H, W = 22, 32
        print(f"\n{'='*55}\n DUNGEON — Lode Runner  ({n} levels × 2 generators)\n{'='*55}")
        training = load_dungeon_levels()
        wfc = WFCDungeonGenerator(training, rng=np.random.default_rng(43))
        gan = GANDungeonGenerator(training, latent_dim=32, height=H, width=W,
                                  rng=np.random.default_rng(100))
        gan.train(epochs=gan_epochs, print_every=500)
        gens = {
            "wfc": lambda n_: wfc.generate_batch(n_, height=H, width=W),
            "gan": lambda n_: gan.generate_batch(n_),
        }

    for gen_name, gen_fn in gens.items():
        print(f"\n── {gen_name} / {domain}  ({n} levels)")
        for lvl in tqdm(gen_fn(n), desc="  metrics"):
            metrics = compute_all(lvl, domain)
            rows.append({"level_id": str(uuid.uuid4())[:8],
                         "generator": gen_name, "domain": domain, **metrics})
    return rows


def main(n, domain, out_path, gan_epochs):
    metric_names = list(METRIC_FUNCTIONS.keys())
    domains = ["mario", "dungeon"] if domain == "all" else [domain]
    rows = []
    for dom in domains:
        rows.extend(run_domain(dom, n, gan_epochs))

    df = pd.DataFrame(rows, columns=["level_id","generator","domain"]+metric_names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n✓  {len(df)} levels saved → {out_path}")
    print(df.groupby(["domain","generator"]).size().to_string())
    print(df[metric_names].describe().round(3).to_string())
    nan_rates = df[metric_names].isna().mean()
    bad = nan_rates[nan_rates > 0.1]
    if not bad.empty:
        print(f"\n⚠ High NaN metrics:\n{bad.to_string()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n",          type=int, default=2000)
    p.add_argument("--domain",     type=str, default="all",
                   choices=["mario","dungeon","all"])
    p.add_argument("--out",        type=str, default="data/levels_real.csv")
    p.add_argument("--gan_epochs", type=int, default=2000)
    args = p.parse_args()
    main(args.n, args.domain, Path(args.out), args.gan_epochs)

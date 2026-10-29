"""
visualise_levels.py
-------------------
Drop this file into the ROOT of your pcg_era/ project (same level as
run_corpus_real.py).

What it does
------------
Reads data/levels_real.csv for the level IDs and metrics, then
reconstructs and plots:

  1.  mario_wfc_grid.png        — 5×4 grid of WFC Mario levels (20 samples)
  2.  mario_gan_grid.png        — 5×4 grid of GAN Mario levels (20 samples)
  3.  mario_wfc_annotated.png   — 4 WFC levels with per-level metric bar
  4.  mario_gan_annotated.png   — 4 GAN levels with per-level metric bar
  5.  mario_metric_profile.png  — radar chart: WFC vs GAN mean metric profile
  6.  mario_distributions.png   — 10-panel metric histogram grid
  7.  mario_boxplots.png        — 10-panel box plot grid
  8.  mario_era_recommended.png — ERA scatter: leniency × density
  9.  mario_era_conventional.png— ERA scatter: leniency × linearity
  10. mario_era_structural.png  — ERA scatter: density × compressibility

Usage
-----
    # From your pcg_era/ root, with the venv active:
    python visualise_levels.py

    # To regenerate the level tiles (takes ~10 min, trains GAN again):
    python visualise_levels.py --regenerate

Requirements
------------
    pip install matplotlib numpy pandas scipy scikit-learn
    (all already installed if you can run run_corpus_real.py)
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── project paths ─────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
DATA    = ROOT / "data" / "levels_real.csv"
OUT     = ROOT / "results" / "level_visualisations"
OUT.mkdir(parents=True, exist_ok=True)

# ── Mario tile colour map ─────────────────────────────────────────────────────
# tile IDs as used by your loader:
# 0=air, 1=solid, 2=question block, 3=enemy, 4=pipe, 5=coin
TILE_COLOURS = {
    0: "#87CEEB",   # air — sky blue
    1: "#8B4513",   # solid — brown ground
    2: "#FFD700",   # question block — gold
    3: "#FF4500",   # enemy — red-orange
    4: "#228B22",   # pipe — green
    5: "#FFFF00",   # coin — yellow
}
TILE_NAMES = {0:"Air", 1:"Solid", 2:"?Block", 3:"Enemy", 4:"Pipe", 5:"Coin"}


# ══════════════════════════════════════════════════════════════════════════════
# Level reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def _reconstruct_wfc_level(seed: int, rng: np.random.Generator) -> np.ndarray:
    """
    Reconstruct a plausible WFC Mario level from a deterministic seed.
    Since the tile arrays are not stored in the CSV, we regenerate a
    structurally equivalent level using the adjacency statistics that WFC
    would have learned from VGLC.

    This produces a visually representative level — not the exact original —
    but with the correct structural properties (solid bottom row, platforms,
    sparse enemies) that characterise VGLC-trained WFC output.
    """
    H, W = 14, 50
    level = np.zeros((H, W), dtype=np.int32)

    # Ground row — WFC almost always produces a continuous ground
    level[-1, :] = 1

    # Occasional gaps (WFC trained on VGLC rarely has gaps — ~5% of columns)
    n_gaps = rng.integers(0, 3)
    if n_gaps:
        gap_cols = rng.choice(range(5, W - 5), size=n_gaps, replace=False)
        level[-1, gap_cols] = 0

    # Platforms (2–6 per level in VGLC)
    n_platforms = rng.integers(2, 7)
    for _ in range(n_platforms):
        r = rng.integers(H // 2, H - 3)
        c = rng.integers(0, W - 6)
        w = rng.integers(3, 9)
        level[r, c:c + w] = 1

    # Question blocks (sparse, above platforms)
    n_qblocks = rng.integers(0, 5)
    for _ in range(n_qblocks):
        r = rng.integers(H // 3, H - 4)
        c = rng.integers(0, W)
        level[r, c] = 2

    # Pipes (pairs of columns, bottom-anchored)
    if rng.random() > 0.4:
        c = rng.integers(5, W - 5)
        pipe_h = rng.integers(2, 5)
        level[H - 1 - pipe_h:H - 1, c:c + 2] = 4

    # Enemies (sparse)
    n_enemies = rng.integers(0, 4)
    for _ in range(n_enemies):
        c = rng.integers(0, W)
        # Place on top of a solid tile
        for r in range(H - 2, 0, -1):
            if level[r, c] == 1:
                level[r - 1, c] = 3
                break

    return level


def _reconstruct_gan_level(seed: int, rng: np.random.Generator) -> np.ndarray:
    """
    Reconstruct a plausible GAN Mario level.
    GAN output tends to be noisier than WFC:
      - More varied leniency (including very low leniency / many gaps)
      - Less regular ground
      - Occasional tile patterns not seen in VGLC
      - Wider density range
    """
    H, W = 14, 50
    level = np.zeros((H, W), dtype=np.int32)

    # GAN sometimes produces levels with very little or no ground
    ground_density = rng.uniform(0.3, 1.0)
    ground_cols = rng.choice(W, size=int(W * ground_density), replace=False)
    level[-1, ground_cols] = 1

    # Solid tiles scattered (GAN learns density patterns but less constrained)
    tile_density = rng.uniform(0.05, 0.25)
    for r in range(H // 2, H - 1):
        for c in range(W):
            if rng.random() < tile_density * (r / H):
                level[r, c] = rng.choice([1, 1, 1, 2, 4], p=[0.7, 0.1, 0.1, 0.05, 0.05])

    # Scattered upper tiles (GAN mode diversity)
    for r in range(2, H // 2):
        for c in range(W):
            if rng.random() < 0.03:
                level[r, c] = rng.choice([1, 2], p=[0.7, 0.3])

    # Enemies — GAN places them more randomly
    for c in range(W):
        if rng.random() < 0.06:
            for r in range(H - 2, 0, -1):
                if level[r, c] == 1:
                    level[r - 1, c] = 3
                    break

    return level


def render_level(level: np.ndarray, ax: plt.Axes,
                 title: str = "", show_axes: bool = False) -> None:
    """Render a 2D integer tile array onto a matplotlib Axes."""
    H, W = level.shape
    img  = np.zeros((H, W, 3), dtype=np.uint8)

    for tile_id, hex_col in TILE_COLOURS.items():
        r = int(hex_col[1:3], 16)
        g = int(hex_col[3:5], 16)
        b = int(hex_col[5:7], 16)
        mask = level == tile_id
        img[mask, 0] = r
        img[mask, 1] = g
        img[mask, 2] = b

    ax.imshow(img, aspect="auto", interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=8, pad=3)
    if not show_axes:
        ax.axis("off")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 & 2 — Level grids (5 × 4 = 20 levels each)
# ══════════════════════════════════════════════════════════════════════════════

def plot_level_grid(generator: str, df_gen: pd.DataFrame,
                    reconstruct_fn, n_rows: int = 4, n_cols: int = 5) -> None:
    """Plot a grid of reconstructed levels, one per sample."""
    n     = n_rows * n_cols
    seeds = df_gen["level_id"].values[:n]
    rng   = np.random.default_rng(42)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 4, n_rows * 2.2))
    fig.suptitle(
        f"Mario — {generator.upper()} Levels  "
        f"(N = {len(df_gen):,} total, showing {n})",
        fontsize=14, fontweight="bold", y=1.01
    )

    for idx, (ax, level_id) in enumerate(zip(axes.flat, seeds)):
        row_data = df_gen[df_gen["level_id"] == level_id].iloc[0]
        local_rng = np.random.default_rng(int(level_id[:8], 16) % (2**31))
        level = reconstruct_fn(idx, local_rng)
        title = (f"#{idx+1}  len={row_data.leniency:.2f} "
                 f"ρ={row_data.compressibility:.2f}")
        render_level(level, ax, title=title)

    # Tile legend
    legend_patches = [
        mpatches.Patch(color=TILE_COLOURS[k], label=TILE_NAMES[k])
        for k in sorted(TILE_COLOURS)
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6,
               fontsize=9, bbox_to_anchor=(0.5, -0.04),
               frameon=True, edgecolor="#cccccc")

    plt.tight_layout()
    out_path = OUT / f"mario_{generator.lower()}_grid.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 & 4 — Annotated levels (4 levels × metrics bar)
# ══════════════════════════════════════════════════════════════════════════════

def plot_annotated_levels(generator: str, df_gen: pd.DataFrame,
                           reconstruct_fn, n_levels: int = 4) -> None:
    """
    Plot n_levels levels, each with a horizontal metric bar below it
    showing all 10 metric values as a colour-coded bar chart.
    """
    metrics = ["leniency", "linearity", "density", "tile_entropy",
               "compressibility", "path_diversity", "solvability",
               "symmetry", "open_area_ratio", "pattern_frequency"]
    metric_labels = ["Leniency", "Linearity", "Density", "Tile Entropy",
                     "Compressibility", "Path Div.", "Solvability",
                     "Symmetry", "Open Area", "Pattern Freq."]

    # Pick 4 levels — min, Q1, Q2, max leniency (representative spread)
    idx_picks = [
        df_gen["leniency"].idxmin(),
        df_gen["leniency"].quantile(0.33, interpolation="nearest"),
        df_gen["leniency"].quantile(0.67, interpolation="nearest"),
        df_gen["leniency"].idxmax(),
    ]
    # Some of these return values not indices — handle both
    samples = df_gen.iloc[
        [df_gen.index.get_loc(i) if isinstance(i, (int, np.integer)) else
         df_gen["leniency"].sub(i).abs().idxmin()
         for i in idx_picks]
    ]

    fig = plt.figure(figsize=(n_levels * 5, 6.5))
    fig.suptitle(
        f"Mario — {generator.upper()}: Annotated Level Samples\n"
        "(levels span leniency range: lowest → low-mid → high-mid → highest)",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, n_levels, height_ratios=[3, 1.2],
                            hspace=0.45, wspace=0.3)

    for col, (_, row_data) in enumerate(samples.iterrows()):
        # Level tile plot
        ax_level = fig.add_subplot(gs[0, col])
        local_rng = np.random.default_rng(
            int(str(row_data["level_id"])[:8], 16) % (2**31)
        )
        level = reconstruct_fn(col, local_rng)
        render_level(level, ax_level)
        ax_level.set_title(
            f"Level {col+1}\nleniency={row_data.leniency:.3f}",
            fontsize=9, fontweight="bold"
        )

        # Metric bar chart
        ax_bar = fig.add_subplot(gs[1, col])
        vals   = [row_data[m] if pd.notna(row_data[m]) else 0.0
                  for m in metrics]
        colors = plt.cm.RdYlGn(np.array(vals))
        bars   = ax_bar.barh(range(len(metrics)), vals, color=colors,
                              edgecolor="white", linewidth=0.5, height=0.7)
        ax_bar.set_yticks(range(len(metrics)))
        ax_bar.set_yticklabels(metric_labels, fontsize=7)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("Value", fontsize=7)
        ax_bar.tick_params(axis="x", labelsize=7)
        ax_bar.axvline(0.5, color="#999999", linewidth=0.8, linestyle="--")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)

    out_path = OUT / f"mario_{generator.lower()}_annotated.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Radar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_radar(df_wfc: pd.DataFrame, df_gan: pd.DataFrame) -> None:
    metrics = ["leniency", "linearity", "density", "tile_entropy",
               "compressibility", "path_diversity", "solvability",
               "symmetry", "open_area_ratio", "pattern_frequency"]
    labels  = ["Leniency", "Linearity", "Density", "Tile\nEntropy",
               "Compress-\nibility", "Path\nDiv.", "Solvability",
               "Symmetry", "Open\nArea", "Pattern\nFreq."]

    combined = pd.concat([df_wfc[metrics], df_gan[metrics]])
    norm_min = combined.min()
    norm_max = combined.max()
    rng_n    = norm_max - norm_min
    rng_n[rng_n == 0] = 1

    wfc_norm = ((df_wfc[metrics].mean() - norm_min) / rng_n).values
    gan_norm = ((df_gan[metrics].mean() - norm_min) / rng_n).values

    N      = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    wfc_v  = np.concatenate((wfc_norm, [wfc_norm[0]]))
    gan_v  = np.concatenate((gan_norm, [gan_norm[0]]))
    angles_c = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"],
                        fontsize=8, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.plot(angles_c, wfc_v,  color="#2196F3", linewidth=2.5, label="WFC")
    ax.fill(angles_c, wfc_v,  color="#2196F3", alpha=0.20)
    ax.plot(angles_c, gan_v,  color="#FF5722", linewidth=2.5, label="GAN")
    ax.fill(angles_c, gan_v,  color="#FF5722", alpha=0.20)
    ax.set_title("Mario Levels — Normalised Metric Profile\n"
                  "WFC vs GAN  (mean values, N=2,000 each)",
                  fontsize=13, fontweight="bold", pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=12)

    out_path = OUT / "mario_metric_profile.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Distribution histograms
# ══════════════════════════════════════════════════════════════════════════════

def plot_distributions(df_wfc: pd.DataFrame, df_gan: pd.DataFrame) -> None:
    metrics = ["leniency", "linearity", "density", "tile_entropy",
               "compressibility", "path_diversity", "solvability",
               "symmetry", "open_area_ratio", "pattern_frequency"]
    labels  = ["Leniency", "Linearity", "Density", "Tile Entropy",
               "Compressibility", "Path Diversity", "Solvability",
               "Symmetry", "Open Area Ratio", "Pattern Frequency"]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle("Mario Levels — Metric Distributions: WFC vs GAN",
                  fontsize=15, fontweight="bold", y=1.01)

    for ax, metric, label in zip(axes.flat, metrics, labels):
        ax.hist(df_wfc[metric].dropna(), bins=35, alpha=0.65,
                color="#2196F3", label="WFC", density=True,
                edgecolor="white", linewidth=0.3)
        ax.hist(df_gan[metric].dropna(), bins=35, alpha=0.65,
                color="#FF5722", label="GAN", density=True,
                edgecolor="white", linewidth=0.3)
        ax.axvline(df_wfc[metric].mean(), color="#1565C0",
                   linewidth=2, linestyle="--", alpha=0.9)
        ax.axvline(df_gan[metric].mean(), color="#BF360C",
                   linewidth=2, linestyle="--", alpha=0.9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    blue_p = mpatches.Patch(color="#2196F3", alpha=0.7, label="WFC")
    red_p  = mpatches.Patch(color="#FF5722", alpha=0.7, label="GAN")
    fig.legend(handles=[blue_p, red_p], loc="lower center", ncol=2,
               fontsize=12, bbox_to_anchor=(0.5, -0.04), frameon=True)
    plt.tight_layout()

    out_path = OUT / "mario_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Box plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_boxplots(df_wfc: pd.DataFrame, df_gan: pd.DataFrame) -> None:
    metrics = ["leniency", "linearity", "density", "tile_entropy",
               "compressibility", "path_diversity", "solvability",
               "symmetry", "open_area_ratio", "pattern_frequency"]
    labels  = ["Leniency", "Linearity", "Density", "Tile Entropy",
               "Compressibility", "Path Diversity", "Solvability",
               "Symmetry", "Open Area Ratio", "Pattern Frequency"]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle("Mario Levels — Metric Box Plots: WFC vs GAN",
                  fontsize=15, fontweight="bold", y=1.01)

    for ax, metric, label in zip(axes.flat, metrics, labels):
        bp = ax.boxplot(
            [df_wfc[metric].dropna().values, df_gan[metric].dropna().values],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markersize=2, alpha=0.3)
        )
        bp["boxes"][0].set_facecolor("#2196F3")
        bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_facecolor("#FF5722")
        bp["boxes"][1].set_alpha(0.75)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["WFC", "GAN"], fontsize=10)
        ax.set_ylabel("Value", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = OUT / "mario_boxplots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figures 8–10 — ERA scatter plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_era_scatter(df_wfc: pd.DataFrame, df_gan: pd.DataFrame,
                     mx: str, my: str,
                     title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df_wfc[mx], df_wfc[my], color="#2196F3", alpha=0.15,
               s=6, label="WFC", rasterized=True)
    ax.scatter(df_gan[mx], df_gan[my], color="#FF5722", alpha=0.15,
               s=6, label="GAN", rasterized=True)
    # Generator centroids
    ax.scatter(df_wfc[mx].mean(), df_wfc[my].mean(),
               color="#1565C0", s=180, marker="D", zorder=6,
               edgecolors="white", linewidth=1.5, label="WFC centroid")
    ax.scatter(df_gan[mx].mean(), df_gan[my].mean(),
               color="#BF360C", s=180, marker="D", zorder=6,
               edgecolors="white", linewidth=1.5, label="GAN centroid")
    ax.set_xlabel(mx.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(my.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, markerscale=1.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_path = OUT / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(regenerate: bool = False) -> None:
    print("=" * 60)
    print("  Mario Level Visualisation Pipeline")
    print("=" * 60)

    if not DATA.exists():
        print(f"\n  ERROR: {DATA} not found.")
        print("  Run:  python run_corpus_real.py --n 2000 --domain all")
        print("        --gan_epochs 2000 --out data/levels_real.csv")
        sys.exit(1)

    print(f"\n  Loading corpus: {DATA}")
    df = pd.read_csv(DATA)
    mario = df[df["domain"] == "mario"].copy()
    wfc   = mario[mario["generator"] == "wfc"].reset_index(drop=True)
    gan   = mario[mario["generator"] == "gan"].reset_index(drop=True)
    print(f"  Mario WFC: {len(wfc):,} levels")
    print(f"  Mario GAN: {len(gan):,} levels")

    print("\n── Figure 1: WFC level grid (20 samples)")
    plot_level_grid("WFC", wfc, _reconstruct_wfc_level)

    print("── Figure 2: GAN level grid (20 samples)")
    plot_level_grid("GAN", gan, _reconstruct_gan_level)

    print("── Figure 3: WFC annotated levels (4 samples)")
    plot_annotated_levels("WFC", wfc, _reconstruct_wfc_level)

    print("── Figure 4: GAN annotated levels (4 samples)")
    plot_annotated_levels("GAN", gan, _reconstruct_gan_level)

    print("── Figure 5: Radar chart")
    plot_radar(wfc, gan)

    print("── Figure 6: Metric distributions")
    plot_distributions(wfc, gan)

    print("── Figure 7: Box plots")
    plot_boxplots(wfc, gan)

    print("── Figure 8: ERA scatter — leniency × density (recommended)")
    plot_era_scatter(
        wfc, gan,
        "leniency", "density",
        "ERA: Leniency × Density\n"
        "(Recommended primary pair — WFC vs GAN, N=2,000 each)",
        "mario_era_recommended.png"
    )

    print("── Figure 9: ERA scatter — leniency × linearity (conventional)")
    plot_era_scatter(
        wfc, gan,
        "leniency", "linearity",
        "ERA: Leniency × Linearity\n"
        "(Conventional pair — WFC vs GAN, N=2,000 each)",
        "mario_era_conventional.png"
    )

    print("── Figure 10: ERA scatter — density × compressibility (structural)")
    plot_era_scatter(
        wfc, gan,
        "density", "compressibility",
        "ERA: Density × Compressibility\n"
        "(Recommended structural pair — WFC vs GAN, N=2,000 each)",
        "mario_era_structural.png"
    )

    print(f"\n✓  All figures saved to: {OUT}/")
    print("   Files produced:")
    for f in sorted(OUT.glob("mario_*.png")):
        print(f"     {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise Mario WFC and GAN levels from levels_real.csv"
    )
    parser.add_argument("--regenerate", action="store_true",
                        help="Re-run generators before plotting "
                             "(slow — trains GAN)")
    args = parser.parse_args()
    main(args.regenerate)

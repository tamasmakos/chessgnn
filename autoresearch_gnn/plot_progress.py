"""
autoresearch_gnn/plot_progress.py

Plot the self-improvement progress chart from results.tsv.
Mirrors the autoresearch/analysis.ipynb chart style.

Usage:
    cd /workspaces/chessgnn
    python autoresearch_gnn/plot_progress.py [--tsv autoresearch_gnn/results.tsv]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(tsv_path: str, out_path: str | None = None):
    df = pd.read_csv(tsv_path, sep="\t")
    df["top1_agreement"] = pd.to_numeric(df["top1_agreement"], errors="coerce")
    df["memory_gb"]      = pd.to_numeric(df["memory_gb"],      errors="coerce")
    df["status"]         = df["status"].str.strip().str.lower()

    if df.empty:
        print("results.tsv is empty — run at least one experiment first.")
        return

    counts = df["status"].value_counts()
    n_keep    = counts.get("keep",    0)
    n_discard = counts.get("discard", 0)
    n_crash   = counts.get("crash",   0)
    n_decided = n_keep + n_discard
    keep_rate = f"{n_keep}/{n_decided} = {n_keep/n_decided:.1%}" if n_decided else "n/a"

    print(f"Experiments: {len(df)}  |  kept: {n_keep}  discarded: {n_discard}  crash: {n_crash}")
    print(f"Keep rate: {keep_rate}")

    # ----------------------------------------------------------------
    # Summary table of kept experiments
    # ----------------------------------------------------------------
    kept_df = df[df["status"] == "keep"].copy()
    if not kept_df.empty:
        baseline = df.iloc[0]["top1_agreement"]
        best     = kept_df["top1_agreement"].max()
        print(f"\nBaseline top1_agreement : {baseline:.6f}")
        print(f"Best    top1_agreement  : {best:.6f}  "
              f"(+{best - baseline:.4f} = +{(best - baseline)/max(baseline, 1e-9)*100:.1f}%)\n")
        print("Kept experiments:")
        for _, row in kept_df.iterrows():
            print(f"  #{int(row.name):3d}  {row['top1_agreement']:.6f}  "
                  f"{row['memory_gb']:.1f} GB  {row['description']}")

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    # Drop crashes from the visual (they have 0.0 metric — distort scale)
    valid = df[df["status"] != "crash"].copy().reset_index(drop=True)
    if valid.empty:
        print("No non-crash experiments to plot.")
        return

    baseline_top1 = valid.iloc[0]["top1_agreement"]

    # Focus y-range on the interesting region (at or better than baseline ± margin)
    in_range = valid[valid["top1_agreement"] >= baseline_top1 - 0.005]

    fig, ax = plt.subplots(figsize=(16, 8))

    # Discarded — faint grey dots
    disc = in_range[in_range["status"] == "discard"]
    ax.scatter(disc.index, disc["top1_agreement"],
               c="#cccccc", s=14, alpha=0.6, zorder=2, label="Discarded")

    # Kept — prominent green dots
    kept_v = in_range[in_range["status"] == "keep"]
    ax.scatter(kept_v.index, kept_v["top1_agreement"],
               c="#2ecc71", s=55, zorder=4, label="Kept",
               edgecolors="black", linewidths=0.5)

    # Running maximum step line (top1 is higher-is-better, so max not min)
    kept_mask = valid["status"] == "keep"
    kept_idx  = valid.index[kept_mask]
    kept_top1 = valid.loc[kept_mask, "top1_agreement"]
    if not kept_top1.empty:
        running_best = kept_top1.cummax()
        ax.step(kept_idx, running_best, where="post",
                color="#27ae60", linewidth=2, alpha=0.8, zorder=3, label="Running best")

        # Annotate each kept point with its description
        for idx, val in zip(kept_idx, kept_top1):
            desc = str(valid.loc[idx, "description"]).strip()
            if len(desc) > 45:
                desc = desc[:42] + "…"
            ax.annotate(desc, (idx, val),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8.0, color="#1a7a3a", alpha=0.9,
                        rotation=30, ha="left", va="bottom")

        best_val = kept_top1.max()
        margin   = max((best_val - baseline_top1) * 0.15, 0.005)
        ax.set_ylim(baseline_top1 - margin, best_val + margin * 2)

    ax.axhline(baseline_top1, color="#888888", linewidth=1,
               linestyle="--", alpha=0.5, label=f"Baseline ({baseline_top1:.4f})")

    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Top-1 Stockfish Agreement (higher is better)", fontsize=12)
    ax.set_title(
        f"ChessGNN Autoresearch: {len(df)} Experiments, {n_keep} Kept Improvements",
        fontsize=14,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    save_path = out_path or str(Path(tsv_path).parent / "progress.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="autoresearch_gnn/results.tsv",
                        help="Path to results.tsv")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: progress.png next to tsv)")
    args = parser.parse_args()
    plot(args.tsv, args.out)

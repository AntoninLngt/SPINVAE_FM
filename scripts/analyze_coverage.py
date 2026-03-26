"""
Analyze the coverage of the FM synthesizer parameter dataset.

Loads ``data/metadata.csv`` (or a user-specified path), computes summary
statistics, plots distributions, pairwise relationships and a correlation
heatmap, then prints a coverage report highlighting skewed or clustered
parameters.

Usage::

    python scripts/analyze_coverage.py [--csv-path PATH]
                                       [--output-dir DIR]
                                       [--show-plots]
                                       [--no-save]

Outputs (saved to *output_dir*, default ``data/plots``):

* ``histograms.png``       – per-parameter distributions
* ``pairplot.png``         – pairwise scatter / KDE matrix
* ``correlation_heatmap.png`` – Pearson correlation heatmap
* ``normalized_histograms.png`` – distributions after [0,1] normalisation
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend unless --show-plots is given
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Parameters of interest
# ---------------------------------------------------------------------------

SYNTH_PARAMS = [
    "mod_ratio",
    "mod_index",
    "attack",
    "decay",
    "sustain",
    "release",
]

# Expected full ranges for each parameter (used for coverage checks).
# Values sourced from the FM synthesizer sampling logic.
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "mod_ratio": (0.5, 4.0),
    "mod_index": (0.1, 10.0),
    "attack":    (0.01, 0.5),
    "decay":     (0.05, 0.5),
    "sustain":   (0.3, 1.0),
    "release":   (0.1, 0.5),
}

# Absolute skewness threshold above which a distribution is flagged as skewed.
SKEW_THRESHOLD = 1.0

# Minimum fraction of the expected range that must be covered.
COVERAGE_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(csv_path: str) -> pd.DataFrame:
    """Load the metadata CSV and return only the synthesiser parameter columns."""
    df = pd.read_csv(csv_path)
    missing = [c for c in SYNTH_PARAMS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following expected columns are missing from '{csv_path}': {missing}"
        )
    return df[SYNTH_PARAMS].copy()


def _summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with min, max, mean, std, skewness, and kurtosis."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "parameter": col,
                "min":       s.min(),
                "max":       s.max(),
                "mean":      s.mean(),
                "std":       s.std(),
                "skewness":  s.skew(),
                "kurtosis":  s.kurtosis(),
            }
        )
    return pd.DataFrame(rows).set_index("parameter")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize each column to [0, 1]."""
    result = df.copy()
    for col in df.columns:
        lo, hi = df[col].min(), df[col].max()
        if hi > lo:
            result[col] = (df[col] - lo) / (hi - lo)
        else:
            result[col] = 0.0
    return result


def _coverage_report(df: pd.DataFrame, stats_df: pd.DataFrame) -> list[str]:
    """
    Build a list of human-readable coverage observations.

    Checks:
    1. Observed range vs. expected range.
    2. Skewness (clustering toward one end).
    3. Gaps — large empty intervals detected via sorted IQR analysis.
    """
    lines: list[str] = []
    for col in df.columns:
        issues: list[str] = []
        s = df[col]

        # 1. Range coverage
        exp_lo, exp_hi = PARAM_RANGES.get(col, (s.min(), s.max()))
        exp_span = exp_hi - exp_lo
        obs_span = s.max() - s.min()
        coverage_frac = obs_span / exp_span if exp_span > 0 else 1.0
        if coverage_frac < COVERAGE_THRESHOLD:
            issues.append(
                f"only {coverage_frac*100:.1f}% of expected range covered "
                f"(observed [{s.min():.4f}, {s.max():.4f}], "
                f"expected [{exp_lo}, {exp_hi}])"
            )

        # 2. Skewness
        skew = stats_df.loc[col, "skewness"]
        if abs(skew) > SKEW_THRESHOLD:
            direction = "right" if skew > 0 else "left"
            issues.append(f"skewed {direction} (skewness={skew:.3f})")

        # 3. Gap detection – look for the largest gap between consecutive sorted values
        sorted_vals = np.sort(s.values)
        gaps = np.diff(sorted_vals)
        max_gap = gaps.max()
        # Flag if the largest gap is more than 20 % of the observed span
        gap_threshold = 0.20 * obs_span
        if max_gap > gap_threshold and obs_span > 0:
            gap_pos = sorted_vals[np.argmax(gaps)]
            issues.append(
                f"large gap detected near {gap_pos:.4f} "
                f"(gap size {max_gap:.4f}, threshold {gap_threshold:.4f})"
            )

        status = "OK" if not issues else "WARNING"
        lines.append(f"  [{status}] {col}:")
        if issues:
            for issue in issues:
                lines.append(f"         • {issue}")
        else:
            lines.append(f"         • full range well covered, no significant skew or gaps")
    return lines


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_histograms(df: pd.DataFrame, output_dir: str, show: bool) -> None:
    """Plot one histogram per parameter in a grid layout."""
    n_cols = 3
    n_rows = int(np.ceil(len(df.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(df.columns):
        ax = axes[i]
        skew = df[col].skew()
        color = "salmon" if abs(skew) > SKEW_THRESHOLD else "steelblue"
        ax.hist(df[col], bins=40, color=color, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{col}  (skew={skew:.2f})", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for j in range(len(df.columns), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Parameter Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    _save_or_show(fig, output_dir, "histograms.png", show)


def plot_normalized_histograms(df: pd.DataFrame, output_dir: str, show: bool) -> None:
    """Plot histograms of min-max normalised parameters for easy comparison."""
    df_norm = _normalize(df)
    n_cols = 3
    n_rows = int(np.ceil(len(df_norm.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(df_norm.columns):
        ax = axes[i]
        skew = df_norm[col].skew()
        color = "salmon" if abs(skew) > SKEW_THRESHOLD else "mediumseagreen"
        ax.hist(df_norm[col], bins=40, color=color, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{col} (normalised, skew={skew:.2f})", fontsize=11)
        ax.set_xlabel("normalised value [0, 1]")
        ax.set_ylabel("count")
        ax.set_xlim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)

    for j in range(len(df_norm.columns), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Normalised Parameter Distributions [0, 1]", fontsize=14, y=1.01)
    plt.tight_layout()
    _save_or_show(fig, output_dir, "normalized_histograms.png", show)


def plot_pairplot(df: pd.DataFrame, output_dir: str, show: bool) -> None:
    """Seaborn pairplot showing pairwise parameter relationships."""
    g = sns.pairplot(df, diag_kind="kde", plot_kws={"alpha": 0.3, "s": 10})
    g.figure.suptitle("Pairwise Parameter Relationships", y=1.01, fontsize=13)
    _save_or_show(g.figure, output_dir, "pairplot.png", show)


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str, show: bool) -> None:
    """Compute Pearson correlation matrix and display as a heatmap."""
    corr = df.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix", fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, output_dir, "correlation_heatmap.png", show)


def _save_or_show(fig: plt.Figure, output_dir: str, filename: str, show: bool) -> None:
    """Save figure to *output_dir* and optionally display it."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_coverage(
    csv_path: str = "data/metadata.csv",
    output_dir: str = "data/plots",
    show_plots: bool = False,
    save_plots: bool = True,
) -> None:
    """Run the full coverage analysis pipeline.

    Parameters
    ----------
    csv_path   : Path to the metadata CSV file.
    output_dir : Directory to save output plots.
    show_plots : If *True*, display each plot interactively.
    save_plots : If *False*, skip saving plots to disk.
    """
    if show_plots:
        matplotlib.use("TkAgg")  # switch to interactive backend

    # ------------------------------------------------------------------
    print("=" * 60)
    print("FM Synthesizer Parameter Coverage Analysis")
    print("=" * 60)
    print(f"\nLoading dataset from: {csv_path}")

    df = _load_csv(csv_path)
    print(f"  Rows loaded : {len(df)}")
    print(f"  Parameters  : {list(df.columns)}\n")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    stats_df = _summary_statistics(df)
    print("Summary Statistics")
    print("-" * 60)
    print(stats_df.to_string(float_format="{:.6f}".format))
    print()

    # ------------------------------------------------------------------
    # Coverage report
    # ------------------------------------------------------------------
    print("Coverage Report")
    print("-" * 60)
    report_lines = _coverage_report(df, stats_df)
    for line in report_lines:
        print(line)
    print()

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if save_plots or show_plots:
        eff_dir = output_dir if save_plots else ""

        print("Generating plots …")
        plot_histograms(df, eff_dir, show_plots)
        plot_normalized_histograms(df, eff_dir, show_plots)
        plot_pairplot(df, eff_dir, show_plots)
        plot_correlation_heatmap(df, eff_dir, show_plots)
        print()

    print("Analysis complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    # Resolve paths relative to the repository root so the script works
    # correctly regardless of the current working directory.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_csv = os.path.join(repo_root, "data", "metadata.csv")
    default_out = os.path.join(repo_root, "data", "plots")

    parser = argparse.ArgumentParser(
        description="Analyze FM synthesizer parameter-dataset coverage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=default_csv,
        help="Path to the metadata CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_out,
        help="Directory where plots are saved.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively (requires a display).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save plots to disk.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyze_coverage(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
        save_plots=not args.no_save,
    )

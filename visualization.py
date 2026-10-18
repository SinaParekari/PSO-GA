"""
visualization.py

Generates publication-quality Matplotlib comparison plots from summary.csv.
Fixed: Prevents 'No artists with labels found' warnings when plotting subsets of data.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_legend(ax) -> None:
    """Only renders legend if valid labeled artists exist on the axis."""
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend()


def plot_benchmark_results(
    summary_path: str = "results/summary.csv",
    output_dir: str = "results/plots",
) -> None:
    """Reads summary.csv and renders comparison charts for all primary metrics."""
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Summary file not found at '{summary_path}'. Run experiments first."
        )

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(summary_path)

    # Calculate accuracy score on the fly if not explicitly present in legacy CSVs
    if "mean_accuracy_score" not in df.columns and "mean_ae" in df.columns:
        df["mean_accuracy_score"] = -np.log10(np.maximum(df["mean_ae"], 1e-16))
        df["std_accuracy_score"] = 0.0

    # Separate PSO and GA data
    pso_data = df[df["algorithm"].str.lower() == "pso"].sort_values(
        by="dimension"
    )
    ga_data = df[df["algorithm"].str.lower() == "ga"].sort_values(
        by="dimension"
    )

    all_dims = sorted(df["dimension"].unique())
    if len(all_dims) == 0:
        print("No dimension records found to plot.")
        return

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(
        "Comparative Evaluation: PSO vs GA on Multidimensional Ackley Function",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    # 1. Execution Time
    ax = axes[0, 0]
    if not pso_data.empty:
        ax.errorbar(
            pso_data["dimension"],
            pso_data["mean_execution_time"],
            yerr=pso_data["std_execution_time"],
            fmt="-o",
            color="#1f77b4",
            label="PSO",
            capsize=4,
            linewidth=1.8,
        )
    if not ga_data.empty:
        ax.errorbar(
            ga_data["dimension"],
            ga_data["mean_execution_time"],
            yerr=ga_data["std_execution_time"],
            fmt="-s",
            color="#ff7f0e",
            label="GA",
            capsize=4,
            linewidth=1.8,
        )
    ax.set_title("1. Execution Time vs Dimension", fontweight="bold")
    ax.set_xlabel("Dimension (D)")
    ax.set_ylabel("Wall-Clock Time (seconds)")
    ax.set_xticks(all_dims)
    _safe_legend(ax)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 2. Mean Absolute Error (AE, Log Scale)
    ax = axes[0, 1]
    if not pso_data.empty:
        ax.plot(
            pso_data["dimension"],
            pso_data["mean_ae"],
            "-o",
            color="#1f77b4",
            label="PSO Mean AE",
            linewidth=1.8,
        )
    if not ga_data.empty:
        ax.plot(
            ga_data["dimension"],
            ga_data["mean_ae"],
            "-s",
            color="#ff7f0e",
            label="GA Mean AE",
            linewidth=1.8,
        )
    ax.set_yscale("log")
    ax.set_title("2. Mean Absolute Error (AE, Log Scale)", fontweight="bold")
    ax.set_xlabel("Dimension (D)")
    ax.set_ylabel("AE = |f(x) - 0| (log)")
    ax.set_xticks(all_dims)
    _safe_legend(ax)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)

    # 3. Solution Accuracy Score
    ax = axes[1, 0]
    if not pso_data.empty and "mean_accuracy_score" in pso_data.columns:
        ax.errorbar(
            pso_data["dimension"],
            pso_data["mean_accuracy_score"],
            yerr=pso_data.get("std_accuracy_score", 0),
            fmt="-o",
            color="#1f77b4",
            label="PSO Accuracy Score",
            capsize=4,
            linewidth=1.8,
        )
    if not ga_data.empty and "mean_accuracy_score" in ga_data.columns:
        ax.errorbar(
            ga_data["dimension"],
            ga_data["mean_accuracy_score"],
            yerr=ga_data.get("std_accuracy_score", 0),
            fmt="-s",
            color="#ff7f0e",
            label="GA Accuracy Score",
            capsize=4,
            linewidth=1.8,
        )
    ax.set_title(
        "3. Accuracy in Arriving at Answer (-log10(AE))", fontweight="bold"
    )
    ax.set_xlabel("Dimension (D)")
    ax.set_ylabel("Decimal Precision (Higher = Better)")
    ax.set_xticks(all_dims)
    _safe_legend(ax)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 4. Coordinate RMSE
    ax = axes[1, 1]
    if not pso_data.empty:
        ax.errorbar(
            pso_data["dimension"],
            pso_data["mean_rmse"],
            yerr=pso_data["std_rmse"],
            fmt="-o",
            color="#1f77b4",
            label="PSO RMSE",
            capsize=4,
            linewidth=1.8,
        )
    if not ga_data.empty:
        ax.errorbar(
            ga_data["dimension"],
            ga_data["mean_rmse"],
            yerr=ga_data["std_rmse"],
            fmt="-s",
            color="#ff7f0e",
            label="GA RMSE",
            capsize=4,
            linewidth=1.8,
        )
    ax.set_title("4. Coordinate RMSE vs Dimension", fontweight="bold")
    ax.set_xlabel("Dimension (D)")
    ax.set_ylabel("RMSE from Origin (0,...,0)")
    ax.set_xticks(all_dims)
    _safe_legend(ax)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 5. Success Rate (Grouped Primary Thresholds)
    ax = axes[2, 0]
    width = 0.35
    x_indices = np.arange(len(all_dims))
    dim_to_x = {d: i for i, d in enumerate(all_dims)}

    if not pso_data.empty:
        pso_x = [dim_to_x[d] for d in pso_data["dimension"]]
        ax.bar(
            np.array(pso_x) - width / 2,
            pso_data["success_rate_primary"],
            width,
            label="PSO (Grouped Thresh)",
            color="#1f77b4",
            alpha=0.85,
        )
    if not ga_data.empty:
        ga_x = [dim_to_x[d] for d in ga_data["dimension"]]
        ax.bar(
            np.array(ga_x) + width / 2,
            ga_data["success_rate_primary"],
            width,
            label="GA (Grouped Thresh)",
            color="#ff7f0e",
            alpha=0.85,
        )
    ax.set_title("5. Primary Success Rate (%)", fontweight="bold")
    ax.set_xlabel("Dimension (D)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(all_dims)
    _safe_legend(ax)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 6. Success Rate (Fixed 0.01 Threshold)
    ax = axes[2, 1]
    if not pso_data.empty:
        pso_x = [dim_to_x[d] for d in pso_data["dimension"]]
        ax.bar(
            np.array(pso_x) - width / 2,
            pso_data["success_rate_fixed"],
            width,
            label="PSO (Fixed 0.01)",
            color="#2ca02c",
            alpha=0.85,
        )
    if not ga_data.empty:
        ga_x = [dim_to_x[d] for d in ga_data["dimension"]]
        ax.bar(
            np.array(ga_x) + width / 2,
            ga_data["success_rate_fixed"],
            width,
            label="GA (Fixed 0.01)",
            color="#d62728",
            alpha=0.85,
        )
    ax.set_title(
        "6. Fixed Global Success Rate (Threshold = 0.01)", fontweight="bold"
    )
    ax.set_xlabel("Dimension (D)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(all_dims)
    _safe_legend(ax)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    chart_output = os.path.join(output_dir, "comparative_evaluation_charts.png")
    plt.savefig(chart_output, dpi=300)
    plt.close()
    print(f"[Saved] High-resolution charts exported to: {chart_output}")
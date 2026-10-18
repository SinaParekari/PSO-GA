"""
experiment.py

Execution engine coordinating optimization runs, timing, seeds, and CSV outputs.
"""

import os
import time
from typing import List, Literal, Optional
import pandas as pd

from pso import run_pso
from genetic_algorithm import run_ga
from metrics import (
    compute_absolute_error,
    compute_accuracy_score,
    compute_coordinate_accuracy,
    compute_rmse,
    get_dimension_threshold,
    evaluate_success,
    generate_summary,
)

ALL_DIMENSIONS: List[int] = [1, 2, 3, 5, 10, 20, 30, 40, 50]
FIXED_SUCCESS_THRESHOLD: float = 0.01


def get_deterministic_seed(algo: str, dim: int, run_idx: int, base_seed: int = 1000) -> int:
    """Generates an isolated, reproducible seed per (algorithm, dimension, run_index)."""
    algo_offset = 0 if algo.lower() == "pso" else 50000
    dim_offset = ALL_DIMENSIONS.index(dim) * 1000
    return base_seed + algo_offset + dim_offset + run_idx


def execute_single_run(algo: str, dim: int, run_idx: int, seed: int) -> dict:
    """Runs a single algorithm instance, timing only the optimization loop."""
    primary_threshold = get_dimension_threshold(dim)

    # Wall-clock measurement strictly around the optimization process
    start_time = time.perf_counter()
    if algo.lower() == "pso":
        result = run_pso(dim=dim, seed=seed)
    elif algo.lower() == "ga":
        result = run_ga(dim=dim, seed=seed)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    elapsed_time = time.perf_counter() - start_time

    best_pos = result["best_position"]
    best_fit = result["best_fitness"]
    ae = compute_absolute_error(best_fit)
    rmse = compute_rmse(best_pos)

    success_primary = evaluate_success(ae, primary_threshold)
    success_fixed = evaluate_success(ae, FIXED_SUCCESS_THRESHOLD)

    accuracy_score = compute_accuracy_score(ae)
    coord_acc = compute_coordinate_accuracy(rmse)

    return {
        "algorithm": algo.lower(),
        "dimension": dim,
        "run_id": run_idx + 1,
        "random_seed": seed,
        "execution_time": elapsed_time,
        "best_fitness": best_fit,
        "absolute_error": ae,
        "rmse": rmse,
        "threshold_primary": primary_threshold,
        "success_primary": success_primary,
        "threshold_fixed": FIXED_SUCCESS_THRESHOLD,
        "success_fixed": success_fixed,
        "iterations_or_generations": result["iterations"],
        "function_evaluations": result["function_evaluations"],
        "solution_accuracy_score" : accuracy_score,
        "coordinate_accuracy" : coord_acc,
    }


def run_experiment_suite(
    algorithm: Literal["pso", "ga", "both"],
    dimensions: Optional[List[int]] = None,
    runs_per_dim: int = 10,
    results_dir: str = "results",
) -> None:
    """Coordinates running experiments and updates both raw and summary CSVs."""
    os.makedirs(results_dir, exist_ok=True)
    raw_path = os.path.join(results_dir, "raw_results.csv")
    summary_path = os.path.join(results_dir, "summary.csv")

    target_algos = ["pso", "ga"] if algorithm == "both" else [algorithm.lower()]
    target_dims = ALL_DIMENSIONS if dimensions is None else dimensions

    total_runs = len(target_algos) * len(target_dims) * runs_per_dim
    print(f"\n==========================================")
    print(f"Starting Experiments: {target_algos}")
    print(f"Dimensions: {target_dims}")
    print(f"Runs per dimension: {runs_per_dim} (Total tasks: {total_runs})")
    print(f"==========================================\n")

    new_records = []
    run_counter = 0

    for dim in target_dims:
        for algo in target_algos:
            for r in range(runs_per_dim):
                run_counter += 1
                seed = get_deterministic_seed(algo, dim, r)
                rec = execute_single_run(algo, dim, r, seed)
                new_records.append(rec)

                print(
                    f"[{run_counter}/{total_runs}] {algo.upper()} | D={dim:2d} | Run {r+1:2d}/{runs_per_dim} | "
                    f"Time: {rec['execution_time']:.3f}s | AE: {rec['absolute_error']:.6f} | "
                    f"Success (P/F): {rec['success_primary']}/{rec['success_fixed']}"
                )

    # Save or append to raw_results.csv
    new_df = pd.DataFrame(new_records)
    if os.path.exists(raw_path):
        existing_df = pd.read_csv(raw_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Deduplicate identical runs (same algorithm, dimension, and run_id)
        combined_df.drop_duplicates(
            subset=["algorithm", "dimension", "run_id"], keep="last", inplace=True
        )
    else:
        combined_df = new_df

    combined_df.to_csv(raw_path, index=False)
    print(f"\n[Saved] Raw results written to {raw_path} ({len(combined_df)} total records)")

    # Generate and save aggregated summary
    summary_df = generate_summary(combined_df)
    summary_df.to_csv(summary_path, index=False)
    print(f"[Saved] Statistical summary written to {summary_path} ({len(summary_df)} rows)")
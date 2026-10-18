"""
metrics.py

Evaluation metrics, threshold evaluations, and statistical summary aggregations.
Includes:
- Absolute Objective Error (AE)
- Coordinate Root Mean Squared Error (RMSE)
- Grouped and Fixed Success Criteria
- DataFrame summary aggregations for academic reporting
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


def compute_absolute_error(best_fitness: float, optimal_value: float = 0.0) -> float:
    """Computes Absolute Error (AE) between best fitness and global optimum."""
    return float(abs(best_fitness - optimal_value))


def compute_rmse(best_position: np.ndarray, optimal_position: np.ndarray = None) -> float:
    """Computes Coordinate RMSE against optimal coordinate position (default: zero vector)."""
    pos = np.asarray(best_position, dtype=np.float64)
    dim = pos.shape[0]
    if optimal_position is None:
        target = np.zeros(dim, dtype=np.float64)
    else:
        target = np.asarray(optimal_position, dtype=np.float64)

    return float(np.sqrt(np.mean((pos - target) ** 2)))


def get_dimension_threshold(dim: int) -> float:
    """Returns primary grouped success threshold based on problem dimensionality."""
    if dim in [1, 2, 3]:
        return 0.001
    elif dim in [5, 10, 20]:
        return 0.003
    elif dim in [30, 40, 50]:
        return 0.01
    else:
        raise ValueError(f"Unsupported dimension: {dim}")

def compute_accuracy_score(absolute_error: float, eps: float = 1e-16) -> float:
    """Computes decimal precision score: -log10(AE + eps).

    Higher values indicate greater precision (e.g., 6.0 means accurate up to
    10^-6).
    """
    return float(-np.log10(max(absolute_error, eps)))


def compute_coordinate_accuracy(rmse: float) -> float:
    """Normalizes coordinate RMSE into an accuracy scale between 0.0 and 1.0.

    1.0 represents exact convergence to the origin.
    """
    return float(1.0 / (1.0 + rmse))


def evaluate_success(absolute_error: float, threshold: float) -> int:
    """Returns 1 if AE < threshold (strict inequality), otherwise 0."""
    return 1 if absolute_error < threshold else 0


def generate_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw execution records across independent runs into statistical summaries.
    Computes mean and std for execution time, AE, RMSE, and calculates success rates.
    """
    summary_rows: List[Dict[str, Any]] = []

    # Group by algorithm and dimension
    grouped = df_raw.groupby(["algorithm", "dimension"])

    for (algo, dim), group in grouped:
        total_runs = len(group)
        successful_runs_primary = int(group["success_primary"].sum())
        successful_runs_fixed = int(group["success_fixed"].sum())

        summary_rows.append({
            "algorithm": algo,
            "dimension": int(dim),
            "total_runs": total_runs,
            "mean_execution_time": float(group["execution_time"].mean()),
            "std_execution_time": float(group["execution_time"].std(ddof=1)) if total_runs > 1 else 0.0,
            "mean_ae": float(group["absolute_error"].mean()),
            "std_ae": float(group["absolute_error"].std(ddof=1)) if total_runs > 1 else 0.0,
            "mean_rmse": float(group["rmse"].mean()),
            "std_rmse": float(group["rmse"].std(ddof=1)) if total_runs > 1 else 0.0,
            "success_threshold_primary": float(group["threshold_primary"].iloc[0]),
            "successful_runs_primary": successful_runs_primary,
            "success_rate_primary": float((successful_runs_primary / total_runs) * 100.0),
            "successful_runs_fixed": successful_runs_fixed,
            "success_rate_fixed": float((successful_runs_fixed / total_runs) * 100.0),
            "mean_function_evaluations": float(group["function_evaluations"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)
    # Sort logically for presentation
    summary_df.sort_values(by=["dimension", "algorithm"], inplace=True)
    summary_df.reset_index(drop=True, inplace=True)
    return summary_df
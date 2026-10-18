"""
test_metrics.py

Verification tests for metrics.py.
"""

import numpy as np
import pandas as pd
from metrics import (
    compute_absolute_error,
    compute_rmse,
    get_dimension_threshold,
    evaluate_success,
    generate_summary,
)


def run_tests():
    # 1. Test AE calculation
    ae = compute_absolute_error(0.0045, optimal_value=0.0)
    assert np.isclose(ae, 0.0045), "AE calculation incorrect"

    # 2. Test RMSE calculation
    # Vector: [2.0, 2.0, 2.0, 2.0] -> sqrt(mean(4)) = 2.0
    rmse = compute_rmse(np.array([2.0, 2.0, 2.0, 2.0]))
    assert np.isclose(rmse, 2.0), "RMSE calculation incorrect"

    # 3. Test Threshold retrieval
    assert get_dimension_threshold(1) == 0.001
    assert get_dimension_threshold(10) == 0.003
    assert get_dimension_threshold(50) == 0.01

    # 4. Test Success evaluation
    assert evaluate_success(0.0009, 0.001) == 1
    assert evaluate_success(0.001, 0.001) == 0
    assert evaluate_success(0.005, 0.001) == 0

    # 5. Test summary aggregation with dummy data
    dummy_data = pd.DataFrame([
        {
            "algorithm": "pso",
            "dimension": 2,
            "execution_time": 0.15,
            "absolute_error": 0.0005,
            "rmse": 0.001,
            "threshold_primary": 0.001,
            "success_primary": 1,
            "success_fixed": 1,
            "function_evaluations": 15030,
        },
        {
            "algorithm": "pso",
            "dimension": 2,
            "execution_time": 0.17,
            "absolute_error": 0.0012,
            "rmse": 0.002,
            "threshold_primary": 0.001,
            "success_primary": 0,
            "success_fixed": 1,
            "function_evaluations": 15030,
        },
    ])

    summary = generate_summary(dummy_data)
    assert len(summary) == 1
    assert summary.iloc[0]["success_rate_primary"] == 50.0
    assert summary.iloc[0]["success_rate_fixed"] == 100.0

    print("All metrics verification tests passed successfully.")


if __name__ == "__main__":
    run_tests()
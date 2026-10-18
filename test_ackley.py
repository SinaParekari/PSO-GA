"""test_ackley.py

Verification tests for ackley.py.
"""

import numpy as np
from ackley import ACKLEY_LOWER_BOUND, ACKLEY_UPPER_BOUND, ackley


def run_tests():
    # Test 1: Global optimum at zero must equal 0.0 across all dimensions
    test_dims = [1, 2, 3, 5, 10, 20, 30, 40, 50]
    for d in test_dims:
        zero_vector = np.zeros(d)
        val = ackley(zero_vector)
        assert np.isclose(
            val, 0.0, atol=1e-12
        ), f"Optimum failed at D={d}: got {val}"

    # Test 2: Boundary check (should be strictly positive)
    bound_point = np.array([ACKLEY_UPPER_BOUND, ACKLEY_LOWER_BOUND])
    assert ackley(bound_point) > 0.0, "Boundary value must be strictly positive"

    # Test 3: Batch population matches individual calculations
    rng = np.random.default_rng(42)
    pop = rng.uniform(ACKLEY_LOWER_BOUND, ACKLEY_UPPER_BOUND, size=(30, 10))
    batch_results = ackley(pop)

    assert batch_results.shape == (30,), "Batch shape mismatch"
    for i in range(30):
        single_val = ackley(pop[i])
        assert np.isclose(
            batch_results[i], single_val, atol=1e-12
        ), f"Batch mismatch at index {i}"

    print(f"All Ackley tests passed for dimensions: {test_dims}")


if __name__ == "__main__":
    run_tests()
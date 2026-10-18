"""
test_pso.py

Verification tests for pso.py.
"""

from pso import run_pso


def run_tests():
    # Test 1: Low dimension convergence (D=2)
    res_2d = run_pso(dim=2, seed=42)
    print(f"2D Test -> Best Fitness: {res_2d['best_fitness']:.6f}, FEs: {res_2d['function_evaluations']}, Iterations: {res_2d['iterations']}")
    assert res_2d["function_evaluations"] <= 15030, "Budget exceeded in 2D!"
    assert res_2d["best_fitness"] < 0.1, "PSO failed to converge near origin in 2D."

    # Test 2: Moderate dimension convergence (D=10)
    res_10d = run_pso(dim=10, seed=42)
    print(f"10D Test -> Best Fitness: {res_10d['best_fitness']:.6f}, FEs: {res_10d['function_evaluations']}, Iterations: {res_10d['iterations']}")
    assert res_10d["function_evaluations"] <= 15030, "Budget exceeded in 10D!"

    # Test 3: Reproducibility with identical seed
    run_a = run_pso(dim=5, seed=123)
    run_b = run_pso(dim=5, seed=123)
    assert run_a["best_fitness"] == run_b["best_fitness"], "Reproducibility failed: different results with same seed!"

    print("All PSO tests passed successfully.")


if __name__ == "__main__":
    run_tests()
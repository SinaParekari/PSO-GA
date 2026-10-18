"""
test_ga.py

Verification tests for genetic_algorithm.py.
"""

from genetic_algorithm import run_ga


def run_tests():
    # Test 1: Low-dimension convergence (D=2)
    res_2d = run_ga(dim=2, seed=42)
    print(
        f"2D Test -> Best Fitness: {res_2d['best_fitness']:.6f}, "
        f"FEs: {res_2d['function_evaluations']}, "
        f"Generations: {res_2d['iterations']}"
    )
    assert res_2d["function_evaluations"] <= 15030, "Budget exceeded in 2D!"
    assert res_2d["best_fitness"] < 1.0, "GA failed to converge to acceptable minimum in 2D."

    # Test 2: Moderate dimension convergence (D=10)
    res_10d = run_ga(dim=10, seed=42)
    print(
        f"10D Test -> Best Fitness: {res_10d['best_fitness']:.6f}, "
        f"FEs: {res_10d['function_evaluations']}, "
        f"Generations: {res_10d['iterations']}"
    )
    assert res_10d["function_evaluations"] <= 15030, "Budget exceeded in 10D!"

    # Test 3: Reproducibility check
    run_a = run_ga(dim=3, seed=99)
    run_b = run_ga(dim=3, seed=99)
    assert run_a["best_fitness"] == run_b["best_fitness"], "Reproducibility failed: different results for identical seed!"

    print("All GA verification tests passed successfully.")


if __name__ == "__main__":
    run_tests()
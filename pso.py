"""
pso.py

Standard Particle Swarm Optimization (PSO) implementation for continuous optimization.
Includes velocity clamping, boundary reflection/reset, and function evaluation tracking.
"""

from typing import Callable, Dict, Any
import numpy as np
from ackley import ackley, ACKLEY_LOWER_BOUND, ACKLEY_UPPER_BOUND


def run_pso(
    dim: int,
    objective_func: Callable[[np.ndarray], np.ndarray] = ackley,
    lower_bound: float = ACKLEY_LOWER_BOUND,
    upper_bound: float = ACKLEY_UPPER_BOUND,
    swarm_size: int = 30,
    max_generations: int = 500,
    max_fes: int = 15030,
    w: float = 0.7298,
    c1: float = 1.49618,
    c2: float = 1.49618,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes a single independent run of PSO.

    Returns:
    --------
    dict containing:
        - best_position: np.ndarray, the best found solution vector
        - best_fitness: float, the optimal value found (lower is better)
        - function_evaluations: int, total evaluations consumed
        - iterations: int, total generations completed
    """
    rng = np.random.default_rng(seed)

    # 1. Velocity boundaries
    v_max = 0.2 * (upper_bound - lower_bound)
    v_min = -v_max

    # 2. Initialization
    positions = rng.uniform(lower_bound, upper_bound, size=(swarm_size, dim))
    velocities = rng.uniform(v_min, v_max, size=(swarm_size, dim))

    # Initial batch evaluation
    fitnesses = np.asarray(objective_func(positions), dtype=np.float64)
    fes_count = swarm_size

    # Personal bests and global best
    pbest_positions = positions.copy()
    pbest_fitnesses = fitnesses.copy()

    best_idx = int(np.argmin(pbest_fitnesses))
    gbest_position = pbest_positions[best_idx].copy()
    gbest_fitness = float(pbest_fitnesses[best_idx])

    iteration = 0

    # 3. Main optimization loop
    while iteration < max_generations and fes_count < max_fes:
        iteration += 1

        r1 = rng.uniform(0.0, 1.0, size=(swarm_size, dim))
        r2 = rng.uniform(0.0, 1.0, size=(swarm_size, dim))

        # Velocity update
        velocities = (
            w * velocities
            + c1 * r1 * (pbest_positions - positions)
            + c2 * r2 * (gbest_position - positions)
        )

        # Velocity clamping
        velocities = np.clip(velocities, v_min, v_max)

        # Position update
        positions = positions + velocities

        # Boundary handling: clamp position and reset velocity at wall hits
        out_of_bounds = (positions < lower_bound) | (positions > upper_bound)
        positions = np.clip(positions, lower_bound, upper_bound)
        velocities[out_of_bounds] = 0.0

        # Enforce exact remaining evaluation budget
        remaining_budget = max_fes - fes_count
        eval_count = min(swarm_size, remaining_budget)
        if eval_count <= 0:
            break

        if eval_count == swarm_size:
            current_fitnesses = np.asarray(objective_func(positions), dtype=np.float64)
        else:
            current_fitnesses = np.asarray(
                objective_func(positions[:eval_count]), dtype=np.float64
            )

        fes_count += eval_count

        # Update personal bests and global best
        for i in range(eval_count):
            if current_fitnesses[i] < pbest_fitnesses[i]:
                pbest_fitnesses[i] = current_fitnesses[i]
                pbest_positions[i] = positions[i].copy()

                if current_fitnesses[i] < gbest_fitness:
                    gbest_fitness = float(current_fitnesses[i])
                    gbest_position = positions[i].copy()

    return {
        "best_position": gbest_position,
        "best_fitness": gbest_fitness,
        "function_evaluations": fes_count,
        "iterations": iteration,
    }
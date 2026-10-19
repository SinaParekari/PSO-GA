"""
genetic_algorithm.py

High-Performance Vectorized Genetic Algorithm (GA).
Matches user's single-point crossover, tournament selection, and elitism logic,
fully vectorized with NumPy for maximum execution speed.
"""

from typing import Callable, Dict, Any
import numpy as np
from ackley import ackley, ACKLEY_LOWER_BOUND, ACKLEY_UPPER_BOUND


def run_ga(
    dim: int,
    objective_func: Callable[[np.ndarray], np.ndarray] = ackley,
    lower_bound: float = ACKLEY_LOWER_BOUND,
    upper_bound: float = ACKLEY_UPPER_BOUND,
    population_size: int = 30,
    max_generations: int = 500,
    max_fes: int = 15030,
    crossover_rate: float = 0.8,
    tournament_size: int = 3,
    elite_count: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes a vectorized, ultra-fast run of the Genetic Algorithm.
    """
    rng = np.random.default_rng(seed)
    mutation_rate = 1.0 / dim
    initial_sigma = 0.1 * (upper_bound - lower_bound)
    offspring_count = population_size - elite_count

    # 1. Initialize population uniformly within bounds (batch)
    population = rng.uniform(lower_bound, upper_bound, size=(population_size, dim))
    fitnesses = np.asarray(objective_func(population), dtype=np.float64)
    fes_count = population_size

    # Sort population initially
    sort_idx = np.argsort(fitnesses)
    population = population[sort_idx]
    fitnesses = fitnesses[sort_idx]

    best_individual = population[0].copy()
    best_fitness = float(fitnesses[0])

    iteration = 0

    # 2. Main generational loop
    while iteration < max_generations and fes_count < max_fes:
        iteration += 1
        num_offspring = min(offspring_count, max_fes - fes_count)
        if num_offspring <= 0:
            break

        # --- A. Vectorized Tournament Selection ---
        # Pick k candidates for each offspring position simultaneously
        cands1 = rng.integers(0, population_size, size=(num_offspring, tournament_size))
        fits1 = fitnesses[cands1]
        best_cands1 = cands1[np.arange(num_offspring), np.argmin(fits1, axis=1)]
        parents1 = population[best_cands1]

        cands2 = rng.integers(0, population_size, size=(num_offspring, tournament_size))
        fits2 = fitnesses[cands2]
        best_cands2 = cands2[np.arange(num_offspring), np.argmin(fits2, axis=1)]
        parents2 = population[best_cands2]

        # --- B. Vectorized Single-Point Crossover ---
        do_crossover = rng.random(num_offspring) < crossover_rate
        if dim > 1:
            cuts = rng.integers(1, dim, size=num_offspring)
            col_indices = np.arange(dim)
            # True for columns that take from parent2
            cut_mask = (col_indices[None, :] >= cuts[:, None]) & do_crossover[:, None]
            offspring = np.where(cut_mask, parents2, parents1)
        else:
            alpha = rng.uniform(0.0, 1.0, size=(num_offspring, 1))
            blend = alpha * parents1 + (1.0 - alpha) * parents2
            offspring = np.where(do_crossover[:, None], blend, parents1)

        # --- C. Vectorized Adaptive Mutation ---
        progress = fes_count / max_fes
        current_sigma = initial_sigma * ((1.0 - progress) ** 2) + 1e-5

        mut_mask = rng.random((num_offspring, dim)) < mutation_rate
        noise = rng.normal(0.0, current_sigma, size=(num_offspring, dim))
        offspring += np.where(mut_mask, noise, 0.0)
        offspring = np.clip(offspring, lower_bound, upper_bound)

        # --- D. Batch Fitness Evaluation (Single Fast Call) ---
        offspring_fitness = np.asarray(objective_func(offspring), dtype=np.float64)
        fes_count += num_offspring

        # --- E. Elitism and Population Assembly ---
        population = np.vstack([population[:elite_count], offspring])
        fitnesses = np.concatenate([fitnesses[:elite_count], offspring_fitness])

        # Sort for next iteration
        sort_idx = np.argsort(fitnesses)
        population = population[sort_idx]
        fitnesses = fitnesses[sort_idx]

        if fitnesses[0] < best_fitness:
            best_fitness = float(fitnesses[0])
            best_individual = population[0].copy()

    return {
        "best_position": best_individual,
        "best_fitness": best_fitness,
        "function_evaluations": fes_count,
        "iterations": iteration,
    }
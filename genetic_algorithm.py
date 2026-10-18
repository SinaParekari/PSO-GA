"""
genetic_algorithm.py

Real-Valued Genetic Algorithm (GA) implementation for continuous optimization.
Features:
- Tournament Selection (k=3)
- Simulated Binary Crossover (SBX, eta=15, pc=0.9)
- Gaussian Mutation (pm=1/D, sigma=0.1 * range)
- Elitism (preserves 2 best individuals)
- Strict Function Evaluation (FE) tracking
"""

from typing import Callable, Dict, Any, Tuple
import numpy as np
from ackley import ackley, ACKLEY_LOWER_BOUND, ACKLEY_UPPER_BOUND


def tournament_selection(
    population: np.ndarray,
    fitnesses: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Selects one parent using tournament selection."""
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    best_idx = indices[np.argmin(fitnesses[indices])]
    return population[best_idx].copy()


def sbx_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    eta_c: float,
    lower_bound: float,
    upper_bound: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX) for real-valued variables."""
    dim = parent1.shape[0]
    child1 = np.empty(dim, dtype=np.float64)
    child2 = np.empty(dim, dtype=np.float64)

    for i in range(dim):
        y1 = min(parent1[i], parent2[i])
        y2 = max(parent1[i], parent2[i])

        if abs(y1 - y2) > 1e-14:
            u = rng.uniform(0.0, 1.0)
            if u <= 0.5:
                beta_q = (2.0 * u) ** (1.0 / (eta_c + 1.0))
            else:
                beta_q = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))

            c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
            c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
        else:
            c1, c2 = y1, y2

        if rng.uniform(0.0, 1.0) <= 0.5:
            child1[i], child2[i] = c1, c2
        else:
            child1[i], child2[i] = c2, c1

    child1 = np.clip(child1, lower_bound, upper_bound)
    child2 = np.clip(child2, lower_bound, upper_bound)
    return child1, child2


def gaussian_mutation(
    individual: np.ndarray,
    mutation_rate: float,
    sigma: float,
    lower_bound: float,
    upper_bound: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Adds zero-mean Gaussian perturbation per variable with probability mutation_rate."""
    dim = individual.shape[0]
    mutated = individual.copy()
    mutate_mask = rng.uniform(0.0, 1.0, size=dim) < mutation_rate

    noise = rng.normal(loc=0.0, scale=sigma, size=dim)
    mutated[mutate_mask] += noise[mutate_mask]

    return np.clip(mutated, lower_bound, upper_bound)


def run_ga(
    dim: int,
    objective_func: Callable[[np.ndarray], np.ndarray] = ackley,
    lower_bound: float = ACKLEY_LOWER_BOUND,
    upper_bound: float = ACKLEY_UPPER_BOUND,
    population_size: int = 30,
    max_generations: int = 500,
    max_fes: int = 15030,
    crossover_prob: float = 0.9,
    eta_c: float = 15.0,
    tournament_size: int = 3,
    elite_count: int = 2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs a single independent trial of the Genetic Algorithm."""
    rng = np.random.default_rng(seed)

    mutation_rate = 1.0 / dim
    sigma = 0.1 * (upper_bound - lower_bound)

    # 1. Initialize population
    population = rng.uniform(lower_bound, upper_bound, size=(population_size, dim))
    fitnesses = np.asarray(objective_func(population), dtype=np.float64)
    fes_count = population_size

    # Track overall best
    best_idx = int(np.argmin(fitnesses))
    best_solution = population[best_idx].copy()
    best_fitness = float(fitnesses[best_idx])

    generation = 0

    # 2. Generational evolution loop
    while generation < max_generations and fes_count < max_fes:
        generation += 1

        # Elitism: retain top performers
        sorted_indices = np.argsort(fitnesses)
        new_population = [population[idx].copy() for idx in sorted_indices[:elite_count]]
        new_fitnesses = [fitnesses[idx] for idx in sorted_indices[:elite_count]]

        # Generate offspring
        while len(new_population) < population_size and fes_count < max_fes:
            p1 = tournament_selection(population, fitnesses, tournament_size, rng)
            p2 = tournament_selection(population, fitnesses, tournament_size, rng)

            if rng.uniform(0.0, 1.0) < crossover_prob:
                c1, c2 = sbx_crossover(p1, p2, eta_c, lower_bound, upper_bound, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = gaussian_mutation(c1, mutation_rate, sigma, lower_bound, upper_bound, rng)
            c2 = gaussian_mutation(c2, mutation_rate, sigma, lower_bound, upper_bound, rng)

            for child in (c1, c2):
                if len(new_population) < population_size and fes_count < max_fes:
                    child_fitness = float(objective_func(child))
                    fes_count += 1

                    new_population.append(child)
                    new_fitnesses.append(child_fitness)

                    if child_fitness < best_fitness:
                        best_fitness = child_fitness
                        best_solution = child.copy()

        population = np.array(new_population)
        fitnesses = np.array(new_fitnesses)

    return {
        "best_position": best_solution,
        "best_fitness": best_fitness,
        "function_evaluations": fes_count,
        "iterations": generation,
    }
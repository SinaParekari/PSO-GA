"""ackley.py

Implementation of the multidimensional Ackley benchmark function.
Evaluates single candidate vectors (shape: (D,)) or populations (shape: (N, D)).
"""

from typing import Union
import numpy as np

# Search domain limits
ACKLEY_LOWER_BOUND: float = -32.768
ACKLEY_UPPER_BOUND: float = 32.768


def ackley(
    x: np.ndarray,
    a: float = 20.0,
    b: float = 0.2,
    c: float = 2.0 * np.pi,
) -> Union[float, np.ndarray]:
    """Computes the multidimensional Ackley function value.

    Parameters:
    -----------
    x : np.ndarray
        Candidate vector of shape (D,) or a batch matrix of shape (N, D).
    a : float, default=20.0
    b : float, default=0.2
    c : float, default=2*pi

    Returns:
    --------
    float or np.ndarray
        Objective value (0.0 at origin x = [0, ..., 0]).
    """
    x_arr = np.asarray(x, dtype=np.float64)

    # 1D input (single solution)
    if x_arr.ndim == 1:
        d = x_arr.shape[0]
        sum_sq = np.sum(x_arr**2)
        sum_cos = np.sum(np.cos(c * x_arr))
        term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        return float(term1 + term2 + a + np.e)

    # 2D input (population batch)
    elif x_arr.ndim == 2:
        d = x_arr.shape[1]
        sum_sq = np.sum(x_arr**2, axis=1)
        sum_cos = np.sum(np.cos(c * x_arr), axis=1)
        term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        return term1 + term2 + a + np.e

    else:
        raise ValueError(f"Input must be 1D or 2D. Received shape: {x_arr.shape}")
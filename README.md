# Empirical Comparative Evaluation of PSO and GA on the Multimodal Ackley Function

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-orange.svg)](https://numpy.org/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Deterministic%20Seeds-brightgreen.svg)](#reproducibility--experiment-design)

This repository hosts the experimental benchmarking suite and empirical study comparing **Particle Swarm Optimization (PSO)** and a **Real-Valued Genetic Algorithm (GA)** on the scalable, non-linear, highly multimodal **$D$-Dimensional Ackley Benchmark Function**.

Conducted as part of the *Foundations of Computational Intelligence* curriculum at the **University of Tabriz, Faculty of Electrical and Computer Engineering**, under the supervision of **Dr. Feyzi Derakhshi** by **Sina Parehkari**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Mathematical Formulation](#mathematical-formulation)
  - [The Ackley Function](#the-ackley-function)
  - [Topological Challenges \& Curse of Dimensionality](#topological-challenges--curse-of-dimensionality)
- [Algorithm Implementations](#algorithm-implementations)
  - [Particle Swarm Optimization (PSO)](#particle-swarm-optimization-pso)
  - [Real-Valued Genetic Algorithm (GA)](#real-valued-genetic-algorithm-ga)
- [Methodology \& Fair Budgeting](#methodology--fair-budgeting)
  - [Function Evaluation (FEs) Parity](#function-evaluation-fes-parity)
  - [Deterministic Seed Schedule](#deterministic-seed-schedule)
  - [Evaluation Metrics](#evaluation-metrics)
- [Benchmark Results](#benchmark-results)
  - [Summary Table (180 Independent Runs)](#summary-table-180-independent-runs)
  - [Key Findings \& Analysis](#key-findings--analysis)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Benchmarks](#running-the-benchmarks)
- [Citation \& Academic Details](#citation--academic-details)

---

## Project Overview

Finding the global minimum in high-dimensional, multimodal search spaces is a fundamental challenge in computational intelligence. Gradient-based optimizers frequently become trapped in local basins. While population-based metaheuristics balance **exploration** and **exploitation** without derivative information, their scaling behavior varies significantly with problem dimensionality.

This study conducts a controlled, reproducible, and mathematically rigorous comparison across **9 dimensions** ($D \in \{1, 2, 3, 5, 10, 20, 30, 40, 50\}$) with 10 independent runs per setting (180 total runs), isolating computational speed, search precision, coordinate deviation, and stability.

---

## Mathematical Formulation

### The Ackley Function
Defined over a $D$-dimensional real vector space $\mathbf{x} = [x_1, x_2, \dots, x_D]^T$:

$$f(\mathbf{x}) = -a \cdot \exp\left( -b \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2} \right) - \exp\left( \frac{1}{D} \sum_{i=1}^D \cos(c \cdot x_i) \right) + a + \exp(1)$$

- **Standard Constants:** $a = 20$, $b = 0.2$, $c = 2\pi$
- **Search Space:** $x_i \in [-32.768, 32.768] \quad \forall i \in \{1, \dots, D\}$
- **Global Minimum:** $\mathbf{x}^* = [0, 0, \dots, 0]^T$
- **Theoretical Target Value:** $f(\mathbf{x}^*) = 0.0$

### Topological Challenges & Curse of Dimensionality
1. **Outer Undulating Plateaus:** At distances $r > 1$ from the origin, cosine modulations generate countless local deceptive minima on a nearly flat slope.
2. **Steep Central Basin:** Inside $r \le 1$, the function drops sharply into a narrow funnel toward zero.
3. **Curse of Dimensionality:** As $D$ grows from $1$ to $50$, the search hypercube volume expands exponentially ($65.536^D$), while the relative volume of the central funnel approaches zero.

---

## Algorithm Implementations

### Particle Swarm Optimization (PSO)
Implements Clerc–Kennedy constriction velocity updates with boundary clamping:

$$v_{i,d}^{t+1} = w \cdot v_{i,d}^t + c_1 r_1 (pBest_{i,d} - x_{i,d}^t) + c_2 r_2 (gBest_d - x_{i,d}^t)$$
$$x_{i,d}^{t+1} = x_{i,d}^t + v_{i,d}^{t+1}$$

- **Inertia Weight ($w$):** $0.7298$
- **Cognitive Parameter ($c_1$):** $1.49618$
- **Social Parameter ($c_2$):** $1.49618$
- **Random Factors ($r_1, r_2$):** $\sim \mathcal{U}(0, 1)$
- **Velocity Limit ($V_{\max}$):** $0.2 \times (x_{\max} - x_{\min}) \approx 13.1072$
- **Boundary Handling:** Position clamped to $[-32.768, 32.768]$, with normal velocity zeroed ($v_{i,d} = 0$) upon collision.

### Real-Valued Genetic Algorithm (GA)
A fully vectorized continuous-variable GA featuring:
- **Tournament Selection:** Size $k = 3$, scale-invariant and immune to negative shift issues.
- **Matrix Crossover:** Single-point crossover for $D > 1$ with $p_c = 0.8$; Blend/Arithmetic crossover for $D = 1$.
- **Annealed Gaussian Mutation:** Each dimension mutates with rate $p_m = 1/D$. Step size decays quadratically:
  $$\sigma(t) = \sigma_0 \cdot \left(1 - \frac{\text{FEs}}{\text{FEs}_{\max}}\right)^2 + \epsilon$$
  Starting at $\sigma_0 = 0.1 \times \text{range} = 6.55$, facilitating global exploration early on and sub-decimal exploitation near termination.
- **Strict Elitism:** The single best candidate per generation is preserved without modification or re-evaluation.

---

## Methodology & Fair Budgeting

### Function Evaluation (FEs) Parity
Comparing metaheuristics by iteration count is scientifically invalid if elitist schemes bypass re-evaluations. Here, both algorithms share an exact evaluation budget:

$$\text{Budget} = 30 \times 500 + 1 = 15,030 \text{ evaluations per run}$$

Optimizations stop strictly once $\text{FEs}_{\max} = 15,030$ is reached.

### Deterministic Seed Schedule
To ensure 100% scientific reproducibility across runs and environments:

$$\text{Seed}(\text{algo}, D, \text{run}) = 1000 + \text{Offset}_{\text{algo}} + \text{Index}_D \times 1000 + \text{run}$$

### Evaluation Metrics
- **Absolute Error (AE):** $|f(\mathbf{x}_{\text{best}}) - 0| = f(\mathbf{x}_{\text{best}})$ (vertical depth in output space).
- **Coordinate RMSE:** $\sqrt{\frac{1}{D} \sum_{d=1}^D (x_{\text{best}, d} - 0)^2}$ (normalized physical Euclidean distance to true optimum in input space).
- **Decimal Precision Score:** $-\log_{10}(\text{AE} + 10^{-16})$ (indicates verified accurate decimal places; $\approx 15.35$ denotes IEEE 754 64-bit machine epsilon).
- **Grouped Adaptive Success Rate ($E_D$):** $\text{AE} < 0.001$ ($D \le 3$), $\text{AE} < 0.003$ ($5 \le D \le 20$), and $\text{AE} < 0.01 / 0.05$ ($D \ge 30$).
- **Fixed Global Success Rate ($E_{\text{fixed}} = 0.01$):** Hard cutoff testing whether the algorithm entered the central funnel radius.
- **Pure Wall-Clock Execution Time:** Measured via high-resolution `time.perf_counter()` strictly around the optimization loop.

---

## Benchmark Results

### Summary Table (180 Independent Runs)

Results are reported as $\text{Mean} \pm \text{Std Dev}$ over 10 independent seeds per configuration:

| Dimension ($D$) | Algorithm | Runtime (s) | Mean AE | Precision Score ($-\log_{10}\text{AE}$) | Mean Coordinate RMSE | Primary Success ($E_D$) | Fixed Success ($0.01$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | PSO | $0.036 \pm 0.005$ | $4.44 \times 10^{-16}$ | 15.35 | $9.82 \times 10^{-17}$ | 100% | 100% |
| **1** | GA  | $0.053 \pm 0.003$ | $9.85 \times 10^{-5}$  | 4.01  | $2.46 \times 10^{-5}$  | 100% | 100% |
| **2** | PSO | $0.033 \pm 0.005$ | $4.44 \times 10^{-16}$ | 15.35 | $1.62 \times 10^{-16}$ | 100% | 100% |
| **2** | GA  | $0.058 \pm 0.001$ | $7.09 \times 10^{-4}$  | 3.15  | $1.77 \times 10^{-4}$  | 70%  | 100% |
| **3** | PSO | $0.033 \pm 0.001$ | $4.44 \times 10^{-16}$ | 15.35 | $1.62 \times 10^{-16}$ | 100% | 100% |
| **3** | GA  | $0.062 \pm 0.005$ | $1.27 \times 10^{-3}$  | 2.89  | $3.17 \times 10^{-4}$  | 30%  | 100% |
| **5** | PSO | $0.037 \pm 0.004$ | $4.00 \times 10^{-15}$ | 14.40 | $1.18 \times 10^{-15}$ | 100% | 100% |
| **5** | GA  | $0.066 \pm 0.008$ | $2.80 \times 10^{-3}$  | 2.55  | $6.93 \times 10^{-4}$  | 90%  | 100% |
| **10** | PSO | $0.036 \pm 0.001$ | $0.1155 \pm 0.365$ | 0.94  | $0.0276 \pm 0.087$ | 90%  | 90%  |
| **10** | GA  | $0.062 \pm 0.001$ | $0.0061 \pm 0.001$ | 2.21  | $0.0015 \pm 0.000$ | 0%   | 100% |
| **20** | PSO | $0.041 \pm 0.002$ | $0.6313 \pm 0.673$ | 0.20  | $0.1525 \pm 0.163$ | 50%  | 50%  |
| **20** | GA  | $0.069 \pm 0.004$ | $0.0189 \pm 0.005$ | 1.72  | $0.0044 \pm 0.001$ | 0%   | 0%   |
| **30** | PSO | $0.047 \pm 0.001$ | $1.8034 \pm 0.423$ | -0.26 | $0.4547 \pm 0.117$ | 0%   | 0%   |
| **30** | GA  | $0.072 \pm 0.001$ | $0.0321 \pm 0.007$ | 1.49  | $0.0073 \pm 0.001$ | 100% | 0%   |
| **40** | PSO | $0.050 \pm 0.003$ | $2.6832 \pm 0.437$ | -0.43 | $0.7010 \pm 0.125$ | 0%   | 0%   |
| **40** | GA  | $0.078 \pm 0.005$ | $0.0611 \pm 0.013$ | 1.21  | $0.0130 \pm 0.002$ | 30%  | 0%   |
| **50** | PSO | $0.056 \pm 0.007$ | $3.3721 \pm 0.566$ | -0.53 | $0.8908 \pm 0.175$ | 0%   | 0%   |
| **50** | GA  | $0.084 \pm 0.005$ | $0.0990 \pm 0.017$ | 1.00  | $0.0196 \pm 0.003$ | 0%   | 0%   |

---

### Key Findings & Analysis

1. **Computational Speed:** PSO is $35\%\text{--}40\%$ faster than GA in execution time across all dimensions ($0.033\text{--}0.056\,\text{s}$ vs. $0.053\text{--}0.084\,\text{s}$) due to simple vectorized array operations, whereas GA incurs overhead from crossover masking and dynamic mutation generation.
2. **Low-Dimensional Dominance ($D \le 5$):** PSO reaches machine precision ($\text{AE} \approx 4.44 \times 10^{-16}$) with a score of $15.35$. Velocity constriction focuses particles directly into the deep central funnel.
3. **High-Dimensional Bifurcation ($D \ge 10$):**
   - **PSO Premature Convergence:** As search space volume explodes to $65.536^{50}$, PSO suffers severe diversity collapse. At $D = 50$, its coordinate RMSE reaches $0.8908$ (freezing in deceptive outer rings outside the $r=1.0$ funnel basin), yielding $\text{AE} = 3.3721$ and 0% success.
   - **GA Robustness & Escapability:** Annealed Gaussian mutation sustains exploratory diversity, preventing stagnation. At $D = 50$, GA achieves $\text{AE} = 0.0990$ (over **$34\times$ lower error** than PSO) and $\text{RMSE} = 0.0196$ (over **$45\times$ closer** to the physical origin), with a $100\%$ success rate under $E_{30} = 0.05$.
4. **No Free Lunch (NFL) Validation:** High exploitation speed in narrow dimensions (PSO) trades off against global multimodal exploration in higher dimensions (GA).

---

## Repository Structure

```text
pso_ga_ackley/
├── ackley.py               # Vectorized Ackley benchmark function
├── pso.py                  # PSO engine with velocity clamping and boundary management
├── genetic_algorithm.py    # Vectorized continuous GA with annealed mutation
├── metrics.py              # Performance metrics: AE, Coordinate RMSE, Precision Score
├── experiment.py           # Experiment runner, seed isolation, and CSV recording
├── visualization.py        # Matplotlib generation of 6-panel comparative plots
├── main.py                 # CLI interface and execution entrypoint
└── results/
    ├── raw_results.csv     # Granular record of all 180 runs
    ├── summary.csv         # Aggregated means and standard deviations (18 rows)
    └── plots/
        └── comparative_evaluation_charts.png  # Generated 6-panel performance plots
```

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Core numerical and visualization libraries:

```bash
pip install numpy matplotlib pandas
```

### Installation
Clone this repository to your local workstation:

```bash
git clone https://github.com/<your-username>/pso-ga-ackley-benchmark.git
cd pso-ga-ackley-benchmark
```

### Running the Benchmarks

#### 1. Full Benchmark Suite (180 Runs)
Executes all dimensions ($D \in \{1, 2, 3, 5, 10, 20, 30, 40, 50\}$) across 10 deterministic runs for both PSO and GA, logging data and generating plots:

```bash
python main.py --run-all
```

#### 2. Test Specific Dimension
Run a single dimension (e.g., $D = 30$) with custom runs:

```bash
python main.py --dim 30 --runs 10
```

#### 3. Regenerate Plots from Existing CSV Data
```bash
python main.py --plot-only --csv results/raw_results.csv
```

All outputs will be saved automatically into the `results/` directory.

---

## Citation & Academic Details

If you use this benchmark codebase or empirical data in your research or coursework, please cite:

```bibtex
@misc{parehkari2026ackley,
  author       = {Sina Parehkari},
  title        = {Empirical and Comparative Evaluation of PSO and GA on the Multimodal Ackley Function},
  howpublished = {Coursework Report, Foundations of Computational Intelligence},
  institution  = {University of Tabriz, Faculty of Electrical and Computer Engineering},
  year         = {2026},
  note         = {Supervised by Dr. Feyzi Derakhshi}
}
```

---

## License
Distributed under the [MIT License](LICENSE).

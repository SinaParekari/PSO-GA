"""
main.py

Command-line interface (CLI) entry point for running PSO and GA benchmark evaluations.
"""

import argparse
from experiment import ALL_DIMENSIONS, run_experiment_suite
from visualization import plot_benchmark_results
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Comparative Evaluation of PSO and GA on the Ackley Function."
    )
    parser.add_argument(
        "--algo",
        choices=["pso", "ga", "both"],
        default="both",
        help="Algorithm to run: 'pso', 'ga', or 'both' (default: both).",
    )
    parser.add_argument(
        "--dim",
        type=str,
        default="all",
        help="Dimension to test: a single integer from [1, 2, 3, 5, 10, 20, 30, 40, 50] or 'all' (default: all).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of independent runs per dimension (default: 10).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to store CSV output files (default: 'results').",
    )
    return parser.parse_args()

def parse_arguments():
    """Parses command-line arguments for running benchmarks and visualizations."""
    parser = argparse.ArgumentParser(
        description="Comparative Evaluation of PSO and GA on the Multidimensional Ackley Function."
    )
    parser.add_argument(
        "--algo",
        choices=["pso", "ga", "both"],
        default="both",
        help="Algorithm to run: 'pso', 'ga', or 'both' (default: 'both').",
    )
    parser.add_argument(
        "--dim",
        type=str,
        default="all",
        help="Dimension to test: a single integer from [1, 2, 3, 5, 10, 20, 30, 40, 50] or 'all' (default: 'all').",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of independent runs per dimension (default: 10).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to store CSV files and plots (default: 'results').",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots using Matplotlib after completing runs.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Resolve dimensions: either all predefined benchmark dimensions or a specific validated integer
    if args.dim.lower() == "all":
        selected_dimensions = ALL_DIMENSIONS
    else:
        try:
            dim_val = int(args.dim)
        except ValueError:
            raise ValueError(
                f"Dimension must be an integer or 'all'. Received: '{args.dim}'"
            )

        if dim_val not in ALL_DIMENSIONS:
            raise ValueError(
                f"Dimension {dim_val} is not in the benchmark set: {ALL_DIMENSIONS}"
            )
        selected_dimensions = [dim_val]

    # Execute the requested experiment runs
    run_experiment_suite(
        algorithm=args.algo,
        dimensions=selected_dimensions,
        runs_per_dim=args.runs,
        results_dir=args.results_dir,
    )

    # Optional automated chart rendering
    if args.plot:
        summary_file = os.path.join(args.results_dir, "summary.csv")
        plot_output_dir = os.path.join(args.results_dir, "plots")

        if os.path.exists(summary_file):
            print("\nGenerating Matplotlib evaluation charts...")
            from visualization import plot_benchmark_results

            plot_benchmark_results(
                summary_path=summary_file, output_dir=plot_output_dir
            )
        else:
            print(
                f"\nWarning: Could not find '{summary_file}' to generate plots."
            )


if __name__ == "__main__":
    main()
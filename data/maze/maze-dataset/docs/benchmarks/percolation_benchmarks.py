"script for running percolation benchmarks"

from pathlib import Path
from typing import Any

from maze_dataset import MazeDatasetConfig

SAVE_DIR: Path = Path("docs/benchmarks/percolation_fractions/")


ANALYSIS_KWARGS: dict[str, dict[str, Any]] = dict(
	test=dict(
		n_mazes=16,
		p_val_count=16,
		grid_sizes=[2, 4],
	),
	small=dict(
		n_mazes=64,
		p_val_count=25,
		grid_sizes=[2, 3, 4, 5, 6],
	),
	medium=dict(
		n_mazes=128,
		p_val_count=50,
		grid_sizes=[2, 3, 4, 5, 6, 8, 10],
	),
	large=dict(
		n_mazes=256,
		p_val_count=100,
		grid_sizes=[2, 3, 4, 5, 6, 8, 10, 16, 32],
	),
	xlarge=dict(
		n_mazes=2048,
		p_val_count=200,
		grid_sizes=[2, 3, 4, 5, 6, 8, 10, 16, 25, 32, 50, 64],
	),
)

if __name__ == "__main__":
	import argparse

	parser: argparse.ArgumentParser = argparse.ArgumentParser()
	parser.add_argument(
		"analysis",
		type=str,
		choices=ANALYSIS_KWARGS.keys(),
		help="The analysis to run",
	)
	parser.add_argument(
		"-p",
		"--parallel",
		type=int,
		default=False,
		help="The number of parallel processes to use",
	)
	parser.add_argument(
		"-s",
		"--save-dir",
		type=str,
		default=None,
		help=f"The directory to save the results. if `None`, will be saved to {SAVE_DIR.as_posix()}/<analysis>",
	)
	args: argparse.Namespace = parser.parse_args()
	kwargs: dict[str, Any] = ANALYSIS_KWARGS[args.analysis]
	save_dir: Path
	if args.save_dir is None:
		save_dir = SAVE_DIR / args.analysis
	else:
		save_dir = Path(args.save_dir)

	print(f"Running analysis: {args.analysis}, saving to: {save_dir.as_posix()}")

	# import here for speed
	from maze_dataset.benchmark.config_sweep import (
		SweepResult,
		full_percolation_analysis,
		plot_grouped,
	)

	# run analysis and save
	results: SweepResult = full_percolation_analysis(
		**kwargs,
		parallel=args.parallel,
		save_dir=save_dir,
		chunksize=1,
	)

	# plot results
	plot_grouped(
		results,
		predict_fn=MazeDatasetConfig.success_fraction_estimate,
		save_dir=save_dir,
		show=False,
	)

"script for benchmarking generation performance"

import platform
from typing import Any

import psutil

from maze_dataset.benchmark.speed import run_benchmark
from maze_dataset.generation.default_generators import DEFAULT_GENERATORS

BENCHMARK_KWARGS: dict[str, dict[str, Any]] = dict(
	test=dict(
		grid_n_vals=[2, 3, 4],
		n_mazes_vals=list(range(1, 5, 2)),
		trials=3,
	),
	default=dict(
		grid_n_vals=[2, 3, 4, 5, 8, 10, 16, 25, 32],
		n_mazes_vals=list(range(1, 12, 2)),
		trials=10,
	),
	large=dict(
		grid_n_vals=[2, 3, 4, 5, 6, 7, 8, 11, 16, 22, 32, 45, 64],
		n_mazes_vals=[1, 2, 4, 8, 16],
		trials=16,
	),
)


def get_system_info() -> dict[str, Any]:
	"""Retrieve system hardware information"""
	info: dict[str, Any] = {}

	# Retrieve CPU frequency (in MHz)
	cpu_freq: psutil._common.scpufreq = psutil.cpu_freq()  # type: ignore
	info["cpu_current_freq"] = cpu_freq.current if cpu_freq else None
	info["cpu_max_freq"] = cpu_freq.max if cpu_freq else None

	# Retrieve CPU model information
	cpu_model: str = platform.processor()
	if not cpu_model:
		# On some systems, platform.processor() returns an empty string.
		cpu_model = platform.uname().processor
	info["cpu_model"] = cpu_model

	# Retrieve CPU core counts
	info["cpu_logical_cores"] = psutil.cpu_count(logical=True)
	info["cpu_physical_cores"] = psutil.cpu_count(logical=False)

	# Retrieve total system RAM in bytes
	virtual_mem: psutil._common.svmem = psutil.virtual_memory()  # type: ignore
	info["total_ram"] = virtual_mem.total if virtual_mem else None

	return info


if __name__ == "__main__":
	print(get_system_info())

	import argparse

	parser: argparse.ArgumentParser = argparse.ArgumentParser()
	parser.add_argument(
		"analysis",
		type=str,
		choices=BENCHMARK_KWARGS.keys(),
		help="The analysis to run",
	)
	parser.add_argument(
		"-s",
		"--save-path",
		type=str,
		default="tests/_temp/benchmark_generation.jsonl",
		help="The path to save the results",
	)
	parser.add_argument(
		"-q",
		"--quiet",
		action="store_true",
		help="disable verbose output",
	)
	args: argparse.Namespace = parser.parse_args()

	run_benchmark(
		save_path=args.save_path,
		base_configs=DEFAULT_GENERATORS,
		verbose=not args.quiet,
		**BENCHMARK_KWARGS[args.analysis],
	)

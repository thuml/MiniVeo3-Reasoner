"benchmark the speed of maze generation"

import functools
import random
import timeit
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation.default_generators import DEFAULT_GENERATORS
from maze_dataset.generation.generators import GENERATORS_MAP

_BASE_CFG_KWARGS: dict = dict(
	grid_n=None,
	n_mazes=None,
)

_GENERATE_KWARGS: dict = dict(
	gen_parallel=False,
	pool_kwargs=None,
	verbose=False,
	# do_generate = True,
	# load_local = False,
	# save_local = False,
	# zanj = None,
	# do_download = False,
	# local_base_path = "INVALID",
	# except_on_config_mismatch = True,
	# verbose = False,
)


def time_generation(
	base_configs: list[tuple[str, dict]],
	grid_n_vals: list[int],
	n_mazes_vals: list[int],
	trials: int = 10,
	verbose: bool = False,
) -> list[dict[str, Any]]:
	"time the generation of mazes for various configurations"
	# assemble configs
	configs: list[MazeDatasetConfig] = list()

	for b_cfg in base_configs:
		for grid_n in grid_n_vals:
			for n_mazes in n_mazes_vals:
				configs.append(
					MazeDatasetConfig(
						name="benchmark",
						grid_n=grid_n,
						n_mazes=n_mazes,
						maze_ctor=GENERATORS_MAP[b_cfg[0]],
						maze_ctor_kwargs=b_cfg[1],
					),
				)

	# shuffle configs (in place) (otherwise progress bar is annoying)
	random.shuffle(configs)

	# time generation for each config
	times: list[dict[str, Any]] = list()
	total: int = len(configs)
	for idx, cfg in tqdm(
		enumerate(configs),
		desc="Timing generation",
		unit="config",
		total=total,
		disable=verbose,
	):
		if verbose:
			print(f"Timing generation for config {idx + 1}/{total}\n{cfg}")

		t: float = (
			timeit.timeit(
				stmt=functools.partial(MazeDataset.generate, cfg, **_GENERATE_KWARGS),  # type: ignore[arg-type]
				number=trials,
			)
			/ trials
		)

		if verbose:
			print(f"avg time: {t:.3f} s")

		times.append(
			dict(
				cfg_name=cfg.name,
				grid_n=cfg.grid_n,
				n_mazes=cfg.n_mazes,
				maze_ctor=cfg.maze_ctor.__name__,
				maze_ctor_kwargs=cfg.maze_ctor_kwargs,
				trials=trials,
				time=t,
			),
		)

	return times


def run_benchmark(
	save_path: str,
	base_configs: list[tuple[str, dict]] | None = None,
	grid_n_vals: Sequence[int] = (2, 3, 4, 5, 8, 10, 16, 25, 32),
	n_mazes_vals: Sequence[int] = tuple(range(1, 12, 2)),
	trials: int = 10,
	verbose: bool = True,
) -> "pd.DataFrame":  # type: ignore[name-defined] # noqa: F821
	"run the benchmark and save the results to a file"
	import pandas as pd

	if base_configs is None:
		base_configs = DEFAULT_GENERATORS

	times: list[dict] = time_generation(
		base_configs=base_configs,
		grid_n_vals=list(grid_n_vals),
		n_mazes_vals=list(n_mazes_vals),
		trials=trials,
		verbose=verbose,
	)

	df: pd.DataFrame = pd.DataFrame(times)

	# print the whole dataframe contents to console as csv
	print(df.to_csv())

	# save to file
	Path(save_path).parent.mkdir(parents=True, exist_ok=True)
	df.to_json(save_path, orient="records", lines=True)

	return df

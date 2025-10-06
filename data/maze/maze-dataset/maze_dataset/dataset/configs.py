"`MAZE_DATASET_CONFIGS` contains some default configs for tests and demos"

import copy
from typing import Callable, Iterator, Mapping

from maze_dataset.dataset.maze_dataset import MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators

_MAZE_DATASET_CONFIGS_SRC: dict[str, MazeDatasetConfig] = {
	cfg.to_fname(): cfg
	for cfg in [
		MazeDatasetConfig(
			name="test",
			grid_n=3,
			n_mazes=5,
			maze_ctor=LatticeMazeGenerators.gen_dfs,
		),
		MazeDatasetConfig(
			name="test-perc",
			grid_n=3,
			n_mazes=5,
			maze_ctor=LatticeMazeGenerators.gen_dfs_percolation,
			maze_ctor_kwargs={"p": 0.7},
		),
		MazeDatasetConfig(
			name="demo_small",
			grid_n=3,
			n_mazes=100,
			maze_ctor=LatticeMazeGenerators.gen_dfs,
		),
		MazeDatasetConfig(
			name="demo",
			grid_n=6,
			n_mazes=10000,
			maze_ctor=LatticeMazeGenerators.gen_dfs,
		),
	]
}


class _MazeDatsetConfigsWrapper(Mapping[str, MazeDatasetConfig]):
	"wrap the default configs in a read-only dict-like object"

	def __init__(self, configs: dict[str, MazeDatasetConfig]) -> None:
		"initialize with a dict of configs"
		self._configs = configs

	def __getitem__(self, item: str) -> MazeDatasetConfig:
		return self._configs[item]

	def __len__(self) -> int:
		return len(self._configs)

	def __iter__(self) -> Iterator:
		"iterate over the keys"
		return iter(self._configs)

	# TYPING: error: Return type "list[str]" of "keys" incompatible with return type "KeysView[str]" in supertype "Mapping"  [override]
	def keys(self) -> list[str]:  # type: ignore[override]
		"return the keys"
		return list(self._configs.keys())

	# TYPING: error: Return type "list[tuple[str, MazeDatasetConfig]]" of "items" incompatible with return type "ItemsView[str, MazeDatasetConfig]" in supertype "Mapping"  [override]
	def items(self) -> list[tuple[str, MazeDatasetConfig]]:  # type: ignore[override]
		"return the items"
		return [(k, copy.deepcopy(v)) for k, v in self._configs.items()]

	# TYPING: error: Return type "list[MazeDatasetConfig]" of "values" incompatible with return type "ValuesView[MazeDatasetConfig]" in supertype "Mapping"  [override]
	def values(self) -> list[MazeDatasetConfig]:  # type: ignore[override]
		return [copy.deepcopy(v) for v in self._configs.values()]


MAZE_DATASET_CONFIGS: _MazeDatsetConfigsWrapper = _MazeDatsetConfigsWrapper(
	_MAZE_DATASET_CONFIGS_SRC,
)


def _get_configs_for_examples() -> list[dict]:
	"""Generate a comprehensive list of diverse maze configurations.

	# Returns:
	- `list[dict]`
		List of configuration dictionaries for maze generation
	"""
	configs: list[dict] = []

	# Define the grid sizes to test
	grid_sizes: list[int] = [5, 8, 12, 15, 20]

	# Define percolation probabilities
	percolation_probs: list[float] = [0.3, 0.5, 0.7]

	# Core algorithms with basic configurations
	basic_algorithms: dict[str, tuple[Callable, dict]] = {
		"dfs": (LatticeMazeGenerators.gen_dfs, {}),
		"wilson": (LatticeMazeGenerators.gen_wilson, {}),
		"kruskal": (LatticeMazeGenerators.gen_kruskal, {}),
		"recursive_division": (LatticeMazeGenerators.gen_recursive_division, {}),
	}

	# Generate basic configurations for each algorithm and grid size
	for grid_n in grid_sizes:
		for algo_name, (maze_ctor, base_kwargs) in basic_algorithms.items():
			configs.append(
				dict(
					name="basic",
					grid_n=grid_n,
					maze_ctor=maze_ctor,
					maze_ctor_kwargs=base_kwargs,
					description=f"Basic {algo_name.upper()} maze ({grid_n}x{grid_n})",
					tags=[f"algo:{algo_name}", "basic", f"grid:{grid_n}"],
				)
			)

	# Generate percolation configurations
	for grid_n in grid_sizes:
		for p in percolation_probs:
			# Pure percolation
			configs.append(
				dict(
					name=f"p{p}",
					grid_n=grid_n,
					maze_ctor=LatticeMazeGenerators.gen_percolation,
					maze_ctor_kwargs=dict(p=p),
					description=f"Pure percolation (p={p}) ({grid_n}x{grid_n})",
					tags=[
						"algo:percolation",
						"percolation",
						f"percolation:{p}",
						f"grid:{grid_n}",
					],
				)
			)

			# DFS with percolation
			configs.append(
				dict(
					name=f"p{p}",
					grid_n=grid_n,
					maze_ctor=LatticeMazeGenerators.gen_dfs_percolation,
					maze_ctor_kwargs=dict(p=p),
					description=f"DFS with percolation (p={p}) ({grid_n}x{grid_n})",
					tags=[
						"algo:dfs_percolation",
						"dfs",
						"percolation",
						f"percolation:{p}",
						f"grid:{grid_n}",
					],
				)
			)

	# Generate specialized constraint configurations
	constraint_base_config: dict = dict(
		grid_n=10,
		maze_ctor=LatticeMazeGenerators.gen_dfs,
	)
	constraint_base_tags: list[str] = [
		"algo:dfs",
		"dfs",
		"constrained_dfs",
		f"grid:{constraint_base_config['grid_n']}",
	]

	constraint_configs: list[dict] = [
		# DFS without forks (simple path)
		dict(
			name="forkless",
			maze_ctor_kwargs=dict(do_forks=False),
			description="DFS without forks (10x10)",
			tags=["forkless"],
		),
		# Accessible cells constraints
		dict(
			name="accessible_cells_count",
			maze_ctor_kwargs=dict(accessible_cells=50),
			description="DFS with limited accessible cells (50)",
			tags=["limited:cells", "limited:absolute"],
		),
		dict(
			name="accessible_cells_ratio",
			maze_ctor_kwargs=dict(accessible_cells=0.6),
			description="DFS with 60% accessible cells",
			tags=["limited:cells", "limited:ratio"],
		),
		# Tree depth constraints
		dict(
			name="max_tree_depth_absolute",
			maze_ctor_kwargs=dict(max_tree_depth=10),
			description="DFS with max tree depth of 10",
			tags=["limited:depth", "limited:absolute"],
		),
		dict(
			name="max_tree_depth_ratio",
			maze_ctor_kwargs=dict(max_tree_depth=0.3),
			description="DFS with max tree depth 30% of grid size",
			tags=["limited:depth", "limited:ratio"],
		),
		# Start position constraint
		dict(
			name="start_center",
			maze_ctor_kwargs=dict(start_coord=[5, 5]),
			description="DFS starting from center of grid",
			tags=["custom_start"],
		),
		dict(
			name="start_corner",
			maze_ctor_kwargs=dict(start_coord=[0, 0]),
			description="DFS starting from corner of grid",
			tags=["custom_start"],
		),
	]

	# Add combined constraints as special case
	configs.append(
		dict(
			name="combined_constraints",
			grid_n=15,
			maze_ctor=LatticeMazeGenerators.gen_dfs,
			maze_ctor_kwargs=dict(
				accessible_cells=100,
				max_tree_depth=25,
				start_coord=[7, 7],
			),
			description="DFS with multiple constraints (100 cells, depth 25, center start)",
			tags=["algo:dfs", "dfs", "constrained_dfs", "grid:15"],
		)
	)

	# Apply the base config to all constraint configs and add to main configs list
	for config in constraint_configs:
		full_config = constraint_base_config.copy()
		full_config.update(config)
		full_config["tags"] = constraint_base_tags + config["tags"]
		configs.append(full_config)

	# Generate endpoint options
	endpoint_variations: list[tuple[bool, bool, str]] = [
		(True, False, "deadend start only"),
		(False, True, "deadend end only"),
		(True, True, "deadend start and end"),
	]

	for deadend_start, deadend_end, desc in endpoint_variations:
		configs.append(
			dict(
				name=f"deadend_s{int(deadend_start)}_e{int(deadend_end)}",
				grid_n=8,
				maze_ctor=LatticeMazeGenerators.gen_dfs,
				maze_ctor_kwargs={},
				endpoint_kwargs=dict(
					deadend_start=deadend_start,
					deadend_end=deadend_end,
					endpoints_not_equal=True,
				),
				description=f"DFS with {desc}",
				tags=["algo:dfs", "dfs", "deadend_endpoints", "grid:8"],
			)
		)

	# Add percolation with deadend endpoints
	configs.append(
		dict(
			name="deadends",
			grid_n=8,
			maze_ctor=LatticeMazeGenerators.gen_dfs_percolation,
			maze_ctor_kwargs=dict(p=0.3),
			endpoint_kwargs=dict(
				deadend_start=True,
				deadend_end=True,
				endpoints_not_equal=True,
				except_on_no_valid_endpoint=False,
			),
			description="DFS percolation (p=0.3) with deadend endpoints",
			tags=[
				"algo:dfs_percolation",
				"dfs",
				"percolation",
				"deadend_endpoints",
				"grid:8",
			],
		)
	)

	return configs

from maze_dataset import MazeDatasetConfig
from maze_dataset.dataset.configs import MAZE_DATASET_CONFIGS


def test_get_configs():
	keys: list[str] = list(MAZE_DATASET_CONFIGS.keys())
	assert len(keys) > 0, "There must be at least one key in the configs"
	assert all([isinstance(key, str) for key in keys]), f"Keys must be strings: {keys}"
	assert all(
		[isinstance(MAZE_DATASET_CONFIGS[key], MazeDatasetConfig) for key in keys],
	), f"Values must be dictionaries: {MAZE_DATASET_CONFIGS}"

	assert len(MAZE_DATASET_CONFIGS.keys()) == len(MAZE_DATASET_CONFIGS)
	assert len(MAZE_DATASET_CONFIGS.items()) == len(MAZE_DATASET_CONFIGS)
	assert len(MAZE_DATASET_CONFIGS.values()) == len(MAZE_DATASET_CONFIGS)

	assert all([isinstance(key, str) for key in MAZE_DATASET_CONFIGS]), (
		f".keys() must be strings: {MAZE_DATASET_CONFIGS.keys()}"
	)
	assert all(
		[
			isinstance(value, MazeDatasetConfig)
			for value in MAZE_DATASET_CONFIGS.values()
		],
	), f".values() must be configs: {MAZE_DATASET_CONFIGS.values()}"
	assert all(
		[
			isinstance(key, str) and isinstance(value, MazeDatasetConfig)
			for key, value in MAZE_DATASET_CONFIGS.items()
		],
	), f".items() must be (str, config) tuples {MAZE_DATASET_CONFIGS.items()}"

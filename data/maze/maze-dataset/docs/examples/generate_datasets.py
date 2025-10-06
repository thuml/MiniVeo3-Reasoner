"script for generating example datasets"

from pathlib import Path

from maze_dataset import MazeDataset
from maze_dataset.dataset.configs import _MAZE_DATASET_CONFIGS_SRC


def main() -> None:
	"generate datasets in _MAZE_DATASET_CONFIGS_SRC"
	for cfg_name, cfg in _MAZE_DATASET_CONFIGS_SRC.items():
		dataset: MazeDataset = MazeDataset.from_config(
			cfg,
			load_local=False,
			save_local=True,
			local_base_path=Path("docs/examples/datasets/"),
		)
		print(f"Dataset {cfg_name} has {len(dataset.mazes)} mazes")


if __name__ == "__main__":
	main()

import sys
from pathlib import Path

import pytest
from muutils.json_serialize.util import _FORMAT_KEY

from maze_dataset import MazeDataset, MazeDatasetConfig

DATASET_PATHS: list[str] = [
	p.as_posix() for p in Path("docs/examples/datasets").rglob("*.zanj")
]


def test_temp():
	cfg = MazeDatasetConfig(
		name="test",
		grid_n=3,
		n_mazes=2,
	)
	print(f"{cfg._fname_loaded = }")
	cfg_ser = cfg.serialize()
	print(f"{list(cfg_ser.keys()) = }")
	print(f"{cfg_ser[_FORMAT_KEY] = }")
	print(f"{cfg_ser['fname'] = }")
	cfg_load: MazeDatasetConfig = MazeDatasetConfig.load(cfg_ser)
	print(f"{cfg.diff(cfg_load) = }")
	print(f"{cfg_load._fname_loaded = }")
	assert cfg == cfg_load


@pytest.mark.parametrize("dataset_path_str", DATASET_PATHS)
def test_validate_fname(dataset_path_str: str):
	dataset_path: Path = Path(dataset_path_str)
	dataset: MazeDataset = MazeDataset.read(dataset_path)
	dataset_cfg: MazeDatasetConfig = dataset.cfg
	dataset_fname_path: str = dataset_path.name.removesuffix(".zanj")
	dataset_fname_loaded: str | None = dataset_cfg._fname_loaded
	dataset_fname_new: str = dataset_cfg.to_fname()
	print(f"{dataset_path.as_posix() = }")
	print(f"{dataset_fname_path = }")
	print(f"{dataset_fname_loaded = }")
	print(f"{dataset_fname_new = }")
	print(f"{dataset_cfg.summary() = }")
	print(f"{dataset_cfg = }")
	print(f"{dataset_cfg._stable_str_dump() = }")

	print(f"{sys.version = }")

	assert dataset_fname_path == dataset_fname_loaded
	assert dataset_fname_path == dataset_fname_new

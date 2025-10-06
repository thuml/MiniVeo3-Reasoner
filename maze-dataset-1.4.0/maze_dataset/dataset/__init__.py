"""`MazeDatasetConfig`s are used to create a `MazeDataset` via `MazeDataset.from_config(cfg)`"

When initializing mazes, further configuration options can be specified through the `from_config()` factory method as necessary. Options include 1) whether to generate the dataset during runtime or load an existing dataset, 2) if and how to parallelize generation, and 3) where to store the generated dataset. Full documentation of configuration options is available in our repository [@maze-dataset-github]. Available maze generation algorithms are static methods of the `LatticeMazeGenerators` class.

Furthermore, a dataset of mazes can be filtered to satisfy certain properties:

```python
dataset_filtered: MazeDataset = dataset.filter_by.path_length(min_length=3)
```

Custom filters can be specified, and several filters are included:

- `path_length(min_length: int)`: shortest length from the origin to target should be at least `min_length`.
- `start_end_distance(min_distance: int)`: Manhattan distance between start and end should be at least `min_distance`, ignoring walls.
- `remove_duplicates(...)`: remove mazes which are similar to others in the dataset, measured via Hamming distance.
- `remove_duplicates_fast()`: remove mazes which are exactly identical to others in the dataset.

All implemented maze generation algorithms are stochastic by nature. For reproducibility, the `seed` parameter of `MazeDatasetConfig` may be set. In practice, we do not find that exact duplicates of mazes are generated with any meaningful frequency,
even when generating large datasets.

"""

from maze_dataset.dataset.collected_dataset import (
	MazeDatasetCollection,
	MazeDatasetCollectionConfig,
)
from maze_dataset.dataset.maze_dataset import MazeDataset
from maze_dataset.dataset.maze_dataset_config import MazeDatasetConfig

__all__ = [
	# submodules
	"collected_dataset",
	"configs",
	"dataset",
	"filters",
	"maze_dataset_config",
	"maze_dataset",
	"rasterized",
	"success_predict_math",
	# dataset classes
	"MazeDataset",
	"MazeDatasetConfig",
	"MazeDatasetCollection",
	"MazeDatasetCollectionConfig",
]

"""a special `RasterizedMazeDataset` that returns 2 images, one for input and one for target, for each maze

this lets you match the input and target format of the [`easy_2_hard`](https://github.com/aks2203/easy-to-hard) dataset


see their paper:

```bibtex
@misc{schwarzschild2021learn,
	title={Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks},
	author={Avi Schwarzschild and Eitan Borgnia and Arjun Gupta and Furong Huang and Uzi Vishkin and Micah Goldblum and Tom Goldstein},
	year={2021},
	eprint={2106.04537},
	archivePrefix={arXiv},
	primaryClass={cs.LG}
}
```
"""

import typing
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from muutils.json_serialize import serializable_dataclass, serializable_field
from zanj import ZANJ

from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.maze import PixelColors, SolvedMaze
from maze_dataset.maze.lattice_maze import PixelGrid, _remove_isolated_cells


def _extend_pixels(
	image: Int[np.ndarray, "x y rgb"],
	n_mult: int = 2,
	n_bdry: int = 1,
) -> Int[np.ndarray, "n_mult*x+2*n_bdry n_mult*y+2*n_bdry rgb"]:
	wall_fill: int = PixelColors.WALL[0]
	assert all(x == wall_fill for x in PixelColors.WALL), (
		"PixelColors.WALL must be a single value"
	)

	output: np.ndarray = np.repeat(
		np.repeat(
			image,
			n_mult,
			axis=0,
		),
		n_mult,
		axis=1,
	)

	# pad on all sides by n_bdry
	return np.pad(
		output,
		pad_width=((n_bdry, n_bdry), (n_bdry, n_bdry), (0, 0)),
		mode="constant",
		constant_values=wall_fill,
	)


_RASTERIZED_CFG_ADDED_PARAMS: list[str] = [
	"remove_isolated_cells",
	"extend_pixels",
	"endpoints_as_open",
]


def process_maze_rasterized_input_target(
	maze: SolvedMaze,
	remove_isolated_cells: bool = True,
	extend_pixels: bool = True,
	endpoints_as_open: bool = False,
) -> Float[np.ndarray, "in/tgt=2 x y rgb=3"]:
	"""turn a single `SolvedMaze` into an array representation

	has extra options for matching the format in https://github.com/aks2203/easy-to-hard

	# Parameters:
	- `maze: SolvedMaze`
		the maze to process
	- `remove_isolated_cells: bool`
		whether to set isolated cells (no connections) to walls
		(default: `True`)
	- `extend_pixels: bool`
		whether to extend pixels to match easy_2_hard dataset (2x2 cells, extra 1 pixel row of wall around maze)
		(default: `True`)
	- `endpoints_as_open: bool`
		whether to set endpoints to open
		(default: `False`)
	"""
	# problem and solution mazes
	maze_pixels: PixelGrid = maze.as_pixels(show_endpoints=True, show_solution=True)
	problem_maze: PixelGrid = maze_pixels.copy()
	solution_maze: PixelGrid = maze_pixels.copy()

	# in problem maze, set path to open
	problem_maze[(problem_maze == PixelColors.PATH).all(axis=-1)] = PixelColors.OPEN

	# wherever solution maze is PixelColors.OPEN, set it to PixelColors.WALL
	solution_maze[(solution_maze == PixelColors.OPEN).all(axis=-1)] = PixelColors.WALL
	# wherever it is solution, set it to PixelColors.OPEN
	solution_maze[(solution_maze == PixelColors.PATH).all(axis=-1)] = PixelColors.OPEN
	if endpoints_as_open:
		for color in (PixelColors.START, PixelColors.END):
			solution_maze[(solution_maze == color).all(axis=-1)] = PixelColors.OPEN

	# postprocess to match original easy_2_hard dataset
	if remove_isolated_cells:
		problem_maze = _remove_isolated_cells(problem_maze)
		solution_maze = _remove_isolated_cells(solution_maze)

	if extend_pixels:
		problem_maze = _extend_pixels(problem_maze)
		solution_maze = _extend_pixels(solution_maze)

	return np.array([problem_maze, solution_maze])


# TYPING: error: Attributes without a default cannot follow attributes with one  [misc]
@serializable_dataclass
class RasterizedMazeDatasetConfig(MazeDatasetConfig):  # type: ignore[misc]
	"""adds options which we then pass to `process_maze_rasterized_input_target`

	- `remove_isolated_cells: bool` whether to set isolated cells to walls
	- `extend_pixels: bool` whether to extend pixels to match easy_2_hard dataset (2x2 cells, extra 1 pixel row of wall around maze)
	- `endpoints_as_open: bool` whether to set endpoints to open
	"""

	remove_isolated_cells: bool = serializable_field(default=True)
	extend_pixels: bool = serializable_field(default=True)
	endpoints_as_open: bool = serializable_field(default=False)


class RasterizedMazeDataset(MazeDataset):
	"subclass of `MazeDataset` that uses a `RasterizedMazeDatasetConfig`"

	cfg: RasterizedMazeDatasetConfig

	# this override here is intentional
	def __getitem__(self, idx: int) -> Float[np.ndarray, "item in/tgt=2 x y rgb=3"]:  # type: ignore[override]
		"""get a single maze"""
		# get the solved maze
		solved_maze: SolvedMaze = self.mazes[idx]

		return process_maze_rasterized_input_target(
			maze=solved_maze,
			remove_isolated_cells=self.cfg.remove_isolated_cells,
			extend_pixels=self.cfg.extend_pixels,
			endpoints_as_open=self.cfg.endpoints_as_open,
		)

	def get_batch(
		self,
		idxs: list[int] | None,
	) -> Float[np.ndarray, "in/tgt=2 item x y rgb=3"]:
		"""get a batch of mazes as a tensor, from a list of indices"""
		if idxs is None:
			idxs = list(range(len(self)))

		inputs: list[Float[np.ndarray, "x y rgb=3"]]
		targets: list[Float[np.ndarray, "x y rgb=3"]]
		inputs, targets = zip(*[self[i] for i in idxs], strict=False)  # type: ignore[assignment]

		return np.array([inputs, targets])

	# override here is intentional
	@classmethod
	def from_config(
		cls,
		cfg: RasterizedMazeDatasetConfig | MazeDatasetConfig,  # type: ignore[override]
		do_generate: bool = True,
		load_local: bool = True,
		save_local: bool = True,
		zanj: ZANJ | None = None,
		do_download: bool = True,
		local_base_path: Path = Path("data/maze_dataset"),
		except_on_config_mismatch: bool = True,
		allow_generation_metadata_filter_mismatch: bool = True,
		verbose: bool = False,
		**kwargs,
	) -> "RasterizedMazeDataset":
		"""create a rasterized maze dataset from a config

		priority of loading:
		1. load from local
		2. download
		3. generate

		"""
		return typing.cast(
			"RasterizedMazeDataset",
			super().from_config(
				cfg=cfg,
				do_generate=do_generate,
				load_local=load_local,
				save_local=save_local,
				zanj=zanj,
				do_download=do_download,
				local_base_path=local_base_path,
				except_on_config_mismatch=except_on_config_mismatch,
				allow_generation_metadata_filter_mismatch=allow_generation_metadata_filter_mismatch,
				verbose=verbose,
				**kwargs,
			),
		)

	@classmethod
	def from_config_augmented(
		cls,
		cfg: RasterizedMazeDatasetConfig,
		**kwargs,
	) -> "RasterizedMazeDataset":
		"""loads either a maze transformer dataset or an easy_2_hard dataset"""
		_cfg_temp: MazeDatasetConfig = MazeDatasetConfig.load(cfg.serialize())
		return cls.from_base_MazeDataset(
			cls.from_config(cfg=_cfg_temp, **kwargs),
			added_params={
				k: v
				for k, v in cfg.serialize().items()
				if k in _RASTERIZED_CFG_ADDED_PARAMS
			},
		)

	@classmethod
	def from_base_MazeDataset(
		cls,
		base_dataset: MazeDataset,
		added_params: dict | None = None,
	) -> "RasterizedMazeDataset":
		"""loads either a maze transformer dataset or an easy_2_hard dataset"""
		if added_params is None:
			added_params = dict(
				remove_isolated_cells=True,
				extend_pixels=True,
			)
		cfg: RasterizedMazeDatasetConfig = RasterizedMazeDatasetConfig.load(
			{
				**base_dataset.cfg.serialize(),
				**added_params,
			},
		)
		output: RasterizedMazeDataset = cls(
			cfg=cfg,
			mazes=base_dataset.mazes,
		)
		return output

	def plot(self, count: int | None = None, show: bool = True) -> tuple | None:
		"""plot the first `count` mazes in the dataset"""
		import matplotlib.pyplot as plt

		print(f"{self[0][0].shape = }, {self[0][1].shape = }")
		count = count or len(self)
		if count == 0:
			print("No mazes to plot for dataset")
			return None
		fig, axes = plt.subplots(2, count, figsize=(15, 5))
		if count == 1:
			axes = [axes]
		for i in range(count):
			axes[0, i].imshow(self[i][0])
			axes[1, i].imshow(self[i][1])
			# remove ticks
			axes[0, i].set_xticks([])
			axes[0, i].set_yticks([])
			axes[1, i].set_xticks([])
			axes[1, i].set_yticks([])

		if show:
			plt.show()

		return fig, axes


def make_numpy_collection(
	base_cfg: RasterizedMazeDatasetConfig,
	grid_sizes: list[int],
	from_config_kwargs: dict | None = None,
	verbose: bool = True,
	key_fmt: str = "{size}x{size}",
) -> dict[
	typing.Literal["configs", "arrays"],
	dict[str, RasterizedMazeDatasetConfig | np.ndarray],
]:
	"""create a collection of configs and arrays for different grid sizes, in plain tensor form

	output is of structure:
	```
	{
		"configs": {
			"<n>x<n>": RasterizedMazeDatasetConfig,
			...
		},
		"arrays": {
			"<n>x<n>": np.ndarray,
			...
		},
	}
	```
	"""
	if from_config_kwargs is None:
		from_config_kwargs = {}

	datasets: dict[int, RasterizedMazeDataset] = {}

	for size in grid_sizes:
		if verbose:
			print(f"Generating dataset for maze size {size}...")

		cfg_temp: RasterizedMazeDatasetConfig = RasterizedMazeDatasetConfig.load(
			base_cfg.serialize(),
		)
		cfg_temp.grid_n = size

		datasets[size] = RasterizedMazeDataset.from_config_augmented(
			cfg=cfg_temp,
			**from_config_kwargs,
		)

	return dict(
		configs={
			key_fmt.format(size=size): dataset.cfg for size, dataset in datasets.items()
		},
		arrays={
			# get_batch(None) returns a single tensor of shape (n, 2, x, y, 3)
			key_fmt.format(size=size): dataset.get_batch(None)
			for size, dataset in datasets.items()
		},
	)

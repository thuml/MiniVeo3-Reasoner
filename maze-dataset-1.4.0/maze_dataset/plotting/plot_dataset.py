"""`plot_dataset_mazes` will plot several mazes using `as_pixels`

`print_dataset_mazes` will use `as_ascii` to print several mazes
"""

import matplotlib.pyplot as plt  # type: ignore[import]

from maze_dataset.dataset.maze_dataset import MazeDataset


def plot_dataset_mazes(
	ds: MazeDataset,
	count: int | None = None,
	figsize_mult: tuple[float, float] = (1.0, 2.0),
	title: bool | str = True,
) -> tuple | None:
	"plot `count` mazes from the dataset `d` in a single figure using `SolvedMaze.as_pixels()`"
	count = count or len(ds)
	if count == 0:
		print("No mazes to plot for dataset")
		return None
	fig, axes = plt.subplots(
		1,
		count,
		figsize=(count * figsize_mult[0], figsize_mult[1]),
	)
	if count == 1:
		axes = [axes]
	for i in range(count):
		axes[i].imshow(ds[i].as_pixels())
		# remove ticks
		axes[i].set_xticks([])
		axes[i].set_yticks([])

	# set title
	if title:
		if isinstance(title, str):
			fig.suptitle(title)
		else:
			kwargs: dict = {
				"grid_n": ds.cfg.grid_n,
				# "n_mazes": ds.cfg.n_mazes,
				**ds.cfg.maze_ctor_kwargs,
			}
			fig.suptitle(
				f"{ds.cfg.to_fname()}\n{ds.cfg.maze_ctor.__name__}({', '.join(f'{k}={v}' for k, v in kwargs.items())})",
			)

	# tight layout
	fig.tight_layout()
	# remove whitespace between title and subplots
	fig.subplots_adjust(top=1.0)

	return fig, axes


def print_dataset_mazes(ds: MazeDataset, count: int | None = None) -> None:
	"print ascii representation of `count` mazes from the dataset `d`"
	count = count or len(ds)
	if count == 0:
		print("No mazes to print for dataset")
		return
	for i in range(count):
		print(ds[i].as_ascii(), "\n\n-----\n")

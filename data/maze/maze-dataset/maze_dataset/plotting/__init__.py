"""utilities for plotting mazes and printing tokens

- any `LatticeMaze` or `SolvedMaze` comes with a `as_pixels()` method that returns
  a 2D numpy array of pixel values, but this is somewhat limited
- `MazePlot` is a class that can be used to plot mazes and paths in a more customizable way
- `print_tokens` contains utilities for printing tokens, colored by their type, position, or some custom weights (i.e. attention weights)
"""

from maze_dataset.plotting.plot_dataset import plot_dataset_mazes, print_dataset_mazes
from maze_dataset.plotting.plot_maze import DEFAULT_FORMATS, MazePlot, PathFormat
from maze_dataset.plotting.print_tokens import (
	color_maze_tokens_AOTP,
	color_tokens_cmap,
	color_tokens_rgb,
)

__all__ = [
	# submodules
	"plot_dataset",
	"plot_maze",
	"plot_svg_fancy",
	"plot_tokens",
	"print_tokens",
	# imports
	"plot_dataset_mazes",
	"print_dataset_mazes",
	"DEFAULT_FORMATS",
	"MazePlot",
	"PathFormat",
	"color_tokens_cmap",
	"color_maze_tokens_AOTP",
	"color_tokens_rgb",
]

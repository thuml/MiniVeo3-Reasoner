"""mostly taken from `demo_latticemaze.ipynb`"""

import os

import matplotlib.pyplot as plt
import numpy as np

from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.maze import SolvedMaze, TargetedLatticeMaze
from maze_dataset.plotting import MazePlot

FIG_SAVE: str = "tests/_temp/figures/"


def test_maze_plot():
	N: int = 10

	os.makedirs(FIG_SAVE, exist_ok=True)  # noqa: PTH103

	maze = LatticeMazeGenerators.gen_dfs(np.array([N, N]))
	tgt_maze: TargetedLatticeMaze = TargetedLatticeMaze.from_lattice_maze(
		maze,
		(0, 0),
		(N - 1, N - 1),
	)
	solved_maze: SolvedMaze = SolvedMaze.from_targeted_lattice_maze(tgt_maze)

	fig, ax = plt.subplots(1, 3, figsize=(15, 5))

	for ax_i, temp_maze in zip(ax, [maze, tgt_maze, solved_maze], strict=False):
		ax_i.set_title(temp_maze.as_ascii(), fontfamily="monospace")
		ax_i.imshow(temp_maze.as_pixels())

		assert temp_maze == temp_maze.__class__.from_pixels(temp_maze.as_pixels())
		assert temp_maze == temp_maze.__class__.from_ascii(temp_maze.as_ascii())

	plt.savefig(FIG_SAVE + "pixels_and_ascii.png")

	MazePlot(maze).plot()
	plt.savefig(FIG_SAVE + "mazeplot-pathless.png")

	true_path = maze.find_shortest_path(c_start=(0, 0), c_end=(3, 3))

	MazePlot(solved_maze).plot()
	plt.savefig(FIG_SAVE + "mazeplot-solvedmaze.png")

	pred_path1: list[tuple[int, int]] = [
		(0, 0),
		(1, 0),
		(2, 0),
		(3, 0),
		(3, 1),
		(3, 2),
		(3, 3),
	]
	pred_path2: list[tuple[int, int]] = [
		(0, 0),
		(0, 1),
		(0, 2),
		(0, 3),
		(1, 3),
		(2, 3),
		(2, 2),
		(3, 2),
		(3, 3),
	]
	(
		MazePlot(maze)
		.add_true_path(true_path)
		.add_predicted_path(pred_path1)
		.add_predicted_path(pred_path2)
		.plot()
	)
	plt.savefig(FIG_SAVE + "mazeplot-fakepaths.png")

	node_values = np.random.uniform(size=maze.grid_shape)

	MazePlot(maze).add_node_values(node_values, color_map="Blues").plot()
	plt.savefig(FIG_SAVE + "mazeplot-nodevalues.png")

	MazePlot(maze).add_node_values(
		node_values,
		color_map="Blues",
		target_token_coord=np.array([2, 0]),
		preceeding_tokens_coords=np.array([[0, 0], [3, 1]]),
	).plot()
	plt.savefig(FIG_SAVE + "mazeplot-nodevalues_target.png")

	pred_paths: list[list[tuple[int, int]]] = [pred_path1, pred_path2]
	MazePlot(maze).add_multiple_paths(pred_paths).plot()
	plt.savefig(FIG_SAVE + "mazeplot-multipath.png")

	ascii_maze = MazePlot(maze).to_ascii()
	print(ascii_maze)

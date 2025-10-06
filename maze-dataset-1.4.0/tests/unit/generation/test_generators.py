import warnings

import numpy as np
import pytest

from maze_dataset.generation.generators import (
	GENERATORS_MAP,
	LatticeMazeGenerators,
	get_maze_with_solution,
)
from maze_dataset.maze import Coord, SolvedMaze


def test_gen_dfs_square():
	three_by_three: Coord = np.array([3, 3])
	maze = LatticeMazeGenerators.gen_dfs(three_by_three)

	assert maze.connection_list.shape == (2, 3, 3)


def test_gen_dfs_oblong():
	three_by_four: Coord = np.array([3, 4])
	maze = LatticeMazeGenerators.gen_dfs(three_by_four)

	assert maze.connection_list.shape == (2, 3, 4)


@pytest.mark.parametrize("gfunc_name", GENERATORS_MAP.keys())
def test_get_maze_with_solution(gfunc_name):
	three_by_three: Coord = np.array([5, 5])

	try:
		maze: SolvedMaze = get_maze_with_solution(gfunc_name, three_by_three)
	except ValueError as e:
		if gfunc_name == "gen_percolation":
			warnings.warn(
				f"Skipping test for {gfunc_name} because percolation is stochastic, and a connected component might not be found",
			)
		else:
			raise e  # noqa: TRY201

	assert maze.connection_list.shape == (2, 5, 5)
	assert len(maze.solution[0]) == 2
	assert len(maze.solution[-1]) == 2

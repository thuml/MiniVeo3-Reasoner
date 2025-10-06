"""testing endpoints

> [!NOTE]
> these are all GPT-4o generated tests, so they might not be all that useful
"""

import numpy as np
import pytest

from maze_dataset import LatticeMaze, LatticeMazeGenerators
from maze_dataset.maze.lattice_maze import NoValidEndpointException


def _get_example_maze():
	connection_list = np.zeros((2, 2, 2), dtype=bool)
	connection_list[0, 0, 1] = True
	connection_list[1, 0, 0] = True
	connection_list[1, 1, 0] = True
	maze = LatticeMaze(connection_list=connection_list)
	print(maze.as_ascii())
	return maze


EXAMPLE_MAZE: LatticeMaze = _get_example_maze()
RANDOM_MAZE: LatticeMaze = LatticeMazeGenerators.gen_dfs(grid_shape=(10, 10))
PARAMETRIZE_KWARGS: dict = dict(
	argnames="maze",
	argvalues=[EXAMPLE_MAZE, RANDOM_MAZE],
	ids=["example", "random"],
)


# parametrize with custom id
@pytest.mark.parametrize(**PARAMETRIZE_KWARGS)
def test_generate_random_path_no_conditions(maze):
	path = maze.generate_random_path()
	assert len(path) > 1


@pytest.mark.parametrize(**PARAMETRIZE_KWARGS)
def test_generate_random_path_allowed_start(maze):
	allowed_start = [(0, 0)]
	path = maze.generate_random_path(allowed_start=allowed_start)
	assert path[0].tolist() == list(allowed_start[0])


@pytest.mark.parametrize(**PARAMETRIZE_KWARGS)
def test_generate_random_path_allowed_end(maze):
	allowed_end = [(1, 1)]
	path = maze.generate_random_path(allowed_end=allowed_end)
	assert path[-1].tolist() == list(allowed_end[0])


@pytest.mark.parametrize(**PARAMETRIZE_KWARGS)
def test_generate_random_path_deadend_start(maze):
	path = maze.generate_random_path(deadend_start=True)
	assert len(maze.get_coord_neighbors(tuple(path[0]))) == 1


@pytest.mark.parametrize(**PARAMETRIZE_KWARGS)
def test_generate_random_path_deadend_end(maze):
	path = maze.generate_random_path(deadend_end=True)
	assert len(maze.get_coord_neighbors(tuple(path[-1]))) == 1


@pytest.mark.parametrize(**PARAMETRIZE_KWARGS)
def test_generate_random_path_allowed_start_and_end(maze):
	allowed_start = [(0, 0)]
	allowed_end = [(1, 1)]
	path = maze.generate_random_path(
		allowed_start=allowed_start,
		allowed_end=allowed_end,
	)
	assert path[0].tolist() == list(allowed_start[0])
	assert path[-1].tolist() == list(allowed_end[0])


@pytest.mark.parametrize(**PARAMETRIZE_KWARGS)
def test_generate_random_path_deadend_start_and_end(maze):
	path = maze.generate_random_path(deadend_start=True, deadend_end=True)
	assert len(maze.get_coord_neighbors(tuple(path[0]))) == 1
	assert len(maze.get_coord_neighbors(tuple(path[-1]))) == 1


@pytest.mark.parametrize("maze", [EXAMPLE_MAZE])
def test_generate_random_path_invalid_conditions(maze):
	with pytest.raises(NoValidEndpointException):
		maze.generate_random_path(allowed_start=[(2, 2)])

	with pytest.raises(NoValidEndpointException):
		maze.generate_random_path(allowed_end=[(2, 2)])


def test_generate_random_path_size_1():
	connection_list = np.zeros((1, 1, 1), dtype=bool)
	maze = LatticeMaze(connection_list=connection_list)
	with pytest.raises(AssertionError):
		maze.generate_random_path()

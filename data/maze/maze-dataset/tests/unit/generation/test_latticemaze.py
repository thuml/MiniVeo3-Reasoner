import numpy as np
import pytest

from maze_dataset.constants import CoordArray
from maze_dataset.generation.default_generators import DEFAULT_GENERATORS
from maze_dataset.generation.generators import GENERATORS_MAP
from maze_dataset.maze import LatticeMaze, PixelColors, SolvedMaze, TargetedLatticeMaze
from maze_dataset.utils import adj_list_to_nested_set, bool_array_from_string


# thanks to gpt for these tests of _from_pixel_grid
@pytest.fixture
def example_pixel_grid():
	return ~np.array(
		[
			[1, 1, 1, 1, 1],
			[1, 0, 0, 0, 1],
			[1, 1, 1, 0, 1],
			[1, 0, 0, 0, 1],
			[1, 1, 1, 1, 1],
		],
		dtype=bool,
	)


@pytest.fixture
def example_rgb_pixel_grid():
	return np.array(
		[
			[
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
			],
			[
				PixelColors.WALL,
				PixelColors.OPEN,
				PixelColors.OPEN,
				PixelColors.OPEN,
				PixelColors.WALL,
			],
			[
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
			],
			[
				PixelColors.WALL,
				PixelColors.OPEN,
				PixelColors.WALL,
				PixelColors.OPEN,
				PixelColors.WALL,
			],
			[
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
			],
		],
		dtype=np.uint8,
	)


def test_from_pixel_grid_bw(example_pixel_grid):
	connection_list, grid_shape = LatticeMaze._from_pixel_grid_bw(example_pixel_grid)

	assert isinstance(connection_list, np.ndarray)
	assert connection_list.shape == (2, 2, 2)
	assert np.all(connection_list[0] == np.array([[False, True], [False, False]]))
	assert np.all(connection_list[1] == np.array([[True, False], [True, False]]))
	assert grid_shape == (2, 2)


def test_from_pixel_grid_with_positions(example_rgb_pixel_grid):
	marked_positions = {
		"start": PixelColors.START,
		"end": PixelColors.END,
		"path": PixelColors.PATH,
	}

	(
		connection_list,
		grid_shape,
		out_positions,
	) = LatticeMaze._from_pixel_grid_with_positions(
		example_rgb_pixel_grid,
		marked_positions,
	)

	assert isinstance(connection_list, np.ndarray)
	assert connection_list.shape == (2, 2, 2)
	assert np.all(connection_list[0] == np.array([[False, False], [False, False]]))
	assert np.all(connection_list[1] == np.array([[True, False], [False, False]]))
	assert grid_shape == (2, 2)

	assert isinstance(out_positions, dict)
	assert len(out_positions) == 3

	assert "start" in out_positions
	assert "end" in out_positions

	assert isinstance(out_positions["start"], np.ndarray)
	assert isinstance(out_positions["end"], np.ndarray)
	assert isinstance(out_positions["path"], np.ndarray)

	assert out_positions["start"].shape == (0,)
	assert out_positions["end"].shape == (0,)
	assert out_positions["path"].shape == (0,)


def test_find_start_end_points_in_rgb_pixel_grid():
	rgb_pixel_grid_with_positions = np.array(
		[
			[
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
			],
			[
				PixelColors.WALL,
				PixelColors.START,
				PixelColors.OPEN,
				PixelColors.END,
				PixelColors.WALL,
			],
			[
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
			],
			[
				PixelColors.WALL,
				PixelColors.OPEN,
				PixelColors.WALL,
				PixelColors.OPEN,
				PixelColors.WALL,
			],
			[
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
				PixelColors.WALL,
			],
		],
		dtype=np.uint8,
	)

	marked_positions = {
		"start": PixelColors.START,
		"end": PixelColors.END,
		"path": PixelColors.PATH,
	}

	(
		connection_list,
		grid_shape,
		out_positions,
	) = LatticeMaze._from_pixel_grid_with_positions(
		rgb_pixel_grid_with_positions,
		marked_positions,
	)

	print(f"{out_positions = }")

	assert isinstance(out_positions, dict)
	assert len(out_positions) == 3
	assert "start" in out_positions
	assert "end" in out_positions
	assert isinstance(out_positions["start"], np.ndarray)
	assert isinstance(out_positions["end"], np.ndarray)
	assert isinstance(out_positions["path"], np.ndarray)

	assert np.all(out_positions["start"] == np.array([[0, 0]]))
	assert np.all(out_positions["end"] == np.array([[0, 1]]))
	assert out_positions["path"].shape == (0,)


@pytest.mark.parametrize(("gfunc_name", "kwargs"), DEFAULT_GENERATORS)
def test_pixels_ascii_roundtrip(gfunc_name, kwargs):
	"""tests all generators work and can be written to/from ascii and pixels"""
	n: int = 5
	maze_gen_func = GENERATORS_MAP[gfunc_name]
	maze: LatticeMaze = maze_gen_func(np.array([n, n]), **kwargs)

	maze_pixels: np.ndarray = maze.as_pixels()
	maze_ascii: str = maze.as_ascii()

	assert maze == LatticeMaze.from_pixels(maze_pixels)
	assert maze == LatticeMaze.from_ascii(maze_ascii)

	expected_shape: tuple = (n * 2 + 1, n * 2 + 1, 3)
	assert maze_pixels.shape == expected_shape, (
		f"{maze_pixels.shape} != {expected_shape}"
	)
	assert all(n * 2 + 1 == len(line) for line in maze_ascii.splitlines()), (
		f"{maze_ascii}"
	)


@pytest.mark.parametrize(("gfunc_name", "kwargs"), DEFAULT_GENERATORS)
def test_targeted_solved_maze(gfunc_name, kwargs):
	n: int = 5
	maze_gen_func = GENERATORS_MAP[gfunc_name]
	maze: LatticeMaze = maze_gen_func(np.array([n, n]), **kwargs)
	solution: CoordArray = maze.generate_random_path()
	tgt_maze: TargetedLatticeMaze = TargetedLatticeMaze.from_lattice_maze(
		maze,
		solution[0],
		solution[-1],
	)

	tgt_maze_pixels: np.ndarray = tgt_maze.as_pixels()
	tgt_maze_ascii: str = tgt_maze.as_ascii()

	assert tgt_maze == TargetedLatticeMaze.from_pixels(tgt_maze_pixels)
	assert tgt_maze == TargetedLatticeMaze.from_ascii(tgt_maze_ascii)

	expected_shape: tuple = (n * 2 + 1, n * 2 + 1, 3)
	assert tgt_maze_pixels.shape == expected_shape, (
		f"{tgt_maze_pixels.shape} != {expected_shape}"
	)
	assert all(n * 2 + 1 == len(line) for line in tgt_maze_ascii.splitlines()), (
		f"{tgt_maze_ascii}"
	)

	solved_maze: SolvedMaze = SolvedMaze.from_targeted_lattice_maze(tgt_maze)

	solved_maze_pixels: np.ndarray = solved_maze.as_pixels()
	solved_maze_ascii: str = solved_maze.as_ascii()

	assert solved_maze == SolvedMaze.from_pixels(solved_maze_pixels)
	assert solved_maze == SolvedMaze.from_ascii(solved_maze_ascii)

	expected_shape: tuple = (n * 2 + 1, n * 2 + 1, 3)
	assert tgt_maze_pixels.shape == expected_shape, (
		f"{tgt_maze_pixels.shape} != {expected_shape}"
	)
	assert all(n * 2 + 1 == len(line) for line in solved_maze_ascii.splitlines()), (
		f"{solved_maze_ascii}"
	)


def test_as_adj_list():
	connection_list = bool_array_from_string(
		"""
        F T
        F F

        T F
        T F
        """,
		shape=[2, 2, 2],
	)

	maze = LatticeMaze(connection_list=connection_list)

	adj_list = maze.as_adj_list(shuffle_d0=False, shuffle_d1=False)

	expected = [[[0, 1], [1, 1]], [[0, 0], [0, 1]], [[1, 0], [1, 1]]]

	assert adj_list_to_nested_set(expected) == adj_list_to_nested_set(adj_list)


@pytest.mark.parametrize(("gfunc_name", "kwargs"), DEFAULT_GENERATORS)
def test_get_nodes(gfunc_name, kwargs):
	maze_gen_func = GENERATORS_MAP[gfunc_name]
	maze = maze_gen_func(np.array((3, 2)), **kwargs)
	assert (
		maze.get_nodes().tolist()
		== np.array([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]).tolist()
	)


@pytest.mark.parametrize(("gfunc_name", "kwargs"), DEFAULT_GENERATORS)
def test_generate_random_path(gfunc_name, kwargs):
	maze_gen_func = GENERATORS_MAP[gfunc_name]
	maze = maze_gen_func(np.array((2, 2)), **kwargs)
	path = maze.generate_random_path()

	# len > 1 ensures that we have unique start and end nodes
	assert len(path) > 1


@pytest.mark.parametrize(("gfunc_name", "kwargs"), DEFAULT_GENERATORS)
def test_generate_random_path_size_1(gfunc_name, kwargs):
	maze_gen_func = GENERATORS_MAP[gfunc_name]
	maze = maze_gen_func(np.array((1, 1)), **kwargs)
	with pytest.raises(AssertionError):
		maze.generate_random_path()

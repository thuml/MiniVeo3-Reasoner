import numpy as np

from maze_dataset.generation.generators import get_neighbors_in_bounds


def test_middle_point():
	coord = np.array([2, 2])
	grid_shape = np.array([5, 5])
	expected_neighbors = np.array([[2, 3], [2, 1], [3, 2], [1, 2]])
	neighbors = get_neighbors_in_bounds(coord, grid_shape)
	assert np.array_equal(neighbors, expected_neighbors), (
		f"{neighbors} != {expected_neighbors}"
	)


def test_corner_point():
	coord = np.array([0, 0])
	grid_shape = np.array([5, 5])
	expected_neighbors = np.array([[0, 1], [1, 0]])
	neighbors = get_neighbors_in_bounds(coord, grid_shape)
	assert np.array_equal(neighbors, expected_neighbors), (
		f"{neighbors} != {expected_neighbors}"
	)


def test_edge_point():
	coord = np.array([0, 2])
	grid_shape = np.array([5, 5])
	expected_neighbors = np.array([[0, 3], [0, 1], [1, 2]])
	neighbors = get_neighbors_in_bounds(coord, grid_shape)
	assert np.array_equal(neighbors, expected_neighbors), (
		f"{neighbors} != {expected_neighbors}"
	)


def test_single_point_grid():
	coord = np.array([0, 0])
	grid_shape = np.array([1, 1])
	expected_neighbors = np.empty((0, 2))
	neighbors = get_neighbors_in_bounds(coord, grid_shape)
	assert np.array_equal(neighbors, expected_neighbors), (
		f"{neighbors} != {expected_neighbors}"
	)

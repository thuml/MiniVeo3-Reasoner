"""Tests for custom maze generator registration system"""

from pathlib import Path

import numpy as np
import pytest
from zanj import ZANJ

from maze_dataset import LatticeMaze, MazeDataset, MazeDatasetConfig
from maze_dataset.constants import Coord, CoordTup
from maze_dataset.generation import (
	GENERATORS_MAP,
	LatticeMazeGenerators,
	get_maze_with_solution,
)
from maze_dataset.generation.registration import (
	MazeGeneratorRegistrationError,
	register_maze_generator,
)

# Temp directory for file operations
TEMP_PATH = Path("tests/_temp/test_custom_generator/")


def test_register_valid_function():
	"""Test that a valid function can be registered successfully"""

	@register_maze_generator
	def gen_test_valid(
		grid_shape: Coord | CoordTup,
		lattice_dim: int = 2,
	) -> LatticeMaze:
		"""Simple test generator - fully connected grid"""
		grid_shape_: Coord = np.array(grid_shape)
		connection_list: np.ndarray = np.zeros(
			(lattice_dim, *grid_shape_), dtype=np.bool_
		)

		# Create fully connected grid
		if grid_shape_[1] > 1:
			connection_list[1, :, : grid_shape_[1] - 1] = True
		if grid_shape_[0] > 1:
			connection_list[0, : grid_shape_[0] - 1, :] = True

		return LatticeMaze(
			connection_list=connection_list,
			generation_meta=dict(
				func_name="gen_test_valid",
				grid_shape=grid_shape_,
				fully_connected=True,
			),
		)

	# Test registration worked
	assert "gen_test_valid" in GENERATORS_MAP
	assert hasattr(LatticeMazeGenerators, "gen_test_valid")

	# Test function works
	maze = get_maze_with_solution("gen_test_valid", (5, 5))
	assert maze.grid_shape == (5, 5)

	# Test via LatticeMazeGenerators
	maze2 = LatticeMazeGenerators.gen_test_valid((4, 4))
	assert maze2.grid_shape == (4, 4)


def test_maze_dataset_config_with_custom_generator():
	"""Test creating, saving, and loading MazeDatasetConfig with custom generator"""

	@register_maze_generator
	def gen_test_config(
		grid_shape: Coord | CoordTup,
		custom_param: float = 0.5,
	) -> LatticeMaze:
		"""Test generator with custom parameter"""
		grid_shape_: Coord = np.array(grid_shape)
		connection_list: np.ndarray = np.zeros((2, *grid_shape_), dtype=np.bool_)

		# Simple fully connected pattern
		if grid_shape_[1] > 1:
			connection_list[1, :, : grid_shape_[1] - 1] = True
		if grid_shape_[0] > 1:
			connection_list[0, : grid_shape_[0] - 1, :] = True

		return LatticeMaze(
			connection_list=connection_list,
			generation_meta=dict(
				func_name="gen_test_config",
				grid_shape=grid_shape_,
				custom_param=custom_param,
				fully_connected=True,
			),
		)

	# Create config with custom generator
	config = MazeDatasetConfig(
		name="test_custom",
		grid_n=5,
		n_mazes=3,
		maze_ctor=gen_test_config,
		maze_ctor_kwargs={"custom_param": 0.7},
	)

	# Test serialization/deserialization
	serialized = config.serialize()
	loaded_config = MazeDatasetConfig.load(serialized)

	assert loaded_config.name == config.name
	assert loaded_config.grid_n == config.grid_n
	assert loaded_config.n_mazes == config.n_mazes
	assert loaded_config.maze_ctor_kwargs == config.maze_ctor_kwargs

	# Test save/load to file using ZANJ
	TEMP_PATH.mkdir(parents=True, exist_ok=True)
	config_path = TEMP_PATH / "test_config.zanj"

	z = ZANJ()
	z.save(config, config_path)
	file_loaded_config = z.read(config_path)

	assert file_loaded_config.name == config.name
	assert file_loaded_config.maze_ctor_kwargs == config.maze_ctor_kwargs


def test_maze_dataset_with_custom_generator():
	"""Test creating, saving, and loading MazeDataset with custom generator"""

	@register_maze_generator
	def gen_test_dataset(
		grid_shape: Coord | CoordTup,
		lattice_dim: int = 2,
	) -> LatticeMaze:
		"""Test generator for dataset creation"""
		grid_shape_: Coord = np.array(grid_shape)
		connection_list: np.ndarray = np.zeros(
			(lattice_dim, *grid_shape_), dtype=np.bool_
		)

		# Create simple pattern - connect every cell to its right/down neighbor
		if grid_shape_[1] > 1:
			connection_list[1, :, : grid_shape_[1] - 1] = True
		if grid_shape_[0] > 1:
			connection_list[0, : grid_shape_[0] - 1, :] = True

		return LatticeMaze(
			connection_list=connection_list,
			generation_meta=dict(
				func_name="gen_test_dataset",
				grid_shape=grid_shape_,
				fully_connected=True,
			),
		)

	# Create config and generate dataset
	config = MazeDatasetConfig(
		name="test_dataset",
		grid_n=4,
		n_mazes=2,
		maze_ctor=gen_test_dataset,
		maze_ctor_kwargs={},
	)

	dataset = MazeDataset.generate(config, gen_parallel=False)

	# Test dataset properties
	assert len(dataset) == 2
	for maze in dataset:
		assert maze.grid_shape == (4, 4)

	# Test save/load dataset
	TEMP_PATH.mkdir(parents=True, exist_ok=True)
	dataset_path = TEMP_PATH / "test_dataset.zanj"

	dataset.save(dataset_path)
	loaded_dataset = MazeDataset.read(dataset_path)

	assert len(loaded_dataset) == len(dataset)
	assert loaded_dataset.cfg.name == dataset.cfg.name
	for original, loaded in zip(dataset, loaded_dataset, strict=True):
		assert original.grid_shape == loaded.grid_shape
		assert np.array_equal(original.connection_list, loaded.connection_list)


# bunch of type ignores here, because we are testing to make sure that
# the registration system raises errors for invalid function signatures


def test_registration_error_missing_grid_shape():
	"""Test error when function is missing grid_shape parameter"""

	def invalid_missing_grid_shape(x):
		assert x  # Use parameter to avoid warning
		return LatticeMaze(np.zeros((2, 3, 3), dtype=np.bool_), {})

	with pytest.raises(
		MazeGeneratorRegistrationError,
		match="must have 'grid_shape' as its first parameter",
	):
		register_maze_generator(invalid_missing_grid_shape)  # type: ignore[type-var]


def test_registration_error_wrong_param_name():
	"""Test error when first parameter has wrong name"""

	def invalid_wrong_param_name(shape):
		assert shape  # Use parameter to avoid warning
		return LatticeMaze(np.zeros((2, 3, 3), dtype=np.bool_), {})

	with pytest.raises(
		MazeGeneratorRegistrationError,
		match="must have 'grid_shape' as its first parameter",
	):
		register_maze_generator(invalid_wrong_param_name)  # type: ignore[type-var]


def test_registration_error_missing_type_annotation():
	"""Test error when grid_shape lacks type annotation"""

	def invalid_missing_type_annotation(grid_shape):
		assert grid_shape  # Use parameter to avoid warning
		return LatticeMaze(np.zeros((2, 3, 3), dtype=np.bool_), {})

	with pytest.raises(
		MazeGeneratorRegistrationError,
		match=r"must be typed as 'Coord \| CoordTup' or compatible type",
	):
		register_maze_generator(invalid_missing_type_annotation)  # type: ignore[type-var]


def test_registration_error_missing_return_annotation():
	"""Test error when function lacks return type annotation"""

	def invalid_missing_return_annotation(grid_shape: Coord | CoordTup):
		assert grid_shape is not None  # Use parameter to avoid warning
		return LatticeMaze(np.zeros((2, 3, 3), dtype=np.bool_), {})

	with pytest.raises(
		MazeGeneratorRegistrationError,
		match="must have a return type annotation of LatticeMaze",
	):
		register_maze_generator(invalid_missing_return_annotation)  # type: ignore[type-var]


def test_registration_error_wrong_return_type():
	"""Test error when function has wrong return type annotation"""

	def invalid_wrong_return_type(grid_shape: Coord | CoordTup) -> str:
		assert grid_shape is not None  # Use parameter to avoid warning
		return "wrong"

	with pytest.raises(MazeGeneratorRegistrationError, match="must return LatticeMaze"):
		register_maze_generator(invalid_wrong_return_type)  # type: ignore[type-var]


def test_registration_error_invalid_grid_shape_type():
	"""Test error when grid_shape has invalid type annotation"""

	def invalid_grid_shape_type(grid_shape: str) -> LatticeMaze:
		assert grid_shape  # Use parameter to avoid warning
		return LatticeMaze(np.zeros((2, 3, 3), dtype=np.bool_), {})

	with pytest.raises(
		MazeGeneratorRegistrationError,
		match=r"must be typed as 'Coord \| CoordTup' or compatible type",
	):
		register_maze_generator(invalid_grid_shape_type)  # type: ignore[type-var]


def test_duplicate_registration_error():
	"""Test that registering a function with an existing name raises an error"""

	@register_maze_generator
	def gen_test_duplicate_unique(
		grid_shape: Coord | CoordTup,
	) -> LatticeMaze:
		"""First registration"""
		assert grid_shape is not None  # Use parameter to avoid warning
		return LatticeMaze(
			np.zeros((2, 3, 3), dtype=np.bool_),
			generation_meta={
				"func_name": "gen_test_duplicate_unique",
				"fully_connected": True,
			},
		)

	# Try to register another function with the same name
	# type ignore because we are intentionally using the same name
	def gen_test_duplicate_unique(  # type: ignore[no-redef] # noqa: F811
		grid_shape: Coord | CoordTup,
	) -> LatticeMaze:
		"""Second registration attempt with same name"""
		assert grid_shape is not None  # Use parameter to avoid warning
		return LatticeMaze(
			np.zeros((2, 3, 3), dtype=np.bool_),
			generation_meta={
				"func_name": "gen_test_duplicate_unique",
				"fully_connected": True,
			},
		)

	with pytest.raises(ValueError, match="already exists in GENERATORS_MAP"):
		register_maze_generator(gen_test_duplicate_unique)

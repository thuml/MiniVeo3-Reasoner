from pathlib import Path

import numpy as np
import pytest
from muutils.json_serialize.util import _FORMAT_KEY

from maze_dataset import (
	MazeDataset,
	MazeDatasetCollection,
	MazeDatasetCollectionConfig,
	MazeDatasetConfig,
)
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.maze import SolvedMaze

# Define a temp path for file operations
TEMP_PATH: Path = Path("tests/_temp/maze_dataset_collection/")


@pytest.fixture(scope="module", autouse=True)
def setup_temp_dir():
	"""Create temporary directory for tests."""
	TEMP_PATH.mkdir(parents=True, exist_ok=True)
	# No cleanup as requested


@pytest.fixture
def small_configs():
	"""Create a list of small MazeDatasetConfig objects for testing."""
	return [
		MazeDatasetConfig(
			name=f"test_{i}",
			grid_n=3,
			n_mazes=2,
			maze_ctor=LatticeMazeGenerators.gen_dfs,
		)
		for i in range(2)
	]


@pytest.fixture
def small_datasets(small_configs):
	"""Create a list of small MazeDataset objects for testing."""
	return [
		MazeDataset.from_config(
			cfg, do_download=False, load_local=False, save_local=False
		)
		for cfg in small_configs
	]


@pytest.fixture
def collection_config(small_configs):
	"""Create a MazeDatasetCollectionConfig for testing."""
	return MazeDatasetCollectionConfig(
		name="test_collection",
		maze_dataset_configs=small_configs,
	)


@pytest.fixture
def collection(small_datasets, collection_config):
	"""Create a MazeDatasetCollection for testing."""
	return MazeDatasetCollection(
		cfg=collection_config,
		maze_datasets=small_datasets,
	)


def test_dataset_lengths(collection, small_datasets):
	"""Test that dataset_lengths returns the correct length for each dataset."""
	expected_lengths = [len(ds) for ds in small_datasets]
	assert collection.dataset_lengths == expected_lengths


def test_dataset_cum_lengths(collection):
	"""Test that dataset_cum_lengths returns the correct cumulative lengths."""
	expected_cum_lengths = np.array([2, 4])  # [2, 2+2]
	assert np.array_equal(collection.dataset_cum_lengths, expected_cum_lengths)


def test_mazes_cached_property(collection, small_datasets):
	"""Test that the mazes cached_property correctly flattens all mazes."""
	expected_mazes = []
	for ds in small_datasets:
		expected_mazes.extend(ds.mazes)

	# Access property
	assert hasattr(collection, "mazes")
	mazes = collection.mazes

	# Check results
	assert len(mazes) == len(expected_mazes)
	assert mazes == expected_mazes


def test_getitem_across_datasets(collection, small_datasets):
	"""Test that __getitem__ correctly accesses mazes across dataset boundaries."""
	# First dataset
	assert collection[0] == small_datasets[0][0]
	assert collection[1] == small_datasets[0][1]

	# Second dataset
	assert collection[2] == small_datasets[1][0]
	assert collection[3] == small_datasets[1][1]


def test_iteration(collection):
	"""Test that the collection is iterable and returns all mazes."""
	mazes = list(collection)
	assert len(mazes) == 4
	assert all(isinstance(maze, SolvedMaze) for maze in mazes)


def test_generate_classmethod(collection_config):
	"""Test the generate class method creates a collection from config."""
	collection = MazeDatasetCollection.generate(
		collection_config, do_download=False, load_local=False, save_local=False
	)

	assert isinstance(collection, MazeDatasetCollection)
	assert len(collection) == 4
	assert collection.cfg == collection_config


def test_serialization_deserialization(collection):
	"""Test serialization and deserialization of the collection."""
	# Serialize
	serialized = collection.serialize()

	# Check keys
	assert _FORMAT_KEY in serialized
	assert serialized[_FORMAT_KEY] == "MazeDatasetCollection"
	assert "cfg" in serialized
	assert "maze_datasets" in serialized

	# Deserialize
	deserialized = MazeDatasetCollection.load(serialized)

	# Check properties
	assert deserialized.cfg.name == collection.cfg.name
	assert len(deserialized) == len(collection)


def test_save_and_read(collection):
	"""Test saving and reading a collection to/from a file."""
	file_path = TEMP_PATH / "test_collection.zanj"

	# Save
	collection.save(file_path)
	assert file_path.exists()

	# Read
	loaded = MazeDatasetCollection.read(file_path)
	assert len(loaded) == len(collection)
	assert loaded.cfg.name == collection.cfg.name


def test_as_tokens(collection):
	"""Test as_tokens method with different parameters."""
	# Create a simple tokenizer for testing
	from maze_dataset.tokenization import MazeTokenizerModular

	tokenizer = MazeTokenizerModular()

	# Test with join_tokens_individual_maze=False
	tokens = collection.as_tokens(tokenizer, limit=2, join_tokens_individual_maze=False)
	assert len(tokens) == 2
	assert all(isinstance(t, list) for t in tokens)

	# Test with join_tokens_individual_maze=True
	tokens_joined = collection.as_tokens(
		tokenizer, limit=2, join_tokens_individual_maze=True
	)
	assert len(tokens_joined) == 2
	assert all(isinstance(t, str) for t in tokens_joined)
	assert all(" " in t for t in tokens_joined)


def test_update_self_config(collection):
	"""Test that update_self_config correctly updates the config."""
	original_n_mazes = collection.cfg.n_mazes

	# Change the dataset size by removing a maze
	collection.maze_datasets[0].mazes.pop()

	# Update config
	collection.update_self_config()

	# Check the config is updated
	assert collection.cfg.n_mazes == original_n_mazes - 1


def test_max_grid_properties(collection_config):
	"""Test max_grid properties are calculated correctly."""
	assert collection_config.max_grid_n == 3
	assert collection_config.max_grid_shape == (3, 3)
	assert np.array_equal(collection_config.max_grid_shape_np, np.array([3, 3]))


def test_config_serialization(collection_config):
	"""Test that the collection config serializes and deserializes correctly."""
	serialized = collection_config.serialize()
	deserialized = MazeDatasetCollectionConfig.load(serialized)

	assert deserialized.name == collection_config.name
	assert len(deserialized.maze_dataset_configs) == len(
		collection_config.maze_dataset_configs
	)

	# Test summary method
	summary = collection_config.summary()
	assert "n_mazes" in summary
	assert "max_grid_n" in summary
	assert summary["n_mazes"] == 4


def test_mixed_grid_sizes():
	"""Test a collection with different grid sizes."""
	configs = [
		MazeDatasetConfig(
			name=f"test_grid_{i}",
			grid_n=i + 3,  # 3, 4
			n_mazes=2,
			maze_ctor=LatticeMazeGenerators.gen_dfs,
		)
		for i in range(2)
	]

	datasets = [
		MazeDataset.from_config(
			cfg, do_download=False, load_local=False, save_local=False
		)
		for cfg in configs
	]

	collection_config = MazeDatasetCollectionConfig(
		name="mixed_grid_collection",
		maze_dataset_configs=configs,
	)

	collection = MazeDatasetCollection(
		cfg=collection_config,
		maze_datasets=datasets,
	)

	# The max grid size should be the largest one
	assert collection.cfg.max_grid_n == 4
	assert collection.cfg.max_grid_shape == (4, 4)


def test_different_generation_methods():
	"""Test a collection with different generation methods."""
	configs = [
		MazeDatasetConfig(
			name="dfs_test",
			grid_n=3,
			n_mazes=2,
			maze_ctor=LatticeMazeGenerators.gen_dfs,
		),
		MazeDatasetConfig(
			name="percolation_test",
			grid_n=3,
			n_mazes=2,
			maze_ctor=LatticeMazeGenerators.gen_percolation,
			maze_ctor_kwargs={"p": 0.7},
		),
	]

	datasets = [
		MazeDataset.from_config(
			cfg, do_download=False, load_local=False, save_local=False
		)
		for cfg in configs
	]

	collection_config = MazeDatasetCollectionConfig(
		name="mixed_gen_collection",
		maze_dataset_configs=configs,
	)

	collection = MazeDatasetCollection(
		cfg=collection_config,
		maze_datasets=datasets,
	)

	# Check that the collection has all mazes
	assert len(collection) == 4

	# Check that the mazes are of different types based on their generation metadata
	# type ignore here since it might be None, but if its None that will cause an error anyways
	# For DFS
	assert collection[0].generation_meta.get("func_name") == "gen_dfs"  # type: ignore[union-attr]
	# For percolation
	assert collection[2].generation_meta.get("func_name") == "gen_percolation"  # type: ignore[union-attr]
	assert collection[2].generation_meta.get("percolation_p") == 0.7  # type: ignore[union-attr]

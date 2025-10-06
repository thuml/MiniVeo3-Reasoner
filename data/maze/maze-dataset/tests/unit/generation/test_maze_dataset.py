import copy
from pathlib import Path

import numpy as np
import pytest
from zanj import ZANJ

from maze_dataset import (
	MazeDataset,
	MazeDatasetConfig,
	register_maze_filter,
	set_serialize_minimal_threshold,
)
from maze_dataset.constants import CoordArray
from maze_dataset.dataset.dataset import (
	register_dataset_filter,
	register_filter_namespace_for_dataset,
)
from maze_dataset.generation.generators import GENERATORS_MAP
from maze_dataset.maze import SolvedMaze
from maze_dataset.utils import bool_array_from_string


class TestMazeDatasetConfig:
	pass


TEST_CONFIGS = [
	MazeDatasetConfig(
		name="test",
		grid_n=grid_n,
		n_mazes=n_mazes,
		maze_ctor=GENERATORS_MAP["gen_dfs"],
		maze_ctor_kwargs=maze_ctor_kwargs,
	)
	for grid_n, n_mazes, maze_ctor_kwargs in [
		(3, 5, {}),
		(3, 1, {}),
		(5, 5, dict(do_forks=False)),
	]
]


def test_generate_serial():
	dataset = MazeDataset.generate(TEST_CONFIGS[0], gen_parallel=False)

	assert len(dataset) == 5
	for maze in dataset:
		assert maze.grid_shape == (3, 3)


def test_generate_parallel():
	dataset = MazeDataset.generate(
		TEST_CONFIGS[0],
		gen_parallel=True,
		verbose=True,
		pool_kwargs=dict(processes=2),
	)

	assert len(dataset) == 5
	for maze in dataset:
		assert maze.grid_shape == (3, 3)


def test_data_hash_wip():
	dataset = MazeDataset.generate(TEST_CONFIGS[0])
	# TODO: dataset.data_hash doesn't work right now
	assert dataset


def test_download():
	with pytest.raises(NotImplementedError):
		MazeDataset.download(TEST_CONFIGS[0])


def test_serialize_load():
	dataset = MazeDataset.generate(TEST_CONFIGS[0])
	dataset_copy = MazeDataset.load(dataset.serialize())

	assert dataset.cfg == dataset_copy.cfg
	for maze, maze_copy in zip(dataset, dataset_copy, strict=False):
		assert maze == maze_copy


@pytest.mark.parametrize(
	"config",
	[
		pytest.param(
			c,
			id=f"{c.grid_n=}; {c.n_mazes=}; {c.maze_ctor_kwargs=}",
		)
		for c in TEST_CONFIGS
	],
)
def test_serialize_load_minimal(config):
	d = MazeDataset.generate(config, gen_parallel=False)
	d_loaded = MazeDataset.load(d._serialize_minimal())
	d_loaded.assert_equal(d)
	assert d_loaded == d


@pytest.mark.parametrize(
	"config",
	[
		pytest.param(
			c,
			id=f"{c.grid_n=}; {c.n_mazes=}; {c.maze_ctor_kwargs=}",
		)
		for c in TEST_CONFIGS
	],
)
def test_save_read_minimal(config):
	def save_and_read(d: MazeDataset, p: str):
		d.save(file_path=p)
		# read as MazeDataset
		roundtrip = MazeDataset.read(p)
		assert roundtrip == d
		# read from zanj
		z = ZANJ()
		roundtrip_zanj = z.read(p)
		assert roundtrip_zanj == d

	d = MazeDataset.generate(config, gen_parallel=False)
	p = Path("tests/_temp/test_maze_dataset/") / (d.cfg.to_fname() + ".zanj")

	# Test with full serialization
	set_serialize_minimal_threshold(None)
	save_and_read(d, p)

	# Test with minimal serialization
	set_serialize_minimal_threshold(0)
	save_and_read(d, p)

	d.save(file_path=p)
	# read as MazeDataset
	roundtrip = MazeDataset.read(p)
	assert d.cfg.diff(roundtrip.cfg) == dict()
	cfg_diff = roundtrip.cfg.diff(d.cfg)
	assert cfg_diff == {}
	assert roundtrip.cfg == d.cfg
	assert roundtrip.mazes == d.mazes
	assert roundtrip == d
	# read from zanj
	z = ZANJ()
	roundtrip_zanj = z.read(p)
	assert roundtrip_zanj == d


def test_custom_maze_filter():
	connection_list = bool_array_from_string(
		"""
        F T
        F F

        T F
        T F
        """,
		shape=[2, 2, 2],
	)
	solutions = [
		[[0, 0], [0, 1], [1, 1]],
		[[0, 0], [0, 1]],
		[[0, 0]],
	]

	def custom_filter_solution_length(maze: SolvedMaze, solution_length: int) -> bool:
		return len(maze.solution) == solution_length

	mazes = [
		SolvedMaze(connection_list=connection_list, solution=solution)
		for solution in solutions
	]
	dataset = MazeDataset(cfg=TEST_CONFIGS[0], mazes=mazes)

	filtered_lambda = dataset.custom_maze_filter(lambda m: len(m.solution) == 1)
	filtered_func = dataset.custom_maze_filter(
		custom_filter_solution_length,
		solution_length=1,
	)

	assert filtered_lambda.mazes == filtered_func.mazes == [mazes[2]]


class TestMazeDatasetFilters:
	config = MazeDatasetConfig(name="test", grid_n=3, n_mazes=5)
	connection_list = bool_array_from_string(
		"""
        F T
        F F

        T F
        T F
        """,
		shape=[2, 2, 2],
	)

	def test_filters(self):
		class TestDataset(MazeDataset): ...

		@register_filter_namespace_for_dataset(TestDataset)
		class TestFilters:
			@register_maze_filter
			@staticmethod
			def solution_match(maze: SolvedMaze, solution: CoordArray) -> bool:
				"""Test for solution equality"""
				return (maze.solution == solution).all()

			@register_dataset_filter
			@staticmethod
			def drop_nth(dataset: TestDataset, n: int) -> TestDataset:
				"""Filter mazes by path length"""
				return copy.deepcopy(
					TestDataset(
						dataset.cfg,
						[maze for i, maze in enumerate(dataset) if i != n],
					),
				)

		maze1 = SolvedMaze(
			connection_list=self.connection_list,
			solution=np.array([[0, 0]]),
		)
		maze2 = SolvedMaze(
			connection_list=self.connection_list,
			solution=np.array([[0, 1]]),
		)

		dataset = TestDataset(self.config, [maze1, maze2])

		maze_filter = dataset.filter_by.solution_match(solution=np.array([[0, 0]]))
		maze_filter2 = dataset.filter_by.solution_match(np.array([[0, 0]]))

		dataset_filter = dataset.filter_by.drop_nth(n=0)
		dataset_filter2 = dataset.filter_by.drop_nth(0)

		assert maze_filter.mazes == maze_filter2.mazes == [maze1]
		assert dataset_filter.mazes == dataset_filter2.mazes == [maze2]

	def test_path_length(self):
		long_maze = SolvedMaze(
			connection_list=self.connection_list,
			solution=np.array([[0, 0], [0, 1], [1, 1]]),
		)

		short_maze = SolvedMaze(
			connection_list=self.connection_list,
			solution=np.array([[0, 0], [0, 1]]),
		)

		dataset = MazeDataset(self.config, [long_maze, short_maze])
		path_length_filtered = dataset.filter_by.path_length(3)
		start_end_filtered = dataset.filter_by.start_end_distance(2)

		assert type(path_length_filtered) == type(dataset)  # noqa: E721
		assert path_length_filtered.mazes == [long_maze]
		assert start_end_filtered.mazes == [long_maze]
		assert dataset.mazes == [long_maze, short_maze]

	def test_cut_percentile_shortest(self):
		solutions = [
			[[0, 0], [0, 1], [1, 1]],
			[[0, 0], [0, 1]],
			[[0, 0]],
		]

		mazes = [
			SolvedMaze(connection_list=self.connection_list, solution=solution)
			for solution in solutions
		]
		dataset = MazeDataset(cfg=self.config, mazes=mazes)
		filtered = dataset.filter_by.cut_percentile_shortest(49.0)

		assert filtered.mazes == mazes[:2]


DUPE_DATASET = [
	"""
#####
#  E#
###X#
#SXX#
#####
""",
	"""
#####
#SXE#
### #
#   #
#####
""",
	"""
#####
#  E#
###X#
#SXX#
#####
""",
	"""
#####
# # #
# # #
#EXS#
#####
""",
	"""
#####
#SXX#
###X#
#EXX#
#####
""",
]


def _helper_dataset_from_ascii(ascii_rep: str) -> MazeDataset:
	mazes: list[SolvedMaze] = list()
	for maze_ascii in ascii_rep:
		# TODO: PERF401 Use `list.extend` to create a transformed list
		mazes.append(SolvedMaze.from_ascii(maze_ascii.strip()))

	return MazeDataset(
		MazeDatasetConfig(
			name="test",
			grid_n=mazes[0].grid_shape[0],
			n_mazes=len(mazes),
		),
		mazes,
	)


def test_remove_duplicates():
	dataset: MazeDataset = _helper_dataset_from_ascii(DUPE_DATASET)
	dataset_deduped: MazeDataset = dataset.filter_by.remove_duplicates()

	assert len(dataset) == 5
	assert dataset_deduped.mazes == [dataset.mazes[3], dataset.mazes[4]]


def test_data_hash():
	dataset: MazeDataset = _helper_dataset_from_ascii(DUPE_DATASET)
	hash_1 = dataset.data_hash()
	hash_2 = dataset.data_hash()

	assert hash_1 == hash_2


def test_remove_duplicates_fast():
	dataset: MazeDataset = _helper_dataset_from_ascii(DUPE_DATASET)
	dataset_deduped: MazeDataset = dataset.filter_by.remove_duplicates_fast()

	assert len(dataset) == 5
	assert dataset_deduped.mazes == [
		dataset.mazes[0],
		dataset.mazes[1],
		dataset.mazes[3],
		dataset.mazes[4],
	]

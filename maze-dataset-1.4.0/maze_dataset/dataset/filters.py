"filtering `MazeDataset`s"

import copy
import functools
import typing
from collections import Counter, defaultdict

import numpy as np

from maze_dataset.constants import CoordTup
from maze_dataset.dataset.dataset import (
	DatasetFilterFunc,
	register_dataset_filter,
	register_filter_namespace_for_dataset,
)
from maze_dataset.dataset.maze_dataset import MazeDataset
from maze_dataset.maze import SolvedMaze


def register_maze_filter(
	method: typing.Callable[[SolvedMaze, typing.Any], bool],
) -> DatasetFilterFunc:
	"""register a maze filter, casting it to operate over the whole list of mazes

	method should be a staticmethod of a namespace class registered with `register_filter_namespace_for_dataset`

	this is a more restricted version of `register_dataset_filter` that removes the need for boilerplate for operating over the arrays
	"""

	@functools.wraps(method)
	def wrapper(dataset: MazeDataset, *args, **kwargs) -> MazeDataset:
		# copy and filter
		new_dataset: MazeDataset = copy.deepcopy(
			MazeDataset(
				cfg=dataset.cfg,
				mazes=[m for m in dataset.mazes if method(m, *args, **kwargs)],
			),
		)
		# update the config
		new_dataset.cfg.applied_filters.append(
			dict(name=method.__name__, args=args, kwargs=kwargs),
		)
		new_dataset.update_self_config()
		return new_dataset

	return wrapper


@register_filter_namespace_for_dataset(MazeDataset)
class MazeDatasetFilters:
	"namespace for filters for `MazeDataset`s"

	@register_maze_filter
	@staticmethod
	def path_length(maze: SolvedMaze, min_length: int) -> bool:
		"""filter out mazes with a solution length less than `min_length`"""
		return len(maze.solution) >= min_length

	@register_maze_filter
	@staticmethod
	def start_end_distance(maze: SolvedMaze, min_distance: int) -> bool:
		"""filter out datasets where the start and end pos are less than `min_distance` apart on the manhattan distance (ignoring walls)"""
		return bool(
			(np.linalg.norm(maze.start_pos - maze.end_pos, 1) >= min_distance).all()
		)

	@register_dataset_filter
	@staticmethod
	def cut_percentile_shortest(
		dataset: MazeDataset,
		percentile: float = 10.0,
	) -> MazeDataset:
		"""cut the shortest `percentile` of mazes from the dataset

		`percentile` is 1-100, not 0-1, as this is what `np.percentile` expects
		"""
		lengths: np.ndarray = np.array([len(m.solution) for m in dataset])
		cutoff: int = int(np.percentile(lengths, percentile))

		filtered_mazes: list[SolvedMaze] = [
			m for m in dataset if len(m.solution) > cutoff
		]
		new_dataset: MazeDataset = MazeDataset(cfg=dataset.cfg, mazes=filtered_mazes)

		return copy.deepcopy(new_dataset)

	@register_dataset_filter
	@staticmethod
	def truncate_count(
		dataset: MazeDataset,
		max_count: int,
	) -> MazeDataset:
		"""truncate the dataset to be at most `max_count` mazes"""
		new_dataset: MazeDataset = MazeDataset(
			cfg=dataset.cfg,
			mazes=dataset.mazes[:max_count],
		)
		return copy.deepcopy(new_dataset)

	@register_dataset_filter
	@staticmethod
	def remove_duplicates(
		dataset: MazeDataset,
		minimum_difference_connection_list: int | None = 1,
		minimum_difference_solution: int | None = 1,
		_max_dataset_len_threshold: int = 1000,
	) -> MazeDataset:
		"""remove duplicates from a dataset, keeping the **LAST** unique maze

		set minimum either minimum difference to `None` to disable checking

		if you want to avoid mazes which have more overlap, set the minimum difference to be greater

		Gotchas:
		- if two mazes are of different sizes, they will never be considered duplicates
		- if two solutions are of different lengths, they will never be considered duplicates

		TODO: check for overlap?
		"""
		if len(dataset) > _max_dataset_len_threshold:
			raise ValueError(
				"this method is currently very slow for large datasets, consider using `remove_duplicates_fast` instead\n",
				"if you know what you're doing, change `_max_dataset_len_threshold`",
			)

		unique_mazes: list[SolvedMaze] = list()

		maze_a: SolvedMaze
		maze_b: SolvedMaze
		for i, maze_a in enumerate(dataset.mazes):
			a_unique: bool = True
			for maze_b in dataset.mazes[i + 1 :]:
				# after all that nesting, more nesting to perform checks
				if (minimum_difference_connection_list is not None) and (  # noqa: SIM102
					maze_a.connection_list.shape == maze_b.connection_list.shape
				):
					if (
						np.sum(maze_a.connection_list != maze_b.connection_list)
						<= minimum_difference_connection_list
					):
						a_unique = False
						break

				if (minimum_difference_solution is not None) and (  # noqa: SIM102
					maze_a.solution.shape == maze_b.solution.shape
				):
					if (
						np.sum(maze_a.solution != maze_b.solution)
						<= minimum_difference_solution
					):
						a_unique = False
						break

			if a_unique:
				unique_mazes.append(maze_a)

		return copy.deepcopy(
			MazeDataset(
				cfg=dataset.cfg,
				mazes=unique_mazes,
				generation_metadata_collected=dataset.generation_metadata_collected,
			),
		)

	@register_dataset_filter
	@staticmethod
	def remove_duplicates_fast(dataset: MazeDataset) -> MazeDataset:
		"""remove duplicates from a dataset"""
		unique_mazes = list(dict.fromkeys(dataset.mazes))
		return copy.deepcopy(
			MazeDataset(
				cfg=dataset.cfg,
				mazes=unique_mazes,
				generation_metadata_collected=dataset.generation_metadata_collected,
			),
		)

	@register_dataset_filter
	@staticmethod
	def strip_generation_meta(dataset: MazeDataset) -> MazeDataset:
		"""strip the generation meta from the dataset"""
		new_dataset: MazeDataset = copy.deepcopy(dataset)
		for maze in new_dataset:
			# hacky because it's a frozen dataclass
			maze.__dict__["generation_meta"] = None
		return new_dataset

	@register_dataset_filter
	@staticmethod
	# yes, this function is complicated hence the noqa
	def collect_generation_meta(  # noqa: C901, PLR0912
		dataset: MazeDataset,
		clear_in_mazes: bool = True,
		inplace: bool = True,
		allow_fail: bool = False,
	) -> MazeDataset:
		"""collect the generation metadata from each maze into a dataset-level metadata (saves space)

		# Parameters:
		- `dataset : MazeDataset`
		- `clear_in_mazes : bool`
			whether to clear the generation meta in the mazes after collecting it, keep it there if `False`
			(defaults to `True`)
		- `inplace : bool`
			whether to modify the dataset in place or return a new one
			(defaults to `True`)
		- `allow_fail : bool`
			whether to allow the collection to fail if the generation meta is not present in a maze
			(defaults to `False`)

		# Returns:
		- `MazeDataset`
			the dataset with the generation metadata collected

		# Raises:
		- `ValueError` : if the generation meta is not present in a maze and `allow_fail` is `False`
		- `ValueError` : if we have other problems converting the generation metadata
		- `TypeError` : if the generation meta on a maze is of an unexpected type
		"""
		if dataset.generation_metadata_collected is not None:
			return dataset
		else:
			assert dataset[0].generation_meta is not None, (
				"generation meta is not collected and original is not present"
			)
		# if the generation meta is already collected, don't collect it again, do nothing

		new_dataset: MazeDataset
		if inplace:
			new_dataset = dataset
		else:
			new_dataset = copy.deepcopy(dataset)

		gen_meta_lists: dict[bool | int | float | str | CoordTup, Counter] = (
			defaultdict(Counter)
		)
		for maze in new_dataset:
			if maze.generation_meta is None:
				if allow_fail:
					break
				raise ValueError(
					"generation meta is not present in a maze, cannot collect generation meta",
				)
			for key, value in maze.generation_meta.items():
				if isinstance(value, (bool, int, float, str)):
					gen_meta_lists[key][value] += 1

				elif isinstance(value, set):
					# special case for visited_cells
					gen_meta_lists[key].update(value)

				elif isinstance(value, (list, np.ndarray)):
					if isinstance(value, list):
						# TODO: `for` loop variable `value` overwritten by assignment target (Ruff PLW2901)
						try:
							value = np.array(value)  # noqa: PLW2901
						except ValueError as convert_to_np_err:
							err_msg = (
								f"Cannot collect generation meta for {key} as it is a list of type '{type(value[0]) = !s}'"
								"\nexpected either a basic type (bool, int, float, str), a numpy coord, or a numpy array of coords"
							)
							raise ValueError(err_msg) from convert_to_np_err

					if (len(value.shape) == 1) and (value.shape[0] == maze.lattice_dim):
						# assume its a single coordinate
						gen_meta_lists[key][tuple(value)] += 1
					# magic value is fine here
					elif (len(value.shape) == 2) and (  # noqa: PLR2004
						value.shape[1] == maze.lattice_dim
					):
						# assume its a list of coordinates
						gen_meta_lists[key].update([tuple(v) for v in value])
					else:
						err_msg = (
							f"Cannot collect generation meta for {key} as it is an ndarray of shape {value.shape}\n"
							"expected either a coord of shape (2,) or a list of coords of shape (n, 2)"
						)
						raise ValueError(err_msg)
				else:
					err_msg = (
						f"Cannot collect generation meta for {key} as it is of type '{type(value)!s}'\n"
						"expected either a basic type (bool, int, float, str), a numpy coord, or a numpy array of coords"
					)
					raise TypeError(err_msg)

			# clear the data
			if clear_in_mazes:
				# hacky because it's a frozen dataclass
				maze.__dict__["generation_meta"] = None

		new_dataset.generation_metadata_collected = {
			key: dict(value) for key, value in gen_meta_lists.items()
		}

		return new_dataset

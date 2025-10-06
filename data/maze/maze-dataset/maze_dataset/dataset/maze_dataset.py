"""`MazeDatasetConfig` is where you decide what your dataset should look like, then pass it to `MazeDataset.from_config` to generate or load the dataset.

see [demo_dataset notebook](../../notebooks/demo_dataset)

"""

import copy
import json
import multiprocessing
import typing
import warnings
from pathlib import Path
from typing import Literal, cast, overload

import numpy as np
import tqdm
from jaxtyping import Int
from muutils.json_serialize import (
	json_serialize,
)
from muutils.json_serialize.util import (
	_FORMAT_KEY,
	JSONdict,
)
from muutils.misc import stable_hash
from zanj import ZANJ
from zanj.loading import LoaderHandler, load_item_recursive, register_loader_handler

from maze_dataset.constants import CoordArray
from maze_dataset.dataset.dataset import (
	GPTDataset,
)
from maze_dataset.dataset.maze_dataset_config import (
	SERIALIZE_MINIMAL_THRESHOLD,
	EndpointKwargsType,
	MazeDatasetConfig,
)
from maze_dataset.generation.seed import GLOBAL_SEED
from maze_dataset.maze import LatticeMaze, SolvedMaze

_GLOBAL_WORKER_CONFIG: MazeDatasetConfig


def _generate_maze_helper(index: int) -> SolvedMaze | None:  # noqa: ARG001
	"""Helper function for generating mazes in parallel.

	> [!CAUTION]
	> don't use this unless generating in parallel!
	"""
	global _GLOBAL_WORKER_CONFIG  # noqa: PLW0602
	# TODO: don't use this unless generating in parallel!
	maze: LatticeMaze = _GLOBAL_WORKER_CONFIG.maze_ctor(
		grid_shape=_GLOBAL_WORKER_CONFIG.grid_shape_np,
		**_GLOBAL_WORKER_CONFIG.maze_ctor_kwargs,
	)

	endpoint_kwargs: EndpointKwargsType = _GLOBAL_WORKER_CONFIG.endpoint_kwargs.copy()

	# Generate the solution
	# mypy doesnt realize EndpointKwargsType has only string keys: `Keywords must be strings  [misc]`
	# TYPING: error: No overload variant of "generate_random_path" of "LatticeMaze" matches argument type "dict[Literal['allowed_start', 'allowed_end', 'deadend_start', 'deadend_end', 'endpoints_not_equal', 'except_on_no_valid_endpoint'], bool | list[tuple[int, int]] | None]"  [call-overload]
	solution: CoordArray | None = maze.generate_random_path(**endpoint_kwargs)  # type: ignore[misc, call-overload]

	# Validate the solution
	if (
		solution is None
		or len(solution) == 0
		or not isinstance(solution, np.ndarray)
		# magic value is fine here
		or len(solution.shape) != 2  # noqa: PLR2004
	):
		return None  # Return None if the solution is invalid

	return SolvedMaze.from_lattice_maze(
		lattice_maze=maze,
		solution=solution,
	)


def _maze_gen_init_worker(config: MazeDatasetConfig) -> None:
	"""special worker helper

	> [!CAUTION]
	> this makes the generation depend both on whether parallelism is used, and on the number of processes. this is bad!

	"""
	# TODO: dont use globals here!
	global _GLOBAL_WORKER_CONFIG  # noqa: PLW0603
	_GLOBAL_WORKER_CONFIG = config

	process_id: tuple[int, ...] = multiprocessing.current_process()._identity
	if len(process_id) == 0:
		# no multiprocessing, seed was already set
		pass
	elif len(process_id) == 1:
		# multiprocessing, adjust seed based on process id
		# only set numpy seed, since we do not use other random gens
		np.random.seed(
			_GLOBAL_WORKER_CONFIG.seed
			or GLOBAL_SEED  # if the seed is None, use the global seed
			+ process_id[0]
		)
	else:
		err_msg = (
			f"unexpected process id: {process_id = }\n{multiprocessing.Process() = }"
		)
		raise ValueError(
			err_msg,
		)


# TODO: we probably don't need to hash datasets, right?
class MazeDataset(GPTDataset[MazeDatasetConfig]):  # noqa: PLW1641
	"""a maze dataset class. This is a collection of solved mazes, and should be initialized via `MazeDataset.from_config`"""

	def __init__(
		self,
		cfg: MazeDatasetConfig,
		mazes: typing.Sequence[SolvedMaze],
		generation_metadata_collected: dict | None = None,
	) -> None:
		"""initialize a maze dataset from a config and a list of solved mazes"""
		super().__init__()
		self.cfg: MazeDatasetConfig = cfg
		self.mazes: list[SolvedMaze] = list(mazes)
		self.generation_metadata_collected: dict | None = generation_metadata_collected

	# TYPING: error: Return type "MazeDataset" of "from_config" incompatible with return type "T_Dataset" in supertype "GPTDataset"  [override]
	@classmethod
	def from_config(  # type: ignore[override]
		cls,
		# TYPING: error: Argument 1 of "from_config" is incompatible with supertype "GPTDataset"; supertype defines the argument type as "T_DatasetConfig"  [override]
		cfg: MazeDatasetConfig,  # type: ignore[override]
		do_generate: bool = True,
		load_local: bool = True,
		save_local: bool = True,
		zanj: ZANJ | None = None,
		do_download: bool = True,
		local_base_path: Path = Path("data/maze_dataset"),
		except_on_config_mismatch: bool = True,
		allow_generation_metadata_filter_mismatch: bool = True,
		verbose: bool = False,
		**kwargs,
	) -> "MazeDataset":
		"""create a maze dataset from a config

		priority of loading:
		1. load from local
		2. download
		3. generate

		"""
		return cast(
			"MazeDataset",
			super().from_config(
				cfg=cfg,
				do_generate=do_generate,
				load_local=load_local,
				save_local=save_local,
				zanj=zanj,
				do_download=do_download,
				local_base_path=local_base_path,
				except_on_config_mismatch=except_on_config_mismatch,
				allow_generation_metadata_filter_mismatch=allow_generation_metadata_filter_mismatch,
				verbose=verbose,
				**kwargs,
			),
		)

	def data_hash(self) -> int:
		"""return a hash of the data"""
		return stable_hash(str(tuple([x.serialize() for x in self.mazes])))

	def __getitem__(self, i: int) -> SolvedMaze:
		"""get a maze by index"""
		return self.mazes[i]

	def __iter__(self) -> typing.Iterator[SolvedMaze]:
		"""iterate over the mazes"""
		return iter(self.mazes)

	def __deepcopy__(self, memo) -> "MazeDataset":  # noqa: ANN001
		"""deepcopy the dataset

		FIX: this isnt actually a deepcopy I think?
		"""
		return MazeDataset.load(self._serialize_full())

	# TYPING: get type hints on the tokenizer here
	@overload
	def as_tokens(
		self,
		maze_tokenizer,  # noqa: ANN001
		limit: int | None = None,
		join_tokens_individual_maze: Literal[False] = False,
	) -> list[list[str]]: ...
	@overload
	def as_tokens(
		self,
		maze_tokenizer,  # noqa: ANN001
		limit: int | None = None,
		join_tokens_individual_maze: Literal[True] = True,
	) -> list[str]: ...
	def as_tokens(
		self,
		maze_tokenizer,  # TODO: MazeTokenizer
		limit: int | None = None,
		join_tokens_individual_maze: bool = False,
	) -> list[list[str]] | list[str]:
		"""return the dataset as tokens according to the passed `maze_tokenizer`

		the `maze_tokenizer` should be either a `MazeTokenizer` or a `MazeTokenizerModular`

		if `join_tokens_individual_maze` is True, then the tokens of each maze are
		joined with a space, and the result is a list of strings.
		i.e.:

			>>> dataset.as_tokens(join_tokens_individual_maze=False)
			[["a", "b", "c"], ["d", "e", "f"]]
			>>> dataset.as_tokens(join_tokens_individual_maze=True)
			["a b c", "d e f"]
		"""
		output: list[list[str]] = [
			maze.as_tokens(maze_tokenizer) for maze in self.mazes[:limit]
		]
		if join_tokens_individual_maze:
			return [" ".join(tokens) for tokens in output]
		else:
			return output

	def __len__(self) -> int:
		"""return the number of mazes in the dataset"""
		return len(self.mazes)

	def __eq__(self, other: object) -> bool:
		"""compare two datasets"""
		if not isinstance(other, MazeDataset):
			raise NotImplementedError(
				"can only compare with other MazeDataset objects",
			)
		# TODO: compare hashes of data instead of the data itself?
		return self.cfg == other.cfg and self.mazes == other.mazes

	def assert_equal(self, other: "MazeDataset") -> None:
		"""assert that two datasets are equal"""
		assert isinstance(other, MazeDataset)
		assert self.cfg == other.cfg, f"{self.cfg.diff(other.cfg) = }"
		assert self.mazes == other.mazes, f"{self.mazes = }, {other.mazes = }"

	@classmethod
	def generate(
		cls,
		cfg: MazeDatasetConfig,
		gen_parallel: bool = False,
		pool_kwargs: dict | None = None,
		verbose: bool = False,
		# TODO: what to do when unexpected kwargs are passed?
		**kwargs,  # noqa: ARG003
	) -> "MazeDataset":
		"""Generate a maze dataset given a config and some generation parameters"""
		# Copy the config to avoid modifying the original
		cfg_cpy: MazeDatasetConfig = MazeDatasetConfig.load(
			json.loads(json.dumps(cfg.serialize())),
		)

		if pool_kwargs is None:
			pool_kwargs = dict()
		maze_indexes: Int[np.ndarray, " maze_index"] = np.arange(cfg_cpy.n_mazes)  # type: ignore[assignment]

		solved_mazes: list[SolvedMaze | None]
		# Configure tqdm for progress bar
		tqdm_kwargs: dict = dict(
			total=cfg_cpy.n_mazes,
			unit="maze",
			desc="generating & solving mazes",
			disable=not verbose,
		)
		# TODO: don't use the global unless generating in parallel!
		if gen_parallel:
			with multiprocessing.Pool(
				**pool_kwargs,
				initializer=_maze_gen_init_worker,
				initargs=(cfg_cpy,),
			) as pool:
				solved_mazes = list(
					tqdm.tqdm(
						pool.imap(_generate_maze_helper, maze_indexes),
						**tqdm_kwargs,
					),
				)

		else:
			_maze_gen_init_worker(cfg_cpy)
			solved_mazes = list(
				tqdm.tqdm(
					map(
						# TYPING:  error: Argument 1 to "map" has incompatible type "Callable[[int], SolvedMaze | None]"; expected "Callable[[str], SolvedMaze | None]"  [arg-type]
						# why does it think tolist() returns a string?
						_generate_maze_helper,  # type: ignore[arg-type]
						maze_indexes.tolist(),
					),
					**tqdm_kwargs,
				),
			)

		# Filter out None values explicitly after ensuring all results are collected
		solved_mazes_: list[SolvedMaze] = [
			maze for maze in solved_mazes if maze is not None
		]
		# solved_mazes_ = list(filter(lambda x: x is not None, solved_mazes))

		# Update the config with the actual number of mazes
		cfg_cpy.n_mazes = len(solved_mazes_)

		dataset: MazeDataset = cls(
			cfg=cfg_cpy,
			mazes=solved_mazes_,
		)

		dataset.update_self_config()  # Call `update_self_config()` to ensure the dataset's config reflects changes

		np.random.seed(cfg_cpy.seed)  # Reset the seed to the value in the config copy

		return dataset

	@classmethod
	def download(cls, cfg: MazeDatasetConfig, **kwargs) -> "MazeDataset":
		"(not implemented yet!) download a maze dataset from the internet"
		raise NotImplementedError("not implemented yet")

	@classmethod
	def load(cls: "type[MazeDataset]", data: JSONdict) -> "MazeDataset":
		"""load from zanj/json"""
		if data[_FORMAT_KEY] == "MazeDataset:minimal":
			return cls._load_minimal(data)
		elif data[_FORMAT_KEY] == "MazeDataset:minimal_soln_cat":
			return cls._load_minimal_soln_cat(data)
		elif data[_FORMAT_KEY] == "MazeDataset":
			if (
				SERIALIZE_MINIMAL_THRESHOLD == -1
			):  # Allow access to `_load_legacy` for profiling
				return cls._load_legacy(data)
			return cls._load_full(data)
		else:
			err_msg: str = f"`_FORMAT_KEY` string {data[_FORMAT_KEY] = } is not a recognized `MazeDataset` format. ({_FORMAT_KEY = })"
			raise KeyError(
				err_msg,
			)

	@classmethod
	def _load_full(cls, data: JSONdict) -> "MazeDataset":
		assert data[_FORMAT_KEY] == "MazeDataset"
		return cls(
			cfg=MazeDatasetConfig.load(data["cfg"]),  # type: ignore[arg-type]
			mazes=load_item_recursive(data["mazes"], tuple()),
			generation_metadata_collected=data["generation_metadata_collected"],  # type: ignore[arg-type]
		)

	@classmethod
	def _load_minimal(cls, data: JSONdict) -> "MazeDataset":
		assert data[_FORMAT_KEY] == "MazeDataset:minimal"
		return cls(
			cfg=MazeDatasetConfig.load(data["cfg"]),  # type: ignore[arg-type]
			generation_metadata_collected=data["generation_metadata_collected"],  # type: ignore[arg-type]
			mazes=[
				SolvedMaze(
					clist,
					soln[:slen, ...],
				)
				for clist, slen, soln in zip(
					load_item_recursive(data["maze_connection_lists"], tuple()),
					load_item_recursive(data["maze_solution_lengths"], tuple()),
					load_item_recursive(data["maze_solutions"], tuple()),
					strict=False,
					# load_item_recursive(data["maze_endpoints"], tuple()),
				)
			],
		)

	@classmethod
	def _load_minimal_soln_cat(cls, data: JSONdict) -> "MazeDataset":
		assert data[_FORMAT_KEY] == "MazeDataset:minimal_soln_cat"

		maze_solution_lengths = load_item_recursive(
			data["maze_solution_lengths"],
			tuple(),
		)
		maze_solutions_concat = load_item_recursive(
			data["maze_solutions_concat"],
			tuple(),
		)
		maze_solutions = np.split(
			maze_solutions_concat,
			np.cumsum(maze_solution_lengths)[:-1],
			axis=0,
		)

		return cls(
			cfg=load_item_recursive(data["cfg"], tuple()),
			generation_metadata_collected=load_item_recursive(
				data["generation_metadata_collected"],
				tuple(),
			),
			mazes=[
				SolvedMaze(
					connection_list=clist,
					solution=soln,
				)
				for clist, soln in zip(
					load_item_recursive(data["maze_connection_lists"], tuple()),
					# load_item_recursive(data["maze_endpoints"], tuple()),
					maze_solutions,
					strict=False,
				)
			],
		)

	@classmethod
	def _load_legacy(cls, data: JSONdict) -> "MazeDataset":
		"""Legacy `load` method from <0.5.2. Used exclusively for profiling comparison."""
		assert data[_FORMAT_KEY] == "MazeDataset"
		return cls(
			**{
				key: load_item_recursive(data[key], tuple())
				for key in ["cfg", "mazes", "generation_metadata_collected"]
			},
		)

	def serialize(self) -> JSONdict:
		"""serialize to zanj/json"""
		if (
			SERIALIZE_MINIMAL_THRESHOLD is not None
			and len(self) >= SERIALIZE_MINIMAL_THRESHOLD
		):
			return self._serialize_minimal()
		return self._serialize_full()

	def _serialize_full(self) -> JSONdict:
		return {
			_FORMAT_KEY: "MazeDataset",
			"cfg": json_serialize(self.cfg),
			"fname": self.cfg.to_fname(),
			"mazes": json_serialize(self.mazes),
			"generation_metadata_collected": json_serialize(
				self.generation_metadata_collected,
			),
		}

	def _serialize_minimal(self) -> JSONdict:
		"alternate serialization where metadata is collected and mazes are stored in concatenated form"
		filtered_meta: MazeDataset
		if self.generation_metadata_collected is None:
			filtered_meta = self.filter_by.collect_generation_meta()
		else:
			filtered_meta = self

		max_solution_len: int = max(m.solution.shape[0] for m in filtered_meta.mazes)
		n_mazes: int = len(filtered_meta.mazes)
		grid_n: int = filtered_meta.cfg.grid_n

		maze_connection_lists: np.ndarray = np.empty(
			(n_mazes, 2, grid_n, grid_n),
			dtype=np.bool_,
		)
		# maze_endpoints: np.ndarray = np.empty((n_mazes, 2, 2), dtype=np.int8)
		maze_solution_lengths: np.ndarray = np.empty((n_mazes,), dtype=np.int32)
		maze_solutions: np.ndarray = np.empty(
			(n_mazes, max_solution_len, 2),
			dtype=np.int8,
		)

		for idx, maze in enumerate(filtered_meta.mazes):
			maze_connection_lists[idx] = maze.connection_list
			# maze_endpoints[idx] = np.array([maze.start_pos, maze.end_pos])
			maze_solution_lengths[idx] = maze.solution.shape[0]
			maze_solutions[idx, : maze.solution.shape[0]] = maze.solution

		return {
			_FORMAT_KEY: "MazeDataset:minimal",
			"cfg": json_serialize(filtered_meta.cfg),
			"fname": filtered_meta.cfg.to_fname(),
			"generation_metadata_collected": json_serialize(
				filtered_meta.generation_metadata_collected,
			),
			"maze_connection_lists": maze_connection_lists,  # type: ignore[dict-item]
			# "maze_endpoints": maze_endpoints,
			"maze_solution_lengths": maze_solution_lengths,  # type: ignore[dict-item]
			"maze_solutions": maze_solutions,  # type: ignore[dict-item]
		}

	def _serialize_minimal_soln_cat(self: "MazeDataset") -> JSONdict:
		"alternate serialization where metadata is collected, and mazes and their solutions are stored in concatenated form"
		filtered_meta: MazeDataset
		if self.generation_metadata_collected is None:
			filtered_meta = self.filter_by.collect_generation_meta()
		else:
			filtered_meta = self

		maze_solution_lengths: np.ndarray = np.array(
			[m.solution.shape[0] for m in filtered_meta.mazes],
			dtype=np.int32,
		)
		n_mazes: int = len(filtered_meta.mazes)
		grid_n: int = filtered_meta.cfg.grid_n
		total_solution_len: int = np.sum(maze_solution_lengths)

		maze_connection_lists: np.ndarray = np.empty(
			(n_mazes, 2, grid_n, grid_n),
			dtype=np.bool_,
		)
		maze_endpoints: np.ndarray = np.empty((n_mazes, 2, 2), dtype=np.int8)
		maze_solutions_concat: np.ndarray = np.empty(
			(total_solution_len, 2),
			dtype=np.int8,
		)

		solutions_running_idx: int = 0
		for idx, maze in enumerate(filtered_meta.mazes):
			maze_connection_lists[idx] = maze.connection_list
			maze_endpoints[idx] = np.array([maze.start_pos, maze.end_pos])
			soln_len: int = maze.solution.shape[0]
			maze_solution_lengths[idx] = soln_len
			maze_solutions_concat[
				solutions_running_idx : solutions_running_idx + soln_len
			] = maze.solution
			solutions_running_idx += soln_len

		return {
			_FORMAT_KEY: "MazeDataset:minimal_soln_cat",
			"cfg": json_serialize(filtered_meta.cfg),
			"fname": filtered_meta.cfg.to_fname(),
			"generation_metadata_collected": json_serialize(
				filtered_meta.generation_metadata_collected,
			),
			"maze_connection_lists": maze_connection_lists,  # type: ignore[dict-item]
			"maze_endpoints": maze_endpoints,  # type: ignore[dict-item]
			"maze_solution_lengths": maze_solution_lengths,  # type: ignore[dict-item]
			"maze_solutions_concat": maze_solutions_concat,  # type: ignore[dict-item]
		}

	def update_self_config(self) -> None:
		"""update the config to match the current state of the dataset (number of mazes, such as after filtering)"""
		if self.cfg.n_mazes != len(self.mazes):
			warnings.warn(
				f"updating config n_mazes from {self.cfg.n_mazes} to {len(self.mazes)}",
			)
			self.cfg.n_mazes = len(self.mazes)

	def custom_maze_filter(
		self,
		method: typing.Callable[[SolvedMaze], bool],
		**kwargs,
	) -> "MazeDataset":
		"""filter the dataset using a custom method"""
		output: MazeDataset = MazeDataset(
			cfg=copy.deepcopy(self.cfg),
			mazes=[m for m in self.mazes if method(m, **kwargs)],
		)
		output.cfg.applied_filters.append(
			{
				"name": f"__custom__:{method.__name__}",
				"kwargs": kwargs,
			},
		)
		output.update_self_config()
		return output


MazeDatasetConfig._dataset_class = property(  # type: ignore[method-assign, assignment]
	lambda self: MazeDataset,  # noqa: ARG005
)

# register things with zanj
register_loader_handler(
	LoaderHandler(
		check=lambda json_item, path=None, z=None: (  # type: ignore[misc] # noqa: ARG005
			isinstance(json_item, typing.Mapping)
			and _FORMAT_KEY in json_item
			and json_item[_FORMAT_KEY].startswith("MazeDataset")
		),
		load=lambda json_item, path=None, z=None: MazeDataset.load(json_item),  # type: ignore[misc] # noqa: ARG005
		uid="MazeDataset",
		source_pckg="maze_dataset.generation.maze_dataset",
		desc="MazeDataset",
	),
)


# TODO: the code below is for doing some smarter collecting and type checking. Probably will delete.
"""
collect either the type at the field, or the shape of the field if it is an array
metadata_types: dict[str, set[type, tuple]] = dict()
for maze in new_dataset:
	for key, value in maze.generation_meta.items():
		if key not in metadata_types:
			metadata_types[key] = set()

		if isinstance(value, np.ndarray):
			metadata_types[key].add(value.shape)
		else:
			metadata_types[key].add(type(value))

# figure out what to do for each field
metadata_actions: dict[str, typing.Callable] = dict()
for key, key_type in metadata_types.items():
	if all(isinstance(kt, tuple) for kt in key_type):
		if all(kt == (2,) for kt in key_type):
			# its all coords, do a statcounter on those coords
			metadata_actions[key] = lambda vals: Counter(tuple(x) for x in vals)
		elif all(
			(len(kt) == 2) and (kt[1] == 2)
			for kt in key_type
		):
			# its a list of coords, do a statcounter on those coords
			metadata_actions[key] = lambda vals: Counter(
				tuple(x) for x in np.concatenate(vals)
			)
		else:
			# its a list of something else, do a counter on those
			# TODO: throw except here?
			metadata_actions[key] = Counter

	elif all(kt in (bool, int, float) for kt in key_type):
		# statcounter for numeric types
		metadata_actions[key] = StatCounter
	elif all(kt == str for kt in key_type):
		# counter for string types
		metadata_actions[key] = Counter
	else:
		# counter for everything else
		# TODO: throw except here?
		metadata_actions[key] = Counter
"""

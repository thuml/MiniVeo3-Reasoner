"implements `MazeDatasetConfig` which is used to generate or load a dataset"

import hashlib
import importlib.metadata
import json
import typing
import warnings
from typing import Callable

import numpy as np
from jaxtyping import Float
from muutils.json_serialize import (
	serializable_dataclass,
	serializable_field,
)
from muutils.json_serialize.util import (
	safe_getsource,
	string_as_lines,
)
from muutils.misc import sanitize_fname, shorten_numerical_to_str

from maze_dataset.constants import Coord, CoordTup
from maze_dataset.dataset.dataset import (
	GPTDatasetConfig,
)
from maze_dataset.dataset.success_predict_math import cfg_success_predict_fn
from maze_dataset.generation.generators import _GENERATORS_PERCOLATED, GENERATORS_MAP

SERIALIZE_MINIMAL_THRESHOLD: int | None = 100
"""If `n_mazes>=SERIALIZE_MINIMAL_THRESHOLD`, then the MazeDataset will use `serialize_minimal`.
Setting to None means that `serialize_minimal` will never be used.
Set to -1 to make calls to `read` use `MazeDataset._load_legacy`. Used for profiling only."""

MAZEDATASETCONFIG_FNAME_HASH_LENGTH: int = 5
"length of the has, in characters, of the hash in the fname of a `MazeDatasetConfig`"

_PercolationSuccessArray = Float[
	np.ndarray,
	"p/grid_n/deadends/endpoints_not_equal/generator_func=5",
]


class NoPercolationInConfigError(ValueError):
	"""raised when trying to predict the success fraction of a config that doesn't have percolation"""

	pass


class SuccessChanceTooSmallError(ValueError):
	"""raised when the success fraction is below the threshold in `MazeDatasetConfig.success_fraction_compensate`"""

	pass


def set_serialize_minimal_threshold(threshold: int | None) -> None:
	"get the global SERIALIZE_MINIMAL_THRESHOLD"
	global SERIALIZE_MINIMAL_THRESHOLD  # noqa: PLW0603
	SERIALIZE_MINIMAL_THRESHOLD = threshold


def _load_maze_ctor(maze_ctor_serialized: str | dict) -> Callable:
	"get the maze constructor from `GENERATORS_MAP`"
	if isinstance(maze_ctor_serialized, dict):
		# this is both the new and old version of the serialization
		return GENERATORS_MAP[maze_ctor_serialized["__name__"]]
	elif isinstance(maze_ctor_serialized, str):
		# this is a version I switched to for a while but now we are switching back
		warnings.warn(
			"you are loading an old model/config in `_load_maze_ctor()`!!! this should not be happening, please report: "
			"https://github.com/understanding-search/maze-dataset/issues/new",
		)
		return GENERATORS_MAP[maze_ctor_serialized]
	else:
		err_msg: str = f"maze_ctor_serialized is of type {type(maze_ctor_serialized) = }, expected str or dict\n{maze_ctor_serialized = }"
		raise TypeError(err_msg)


EndpointKwargsType = dict[
	typing.Literal[
		"allowed_start",
		"allowed_end",
		"deadend_start",
		"deadend_end",
		"endpoints_not_equal",
		"except_on_no_valid_endpoint",
	],
	bool | None | list[tuple[int, int]],
]
"""type hint for `MazeDatasetConfig.endpoint_kwargs`

- `except_on_no_valid_endpoint : bool` (default: `True`)
	some of the conditions (dead ends if a maze is very open, no path between given start and end) can cause the maze generation to fail.
	if `except_on_no_valid_endpoint` is `True`, then the maze generation will raise an error if it fails to generate a valid maze.
	however, if `False`, then the maze generation will return a dataset with fewer mazes than requested.
	If you are generating large datasets, consider using `MazeDatasetConfig.success_fraction_compensate()`
	this uses a pysr-created function to roughly estimate the success fraction of the dataset.
- `allowed_start : list[tuple[int, int]]` (default: `None`)
	list of allowed starting position coordinates
- `allowed_end : list[tuple[int, int]]` (default: `None`)
	list of allowed ending position coordinates
- `deadend_start : bool` (default: `False`)
	if `True`, the starting position must be a dead end
- `deadend_end : bool` (default: `False`)
	if `True`, the ending position must be a dead end
- `endpoints_not_equal : bool` (default: `True`)
	if `True`, the starting and ending positions must be different



"""


def _load_endpoint_kwargs(data: dict) -> EndpointKwargsType:
	if data.get("endpoint_kwargs") is None:
		return dict()

	else:
		return {
			k: (
				# bools and Nones are fine
				v
				if (isinstance(v, bool) or v is None)
				# assume its a CoordList
				else [tuple(x) for x in v]  # muutils/zanj saves tuples as lists
			)
			for k, v in data["endpoint_kwargs"].items()
		}


# not private because we need this to show up in docs
@serializable_dataclass(kw_only=True, properties_to_serialize=["grid_shape"])
class MazeDatasetConfig_base(GPTDatasetConfig):  # noqa: N801
	"""base config -- we serialize, dump to json, and hash this to get the fname. all actual variables we want to be hashed are here"""

	# NOTE: type: ignore[misc] is because it tells us non-default attributes aren't allowed after ones with defaults, but everything is kw_only

	grid_n: int = serializable_field()  # type: ignore[misc]

	# not comparing n_mazes is done primarily to avoid conflicts which happen during `from_config` when we have applied filters
	n_mazes: int = serializable_field(compare=False)  # type: ignore[misc]

	maze_ctor: Callable = serializable_field(
		default=GENERATORS_MAP["gen_dfs"],
		serialization_fn=lambda gen_func: {
			"__name__": gen_func.__name__,
			"__module__": gen_func.__module__,
			# NOTE: this was causing hashing issues on 3.13 vs older versions because somehow,
			# the `__doc__` variable is different across versions??????? WHY???????? IT TREATS WHITESPACE DIFFERENTLY
			# so we just uh. strip it all now.
			# see:
			# https://github.com/understanding-search/maze-dataset/actions/runs/14028046497/job/39270080746?pr=53
			# https://github.com/understanding-search/maze-dataset/actions/runs/14028046497/job/39270080742?pr=53
			# https://www.diffchecker.com/tqIMSevy/
			# update: we also need to filter for empty lines. B)
			"__doc__": [
				line.strip()
				for line in string_as_lines(gen_func.__doc__)
				if line.strip()
			],
			"source_code": safe_getsource(gen_func),
		},
		loading_fn=lambda data: _load_maze_ctor(data["maze_ctor"]),
		assert_type=False,  # TODO: check the type here once muutils supports checking Callable signatures
	)

	maze_ctor_kwargs: dict = serializable_field(
		default_factory=dict,
		serialization_fn=lambda kwargs: kwargs,
		loading_fn=lambda data: (
			dict()
			if data.get("maze_ctor_kwargs", None)
			is None  # this should handle the backwards compatibility
			else data["maze_ctor_kwargs"]
		),
	)

	endpoint_kwargs: EndpointKwargsType = serializable_field(
		default_factory=dict,
		serialization_fn=lambda kwargs: kwargs,
		loading_fn=_load_endpoint_kwargs,
		assert_type=False,
	)

	# NOTE: this part is very hacky. the way muutils works is that it iterates over the *keys in the serialized data*,
	# and so we need to save an `None` here or this wont load the `fname` field on load
	# this is a total mess, and very confusing, and entirely my fault
	_fname_loaded: str | None = serializable_field(
		default=None,
		compare=False,
		serialization_fn=lambda _: None,
		loading_fn=lambda data: data.get("fname", None),
	)

	@property
	def grid_shape(self) -> CoordTup:
		"""return the shape of the grid as a tuple"""
		return (self.grid_n, self.grid_n)

	@property
	def grid_shape_np(self) -> Coord:
		"""return the shape of the grid as a numpy array"""
		return np.array(self.grid_shape)

	@property
	def max_grid_n(self) -> int:
		"""return the maximum of the grid shape"""
		return max(self.grid_shape)

	def _serialize_base(
		self, applied_filters__skip__collect_generation_meta: bool = True
	) -> dict:
		"""serialize the base config for user in `stable_hash_cfg()` and `to_fname()`

		- note that the _fname_loaded will always be `None` to avoid infinite recursion
		- note that we **do not** by default include information about metadata collection here,
		since otherwise loading a dataset that we minified by collecting the metadata would be impossible
		but for comparing things, we do store it when serializing properly by setting
		`applied_filters__skip__collect_generation_meta=False`
		"""
		serialized: dict = MazeDatasetConfig_base.serialize(self)
		if applied_filters__skip__collect_generation_meta:
			serialized["applied_filters"] = [
				x
				for x in serialized["applied_filters"]
				if x.get("name", None) != "collect_generation_meta"
			]
		return serialized

	def _stable_str_dump(self) -> str:
		return json.dumps(
			self._serialize_base(),
			sort_keys=True,
			indent=None,
		)

	def stable_hash_cfg(self) -> int:
		"""return a stable hash of the config"""
		return int.from_bytes(
			hashlib.md5(  # noqa: S324
				bytes(self._stable_str_dump(), "ascii")
			).digest(),
			"big",
		)

	def to_fname(self) -> str:
		"""return a unique identifier (valid as a filename) for this config"""
		n_mazes_str: str = shorten_numerical_to_str(self.n_mazes)
		maze_ctor_name: str = self.maze_ctor.__name__.removeprefix("gen_")
		hash_id: int = self.stable_hash_cfg() % 10**MAZEDATASETCONFIG_FNAME_HASH_LENGTH
		return sanitize_fname(
			f"{self.name}-g{self.grid_n}-n{n_mazes_str}-a_{maze_ctor_name}-h{hash_id}",
		)


# NOTE: type: ignore[misc] is because it tells us non-default attributes aren't allowed after ones with defaults, but everything is kw_only
@serializable_dataclass(kw_only=True, methods_no_override=["serialize"])
class MazeDatasetConfig(MazeDatasetConfig_base):  # type: ignore[misc]
	"""config object which is passed to `MazeDataset.from_config` to generate or load a dataset

	# Parameters:
	- `name : str`
		name of the dataset -- this can be anything, but should be filesystem safe since we use it in the `fname`
	- `grid_n : int`
		grid size of the maze (number of rows/columns)
	- `n_mazes : int`
		number of mazes to request. For some combinations of `endpoint_kwargs` and `maze_ctor`, not all mazes might successfully generate.
		see `EndpointKwargsType` for more details.
	- `maze_ctor : Callable`
		maze generator function. This should be a function that takes a grid size and returns a maze.
		This will usually be one of the functions in `LatticeMazeGenerators`.
	- `maze_ctor_kwargs : dict`
		keyword arguments to pass to the maze generator function. Specific to the `maze_ctor` you are using.
	- `endpoint_kwargs : EndpointKwargsType`
		keyword arguments passed to `LatticeMaze.generate_random_path()`. see `EndpointKwargsType` for more info.
	- `applied_filters : list[dict]`
		list of filters that have been applied to the dataset. We recommend applying filters to datasets directly,
		but these are stored with the config in case you want to re-generate the dataset with the same filters.

	"""

	@property
	def config_version(self) -> str:
		"""return the version of the config. added in maze_dataset v1.3.0, previous versions had no dataset config"""
		return "1.0"

	@property
	def versions(self) -> dict:
		"""return the versions of the config and the maze_dataset"""
		return dict(
			config=self.config_version,
			maze_dataset=importlib.metadata.version("maze_dataset"),
		)

	def serialize(self) -> dict:
		"serialize the MazeDatasetConfig with all fields and fname"
		return {
			**self._serialize_base(
				applied_filters__skip__collect_generation_meta=False
			),
			"fname": self.to_fname(),
			"versions": self.versions,
		}

	def summary(self) -> dict:
		"""return a summary of the config"""
		# do we run this to make sure it doesn't error?
		super_summary: dict = super().summary()
		assert super_summary
		self_ser: dict = self.serialize()
		return dict(
			name=self.name,
			fname=self.to_fname(),
			sdc_hash=self.stable_hash_cfg(),
			seed=self.seed,
			seq_len_min=self.seq_len_min,
			seq_len_max=self.seq_len_max,
			applied_filters=self.applied_filters,
			grid_n=self_ser["grid_n"],
			n_mazes=self_ser["n_mazes"],
			maze_ctor_name=self_ser["maze_ctor"]["__name__"],
			maze_ctor_kwargs=self_ser["maze_ctor_kwargs"],
			endpoint_kwargs=self_ser["endpoint_kwargs"],
		)

	def _to_ps_array(self) -> _PercolationSuccessArray:
		"""Convert this config to a [p, grid_n, deadends, endpoints_not_equal, generator_func] vector.

		used in predicting the success rate
		"""
		try:
			assert self.maze_ctor.__name__ in _GENERATORS_PERCOLATED, (
				f"generator not supported, must be a percolation generator\n{self.maze_ctor.__name__ = }, {_GENERATORS_PERCOLATED = }"
			)
			assert "p" in self.maze_ctor_kwargs, (
				f"maze_ctor_kwargs must have a 'p' (percolation value) key: {self.maze_ctor_kwargs = }"
			)
			assert not self.endpoint_kwargs.get("except_on_no_valid_endpoint", True), (
				f"except_on_no_valid_endpoint must be False, or else if any maze fails to generate, the whole dataset will fail: {self.endpoint_kwargs = }"
			)
		except AssertionError as e:
			err_msg: str = f"invalid config for percolation success prediction: {self.summary() = }"
			raise NoPercolationInConfigError(
				err_msg,
			) from e

		endpoints_unique_flag: int = int(
			# we are pretty sure it will be an int or bool here
			self.endpoint_kwargs.get("endpoints_not_equal", True),  # type: ignore[arg-type]
		)

		# adjustment for bknutson0
		if not (
			self.endpoint_kwargs.get("deadend_start", False)
			and self.endpoint_kwargs.get("deadend_end", False)
		):
			# we didnt train on this, but if either endpoint is not required to be in a dead end
			# then  requiring the endpoints to be unique does not really affect the success rate
			# (except for very small percolation values, pure percolation generation)
			endpoints_unique_flag = 0

		return np.array(
			[
				float(self.maze_ctor_kwargs["p"]),
				float(self.grid_n),
				float(
					int(
						self.endpoint_kwargs.get("deadend_start", False)  # type: ignore[arg-type]
						or self.endpoint_kwargs.get("deadend_end", False),
					),
				),
				float(endpoints_unique_flag),
				float(_GENERATORS_PERCOLATED.index(self.maze_ctor.__name__)),
			],
			dtype=np.float64,
		)

	@classmethod
	def _from_ps_array(
		cls,
		arr: _PercolationSuccessArray,
		name: str = "predict",
		n_mazes: int = 100,
		**kwargs,
	) -> "MazeDatasetConfig":
		"""Reconstruct a config from an array [p, grid_n, deadends, endpoints_not_equal, generator_func] and other config parameters.

		# Returns:
		- `MazeDatasetConfig`
			Config corresponding to `arr`
		"""
		return cls(
			name=name,
			grid_n=int(arr[1]),
			n_mazes=n_mazes,
			maze_ctor=GENERATORS_MAP[_GENERATORS_PERCOLATED[int(arr[4])]],
			maze_ctor_kwargs={"p": float(arr[0])},
			endpoint_kwargs=dict(
				deadend_start=bool(arr[2]),
				deadend_end=bool(arr[2]),
				endpoints_not_equal=bool(arr[3]),
				except_on_no_valid_endpoint=False,
			),
			**kwargs,
		)

	def success_fraction_estimate(
		self,
		except_if_all_success_expected: bool = False,
	) -> float:
		"""Estimate the success fraction of this config.

		only valid when the generator is a percolation generator,
		and endpoints are enforced to be dead ends

		more information on where this comes from can be found in
		- `cfg_success_predict_fn()` from `maze_dataset.dataset.success_predict_math`
		- `estimate_dataset_fractions.ipynb`
		- `maze_dataset.benchmarks.sweep_fit`

		# Parameters:
		- `except_if_all_success_expected : bool`
			if `True`, don't raise an error if the success fraction is below the threshold.
			will always return `1.0` if the config is not expected to fail

		# Returns:
		- `float`
			estimated success fraction

		# Raises:
		- `NoPercolationInConfigError` : if the config is not expected to fail, and `except_if_all_success_expected` is `False`
		"""
		try:
			return cfg_success_predict_fn(self)

		except NoPercolationInConfigError as e:
			if except_if_all_success_expected:
				raise e  # noqa: TRY201
			return 1.0

	def success_fraction_compensate(
		self,
		safety_margin: float = 1.2,
		except_if_all_success_expected: bool = False,
		epsilon: float = 1e-2,
	) -> "MazeDatasetConfig":
		"""return a new `MazeDatasetConfig` like this one with `n_mazes` adjusted to compensate for the success fraction

		calls `MazeDatasetConfig.success_fraction_estimate()` to get the success fraction, and then
		computes the new number of mazes as `n_mazes = n_mazes * safety_margin / success_fraction + 1`

		more information on where this comes from can be found in
		- `cfg_success_predict_fn()` from `maze_dataset.dataset.success_predict_math`
		- `estimate_dataset_fractions.ipynb`
		- `maze_dataset.benchmarks.sweep_fit`

		# Parameters:
		- `safety_margin : float`
			safety margin to apply to the success fraction estimate
			(defaults to `1.2`, or 20% more mazes than estimated)
		- `except_if_all_success_expected : bool`
			if `True`, don't raise an error if the success fraction is below the threshold.
			this is passed to `MazeDatasetConfig.success_fraction_estimate`.
			if your config isn't expected to fail, passing this might mean you generate more mazes than needed
			since `safety_margin` is still applied.
			(defaults to `False`)
		- `epsilon : float`
			raise `SuccessChanceTooSmallError` if the success fraction is below this threshold
			(defaults to `1e-2`)

		# Returns:
		- `MazeDatasetConfig`
			new config with adjusted `n_mazes`

		# Raises:
		- `SuccessChanceTooSmallError` : if the computed success fraction is below `epsilon`
		"""
		# compute and check the success fraction
		success_fraction: float = self.success_fraction_estimate(
			except_if_all_success_expected=except_if_all_success_expected,
		)
		if success_fraction < epsilon:
			err_msg: str = (
				f"{success_fraction = } is below the threshold of {epsilon = }"
			)
			raise SuccessChanceTooSmallError(
				err_msg,
			)

		# compute the new number of mazes
		n_mazes: int = self.n_mazes
		new_n_mazes: int = int((n_mazes * safety_margin) / success_fraction) + 1

		# put it in a new config and return
		cfg_dict: dict = self.serialize()
		cfg_dict["n_mazes"] = new_n_mazes
		return MazeDatasetConfig.load(cfg_dict)

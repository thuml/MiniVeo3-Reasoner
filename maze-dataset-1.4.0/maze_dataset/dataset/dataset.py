"""`GPTDatasetConfig` and `GPTDataset` are base classes for datasets

they implement some basic functionality, saving/loading, the `from_config` pipeline, and filtering

> [!NOTE]
> these should probably be moved into a different package, so don't rely on them being here
"""

import functools
import json
import random
import typing
import warnings
from pathlib import Path
from typing import Callable, Type, TypeVar

import numpy as np
from muutils.json_serialize import (
	JSONitem,
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)
from muutils.json_serialize.util import (
	JSONdict,
)
from muutils.misc import sanitize_fname, shorten_numerical_to_str, stable_hash
from typing_extensions import Self
from zanj import ZANJ

from maze_dataset.generation.seed import GLOBAL_SEED


def set_reproducibility(seed: int) -> None:
	"set reproducibility in stdlib random and numpy (but not torch)"
	random.seed(seed)
	np.random.seed(seed)


class FilterInfoMismatchError(ValueError):
	"""raised when the filter info in a dataset config does not match the filter info in the dataset"""

	pass


def _load_applied_filters(
	filters: list[dict[typing.Literal["name", "args", "kwargs"], str | tuple | dict]],
) -> list[dict[typing.Literal["name", "args", "kwargs"], str | tuple | dict]]:
	try:
		return [
			dict(
				name=filter_info["name"],
				args=tuple(
					filter_info["args"],
				),  # muutils/zanj save tuples as lists, and this causes problems
				kwargs=dict(filter_info["kwargs"]),  # type: ignore[arg-type]
			)
			for filter_info in filters
		]
	except Exception as e:
		err_msg: str = f"failed to load applied filters:\n{filters}"
		raise ValueError(err_msg) from e


@serializable_dataclass(kw_only=True)
class GPTDatasetConfig(SerializableDataclass):
	"""base GPTDatasetConfig class"""

	name: str

	# TODO: get rid of all these things as part of migration to tokenizer-free dataset config
	# --------------------------------------------------
	seq_len_min: int = serializable_field(default=1)
	seq_len_max: int = serializable_field(default=512)
	# --------------------------------------------------

	seed: int | None = serializable_field(default=GLOBAL_SEED)
	applied_filters: list[
		dict[typing.Literal["name", "args", "kwargs"], str | list | tuple | dict]
	] = serializable_field(
		default_factory=list,
		deserialize_fn=_load_applied_filters,
		assert_type=False,  # TODO: check the type here once muutils supports checking Callable signatures
	)

	def __post_init__(self) -> None:
		"post init, where we set a random seed if none is set"
		assert self.seq_len_min <= self.seq_len_max
		# if seed set to None, then generate a new random seed
		if self.seed is None:
			self.seed = np.random.randint(2**31)

		# TODO: something here is broken
		if self.seed != GLOBAL_SEED:
			warnings.warn(
				f"in GPTDatasetConfig {self.name=}, {self.seed=} is trying to override {GLOBAL_SEED = }",
			)

		set_reproducibility(self.seed)

	def summary(self) -> dict:
		"""return a summary of the config"""
		# do we run this to make sure it doesn't error?
		self_ser: dict = self.serialize()
		assert self_ser
		return dict(
			name=self.name,
			seq_len_min=self.seq_len_min,
			seq_len_max=self.seq_len_max,
			seed=self.seed,
			applied_filters=self.applied_filters,
		)

	@property
	def _dataset_class(self) -> type:
		raise NotImplementedError("this should be implemented by subclasses!")

	def to_fname(self) -> str:
		"""convert config to a filename"""
		self_json_str: str = json.dumps(self.serialize())
		self_json_hash: int = int(abs(stable_hash(self_json_str)) % 1e10)
		warnings.warn(
			f"using fallblack to_fname() method for {self.__class__.__name__}, this should be implemented by subclasses!",
		)
		return sanitize_fname(
			# TYPING: error: Argument 1 to "len" has incompatible type "GPTDatasetConfig"; expected "Sized"  [arg-type]
			f"f{self.name}-n{shorten_numerical_to_str(len(self))}-h{self_json_hash}",  # type: ignore[arg-type]
		)


def _dataset_config_load(*args, **kwargs) -> "GPTDatasetConfig":
	err_msg: str = f"this `load` function should be implemented by subclasses! got: {args=}, {kwargs=}"
	raise NotImplementedError(
		err_msg,
	)


# abstract function, hence we dont care that `self` is unused
def _dataset_config_serialize(self, *args, **kwargs) -> JSONitem:  # noqa: ANN001, ARG001
	err_msg: str = f"this `serialize` function should be implemented by subclasses! got: {args=}, {kwargs=}"
	raise NotImplementedError(
		err_msg,
	)


GPTDatasetConfig.load = _dataset_config_load  # type: ignore[method-assign]
GPTDatasetConfig.serialize = _dataset_config_serialize  # type: ignore[method-assign,assignment]
T_DatasetConfig = TypeVar("T_DatasetConfig", bound=GPTDatasetConfig)


class GPTDataset(typing.Generic[T_DatasetConfig]):
	"""wrapper for torch dataset with some extra functionality

	(meaning the functionality should be inherited in downstream classes)

	> [!NOTE]
	> `GPTDatasetConfig` should implement a `to_fname` method that returns a unique filename for the config

	# Requires:
	the following methods should be implemented in subclasses:
	- `__init__(self, cfg: GPTDatasetConfig, **kwargs)`
		initialize the dataset from a given config. kwargs are not passed through, the kwargs should take the actual generated or loaded data (a list of objects or sequences probably)
	- `generate(cls, cfg: GPTDatasetConfig, **kwargs) -> GPTDataset`
		generate the dataset from a given config. kwargs are passed through from `from_config`, and should only contain things that dont belong in the config (i.e. how many threads to use for generation)
	- `serialize(self) -> JSONitem`
		serialize the dataset to a ZANJ-serializable object, including:
		- config
		- data in formats specified by `self.save_formats`
	- `load(cls, data: JSONitem) -> GPTDataset`
		load the dataset from a ZANJ-serializable object
	- `download(cls, cfg: GPTDatasetConfig, **kwargs) -> GPTDataset`
		given a config, try to download a dataset from some source. kwargs are passed through from `from_config`, and should only contain things that dont belong in the config (i.e. some kind of auth token or source url)
	- `__len__(self) -> int`
		return the length of the dataset, required to match interface of `torch.utils.data.Dataset`
	- `__getitem__(self, i: int) -> list[str]`
		return the ith item in the dataset, required to match interface of `torch.utils.data.Dataset`
	-  `update_self_config(self) -> None`
		update the config of the dataset to match the current state of the dataset, used primarily in filtering and validation
	-  decorating the appropriate filter namespace with `register_filter_namespace_for_dataset(your_dataset_class)` if you want to use filters

	# Parameters:
	- `cfg : GPTDatasetConfig`
		config for the dataset, used to generate the dataset
	- `do_generate : bool`
		whether to generate the dataset if it isn't found
		(defaults to `True`)
	- `load_local : bool`
		whether to try finding the dataset locally
		(defaults to `True`)
	- `save_local : bool`
		whether to save the dataset locally if it is generated or downloaded
		(defaults to `True`)
	- `do_download : bool`
		whether to try downloading the dataset
		(defaults to `True`)
	- `local_base_path : Path`
		where to save the dataset
		(defaults to `Path("data/maze_dataset")`)

	# Returns:
	- `GPTDataset`
		the dataset, as you wanted it

	# Implements:
	- `save(self, file_path: str) -> None`
		save the dataset to a file, using ZANJ
	- `read(cls, file_path: str) -> GPTDataset`
		read the dataset from a file, using ZANJ
		get all items in the dataset, in the specified format
	- `filter_by(self)`
		returns a namespace class
	-  `_filter_namespace(self) -> Class`
		returns a namespace class for filtering the dataset, checking that method
	- `_apply_filters_from_config(self) -> None`
		apply filters to the dataset, as specified in the config. used in `from_config()` but only when generating

	"""

	_FILTER_NAMESPACE: type = "this isn't a filter namespace! you have to initialize this by registering with `register_filter_namespace_for_dataset`"  # type: ignore

	cfg: "T_DatasetConfig"

	@classmethod
	def from_config(  # noqa: C901, PLR0912
		cls,
		cfg: "T_DatasetConfig",
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
	) -> "Self":
		"""base class for gpt datasets

		priority of loading:
		1. load from local
		2. download
		3. generate

		"""
		print_log: Callable = print if verbose else lambda *_a, **_kw: None

		local_base_path = Path(local_base_path)
		fname: Path = Path(f"{cfg.to_fname()}.zanj")
		output: Self | None = None
		did_load_local: bool = False
		if zanj is None:
			zanj = ZANJ()

		print_log(f"trying to get the dataset '{cfg.to_fname()}'")

		if not (load_local or do_download or do_generate):
			raise ValueError(
				"no way to load dataset! you said not to load local, not to download, and not to generate",
			)

		dataset_path: Path = local_base_path / fname

		# try loading
		if load_local:  # noqa: SIM102
			if dataset_path.exists():
				print_log(f"loading dataset from {dataset_path.as_posix()}")
				try:
					output = cls.read(dataset_path, zanj=zanj)
					did_load_local = True
					print_log("load successful!")
				except Exception as e:  # noqa: BLE001
					print_log(f"failed to load dataset: {e}")

		if do_download and output is None:
			print_log("seeing if we can download the dataset...")
			try:
				output = cls.download(cfg, **kwargs)
				print_log("download successful!")
			except NotImplementedError:
				print_log("no download found, or download failed")

		if do_generate and output is None:
			print_log("generating dataset...")
			output = cls.generate(cfg, verbose=verbose, **kwargs)
			# only if we generated it, apply filters
			output = output._apply_filters_from_config()

		# check and save
		if output is None:
			raise ValueError("failed to load dataset!")

		cfg_diff: dict = cfg.diff(output.cfg, of_serialized=True)
		if cfg_diff:
			if except_on_config_mismatch:
				if allow_generation_metadata_filter_mismatch and (
					cfg_diff
					== {
						"applied_filters": {
							"self": [],
							"other": [
								{
									"name": "collect_generation_meta",
									"args": (),
									"kwargs": {},
								},
							],
						},
					}
				):
					pass
				else:
					err_msg: str = f"config mismatch: {cfg_diff = }"
					raise ValueError(err_msg)
			else:
				warnings.warn(f"config mismatch: {cfg_diff = }")

		if save_local and not did_load_local:
			print_log(f"saving dataset to {dataset_path}")
			output.save(dataset_path, zanj=zanj)

		print_log(
			f"Got dataset {output.cfg.name} with {len(output)} items. {output.cfg.to_fname() = }",
		)
		return output

	def save(self, file_path: Path | str, zanj: ZANJ | None = None) -> None:
		"save dataset to a file with zanj"
		if zanj is None:
			zanj = ZANJ()
		zanj.save(self.serialize(), file_path)

	# serialization & loading
	@classmethod
	def read(cls, file_path: str | Path, zanj: ZANJ | None = None) -> "Self":
		"read dataset from a file with zanj"
		if zanj is None:
			zanj = ZANJ()
		return zanj.read(file_path)

	def serialize(self) -> JSONdict:
		"(implement in subclass!) serialize to something we can save with zanj"
		raise NotImplementedError

	def data_hash(self) -> int:
		"(implement in subclass!) return a hash of the data"
		raise NotImplementedError

	@classmethod
	def load(cls, data: JSONdict) -> "Self":
		"(implement in subclass!) load a dataset from what we made with `.serialize()`"
		raise NotImplementedError

	# generating & downloading
	@classmethod
	def generate(cls, cfg: "T_DatasetConfig", **kwargs) -> "Self":
		"(implement in subclass!) generative given the config"
		raise NotImplementedError

	@classmethod
	def download(cls, cfg: "T_DatasetConfig", **kwargs) -> "Self":
		"(implement in subclass!) download the dataset given the config"
		raise NotImplementedError

	# filtering
	def update_self_config(self) -> None:
		"""(implement in subclass!) update the config of the dataset to match the actual data, if needed

		for example, adjust number of mazes after filtering
		"""
		pass

	def __len__(self) -> int:
		"return the length of the dataset"
		raise NotImplementedError("implement in subclass!")

	class FilterBy:
		"""thanks GPT-4"""

		def __init__(self, dataset: "T_Dataset") -> None:
			"mock class so we can call `my_dataset.filter_by.some_registered_filter()`"
			self.dataset: T_Dataset = dataset

		def __getattr__(self, name: str) -> typing.Callable[..., "T_Dataset"]:
			"override getattr so we can call `my_dataset.filter_by.some_registered_filter()`"
			filter_func: DatasetFilterFunc = getattr(
				self.dataset._FILTER_NAMESPACE,
				name,
			)

			def wrapped_filter_func(*args, **kwargs):  # noqa: ANN202
				return filter_func(self.dataset, *args, **kwargs)

			return wrapped_filter_func

	@property
	def filter_by(self) -> "FilterBy":
		"can call `my_dataset.filter_by.some_registered_filter()` to filter the dataset"
		return self.FilterBy(self)

	def _apply_filters_from_config(self) -> "Self":
		"""apply filters to the dataset, as specified in the config. used in `from_config()`"""
		output: Self = self
		# copy the list, and then clear it in the config. we do this because each time we apply a filter it will update config.applied_filters
		applied_filters_old: list[
			dict[typing.Literal["name", "args", "kwargs"], typing.Any]
		] = self.cfg.applied_filters
		output.cfg.applied_filters = list()
		# apply the filters
		for filter_info in applied_filters_old:
			filter_name: str = filter_info["name"]
			if filter_name not in output._FILTER_NAMESPACE.__dict__:
				if filter_name.startswith("__custom__:"):
					err_msg = f"the dataset {output.cfg.to_fname()} was filtering using a custom filter: '{filter_name}', which we don't know about. add it to MazeDatasetFilters!"
					raise ValueError(
						err_msg,
					)
				err_msg = f"the dataset {output.cfg.to_fname()} was filtering using an unknown filter: '{filter_name}'"
				raise ValueError(
					err_msg,
				)
			filter_args: list = filter_info.get("args", list())
			filter_kwargs: dict = filter_info.get("kwargs", dict())
			output = getattr(output.filter_by, filter_name)(
				*filter_args,
				**filter_kwargs,
			)

		# update the config, perform checks
		# TODO: some funny business with manually specified filters here?
		output.update_self_config()
		_check_filter_equality(
			filters_old=applied_filters_old,
			filters_new=output.cfg.applied_filters,  # type: ignore[arg-type]
		)
		return output


def _check_filter_equality(
	filters_old: list[
		dict[typing.Literal["name", "args", "kwargs"], str | list | dict]
	],
	filters_new: list[
		dict[typing.Literal["name", "args", "kwargs"], str | list | dict]
	],
) -> None:
	try:
		assert len(filters_old) == len(filters_new)

		for filterinfo_new, filterinfo_old in zip(
			filters_old,
			filters_new,
			strict=False,
		):
			# basic checks
			assert isinstance(filterinfo_new, dict), "filterinfo_new is not a dict"
			assert isinstance(filterinfo_old, dict), "filterinfo_old is not a dict"
			assert all(key in filterinfo_new for key in ["name", "args", "kwargs"]), (
				"missing keys in filterinfo_new"
			)
			assert all(key in filterinfo_old for key in ["name", "args", "kwargs"]), (
				"missing keys in filterinfo_old"
			)

			# name
			assert filterinfo_new["name"] == filterinfo_old["name"], (
				"filter names don't match"
			)

			# args
			assert len(filterinfo_new["args"]) == len(filterinfo_old["args"]), (
				"filter args of different lengths"
			)
			for arg_new, arg_old in zip(
				filterinfo_new["args"],
				filterinfo_old["args"],
				strict=False,
			):
				assert arg_new == arg_old, "filter args don't match"

			# kwargs
			assert len(filterinfo_new["kwargs"]) == len(filterinfo_old["kwargs"]), (
				"filter kwargs of different lengths"
			)
			for key in filterinfo_old["kwargs"]:
				assert key in filterinfo_new["kwargs"], (
					f"filter kwargs don't match: missing key '{key}'"
				)
				assert filterinfo_new["kwargs"][key] == filterinfo_old["kwargs"][key], (  # type: ignore[index]
					f"filter kwargs don't match: values for key '{key}' don't match"
				)

	except AssertionError as e:
		err_msg: str = (
			f"config mismatch in applied filters: {filters_new} != {filters_old}"
		)
		raise FilterInfoMismatchError(
			err_msg,
		) from e


def register_filter_namespace_for_dataset(
	dataset_cls: Type[GPTDataset],
) -> Callable[[Type], Type]:
	"""register the namespace class with the given dataset class"""

	def decorator(filter_namespace_cls: Type) -> Type:
		dataset_cls._FILTER_NAMESPACE = filter_namespace_cls
		filter_namespace_cls._BASE_DATASET = dataset_cls

		return filter_namespace_cls

	return decorator


T_Dataset = TypeVar("T_Dataset", bound=GPTDataset)
P_FilterKwargs = typing.ParamSpec("P_FilterKwargs")
DatasetFilterFunc = Callable[typing.Concatenate[T_Dataset, P_FilterKwargs], T_Dataset]


def register_dataset_filter(
	method: DatasetFilterFunc,
) -> DatasetFilterFunc:
	"""register a dataset filter, copying the underlying dataset and updating the config

	be sure to return a COPY, not the original?
	# TODO: what the heck do we mean by the above? why the question mark? it should be a copy right?

	method should be a staticmethod of a namespace class registered with `register_filter_namespace_for_dataset`
	"""

	@functools.wraps(method)
	def wrapper(
		# TYPING: error: ParamSpec "P_FilterKwargs" is unbound  [valid-type]
		dataset: T_Dataset,
		*args: P_FilterKwargs.args,  # type: ignore[valid-type]
		**kwargs: P_FilterKwargs.kwargs,  # type: ignore[valid-type]
	) -> T_Dataset:
		new_dataset = method(dataset, *args, **kwargs)
		# update the config
		new_dataset.cfg.applied_filters.append(
			dict(name=method.__name__, args=args, kwargs=kwargs),  # type: ignore[attr-defined]
		)
		new_dataset.update_self_config()
		return new_dataset

	# TYPING: error: Incompatible return value type (got "_Wrapped[[Any, KwArg(Any)], Any, [Never, VarArg(Any), KwArg(Any)], Never]", expected "DatasetFilterProtocol[Any]")  [return-value]
	return wrapper  # type: ignore[return-value]

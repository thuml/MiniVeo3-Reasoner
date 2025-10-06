"""Benchmarking of how successful maze generation is for various values of percolation"""

import functools
import json
import warnings
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Sequence, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from muutils.dictmagic import dotlist_to_nested_dict, update_with_nested_dict
from muutils.json_serialize import (
	JSONitem,
	SerializableDataclass,
	json_serialize,
	serializable_dataclass,
	serializable_field,
)
from muutils.parallel import run_maybe_parallel
from zanj import ZANJ

from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators

SweepReturnType = TypeVar("SweepReturnType")
ParamType = TypeVar("ParamType")
AnalysisFunc = Callable[[MazeDatasetConfig], SweepReturnType]


def dataset_success_fraction(cfg: MazeDatasetConfig) -> float:
	"""empirical success fraction of maze generation

	for use as an `analyze_func` in `sweep()`
	"""
	dataset: MazeDataset = MazeDataset.from_config(
		cfg,
		do_download=False,
		load_local=False,
		save_local=False,
		verbose=False,
	)

	return len(dataset) / cfg.n_mazes


ANALYSIS_FUNCS: dict[str, AnalysisFunc] = dict(
	dataset_success_fraction=dataset_success_fraction,
)


def sweep(
	cfg_base: MazeDatasetConfig,
	param_values: list[ParamType],
	param_key: str,
	analyze_func: Callable[[MazeDatasetConfig], SweepReturnType],
) -> list[SweepReturnType]:
	"""given a base config, parameter values list, key, and analysis function, return the results of the analysis function for each parameter value

	# Parameters:
	- `cfg_base : MazeDatasetConfig`
		base config on which we will modify the value at `param_key` with values from `param_values`
	- `param_values : list[ParamType]`
		list of values to try
	- `param_key : str`
		value to modify in `cfg_base`
	- `analyze_func : Callable[[MazeDatasetConfig], SweepReturnType]`
		function which analyzes the resulting config. originally built for `dataset_success_fraction`

	# Returns:
	- `list[SweepReturnType]`
		_description_
	"""
	outputs: list[SweepReturnType] = []

	for p in param_values:
		# update the config
		cfg_dict: dict = cfg_base.serialize()
		update_with_nested_dict(
			cfg_dict,
			dotlist_to_nested_dict({param_key: p}),
		)
		cfg_test: MazeDatasetConfig = MazeDatasetConfig.load(cfg_dict)

		outputs.append(analyze_func(cfg_test))

	return outputs


@serializable_dataclass()
class SweepResult(SerializableDataclass, Generic[ParamType, SweepReturnType]):
	"""result of a parameter sweep"""

	configs: list[MazeDatasetConfig] = serializable_field(
		serialization_fn=lambda cfgs: [cfg.serialize() for cfg in cfgs],
		deserialize_fn=lambda cfgs: [MazeDatasetConfig.load(cfg) for cfg in cfgs],
	)
	param_values: list[ParamType] = serializable_field(
		serialization_fn=lambda x: json_serialize(x),
		deserialize_fn=lambda x: x,
		assert_type=False,
	)
	result_values: dict[str, Sequence[SweepReturnType]] = serializable_field(
		serialization_fn=lambda x: json_serialize(x),
		deserialize_fn=lambda x: x,
		assert_type=False,
	)
	param_key: str
	analyze_func: Callable[[MazeDatasetConfig], SweepReturnType] = serializable_field(
		serialization_fn=lambda f: f.__name__,
		deserialize_fn=ANALYSIS_FUNCS.get,
		assert_type=False,
	)

	def summary(self) -> JSONitem:
		"human-readable and json-dumpable short summary of the result"
		return {
			"len(configs)": len(self.configs),
			"len(param_values)": len(self.param_values),
			"len(result_values)": len(self.result_values),
			"param_key": self.param_key,
			"analyze_func": self.analyze_func.__name__,
		}

	def save(self, path: str | Path, z: ZANJ | None = None) -> None:
		"save to a file with zanj"
		if z is None:
			z = ZANJ()

		z.save(self, path)

	@classmethod
	def read(cls, path: str | Path, z: ZANJ | None = None) -> "SweepResult":
		"read from a file with zanj"
		if z is None:
			z = ZANJ()

		return z.read(path)

	def configs_by_name(self) -> dict[str, MazeDatasetConfig]:
		"return configs by name"
		return {cfg.name: cfg for cfg in self.configs}

	def configs_by_key(self) -> dict[str, MazeDatasetConfig]:
		"return configs by the key used in `result_values`, which is the filename of the config"
		return {cfg.to_fname(): cfg for cfg in self.configs}

	def configs_shared(self) -> dict[str, Any]:
		"return key: value pairs that are shared across all configs"
		# we know that the configs all have the same keys,
		# so this way of doing it is fine
		config_vals: dict[str, set[Any]] = dict()
		for cfg in self.configs:
			for k, v in cfg.serialize().items():
				if k not in config_vals:
					config_vals[k] = set()
				config_vals[k].add(json.dumps(v))

		shared_vals: dict[str, Any] = dict()

		cfg_ser: dict = self.configs[0].serialize()
		for k, v in config_vals.items():
			if len(v) == 1:
				shared_vals[k] = cfg_ser[k]

		return shared_vals

	def configs_differing_keys(self) -> set[str]:
		"return keys that differ across configs"
		shared_vals: dict[str, Any] = self.configs_shared()
		differing_keys: set[str] = set()

		for k in MazeDatasetConfig.__dataclass_fields__:
			if k not in shared_vals:
				differing_keys.add(k)

		return differing_keys

	def configs_value_set(self, key: str) -> list[Any]:
		"return a list of the unique values for a given key"
		d: dict[str, Any] = {
			json.dumps(json_serialize(getattr(cfg, key))): getattr(cfg, key)
			for cfg in self.configs
		}

		return list(d.values())

	def get_where(self, key: str, val_check: Callable[[Any], bool]) -> "SweepResult":
		"get a subset of this `Result` where the configs has `key` satisfying `val_check`"
		configs_list: list[MazeDatasetConfig] = [
			cfg for cfg in self.configs if val_check(getattr(cfg, key))
		]
		configs_keys: set[str] = {cfg.to_fname() for cfg in configs_list}
		result_values: dict[str, Sequence[SweepReturnType]] = {
			k: self.result_values[k] for k in configs_keys
		}

		return SweepResult(
			configs=configs_list,
			param_values=self.param_values,
			result_values=result_values,
			param_key=self.param_key,
			analyze_func=self.analyze_func,
		)

	@classmethod
	def analyze(
		cls,
		configs: list[MazeDatasetConfig],
		param_values: list[ParamType],
		param_key: str,
		analyze_func: Callable[[MazeDatasetConfig], SweepReturnType],
		parallel: bool | int = False,
		**kwargs,
	) -> "SweepResult":
		"""Analyze success rate of maze generation for different percolation values

		# Parameters:
		- `configs : list[MazeDatasetConfig]`
		configs to try
		- `param_values : np.ndarray`
		numpy array of values to try

		# Returns:
		- `SweepResult`
		"""
		n_pvals: int = len(param_values)

		result_values_list: list[float] = run_maybe_parallel(
			# TYPING: error: Argument "func" to "run_maybe_parallel" has incompatible type "partial[list[SweepReturnType]]"; expected "Callable[[MazeDatasetConfig], float]"  [arg-type]
			func=functools.partial(  # type: ignore[arg-type]
				sweep,
				param_values=param_values,
				param_key=param_key,
				analyze_func=analyze_func,
			),
			iterable=configs,
			keep_ordered=True,
			parallel=parallel,
			pbar_kwargs=dict(total=len(configs)),
			**kwargs,
		)
		result_values: dict[str, Float[np.ndarray, n_pvals]] = {
			cfg.to_fname(): np.array(res)
			for cfg, res in zip(configs, result_values_list, strict=False)
		}
		return cls(
			configs=configs,
			param_values=param_values,
			# TYPING: error: Argument "result_values" to "SweepResult" has incompatible type "dict[str, ndarray[Any, Any]]"; expected "dict[str, Sequence[SweepReturnType]]"  [arg-type]
			result_values=result_values,  # type: ignore[arg-type]
			param_key=param_key,
			analyze_func=analyze_func,
		)

	def plot(
		self,
		save_path: str | None = None,
		cfg_keys: list[str] | None = None,
		cmap_name: str | None = "viridis",
		plot_only: bool = False,
		show: bool = True,
		ax: plt.Axes | None = None,
		minify_title: bool = False,
		legend_kwargs: dict[str, Any] | None = None,
	) -> plt.Axes:
		"""Plot the results of percolation analysis"""
		# set up figure
		if not ax:
			fig: plt.Figure
			ax_: plt.Axes
			fig, ax_ = plt.subplots(1, 1, figsize=(22, 10))
		else:
			ax_ = ax

		# plot
		cmap = plt.get_cmap(cmap_name)
		n_cfgs: int = len(self.result_values)
		for i, (ep_cfg_name, result_values) in enumerate(
			sorted(
				self.result_values.items(),
				# HACK: sort by grid size
				#                 |--< name of config
				#                 |    |-----------< gets 'g{n}'
				#                 |    |            |--< gets '{n}'
				#                 |    |            |
				key=lambda x: int(x[0].split("-")[0][1:]),
			),
		):
			ax_.plot(
				# TYPING: error: Argument 1 to "plot" of "Axes" has incompatible type "list[ParamType]"; expected "float | Buffer | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes] | str"  [arg-type]
				self.param_values,  # type: ignore[arg-type]
				# TYPING: error: Argument 2 to "plot" of "Axes" has incompatible type "Sequence[SweepReturnType]"; expected "float | Buffer | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes] | str"  [arg-type]
				result_values,  # type: ignore[arg-type]
				".-",
				label=self.configs_by_key()[ep_cfg_name].name,
				color=cmap((i + 0.5) / (n_cfgs - 0.5)),
			)

		# repr of config
		cfg_shared: dict = self.configs_shared()
		if minify_title:
			cfg_shared["endpoint_kwargs"] = {
				k: v
				for k, v in cfg_shared["endpoint_kwargs"].items()
				if k != "except_on_no_valid_endpoint"
			}
		cfg_repr: str = (
			str(cfg_shared)
			if cfg_keys is None
			else (
				"MazeDatasetConfig("
				+ ", ".join(
					[
						f"{k}={cfg_shared[k].__name__}"
						# TYPING: error: Argument 2 to "isinstance" has incompatible type "<typing special form>"; expected "_ClassInfo"  [arg-type]
						if isinstance(cfg_shared[k], Callable)  # type: ignore[arg-type]
						else f"{k}={cfg_shared[k]}"
						for k in cfg_keys
					],
				)
				+ ")"
			)
		)

		# add title and stuff
		if not plot_only:
			ax_.set_xlabel(self.param_key)
			ax_.set_ylabel(self.analyze_func.__name__)
			ax_.set_title(
				f"{self.param_key} vs {self.analyze_func.__name__}\n{cfg_repr}",
			)
			ax_.grid(True)
			# ax_.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.1))
			legend_kwargs = {
				**dict(loc="center left"),
				**(legend_kwargs or dict()),
			}
			ax_.legend(**legend_kwargs)

		# save and show
		if save_path:
			plt.savefig(save_path)

		if show:
			plt.show()

		return ax_


DEFAULT_ENDPOINT_KWARGS: list[tuple[str, dict]] = [
	(
		"any",
		dict(deadend_start=False, deadend_end=False, except_on_no_valid_endpoint=False),
	),
	(
		"deadends",
		dict(
			deadend_start=True,
			deadend_end=True,
			endpoints_not_equal=False,
			except_on_no_valid_endpoint=False,
		),
	),
	(
		"deadends_unique",
		dict(
			deadend_start=True,
			deadend_end=True,
			endpoints_not_equal=True,
			except_on_no_valid_endpoint=False,
		),
	),
]


def endpoint_kwargs_to_name(ep_kwargs: dict) -> str:
	"""convert endpoint kwargs options to a human-readable name"""
	if ep_kwargs.get("deadend_start", False) or ep_kwargs.get("deadend_end", False):
		if ep_kwargs.get("endpoints_not_equal", False):
			return "deadends_unique"
		else:
			return "deadends"
	else:
		return "any"


def full_percolation_analysis(
	n_mazes: int,
	p_val_count: int,
	grid_sizes: list[int],
	ep_kwargs: list[tuple[str, dict]] | None = None,
	generators: Sequence[Callable] = (
		LatticeMazeGenerators.gen_percolation,
		LatticeMazeGenerators.gen_dfs_percolation,
	),
	save_dir: Path = Path("../docs/benchmarks/percolation_fractions"),
	parallel: bool | int = False,
	**analyze_kwargs,
) -> SweepResult:
	"run the full analysis of how percolation affects maze generation success"
	if ep_kwargs is None:
		ep_kwargs = DEFAULT_ENDPOINT_KWARGS

	# configs
	configs: list[MazeDatasetConfig] = list()

	# TODO: B007 noqaed because we dont use `ep_kw_name` or `gf_idx`
	for ep_kw_name, ep_kw in ep_kwargs:  # noqa: B007
		for gf_idx, gen_func in enumerate(generators):  # noqa: B007
			configs.extend(
				[
					MazeDatasetConfig(
						name=f"g{grid_n}-{gen_func.__name__.removeprefix('gen_').removesuffix('olation')}",
						grid_n=grid_n,
						n_mazes=n_mazes,
						maze_ctor=gen_func,
						maze_ctor_kwargs=dict(p=float("nan")),
						endpoint_kwargs=ep_kw,
					)
					for grid_n in grid_sizes
				],
			)

	# get results
	result: SweepResult = SweepResult.analyze(
		configs=configs,  # type: ignore[misc]
		# TYPING: error: Argument "param_values" to "analyze" of "SweepResult" has incompatible type "float | list[float] | list[list[float]] | list[list[list[Any]]]"; expected "list[Any]"  [arg-type]
		param_values=np.linspace(0.0, 1.0, p_val_count).tolist(),  # type: ignore[arg-type]
		param_key="maze_ctor_kwargs.p",
		analyze_func=dataset_success_fraction,
		parallel=parallel,
		**analyze_kwargs,
	)

	# save the result
	results_path: Path = (
		save_dir / f"result-n{n_mazes}-c{len(configs)}-p{p_val_count}.zanj"
	)
	print(f"Saving results to {results_path.as_posix()}")
	result.save(results_path)

	return result


def _is_eq(a, b) -> bool:  # noqa: ANN001
	"""check if two objects are equal"""
	return a == b


def plot_grouped(  # noqa: C901
	results: SweepResult,
	predict_fn: Callable[[MazeDatasetConfig], float] | None = None,
	prediction_density: int = 50,
	save_dir: Path | None = None,
	show: bool = True,
	logy: bool = False,
	save_fmt: str = "svg",
	figsize: tuple[int, int] = (22, 10),
	minify_title: bool = False,
	legend_kwargs: dict[str, Any] | None = None,
	manual_titles: dict[Literal["x", "y", "title"], str] | None = None,
) -> None:
	"""Plot grouped sweep percolation value results for each distinct `endpoint_kwargs` in the configs

	with separate colormaps for each maze generator function

	# Parameters:
	- `results : SweepResult`
		The sweep results to plot
	- `predict_fn : Callable[[MazeDatasetConfig], float] | None`
		Optional function that predicts success rate from a config. If provided, will plot predictions as dashed lines.
	- `prediction_density : int`
		Number of points to use for prediction curves (default: 50)
	- `save_dir : Path | None`
		Directory to save plots (defaults to `None`, meaning no saving)
	- `show : bool`
		Whether to display the plots (defaults to `True`)

	# Usage:
	```python
	>>> result = full_analysis(n_mazes=100, p_val_count=11, grid_sizes=[8,16])
	>>> plot_grouped(result, save_dir=Path("./plots"), show=False)
	```
	"""
	# groups
	endpoint_kwargs_set: list[dict] = results.configs_value_set("endpoint_kwargs")  # type: ignore[assignment]
	generator_funcs_names: list[str] = list(
		{cfg.maze_ctor.__name__ for cfg in results.configs},
	)

	# if predicting, create denser p values
	if predict_fn is not None:
		p_dense = np.linspace(0.0, 1.0, prediction_density)

	# separate plot for each set of endpoint kwargs
	for ep_kw in endpoint_kwargs_set:
		results_epkw: SweepResult = results.get_where(
			"endpoint_kwargs",
			functools.partial(_is_eq, b=ep_kw),
			# lambda x: x == ep_kw,
		)
		shared_keys: set[str] = set(results_epkw.configs_shared().keys())
		cfg_keys: set[str] = shared_keys.intersection({"n_mazes", "endpoint_kwargs"})
		fig, ax = plt.subplots(1, 1, figsize=figsize)
		for gf_idx, gen_func in enumerate(generator_funcs_names):
			results_filtered: SweepResult = results_epkw.get_where(
				"maze_ctor",
				# HACK: big hassle to do this without a lambda, is it really that bad?
				lambda x: x.__name__ == gen_func,  # noqa: B023
			)
			if len(results_filtered.configs) < 1:
				warnings.warn(
					f"No results for {gen_func} and {ep_kw}. Skipping.",
				)
				continue

			cmap_name = "Reds" if gf_idx == 0 else "Blues"
			cmap = plt.get_cmap(cmap_name)

			# Plot actual results
			ax = results_filtered.plot(
				cfg_keys=list(cfg_keys),
				ax=ax,
				show=False,
				cmap_name=cmap_name,
				minify_title=minify_title,
				legend_kwargs=legend_kwargs,
			)
			if logy:
				ax.set_yscale("log")

			# Plot predictions if function provided
			if predict_fn is not None:
				for cfg_idx, cfg in enumerate(results_filtered.configs):
					predictions = []
					for p in p_dense:
						cfg_temp = MazeDatasetConfig.load(cfg.serialize())
						cfg_temp.maze_ctor_kwargs["p"] = p
						predictions.append(predict_fn(cfg_temp))

					# Get the same color as the actual data
					n_cfgs: int = len(results_filtered.configs)
					color = cmap((cfg_idx + 0.5) / (n_cfgs - 0.5))

					# Plot prediction as dashed line
					ax.plot(p_dense, predictions, "--", color=color, alpha=0.8)

			if manual_titles:
				ax.set_xlabel(manual_titles["x"])
				ax.set_ylabel(manual_titles["y"])
				ax.set_title(manual_titles["title"])

		# save and show
		if save_dir:
			save_path: Path = (
				save_dir / f"ep_{endpoint_kwargs_to_name(ep_kw)}.{save_fmt}"
			)
			print(f"Saving plot to {save_path.as_posix()}")
			save_path.parent.mkdir(exist_ok=True, parents=True)
			plt.savefig(save_path)

		if show:
			plt.show()

"""Fit a PySR model to a sweep result and plot the results"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import sympy as sp  # type: ignore[import-untyped]
from jaxtyping import Float
from pysr import PySRRegressor  # type: ignore[import-untyped]

from maze_dataset import MazeDatasetConfig
from maze_dataset.benchmark.config_sweep import (
	SweepResult,
	plot_grouped,
)


def extract_training_data(
	sweep_result: SweepResult,
) -> tuple[Float[np.ndarray, "num_rows 5"], Float[np.ndarray, " num_rows"]]:
	"""Extract data (X, y) from a SweepResult.

	# Parameters:
	- `sweep_result : SweepResult`
		The sweep result holding configs and success arrays.

	# Returns:
	- `X : Float[np.ndarray, "num_rows 5"]`
		Stacked [p, grid_n, deadends, endpoints_not_equal, generator_func] for each config & param-value
	- `y : Float[np.ndarray, "num_rows"]`
		The corresponding success rate
	"""
	x_list: list[list[float]] = []
	y_list: list[float] = []
	for cfg in sweep_result.configs:
		# success_arr is an array of success rates for param_values
		success_arr = sweep_result.result_values[cfg.to_fname()]
		for i, p in enumerate(sweep_result.param_values):
			# Temporarily override p in the config's array representation:
			arr = cfg._to_ps_array().copy()
			arr[0] = p  # index 0 is 'p'
			x_list.append(arr)  # type: ignore[arg-type]
			y_list.append(success_arr[i])

	return np.array(x_list, dtype=np.float64), np.array(y_list, dtype=np.float64)


DEFAULT_PYSR_KWARGS: dict = dict(
	niterations=50,
	unary_operators=[
		"exp",
		"log",
		"square(x) = x^2",
		"cube(x) = x^3",
		"sigmoid(x) = 1/(1 + exp(-x))",
	],
	extra_sympy_mappings={
		"square": lambda x: x**2,
		"cube": lambda x: x**3,
		"sigmoid": lambda x: 1 / (1 + sp.exp(-x)),
	},
	binary_operators=["+", "-", "*", "/", "^"],
	# populations=50,
	progress=True,
	model_selection="best",
)


def train_pysr_model(
	data: SweepResult,
	**pysr_kwargs,
) -> PySRRegressor:
	"""Train a PySR model on the given sweep result data"""
	# Convert to arrays
	x, y = extract_training_data(data)

	print(f"training data extracted: {x.shape = }, {y.shape = }")

	# Fit the PySR model
	model: PySRRegressor = PySRRegressor(**{**DEFAULT_PYSR_KWARGS, **pysr_kwargs})
	model.fit(x, y)

	return model


def plot_model(
	data: SweepResult,
	model: PySRRegressor,
	save_dir: Path,
	show: bool = True,
) -> None:
	"""Plot the model predictions against the sweep data"""
	# save all the equations
	save_dir.mkdir(parents=True, exist_ok=True)
	equations_file: Path = save_dir / "equations.txt"
	equations_file.write_text(repr(model))
	print(f"Equations saved to: {equations_file = }")

	# Create a callable that predicts from MazeDatasetConfig
	predict_fn: Callable = model.get_best()["lambda_format"]
	print(f"Best PySR Equation: {model.get_best()['equation'] = }")
	print(f"{predict_fn =}")

	def predict_config(cfg: MazeDatasetConfig) -> float:
		arr = cfg._to_ps_array()
		result = predict_fn(arr)[0]
		return float(result)  # pass the array as separate args

	plot_grouped(
		data,
		predict_fn=predict_config,
		save_dir=save_dir,
		show=show,
	)


def sweep_fit(
	data_path: Path,
	save_dir: Path,
	**pysr_kwargs,
) -> None:
	"""read a sweep result, train a PySR model, and plot the results"""
	# Load the sweep result
	data: SweepResult = SweepResult.read(data_path)
	print(f"loaded data: {data.summary() = }")

	# Train the PySR model
	model: PySRRegressor = train_pysr_model(data, **pysr_kwargs)

	# Plot the model
	plot_model(data, model, save_dir, show=False)


if __name__ == "__main__":
	import argparse

	argparser: argparse.ArgumentParser = argparse.ArgumentParser()
	argparser.add_argument(
		"data_path",
		type=Path,
		help="Path to the sweep result file",
	)
	argparser.add_argument(
		"--save_dir",
		type=Path,
		default=Path("tests/_temp/percolation_fractions/fit_plots/"),
		help="Path to save the plots",
	)
	argparser.add_argument(
		"--niterations",
		type=int,
		default=50,
		help="Number of iterations for PySR",
	)
	args: argparse.Namespace = argparser.parse_args()

	sweep_fit(
		args.data_path,
		args.save_dir,
		niterations=args.niterations,
		# add any additional kwargs here if running in CLI
		populations=50,
		# ^ Assuming we have 4 cores, this means 2 populations per core, so one is always running.
		population_size=50,
		# ^ Generations between migrations.
		timeout_in_seconds=60 * 60 * 7,
		# ^ stop after 7 hours have passed.
		maxsize=50,
		# ^ Allow greater complexity.
		weight_randomize=0.01,
		# ^ Randomize the tree much more frequently
		turbo=True,
		# ^ Faster evaluation (experimental)
	)


def create_interactive_plot(heatmap: bool = True) -> None:  # noqa: C901, PLR0915
	"""Create an interactive plot with the specified grid layout

	# Parameters:
	- `heatmap : bool`
		Whether to show heatmaps (defaults to `True`)
	"""
	import ipywidgets as widgets  # type: ignore[import-untyped]
	import matplotlib.pyplot as plt
	from ipywidgets import FloatSlider, HBox, Layout, VBox
	from matplotlib.gridspec import GridSpec

	from maze_dataset.dataset.success_predict_math import soft_step

	# Create sliders with better layout
	x_slider = FloatSlider(
		min=0.0,
		max=1.0,
		step=0.01,
		value=0.5,
		description="x:",
		style={"description_width": "30px"},
		layout=Layout(width="98%"),
	)

	p_slider = FloatSlider(
		min=0.0,
		max=1.0,
		step=0.01,
		value=0.5,
		description="p:",
		style={"description_width": "30px"},
		layout=Layout(width="98%"),
	)

	alpha_slider = FloatSlider(
		min=0.1,
		max=30.0,
		step=0.1,
		value=10.0,
		description="α:",  # noqa: RUF001
		style={"description_width": "30px"},
		layout=Layout(width="98%"),
	)

	w_slider = FloatSlider(
		min=0.0,
		max=20,
		step=0.5,
		value=4.0 / 7.0,
		description="w:",
		style={"description_width": "30px"},
		layout=Layout(width="98%"),
	)

	# Slider layout control
	slider_box = VBox(
		[
			widgets.Label("Adjust parameters:"),
			HBox(
				[x_slider, w_slider],
				layout=Layout(width="100%", justify_content="space-between"),
			),
			HBox(
				[p_slider, alpha_slider],
				layout=Layout(width="100%", justify_content="space-between"),
			),
		],
	)

	def update_plot(x: float, p: float, alpha: float, w: float) -> None:  # noqa: PLR0915
		"""Update the plot with current slider values

		# Parameters:
		- `x : float`
			x value
		- `p : float`
			p value
		- `k : float`
			k value
		- `alpha : float`
			alpha value
		"""
		# Set up the figure and grid - now 2x2 grid
		fig = plt.figure(figsize=(14, 10))
		gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

		# Create x and p values focused on [0,1] range
		xs = np.linspace(0.0, 1.0, 500)
		ps = np.linspace(0.0, 1.0, 500)

		# Plot 1: f(x) vs x (top left)
		ax1 = fig.add_subplot(gs[0, 0])
		ys = soft_step(xs, p, alpha, w)
		ax1.plot(xs, ys, "b-", linewidth=2.5)

		# Add guidelines
		ax1.axvline(x=p, color="red", linestyle="--", alpha=0.7, label=f"p = {p:.2f}")
		ax1.axvline(x=w, color="green", linestyle="--", alpha=0.7, label=f"w = {w:.2f}")
		ax1.axvline(x=x, color="blue", linestyle=":", alpha=0.7, label=f"x = {x:.2f}")

		# Add identity line for reference
		ax1.plot(xs, xs, "k--", alpha=0.3, label="f(x) = x")

		ax1.set_xlim(0, 1)
		ax1.set_ylim(0, 1)
		ax1.set_xlabel("x")
		ax1.set_ylabel("f(x)")
		ax1.set_title(f"f(x) with p={p:.2f}, w={w:.2f}, α={alpha:.1f}")  # noqa: RUF001
		ax1.grid(True, alpha=0.3)
		ax1.legend(loc="best")

		# Plot 2: f(p) vs p with fixed x (top right)
		ax2 = fig.add_subplot(gs[0, 1])

		# Plot the main curve with current x value
		f_p_values = np.array([soft_step(x, p_val, alpha, w) for p_val in ps])
		ax2.plot(ps, f_p_values, "blue", linewidth=2.5, label=f"x = {x:.2f}")

		# Create additional curves for different x values
		x_values = [0.2, 0.4, 0.6, 0.8]
		colors = ["purple", "orange", "magenta", "green"]

		for x_val, color in zip(x_values, colors, strict=False):
			# Don't draw if too close to current x
			if abs(x_val - x) > 0.05:  # noqa: PLR2004
				f_p_values = np.array(
					[soft_step(x_val, p_val, alpha, w) for p_val in ps],
				)
				ax2.plot(
					ps,
					f_p_values,
					color=color,
					linewidth=1.5,
					alpha=0.4,
					label=f"x = {x_val}",
				)

		# Add guideline for current p value
		ax2.axvline(x=p, color="red", linestyle="--", alpha=0.7)

		ax2.set_xlim(0, 1)
		ax2.set_ylim(0, 1)
		ax2.set_xlabel("p")
		ax2.set_ylabel("f(x,p)")
		ax2.set_title(f"f(x,p) for fixed x={x:.2f}, w={w:.2f}, α={alpha:.1f}")  # noqa: RUF001
		ax2.grid(True, alpha=0.3)
		ax2.legend(loc="best")

		if heatmap:
			# Plot 3: Heatmap of f(x,p) (bottom left)
			ax3 = fig.add_subplot(gs[1, 0])
			X, P = np.meshgrid(xs, ps)  # noqa: N806
			Z = np.zeros_like(X)  # noqa: N806

			# Calculate f(x,p) for all combinations
			for i, p_val in enumerate(ps):
				# TYPING: error: Incompatible types in assignment (expression has type "floating[Any]", variable has type "float")  [assignment]
				for j, x_val in enumerate(xs):  # type: ignore[assignment]
					Z[i, j] = soft_step(x_val, p_val, alpha, w)

			c = ax3.pcolormesh(X, P, Z, cmap="viridis", shading="auto")

			# Add current parameter values as lines
			ax3.axhline(y=p, color="red", linestyle="--", label=f"p = {p:.2f}")
			ax3.axvline(x=w, color="green", linestyle="--", label=f"w = {w:.2f}")
			ax3.axvline(x=x, color="blue", linestyle="--", label=f"x = {x:.2f}")

			# Add lines for the reference x values used in the top-right plot
			for x_val, color in zip(x_values, colors, strict=False):
				# Don't draw if too close to current x, magic value is fine
				if abs(x_val - x) > 0.05:  # noqa: PLR2004
					ax3.axvline(x=x_val, color=color, linestyle=":", alpha=0.4)

			# Mark the specific point corresponding to the current x and p values
			ax3.plot(x, p, "ro", markersize=8)

			# yes we mean to use alpha here (RUF001)
			ax3.set_xlabel("x")
			ax3.set_ylabel("p")
			ax3.set_title(f"f(x,p) heatmap with w={w:.2f}, α={alpha:.1f}")  # noqa: RUF001
			fig.colorbar(c, ax=ax3, label="f(x,p)")

			# Plot 4: NEW Heatmap of f(x,p) as function of k and alpha (bottom right)
			ax4 = fig.add_subplot(gs[1, 1])

			# Create k and alpha ranges
			ws = np.linspace(0.0, 1.0, 100)
			alphas = np.linspace(0.1, 30.0, 100)

			K, A = np.meshgrid(ws, alphas)  # noqa: N806
			Z_ka = np.zeros_like(K)  # noqa: N806

			# Calculate f(x,p) for all combinations of k and alpha
			for i, alpha_val in enumerate(alphas):
				for j, w_val in enumerate(ws):
					Z_ka[i, j] = soft_step(x, p, alpha_val, w_val)

			c2 = ax4.pcolormesh(K, A, Z_ka, cmap="plasma", shading="auto")

			# Add current parameter values as lines
			# yes we mean to use alpha here (RUF001)
			ax4.axhline(
				y=alpha,
				color="purple",
				linestyle="--",
				label=f"α = {alpha:.1f}",  # noqa: RUF001
			)
			ax4.axvline(x=w, color="green", linestyle="--", label=f"w = {w:.2f}")

			# Mark the specific point corresponding to the current w and alpha values
			ax4.plot(w, alpha, "ro", markersize=8)

			# yes we mean to use alpha here (RUF001)
			ax4.set_xlabel("w")
			ax4.set_ylabel("α")  # noqa: RUF001
			ax4.set_title(f"f(x,p) heatmap with fixed x={x:.2f}, p={p:.2f}")
			fig.colorbar(c2, ax=ax4, label="f(x,p,w,α)")  # noqa: RUF001

		plt.tight_layout()
		plt.show()

	# Display the interactive widget
	interactive_output = widgets.interactive_output(
		update_plot,
		{"x": x_slider, "p": p_slider, "w": w_slider, "alpha": alpha_slider},
	)

	# we noqa here because we will only call this function inside a notebook
	if not TYPE_CHECKING:
		display(VBox([slider_box, interactive_output]))  # noqa: F821

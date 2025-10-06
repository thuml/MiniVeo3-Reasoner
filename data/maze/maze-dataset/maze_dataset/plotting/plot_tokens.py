"`plot_colored_text` function to plot tokens on a matplotlib axis with colored backgrounds"

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_colored_text(
	tokens: Sequence[str],
	weights: Sequence[float],
	# assume its a colormap if not a string
	cmap: str | Any,  # noqa: ANN401
	ax: plt.Axes | None = None,
	width_scale: float = 0.023,
	width_offset: float = 0.005,
	height_offset: float = 0.1,
	rect_height: float = 0.7,
	token_height: float = 0.7,
	label_height: float = 0.3,
	word_gap: float = 0.01,
	fontsize: int = 12,
	fig_height: float = 0.7,
	fig_width_scale: float = 0.25,
	char_min: int = 4,
) -> plt.Axes:
	"hacky function to plot tokens on a matplotlib axis with colored backgrounds"
	assert len(tokens) == len(weights), (
		f"The number of tokens and weights must be the same: {len(tokens)} != {len(weights)}"
	)
	total_len_estimate: float = sum([max(len(tok), char_min) for tok in tokens])
	# set up figure if needed
	if ax is None:
		fig, ax = plt.subplots(
			figsize=(total_len_estimate * fig_width_scale, fig_height),
		)
	ax.axis("off")

	# Normalize the weights to be between 0 and 1
	norm_weights: Sequence[float] = (weights - np.min(weights)) / (
		np.max(weights) - np.min(weights)
	)

	# Create a colormap instance
	if isinstance(cmap, str):
		colormap = plt.get_cmap(cmap)
	else:
		colormap = cmap

	x_pos: float = 0.0
	for i, (tok, weight, norm_wgt) in enumerate(  # noqa: B007
		zip(tokens, weights, norm_weights, strict=False),
	):
		color = colormap(norm_wgt)[:3]

		# Plot the background color
		rect_width = width_scale * max(len(tok), char_min)
		ax.add_patch(
			plt.Rectangle(
				(x_pos, height_offset),
				rect_width,
				height_offset + rect_height,
				fc=color,
				ec="none",
			),
		)

		# Plot the token
		ax.text(
			x_pos + width_offset,
			token_height,
			tok,
			fontsize=fontsize,
			va="center",
			ha="left",
		)

		# Plot the weight below the token
		ax.text(
			x_pos + width_offset,
			label_height,
			f"{weight:.2f}",
			fontsize=fontsize,
			va="center",
			ha="left",
		)

		x_pos += rect_width + word_gap

	return ax

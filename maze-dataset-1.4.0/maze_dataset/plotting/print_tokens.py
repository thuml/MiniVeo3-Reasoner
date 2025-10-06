"""Functions to print tokens with colors in different formats

you can color the tokens by their:

- type (i.e. adjacency list, origin, target, path) using `color_maze_tokens_AOTP`
- custom weights (i.e. attention weights) using `color_tokens_cmap`
- entirely custom colors using `color_tokens_rgb`

and the output can be in different formats, specified by `FormatType` (html, latex, terminal)

"""

import html
import textwrap
from typing import Literal, Sequence

import matplotlib  # noqa: ICN001
import numpy as np
from IPython.display import HTML, display
from jaxtyping import Float, UInt8
from muutils.misc import flatten

from maze_dataset.constants import SPECIAL_TOKENS
from maze_dataset.token_utils import tokens_between

RGBArray = UInt8[np.ndarray, "n 3"]
"1D array of RGB values"

FormatType = Literal["html", "latex", "terminal", None]
"output format for the tokens"

TEMPLATES: dict[FormatType, str] = {
	"html": '<span style="color: black; background-color: rgb({clr})">&nbsp{tok}&nbsp</span>',
	"latex": "\\colorbox[RGB]{{ {clr} }}{{ \\texttt{{ {tok} }} }}",
	"terminal": "\033[30m\033[48;2;{clr}m{tok}\033[0m",
}
"templates of printing tokens in different formats"

_COLOR_JOIN: dict[FormatType, str] = {
	"html": ",",
	"latex": ",",
	"terminal": ";",
}
"joiner for colors in different formats"


def _escape_tok(
	tok: str,
	fmt: FormatType,
) -> str:
	"escape token based on format"
	if fmt == "html":
		return html.escape(tok)
	elif fmt == "latex":
		return tok.replace("_", "\\_").replace("#", "\\#")
	elif fmt == "terminal":
		return tok
	else:
		err_msg: str = f"Unexpected format: {fmt}"
		raise ValueError(err_msg)


def color_tokens_rgb(
	tokens: list,
	colors: Sequence[Sequence[int]] | Float[np.ndarray, "n 3"],
	fmt: FormatType = "html",
	template: str | None = None,
	clr_join: str | None = None,
	max_length: int | None = None,
) -> str:
	"""color tokens from a list with an RGB color array

	tokens will not be escaped if `fmt` is None

	# Parameters:
	- `max_length: int | None`: Max number of characters before triggering a line wrap, i.e., making a new colorbox. If `None`, no limit on max length.
	"""
	# process format
	if fmt is None:
		assert template is not None
		assert clr_join is not None
	else:
		assert template is None
		assert clr_join is None
		template = TEMPLATES[fmt]
		clr_join = _COLOR_JOIN[fmt]

	if max_length is not None:
		# TODO: why are we using a map here again?
		# TYPING: this is missing a lot of type hints
		wrapped: list = list(  # noqa: C417
			map(
				lambda x: textwrap.wrap(
					x,
					width=max_length,
					break_long_words=False,
					break_on_hyphens=False,
				),
				tokens,
			),
		)
		colors = list(
			flatten(
				[[colors[i]] * len(wrapped[i]) for i in range(len(wrapped))],
				levels_to_flatten=1,
			),
		)
		wrapped = list(flatten(wrapped, levels_to_flatten=1))
		tokens = wrapped

	# put everything together
	output = [
		template.format(
			clr=clr_join.join(map(str, map(int, clr))),
			tok=_escape_tok(tok, fmt),
		)
		for tok, clr in zip(tokens, colors, strict=False)
	]

	return " ".join(output)


# TYPING: would be nice to type hint as html, latex, or terminal string and overload depending on `FormatType`
def color_tokens_cmap(
	tokens: list[str],
	weights: Sequence[float],
	cmap: str | matplotlib.colors.Colormap = "Blues",
	fmt: FormatType = "html",
	template: str | None = None,
	labels: bool = False,
) -> str:
	"color tokens given a list of weights and a colormap"
	n_tok: int = len(tokens)
	assert n_tok == len(weights), f"'{len(tokens) = }' != '{len(weights) = }'"
	weights_np: Float[np.ndarray, " n_tok"] = np.array(weights)
	# normalize weights to [0, 1]
	weights_norm = matplotlib.colors.Normalize()(weights_np)

	if isinstance(cmap, str):
		cmap = matplotlib.colormaps.get_cmap(cmap)

	colors: RGBArray = cmap(weights_norm)[:, :3] * 255

	output: str = color_tokens_rgb(
		tokens=tokens,
		colors=colors,
		fmt=fmt,
		template=template,
	)

	if labels:
		if fmt != "terminal":
			raise NotImplementedError("labels only supported for terminal")
		# align labels with the tokens
		output += "\n"
		for tok, weight in zip(tokens, weights_np, strict=False):
			# 2 decimal points, left-aligned and trailing spaces to match token length
			weight_str: str = f"{weight:.1f}"
			# omit if longer than token
			if len(weight_str) > len(tok):
				weight_str = " " * len(tok)
			else:
				weight_str = weight_str.ljust(len(tok))
			output += f"{weight_str} "

	return output


# colors roughly made to be similar to visual representation
_MAZE_TOKENS_DEFAULT_COLORS: dict[tuple[str, str], tuple[int, int, int]] = {
	(SPECIAL_TOKENS.ADJLIST_START, SPECIAL_TOKENS.ADJLIST_END): (
		176,
		152,
		232,
	),  # purple
	(SPECIAL_TOKENS.ORIGIN_START, SPECIAL_TOKENS.ORIGIN_END): (154, 239, 123),  # green
	(SPECIAL_TOKENS.TARGET_START, SPECIAL_TOKENS.TARGET_END): (246, 136, 136),  # red
	(SPECIAL_TOKENS.PATH_START, SPECIAL_TOKENS.PATH_END): (111, 187, 254),  # blue
}
"default colors for maze tokens, roughly matches the format of `as_pixels`"


def color_maze_tokens_AOTP(
	tokens: list[str],
	fmt: FormatType = "html",
	template: str | None = None,
	**kwargs,
) -> str:
	"""color tokens assuming AOTP format

	i.e: adjaceny list, origin, target, path

	"""
	output: list[str] = [
		" ".join(
			tokens_between(
				tokens,
				start_tok,
				end_tok,
				include_start=True,
				include_end=True,
			),
		)
		for start_tok, end_tok in _MAZE_TOKENS_DEFAULT_COLORS
	]

	colors: RGBArray = np.array(
		list(_MAZE_TOKENS_DEFAULT_COLORS.values()),
		dtype=np.uint8,
	)

	return color_tokens_rgb(
		tokens=output,
		colors=colors,
		fmt=fmt,
		template=template,
		**kwargs,
	)


def display_html(html: str) -> None:
	"display html string"
	display(HTML(html))


def display_color_tokens_rgb(
	tokens: list[str],
	colors: RGBArray,
) -> None:
	"""display tokens (as html) with custom colors"""
	html: str = color_tokens_rgb(tokens, colors, fmt="html")
	display_html(html)


def display_color_tokens_cmap(
	tokens: list[str],
	weights: Sequence[float],
	cmap: str | matplotlib.colors.Colormap = "Blues",
) -> None:
	"""display tokens (as html) with color based on weights"""
	html: str = color_tokens_cmap(tokens, weights, cmap)
	display_html(html)


def display_color_maze_tokens_AOTP(
	tokens: list[str],
) -> None:
	"""display maze tokens (as html) with AOTP coloring"""
	html: str = color_maze_tokens_AOTP(tokens)
	display_html(html)

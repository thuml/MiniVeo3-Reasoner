"""a whole bunch of utilities for tokenization"""

import re
import typing
import warnings
from collections import Counter
from typing import Callable, Literal, overload

import numpy as np
from jaxtyping import Bool, Float, Int, Int8
from muutils.errormode import ErrorMode
from muutils.misc import list_join
from muutils.misc.sequence import WhenMissing

from maze_dataset.constants import (
	CARDINAL_MAP,
	SPECIAL_TOKENS,
	VOCAB,
	ConnectionArray,
	ConnectionList,
	CoordTup,
)

# filtering things from a prompt or generated text
# ==================================================


def remove_padding_from_token_str(token_str: str) -> str:
	"""remove padding tokens from a joined token string"""
	token_str = token_str.replace(f"{SPECIAL_TOKENS.PADDING} ", "")
	token_str = token_str.replace(f"{SPECIAL_TOKENS.PADDING}", "")
	return token_str  # noqa: RET504


def tokens_between(
	tokens: list[str],
	start_value: str,
	end_value: str,
	include_start: bool = False,
	include_end: bool = False,
	except_when_tokens_not_unique: bool = False,
) -> list[str]:
	"""given a list `tokens`, get the tokens between `start_value` and `end_value`

	_extended_summary_

	# Parameters:
	- `tokens : list[str]`
	- `start_value : str`
	- `end_value : str`
	- `include_start : bool`
		(defaults to `False`)
	- `include_end : bool`
		(defaults to `False`)
	- `except_when_tokens_not_unique : bool`
		when `True`, raise an error if `start_value` or `end_value` are not unique in the input tokens
		(defaults to `False`)

	# Returns:
	- `list[str]`

	# Raises:
	- `ValueError` : if `start_value` and `end_value` are the same
	- `ValueError` : if `except_when_tokens_not_unique` is `True` and `start_value` or `end_value` are not unique in the input tokens
	- `ValueError` : if `start_value` or `end_value` are not present in the input tokens
	"""
	if start_value == end_value:
		err_msg: str = f"start_value and end_value cannot be the same: {start_value = } {end_value = }"
		raise ValueError(
			err_msg,
		)
	if except_when_tokens_not_unique:
		if (tokens.count(start_value) != 1) or (tokens.count(end_value) != 1):
			err_msg: str = (
				"start_value or end_value is not unique in the input tokens:"
				f"\n{tokens.count(start_value) = } {tokens.count(end_value) = }"
				f"\n{start_value = } {end_value = }"
				f"\n{tokens = }"
			)
			raise ValueError(err_msg)
	else:
		if (tokens.count(start_value) < 1) or (tokens.count(end_value) < 1):
			err_msg: str = (
				"start_value or end_value is not present in the input tokens:"
				f"\n{tokens.count(start_value) = } {tokens.count(end_value) = }"
				f"\n{start_value = } {end_value = }"
				f"\n{tokens = }"
			)
			raise ValueError(err_msg)

	start_idx: int = tokens.index(start_value) + int(not include_start)
	end_idx: int = tokens.index(end_value) + int(include_end)

	assert start_idx < end_idx, "Start must come before end"

	return tokens[start_idx:end_idx]


def get_adj_list_tokens(tokens: list[str]) -> list[str]:
	"get tokens between ADJLIST_START and ADJLIST_END, without the special tokens themselves"
	return tokens_between(
		tokens,
		SPECIAL_TOKENS.ADJLIST_START,
		SPECIAL_TOKENS.ADJLIST_END,
	)


def get_path_tokens(tokens: list[str], trim_end: bool = False) -> list[str]:
	"""The path is considered everything from the first path coord to the path_end token, if it exists."""
	if SPECIAL_TOKENS.PATH_START not in tokens:
		err_msg: str = f"Path start token {SPECIAL_TOKENS.PATH_START} not found in tokens:\n{tokens}"
		raise ValueError(
			err_msg,
		)
	start_idx: int = tokens.index(SPECIAL_TOKENS.PATH_START) + int(trim_end)
	end_idx: int | None = None
	if trim_end and (SPECIAL_TOKENS.PATH_END in tokens):
		end_idx = tokens.index(SPECIAL_TOKENS.PATH_END)
	return tokens[start_idx:end_idx]


def get_context_tokens(tokens: list[str]) -> list[str]:
	"get tokens between ADJLIST_START and PATH_START"
	return tokens_between(
		tokens,
		SPECIAL_TOKENS.ADJLIST_START,
		SPECIAL_TOKENS.PATH_START,
		include_start=True,
		include_end=True,
	)


def get_origin_tokens(tokens: list[str]) -> list[str]:
	"get tokens_between ORIGIN_START and ORIGIN_END"
	return tokens_between(
		tokens,
		SPECIAL_TOKENS.ORIGIN_START,
		SPECIAL_TOKENS.ORIGIN_END,
		include_start=False,
		include_end=False,
	)


def get_target_tokens(tokens: list[str]) -> list[str]:
	"get tokens_between TARGET_START and TARGET_END"
	return tokens_between(
		tokens,
		SPECIAL_TOKENS.TARGET_START,
		SPECIAL_TOKENS.TARGET_END,
		include_start=False,
		include_end=False,
	)


def get_cardinal_direction(coords: Int[np.ndarray, "start_end=2 row_col=2"]) -> str:
	"""Returns the cardinal direction token corresponding to traveling from `coords[0]` to `coords[1]`."""
	return CARDINAL_MAP[tuple(coords[1] - coords[0])]


def get_relative_direction(coords: Int[np.ndarray, "prev_cur_next=3 row_col=2"]) -> str:
	"""Returns the relative first-person direction token corresponding to traveling from `coords[1]` to `coords[2]`.

	# Parameters
	- `coords`: Contains 3 Coords, each of which must neighbor the previous Coord.
		- `coords[0]`: The previous location, used to determine the current absolute direction that the "agent" is facing.
		- `coords[1]`: The current location
		- `coords[2]`: The next location. May be equal to the current location.
	"""
	if coords.shape != (3, 2):
		err_msg: str = f"`coords` must have shape (3,2). Got {coords.shape} instead."
		raise ValueError(err_msg)
	directions = coords[1:] - coords[:-1]
	if not np.all(np.linalg.norm(directions, axis=1) <= np.array([1.1, 1.1])):
		# Use floats as constant since `np.linalg.norm` returns float array
		err_msg: str = f"Adjacent `coords` must be neighboring or equivalent. Got {coords} instead."
		raise ValueError(
			err_msg,
		)
	if np.array_equal(coords[1], coords[2]):
		return VOCAB.PATH_STAY
	if np.array_equal(coords[0], coords[2]):
		return VOCAB.PATH_BACKWARD
	if np.array_equal(coords[0], coords[1]):
		err_msg: str = f"Previous first-person direction indeterminate from {coords=}."
		raise ValueError(
			err_msg,
		)
	if np.array_equal(directions[0], directions[1]):
		return VOCAB.PATH_FORWARD
	directions = np.append(
		directions,
		[[0], [0]],
		axis=1,
	)  # Augment to represent unit basis vectors in 3D
	match np.cross(directions[0], directions[1])[-1]:
		case 1:
			return VOCAB.PATH_LEFT
		case -1:
			return VOCAB.PATH_RIGHT


class TokenizerPendingDeprecationWarning(PendingDeprecationWarning):
	"""Pending deprecation warnings related to the `MazeTokenizerModular` upgrade."""

	pass


def str_is_coord(coord_str: str, allow_whitespace: bool = True) -> bool:
	"""return True if the string represents a coordinate, False otherwise"""
	warnings.warn(
		"`util.str_is_coord` only supports legacy UT strings. Function will be replaced with a generalized version in a future release.",
		TokenizerPendingDeprecationWarning,
	)
	strip_func: Callable[[str], str] = lambda x: x.strip() if allow_whitespace else x  # noqa: E731

	coord_str = strip_func(coord_str)

	return all(
		[
			coord_str.startswith("("),
			coord_str.endswith(")"),
			"," in coord_str,
			all(
				strip_func(x).isdigit()
				for x in strip_func(coord_str.lstrip("(").rstrip(")")).split(",")
			),
		],
	)


class TokenizerDeprecationWarning(DeprecationWarning):
	"""Deprecation warnings related to the `MazeTokenizerModular` upgrade."""

	pass


# coordinate to strings
# ==================================================


def _coord_to_strings_UT(coord: typing.Sequence[int]) -> list[str]:
	"""convert a coordinate to a string: `(i,j)`->"(i,j)"

	always returns a list of length 1
	"""
	return [f"({','.join(str(c) for c in coord)})"]


def _coord_to_strings_indexed(coord: typing.Sequence[int]) -> list[str]:
	"""convert a coordinate to a list of indexed strings: `(i,j)`->"(", "i", ",", "j", ")"

	always returns a list of length 5
	"""
	return [
		"(",
		*list_join([str(c) for c in coord], lambda: ","),
		")",
	]


def coord_str_to_tuple(
	coord_str: str,
	allow_whitespace: bool = True,
) -> tuple[int, ...]:
	"""convert a coordinate string to a tuple"""
	strip_func: Callable[[str], str] = lambda x: x.strip() if allow_whitespace else x  # noqa: E731
	coord_str = strip_func(coord_str)
	stripped: str = strip_func(coord_str.lstrip("(").rstrip(")"))
	return tuple(int(strip_func(x)) for x in stripped.split(","))


def coord_str_to_coord_np(coord_str: str, allow_whitespace: bool = True) -> np.ndarray:
	"""convert a coordinate string to a numpy array"""
	return np.array(coord_str_to_tuple(coord_str, allow_whitespace=allow_whitespace))


def coord_str_to_tuple_noneable(coord_str: str) -> CoordTup | None:
	"""convert a coordinate string to a tuple, or None if the string is not a coordinate string"""
	if not str_is_coord(coord_str):
		return None
	return coord_str_to_tuple(coord_str)


def coords_string_split_UT(coords: str) -> list[str]:
	"""Splits a string of tokens into a list containing the UT tokens for each coordinate.

	Not capable of producing indexed tokens ("(", "1", ",", "2", ")"), only unique tokens ("(1,2)").
	Non-whitespace portions of the input string not matched are preserved in the same list:
	"(1,2) <SPECIAL_TOKEN> (5,6)" -> ["(1,2)", "<SPECIAL_TOKEN>", "(5,6)"]
	"""
	# ty gpt4
	return re.findall(r"\([^)]*\)|\S+", coords)


# back and forth in wrapped form
# ==================================================
@overload
def strings_to_coords(
	text: str | list[str],
	when_noncoord: Literal["skip"] = "skip",
) -> list[CoordTup]: ...
@overload
def strings_to_coords(
	text: str | list[str],
	when_noncoord: Literal["error"] = "error",
) -> list[CoordTup]: ...
@overload
def strings_to_coords(
	text: str | list[str],
	when_noncoord: Literal["include"] = "include",
) -> list[str | CoordTup]: ...
def strings_to_coords(
	text: str | list[str],
	when_noncoord: WhenMissing = "skip",
) -> list[str | CoordTup]:
	"""converts a list of tokens to a list of coordinates

	returns list[CoordTup] if `when_noncoord` is "skip" or "error"
	returns list[str | CoordTup] if `when_noncoord` is "include"
	"""
	warnings.warn(
		"`util.strings_to_coords` only supports legacy UT strings. Function will be replaced with a generalized version in a future release.",
		TokenizerPendingDeprecationWarning,
	)
	tokens_joined: str = text if isinstance(text, str) else " ".join(text)
	tokens_processed: list[str] = coords_string_split_UT(tokens_joined)
	result: list[str] = list()
	for token in tokens_processed:
		coord: CoordTup | None = coord_str_to_tuple_noneable(token)
		if coord is None:
			if when_noncoord == "skip":
				continue
			if when_noncoord == "error":
				err_msg: str = (
					f"Invalid non-coordinate token '{token}' in text: '{text}'"
				)
				raise ValueError(
					err_msg,
				)
			if when_noncoord == "include":
				result.append(token)
			else:
				err_msg: str = f"Invalid when_noncoord value '{when_noncoord}'"
				raise ValueError(err_msg)
		else:
			result.append(coord)
	return result


@overload
def coords_to_strings(
	coords: list[str | CoordTup],
	coord_to_strings_func: Callable[[CoordTup], list[str]],
	when_noncoord: Literal["include", "skip"] = "skip",
) -> list[str]: ...
@overload
def coords_to_strings(
	coords: list[CoordTup],
	coord_to_strings_func: Callable[[CoordTup], list[str]],
	when_noncoord: Literal["error"] = "error",
) -> list[str]: ...
def coords_to_strings(
	coords: list[str | CoordTup],
	coord_to_strings_func: Callable[[CoordTup], list[str]],
	when_noncoord: WhenMissing = "skip",
) -> list[str]:
	"""converts a list of coordinates to a list of strings (tokens)

	expects list[CoordTup] if `when_noncoord` is "error"
	expects list[str | CoordTup] if `when_noncoord` is "include" or "skip"
	"""
	result: list[str] = list()
	for coord in coords:
		if isinstance(coord, str):
			if when_noncoord == "skip":
				continue
			if when_noncoord == "error":
				err_msg: str = (
					f"Invalid non-coordinate '{coord}' in list of coords: '{coords}'"
				)
				raise ValueError(
					err_msg,
				)
			if when_noncoord == "include":
				result.append(coord)
			else:
				err_msg: str = f"Invalid when_noncoord value '{when_noncoord}'"
				raise ValueError(err_msg)
		else:
			result.extend(coord_to_strings_func(coord))
	return result


def get_token_regions(toks: list[str]) -> tuple[list[str], list[str]]:
	"""Splits a list of tokens into adjacency list tokens and non-adjacency list tokens."""
	adj_list_start, adj_list_end = (
		toks.index("<ADJLIST_START>") + 1,
		toks.index("<ADJLIST_END>"),
	)
	adj_list = toks[adj_list_start:adj_list_end]
	non_adj_list = toks[:adj_list_start] + toks[adj_list_end:]
	return adj_list, non_adj_list


def equal_except_adj_list_sequence(  # noqa: C901
	rollout1: list[str],
	rollout2: list[str],
	do_except: bool = False,
	when_counter_mismatch: ErrorMode = ErrorMode.EXCEPT,
	when_len_mismatch: ErrorMode = ErrorMode.EXCEPT,
) -> bool:
	"""Returns if the rollout strings are equal, allowing for differently sequenced adjacency lists.

	<ADJLIST_START> and <ADJLIST_END> tokens must be in the rollouts.
	Intended ONLY for determining if two tokenization schemes are the same for rollouts generated from the same maze.
	This function should NOT be used to determine if two rollouts encode the same `LatticeMaze` object.

	# Warning: CTT False Positives
	This function is not robustly correct for some corner cases using `CoordTokenizers.CTT`.
	If rollouts are passed for identical tokenizers processing two slightly different mazes, a false positive is possible.
	More specifically, some cases of zero-sum adding and removing of connections in a maze within square regions along the diagonal will produce a false positive.
	"""
	if len(rollout1) != len(rollout2):
		if do_except:
			when_len_mismatch.process(
				f"Rollouts are not the same length: {len(rollout1)} != {len(rollout2)}",
			)
		return False
	if ("<ADJLIST_START>" in rollout1) ^ ("<ADJLIST_START>" in rollout2):
		if do_except:
			err_msg: str = f"Rollouts do not have the same <ADJLIST_START> token: `{'<ADJLIST_START>' in rollout1 = }` != `{'<ADJLIST_START>' in rollout2 = }`"
			raise ValueError(
				err_msg,
			)
		return False
	if ("<ADJLIST_END>" in rollout1) ^ ("<ADJLIST_END>" in rollout2):
		if do_except:
			err_msg: str = f"Rollouts do not have the same <ADJLIST_END> token: `{'<ADJLIST_END>' in rollout1 = }` != `{'<ADJLIST_END>' in rollout2 = }`"
			raise ValueError(
				err_msg,
			)
		return False

	adj_list1, non_adj_list1 = get_token_regions(rollout1)
	adj_list2, non_adj_list2 = get_token_regions(rollout2)
	if non_adj_list1 != non_adj_list2:
		if do_except:
			when_len_mismatch.process(
				f"Non-adjacency list tokens are not the same:\n{non_adj_list1}\n!=\n{non_adj_list2}",
			)
			err_msg: str = f"Non-adjacency list tokens are not the same:\n{non_adj_list1}\n!=\n{non_adj_list2}"
			raise ValueError(
				err_msg,
			)
		return False
	counter1: Counter = Counter(adj_list1)
	counter2: Counter = Counter(adj_list2)
	counters_eq: bool = counter1 == counter2
	if not counters_eq:
		if do_except:
			when_counter_mismatch.process(
				f"Adjacency list counters are not the same:\n{counter1}\n!=\n{counter2}\n{counter1 - counter2 = }",
			)
		return False

	return True


def connection_list_to_adj_list(
	conn_list: ConnectionList,
	shuffle_d0: bool = True,
	shuffle_d1: bool = True,
) -> Int8[np.ndarray, "conn start_end=2 coord=2"]:
	"""converts a `ConnectionList` (special lattice format) to a shuffled adjacency list

	# Parameters:
	- `conn_list: ConnectionList`
		special internal format for graphs which are subgraphs of a lattice
	- `shuffle_d0: bool`
		shuffle the adjacency list along the 0th axis (order of pairs)
	- `shuffle_d1: bool`
		shuffle the adjacency list along the 1st axis (order of coordinates in each pair).
		If `False`, all pairs have the smaller coord first.


	# Returns:
	- `Int8[np.ndarray, "conn start_end=2 coord=2"]`
		adjacency list in the shape `(n_connections, 2, 2)`
	"""
	n_connections: int = conn_list.sum()
	adj_list: Int8[np.ndarray, "conn start_end=2 coord=2"] = np.full(
		(n_connections, 2, 2),
		-1,
		dtype=np.int8,
	)

	if shuffle_d1:
		flip_d1: Float[np.ndarray, " conn"] = np.random.rand(n_connections)

	# loop over all nonzero elements of the connection list
	i: int = 0
	for d, x, y in np.ndindex(conn_list.shape):
		if conn_list[d, x, y]:
			c_start: CoordTup = (x, y)
			c_end: CoordTup = (
				x + (1 if d == 0 else 0),
				y + (1 if d == 1 else 0),
			)
			adj_list[i, 0] = np.array(c_start, dtype=np.int8)
			adj_list[i, 1] = np.array(c_end, dtype=np.int8)

			# flip if shuffling
			# magic value is fine here
			if shuffle_d1 and (flip_d1[i] > 0.5):  # noqa: PLR2004
				c_s, c_e = adj_list[i, 0].copy(), adj_list[i, 1].copy()
				adj_list[i, 0] = c_e
				adj_list[i, 1] = c_s

			i += 1

	if shuffle_d0:
		np.random.shuffle(adj_list)

	return adj_list


def is_connection(
	edges: ConnectionArray,
	connection_list: ConnectionList,
) -> Bool[np.ndarray, "is_connection=edges"]:
	"""Returns if each edge in `edges` is a connection (`True`) or wall (`False`) in `connection_list`."""
	sorted_edges = np.sort(edges, axis=1)
	edge_direction = (
		(sorted_edges[:, 1, :] - sorted_edges[:, 0, :])[:, 0] == 0
	).astype(np.int8)
	return connection_list[edge_direction, sorted_edges[:, 0, 0], sorted_edges[:, 0, 1]]


# string to coordinate representation
# ==================================================

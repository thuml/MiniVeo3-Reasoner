"misc utilities for the `maze_dataset` package"

import math
from typing import (
	overload,
)

import numpy as np
from jaxtyping import Bool, Int, Int8


def bool_array_from_string(
	string: str,
	shape: list[int],
	true_symbol: str = "T",
) -> Bool[np.ndarray, "*shape"]:
	"""Transform a string into an ndarray of bools.

	Parameters
	----------
	string: str
		The string representation of the array
	shape: list[int]
		The shape of the resulting array
	true_symbol:
		The character to parse as True. Whitespace will be removed. All other characters will be parsed as False.

	Returns
	-------
	np.ndarray
		A ndarray with dtype bool of shape `shape`

	Examples
	--------
	>>> bool_array_from_string(
	...	 "TT TF", shape=[2,2]
	... )
	array([[ True,  True],
		[ True, False]])

	"""
	stripped = "".join(string.split())

	expected_symbol_count = math.prod(shape)
	symbol_count = len(stripped)
	if len(stripped) != expected_symbol_count:
		err_msg: str = f"Connection List contains the wrong number of symbols. Expected {expected_symbol_count}. Found {symbol_count} in {stripped}."
		raise ValueError(err_msg)

	bools = [(symbol == true_symbol) for symbol in stripped]
	return np.array(bools).reshape(*shape)


def corner_first_ndindex(n: int, ndim: int = 2) -> list[tuple]:
	"""returns an array of indices, sorted by distance from the corner

	this gives the property that `np.ndindex((n,n))` is equal to
	the first n^2 elements of `np.ndindex((n+1, n+1))`

	```
	>>> corner_first_ndindex(1)
	[(0, 0)]
	>>> corner_first_ndindex(2)
	[(0, 0), (0, 1), (1, 0), (1, 1)]
	>>> corner_first_ndindex(3)
	[(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2)]
	```
	"""
	unsorted: list = list(np.ndindex(tuple([n for _ in range(ndim)])))
	return sorted(unsorted, key=lambda x: (max(x), x if x[0] % 2 == 0 else x[::-1]))


# alternate numpy version from GPT-4:
"""
# Create all index combinations
indices = np.indices([n]*ndim).reshape(ndim, -1).T
# Find the max value for each index
max_indices = np.max(indices, axis=1)
# Identify the odd max values
odd_mask = max_indices % 2 != 0
# Make a copy of indices to avoid changing the original one
indices_copy = indices.copy()
# Reverse the order of the coordinates for indices with odd max value
indices_copy[odd_mask] = indices_copy[odd_mask, ::-1]
# Sort by max index value, then by coordinates
sorted_order = np.lexsort((*indices_copy.T, max_indices))
return indices[sorted_order]
"""


@overload
def manhattan_distance(
	edges: Int[np.ndarray, "edges coord=2 row_col=2"],
) -> Int8[np.ndarray, " edges"]: ...
# TYPING: error: Overloaded function signature 2 will never be matched: signature 1's parameter type(s) are the same or broader  [overload-cannot-match]
# this is because mypy doesn't play nice with jaxtyping
@overload
def manhattan_distance(  # type: ignore[overload-cannot-match]
	edges: Int[np.ndarray, "coord=2 row_col=2"],
) -> int: ...
def manhattan_distance(
	edges: (
		Int[np.ndarray, "edges coord=2 row_col=2"]
		| Int[np.ndarray, "coord=2 row_col=2"]
	),
) -> Int8[np.ndarray, " edges"] | int:
	"""Returns the Manhattan distance between two coords."""
	# magic values for dims fine here
	if len(edges.shape) == 3:  # noqa: PLR2004
		return np.linalg.norm(edges[:, 0, :] - edges[:, 1, :], axis=1, ord=1).astype(
			np.int8,
		)
	elif len(edges.shape) == 2:  # noqa: PLR2004
		return int(np.linalg.norm(edges[0, :] - edges[1, :], ord=1).astype(np.int8))
	else:
		err_msg: str = f"{edges} has shape {edges.shape}, but must be match the shape in the type hints."
		raise ValueError(err_msg)


def lattice_max_degrees(n: int) -> Int8[np.ndarray, "row col"]:
	"""Returns an array with the maximum possible degree for each coord."""
	out = np.full((n, n), 2)
	out[1:-1, :] += 1
	out[:, 1:-1] += 1
	return out


def lattice_connection_array(
	n: int,
) -> Int8[np.ndarray, "edges=2*n*(n-1) leading_trailing_coord=2 row_col=2"]:
	"""Returns a 3D NumPy array containing all the edges in a 2D square lattice of size n x n.

	Thanks Claude.

	# Parameters
	- `n`: The size of the square lattice.

	# Returns
	np.ndarray: A 3D NumPy array of shape containing the coordinates of the edges in the 2D square lattice.
	In each pair, the coord with the smaller sum always comes first.
	"""
	row_coords, col_coords = np.meshgrid(
		np.arange(n, dtype=np.int8),
		np.arange(n, dtype=np.int8),
		indexing="ij",
	)

	# Horizontal edges
	horiz_edges = np.column_stack(
		(
			row_coords[:, :-1].ravel(),
			col_coords[:, :-1].ravel(),
			row_coords[:, 1:].ravel(),
			col_coords[:, 1:].ravel(),
		),
	)

	# Vertical edges
	vert_edges = np.column_stack(
		(
			row_coords[:-1, :].ravel(),
			col_coords[:-1, :].ravel(),
			row_coords[1:, :].ravel(),
			col_coords[1:, :].ravel(),
		),
	)

	return np.concatenate(
		(horiz_edges.reshape(n**2 - n, 2, 2), vert_edges.reshape(n**2 - n, 2, 2)),
		axis=0,
	)


def adj_list_to_nested_set(adj_list: list) -> set:
	"""Used for comparison of adj_lists

	Adj_list looks like [[[0, 1], [1, 1]], [[0, 0], [0, 1]], ...]
	We don't care about order of coordinate pairs within
	the adj_list or coordinates within each coordinate pair.
	"""
	return {
		frozenset([tuple(start_coord), tuple(end_coord)])
		for start_coord, end_coord in adj_list
	}

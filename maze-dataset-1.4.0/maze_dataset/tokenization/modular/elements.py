"""implements subclasses of `_TokenizerElement` to be used in `MazeTokenizerModular`"""

import abc
import random
from typing import (
	Callable,
	Literal,
	Sequence,
	TypedDict,
)

import numpy as np
from jaxtyping import Bool, Int
from muutils.json_serialize import (
	serializable_dataclass,
	serializable_field,
)
from muutils.misc import empty_sequence_if_attr_false, flatten

# from maze_dataset import SolvedMaze
from maze_dataset.constants import (
	VOCAB,
	ConnectionArray,
	ConnectionList,
	Coord,
	CoordTup,
)
from maze_dataset.generation import _NUMPY_RNG
from maze_dataset.maze.lattice_maze import LatticeMaze, SolvedMaze
from maze_dataset.token_utils import (
	connection_list_to_adj_list,
	get_cardinal_direction,
	get_relative_direction,
	is_connection,
	tokens_between,
)
from maze_dataset.tokenization.modular.element_base import (
	__TokenizerElementNamespace,
	_load_tokenizer_element,
	_TokenizerElement,
	_unsupported_is_invalid,
	mark_as_unsupported,
)
from maze_dataset.utils import lattice_connection_array


class CoordTokenizers(__TokenizerElementNamespace):
	"""Namespace for `_CoordTokenizer` subclass hierarchy used by `MazeTokenizerModular`."""

	key = "coord_tokenizer"

	@serializable_dataclass(frozen=True, kw_only=True)
	class _CoordTokenizer(_TokenizerElement, abc.ABC):
		"""Superclass for classes which tokenize singular coords in a maze."""

		@abc.abstractmethod
		def to_tokens(self, coord: Coord | CoordTup) -> list[str]:
			pass

		@classmethod
		def attribute_key(cls) -> str:
			return CoordTokenizers.key

		def is_valid(self, do_except: bool = False) -> bool:
			# No invalid instances possible within data member type hint bounds
			return True

	@serializable_dataclass(frozen=True, kw_only=True)
	class UT(_CoordTokenizer):
		"""Unique token coordinate tokenizer."""

		# inherit docstring
		def to_tokens(self, coord: Coord | CoordTup) -> list[str]:  # noqa: D102
			return ["".join(["(", str(coord[0]), ",", str(coord[1]), ")"])]

	@serializable_dataclass(frozen=True, kw_only=True)
	class CTT(_CoordTokenizer):
		"""Coordinate tuple tokenizer

		# Parameters
		- `pre`: Whether all coords include an integral preceding delimiter token
		- `intra`: Whether all coords include a delimiter token between coordinates
		- `post`: Whether all coords include an integral following delimiter token
		"""

		pre: bool = serializable_field(default=True)
		intra: bool = serializable_field(default=True)
		post: bool = serializable_field(default=True)
		# Implement methods

		# inherit docstring
		def to_tokens(self, coord: Coord | CoordTup) -> list[str]:  # noqa: D102
			return [
				*empty_sequence_if_attr_false([VOCAB.COORD_PRE], self, "pre"),
				str(coord[0]),
				*empty_sequence_if_attr_false([VOCAB.COORD_INTRA], self, "intra"),
				str(coord[1]),
				*empty_sequence_if_attr_false([VOCAB.COORD_POST], self, "post"),
			]


class EdgeGroupings(__TokenizerElementNamespace):
	"""Namespace for `_EdgeGrouping` subclass hierarchy used by `_AdjListTokenizer`."""

	key = "edge_grouping"

	class _GroupingTokenParams(TypedDict):
		"""A uniform private hyperparameter interface used by `AdjListTokenizer`."""

		connection_token_ordinal: Literal[0, 1, 2]
		intra: bool
		grouped: bool

	@serializable_dataclass(frozen=True, kw_only=True)
	class _EdgeGrouping(_TokenizerElement, abc.ABC):
		"""Specifies if/how multiple coord-coord connections are grouped together in a token subsequence called a edge grouping."""

		@classmethod
		def attribute_key(cls) -> str:
			return EdgeGroupings.key

		def is_valid(self, do_except: bool = False) -> bool:
			return True

		@abc.abstractmethod
		def _group_edges(self, edges: ConnectionArray) -> Sequence[ConnectionArray]:
			"""Divides a ConnectionArray into groups of edges.

			Shuffles/sequences within each group if applicable.
			"""
			pass

		@abc.abstractmethod
		def _token_params(self) -> "EdgeGroupings._GroupingTokenParams":
			"""Returns the tok.nization hyperparameters necessary for an `AdjListTokenizer` to tokenize.

			These hyperparameters are not used by `_EdgeGrouping` internally.
			They are located in `_EdgeGrouping` rather than in `AdjListTokenizer`
			since the hyperparameter space is a function of the `_EdgeGrouping` subclass.
			This function resolves the `_EdgeGrouping` hyperparameter space which is non-uniform across subclasses
			into a uniform private interface used by `AdjListTokenizer`.
			"""
			pass

	@serializable_dataclass(frozen=True, kw_only=True)
	class Ungrouped(_EdgeGrouping):
		"""No grouping occurs, each edge is tokenized individually.

		# Parameters
		- `connection_token_ordinal`: At which index in the edge tokenization the connector (or wall) token appears.
		Edge tokenizations contain 3 parts: a leading coord, a connector (or wall) token, and either a second coord or cardinal direction tokenization.
		"""

		connection_token_ordinal: Literal[0, 1, 2] = serializable_field(
			default=1,
			assert_type=False,
		)

		def _token_params(self) -> "EdgeGroupings._GroupingTokenParams":
			return EdgeGroupings._GroupingTokenParams(
				connection_token_ordinal=self.connection_token_ordinal,
				intra=False,
				grouped=False,
			)

		def _group_edges(self, edges: ConnectionList) -> Sequence[ConnectionList]:
			return np.expand_dims(edges, 1)

	@serializable_dataclass(frozen=True, kw_only=True)
	@mark_as_unsupported(_unsupported_is_invalid)
	class ByLeadingCoord(_EdgeGrouping):
		"""All edges with the same leading coord are grouped together.

		# Parameters
		- `intra`: Whether all edge groupings include a delimiter token between individual edge representations.
		Note that each edge representation will already always include a connector token (`VOCAB.CONNECTOR`, or possibly `)
		- `shuffle_group`: Whether the sequence of edges within the group should be shuffled or appear in a fixed order.
		If false, the fixed order is lexicographical by (row, col).
		In effect, lexicographical sorting sorts edges by their cardinal direction in the sequence NORTH, WEST, EAST, SOUTH, where the directions indicate the position of the trailing coord relative to the leading coord.
		- `connection_token_ordinal`: At which index in token sequence representing a single edge the connector (or wall) token appears.
		Edge tokenizations contain 2 parts: a connector (or wall) token and a coord or cardinal tokenization.
		"""

		intra: bool = serializable_field(default=True)
		shuffle_group: bool = serializable_field(default=True)
		connection_token_ordinal: Literal[0, 1] = serializable_field(
			default=0,
			assert_type=False,
		)

		def _token_params(self) -> "EdgeGroupings._GroupingTokenParams":
			return EdgeGroupings._GroupingTokenParams(
				connection_token_ordinal=self.connection_token_ordinal,
				intra=self.intra,
				grouped=True,
			)

		def _group_edges(self, edges: ConnectionArray) -> Sequence[ConnectionArray]:
			# Adapted from: https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function
			index_array: Int[np.ndarray, "sort_indices=edges"] = np.lexsort(
				(edges[:, 1, 1], edges[:, 1, 0], edges[:, 0, 1], edges[:, 0, 0]),
			)
			sorted_edges: ConnectionArray = edges[index_array, ...]
			groups: list[ConnectionArray] = np.split(
				sorted_edges,
				np.unique(sorted_edges[:, 0, :], return_index=True, axis=0)[1][1:],
			)
			if self.shuffle_group:
				[_NUMPY_RNG.shuffle(g, axis=0) for g in groups]
			return groups


class EdgePermuters(__TokenizerElementNamespace):
	"""Namespace for `_EdgePermuter` subclass hierarchy used by `_AdjListTokenizer`."""

	key = "edge_permuter"

	@serializable_dataclass(frozen=True, kw_only=True)
	class _EdgePermuter(_TokenizerElement, abc.ABC):
		"""Specifies how to sequence the two coords that encode a lattice edge."""

		@classmethod
		def attribute_key(cls) -> str:
			return EdgePermuters.key

		def is_valid(self, do_except: bool = False) -> bool:
			# No invalid instances possible within data member type hint bounds
			return True

		@staticmethod
		@abc.abstractmethod
		def _permute(lattice_edges: ConnectionArray) -> ConnectionArray:
			"""Executes a permutation.

			Warning: Caller should be aware that `lattice_edges` may be modified in-place depending on the subclass's implementation.

			# Parameters
			- `lattice_edges`: Array of lattice edges.
			The two coords in shape[1] must be adjacent in the lattice.

			# Returns
			- Array of lattice edges with entries along shape[1] systematically permuted.
			- shape[0] of the returned array is NOT guaranteed to match `lattice_edges.shape[1]`.
			"""
			pass

	@serializable_dataclass(frozen=True, kw_only=True)
	class SortedCoords(_EdgePermuter):
		"""returns a sorted representation. useful for checking consistency"""

		@staticmethod
		def _permute(lattice_edges: ConnectionArray) -> ConnectionArray:
			return lattice_edges[
				np.lexsort(
					(
						lattice_edges[:, 1, 1],
						lattice_edges[:, 1, 0],
						lattice_edges[:, 0, 1],
						lattice_edges[:, 0, 0],
					),
				),
				...,
			]

	@serializable_dataclass(frozen=True, kw_only=True)
	class RandomCoords(_EdgePermuter):
		"""Permutes each edge randomly."""

		@staticmethod
		def _permute(lattice_edges: ConnectionArray) -> ConnectionArray:
			_NUMPY_RNG.permuted(lattice_edges, axis=1, out=lattice_edges)
			return lattice_edges

	@serializable_dataclass(frozen=True, kw_only=True)
	class BothCoords(_EdgePermuter):
		"""Includes both possible permutations of every edge in the output.

		Since input ConnectionList has only 1 instance of each edge,
		a call to `BothCoords._permute` will modify `lattice_edges` in-place, doubling `shape[0]`.
		"""

		@staticmethod
		def _permute(lattice_edges: ConnectionArray) -> ConnectionArray:
			return np.append(lattice_edges, np.flip(lattice_edges, axis=1), axis=0)


class EdgeSubsets(__TokenizerElementNamespace):
	"""Namespace for `_EdgeSubset` subclass hierarchy used by `_AdjListTokenizer`."""

	key = "edge_subset"

	@serializable_dataclass(frozen=True, kw_only=True)
	class _EdgeSubset(_TokenizerElement, abc.ABC):
		"""Component of an `AdjListTokenizers._AdjListTokenizer` which specifies the subset of lattice edges to be tokenized."""

		@classmethod
		def attribute_key(cls) -> str:
			return EdgeSubsets.key

		def is_valid(self, do_except: bool = False) -> bool:
			return True

		@abc.abstractmethod
		def _get_edges(self, maze: LatticeMaze) -> ConnectionArray:
			"""Returns the set of lattice edges to be tokenized."""
			pass

	@serializable_dataclass(frozen=True, kw_only=True)
	class AllLatticeEdges(_EdgeSubset):
		"""All 2n**2-2n edges of the lattice are tokenized.

		If a wall exists on that edge, the edge is tokenized in the same manner, using `VOCAB.ADJLIST_WALL` in place of `VOCAB.CONNECTOR`.
		"""

		def _get_edges(self, maze: LatticeMaze) -> ConnectionArray:
			return lattice_connection_array(maze.grid_n)

	@serializable_dataclass(frozen=True, kw_only=True)
	class ConnectionEdges(_EdgeSubset):
		"""Only edges which contain a connection are tokenized.

		Alternatively, only edges which contain a wall are tokenized.

		# Parameters
		- `walls`: Whether wall edges or connection edges are tokenized.
		If true, `VOCAB.ADJLIST_WALL` is used in place of `VOCAB.CONNECTOR`.
		"""

		walls: bool = serializable_field(default=False)

		def _get_edges(self, maze: LatticeMaze) -> ConnectionArray:
			conn_list: ConnectionList = maze.connection_list
			if self.walls:
				conn_list = np.logical_not(conn_list)
				conn_list[0, -1, :] = False
				conn_list[1, :, -1] = False
			return connection_list_to_adj_list(
				conn_list,
				shuffle_d0=False,
				shuffle_d1=False,
			)


def _adjlist_no_pre_unsupported(self_, do_except: bool = False) -> bool:  # noqa: ANN001
	"""Returns False if `pre` is True, True otherwise."""
	output: bool = self_.pre is False
	if do_except and not output:
		raise ValueError(
			"AdjListCoord does not support `pre == False`.",
		)

	return output


class AdjListTokenizers(__TokenizerElementNamespace):
	"""Namespace for `_AdjListTokenizer` subclass hierarchy used by `MazeTokenizerModular`."""

	key = "adj_list_tokenizer"

	@serializable_dataclass(frozen=True, kw_only=True)
	@mark_as_unsupported(_adjlist_no_pre_unsupported)
	class _AdjListTokenizer(_TokenizerElement, abc.ABC):
		"""Specifies how the adjacency list is tokenized.

		Tokenization behavior is decomposed into specification of edge subsets, groupings, and permutations.
		See documentation of `EdgeSubset` and `EdgeGrouping` classes for more details.

		# Parameters
		- `pre`: Whether all edge groupings include a preceding delimiter token
		- `post`: Whether all edge groupings include a following delimiter token
		- `shuffle_d0`: Specifies how to sequence the edge groupings.
			If true, groupings are shuffled randomly. If false, groupings are sorted by the leading coord of each group.
		- `edge_grouping`: Specifies if/how multiple coord-coord connections are grouped together in a token subsequence called an edge grouping.
		- `edge_subset`: Specifies the subset of lattice edges to be tokenized.
		- `edge_permuter`: Specifies, in each edge tokenization, which coord either:
			1. Appears first in the tokenization, for `AdjListCoord`.
			2. Is tokenized directly as a coord, for `AdjListCardinal`.
			- `shuffle`: For each edge, the leading coord is selected randomly.
			- `all`: Each edge appears twice in the tokenization, appearing with both leading coords.
			- `evens`, `odds`: The leading coord is the one belonging to that coord subset. See `EdgeSubsets.ChessboardSublattice` for details.
		"""

		pre: bool = serializable_field(default=False, assert_type=False)
		post: bool = serializable_field(default=True)
		shuffle_d0: bool = serializable_field(default=True)
		edge_grouping: EdgeGroupings._EdgeGrouping = serializable_field(
			default=EdgeGroupings.Ungrouped(),
			loading_fn=lambda x: _load_tokenizer_element(x, EdgeGroupings),
		)
		edge_subset: EdgeSubsets._EdgeSubset = serializable_field(
			default=EdgeSubsets.ConnectionEdges(),
			loading_fn=lambda x: _load_tokenizer_element(x, EdgeSubsets),
		)
		edge_permuter: EdgePermuters._EdgePermuter = serializable_field(
			default=EdgePermuters.RandomCoords(),
			loading_fn=lambda x: _load_tokenizer_element(x, EdgePermuters),
		)

		@classmethod
		def attribute_key(cls) -> str:
			return AdjListTokenizers.key

		def is_valid(self, do_except: bool = False) -> bool:
			# No invalid instances possible within data member type hint bounds
			return True

		@abc.abstractmethod
		def _tokenization_callables(
			self,
			edges: ConnectionArray,
			is_conn: Bool[np.ndarray, " edges"],
			coord_tokenizer: CoordTokenizers._CoordTokenizer,
			*args,
			**kwargs,
		) -> list[Callable]:
			"""Returns a sequence of callables which take an index in `edges` and return parts of that edge tokenization.

			# Returns
			- `[0]`: leading coord tokens
			- `[1]`: connector tokens
			- `[2]`: trailing coord tokens
			"""
			pass

		def _tokenize_edge_grouping(
			self,
			edges: ConnectionArray,
			maze: LatticeMaze,
			coord_tokenizer: CoordTokenizers._CoordTokenizer,
			group_params: EdgeGroupings._GroupingTokenParams,
		) -> Sequence[str]:
			"""Tokenizes a single edge grouping."""
			cxn_ord: int = group_params["connection_token_ordinal"]
			is_conn: Bool[np.ndarray, edges] = is_connection(
				edges,
				maze.connection_list,
			)
			tokenize_callables = self._tokenization_callables(
				edges,
				is_conn,
				coord_tokenizer,
			)

			if group_params["grouped"]:
				# If grouped
				callable_permutation: list[int] = [1, 2] if cxn_ord == 0 else [2, 1]
				repeated_callables = [
					tokenize_callables[i] for i in callable_permutation
				]
				return flatten(
					[
						tokenize_callables[0](0),
						[
							[
								*[
									tok_callable(i)
									for tok_callable in repeated_callables
								],
								*(
									(VOCAB.ADJLIST_INTRA,)
									if group_params["intra"]
									else ()
								),
							]
							for i in range(edges.shape[0])
						],
					],
				)
			else:
				# If ungrouped
				callable_permutation = [0, 2]
				callable_permutation.insert(cxn_ord, 1)
				tokenize_callables = [
					tokenize_callables[i] for i in callable_permutation
				]

				return flatten(
					[
						[
							[
								*[
									tok_callable(i)
									for tok_callable in tokenize_callables
								],
								*empty_sequence_if_attr_false(
									(VOCAB.ADJLIST_INTRA,),
									group_params,
									"intra",
								),
							]
							for i in range(edges.shape[0])
						],
					],
				)

		def to_tokens(
			self,
			maze: LatticeMaze,
			coord_tokenizer: CoordTokenizers._CoordTokenizer,
		) -> list[str]:
			# Get the set of edges to be tokenized
			edges: ConnectionArray = self.edge_subset._get_edges(maze)
			# Systematically permute the leading coord of each edge
			edges: ConnectionArray = self.edge_permuter._permute(edges)
			group_params: EdgeGroupings._GroupingTokenParams = (
				self.edge_grouping._token_params()
			)
			# then, we need to group the edges
			groups: Sequence[ConnectionArray] = self.edge_grouping._group_edges(edges)
			# shuffle the groups if specified
			if self.shuffle_d0:
				if isinstance(groups, np.ndarray):
					_NUMPY_RNG.shuffle(groups, axis=0)
				elif isinstance(groups, list):
					random.shuffle(groups)
				else:
					err_msg: str = f"`groups` is an unexpected type {type(groups)}. Only types `list` and `np.ndarray` are currently supported."
					raise TypeError(err_msg)
			# Tokenize each group with optional delimiters
			tokens: list[str] = list(
				flatten(
					[
						[
							*empty_sequence_if_attr_false(
								(VOCAB.ADJLIST_PRE,),
								self,
								"pre",
							),
							*self._tokenize_edge_grouping(
								group,
								maze,
								coord_tokenizer,
								group_params,
							),
							*empty_sequence_if_attr_false(
								(VOCAB.ADJACENCY_ENDLINE,),
								self,
								"post",
							),
						]
						for group in groups
					],
				),
			)
			return tokens

	@serializable_dataclass(frozen=True, kw_only=True)
	class AdjListCoord(_AdjListTokenizer):
		"""Represents an edge group as tokens for the leading coord followed by coord tokens for the other group members."""

		edge_permuter: EdgePermuters._EdgePermuter = serializable_field(
			default=EdgePermuters.RandomCoords(),
			loading_fn=lambda x: _load_tokenizer_element(x, EdgePermuters),
		)

		def _tokenization_callables(
			self,
			edges: ConnectionArray,
			is_conn: Bool[np.ndarray, " edges"],
			coord_tokenizer: CoordTokenizers._CoordTokenizer,
			*args,
			**kwargs,
		) -> list[Callable]:
			# Map from `is_conn` to the tokens which represent connections and walls
			conn_token_map: dict[bool, str] = {
				True: VOCAB.CONNECTOR,
				False: VOCAB.ADJLIST_WALL,
			}
			return [
				lambda i: coord_tokenizer.to_tokens(edges[i, 0]),
				lambda i: conn_token_map[is_conn[i]],
				lambda i: coord_tokenizer.to_tokens(edges[i, 1]),
			]

	@serializable_dataclass(frozen=True, kw_only=True)
	class AdjListCardinal(_AdjListTokenizer):
		"""Represents an edge group as coord tokens for the leading coord and cardinal tokens relative to the leading coord for the other group members.

		# Parameters
		- `coord_first`: Whether the leading coord token(s) should come before or after the sequence of cardinal tokens.
		"""

		edge_permuter: EdgePermuters._EdgePermuter = serializable_field(
			default=EdgePermuters.BothCoords(),
			loading_fn=lambda x: _load_tokenizer_element(x, EdgePermuters),
		)

		def _tokenization_callables(
			self,
			edges: ConnectionArray,
			is_conn: Bool[np.ndarray, " edges"],
			coord_tokenizer: CoordTokenizers._CoordTokenizer,
			*args,
			**kwargs,
		) -> list[Callable]:
			# Map from `is_conn` to the tokens which represent connections and walls
			conn_token_map: dict[bool, str] = {
				True: VOCAB.CONNECTOR,
				False: VOCAB.ADJLIST_WALL,
			}
			return [
				lambda i: coord_tokenizer.to_tokens(edges[i, 0]),
				lambda i: conn_token_map[is_conn[i]],
				lambda i: get_cardinal_direction(edges[i]),
			]


class TargetTokenizers(__TokenizerElementNamespace):
	"""Namespace for `_TargetTokenizer` subclass hierarchy used by `MazeTokenizerModular`."""

	key = "target_tokenizer"

	@serializable_dataclass(frozen=True, kw_only=True)
	class _TargetTokenizer(_TokenizerElement, abc.ABC):
		"""Superclass of tokenizers for maze targets."""

		@abc.abstractmethod
		def to_tokens(
			self,
			targets: Sequence[Coord],
			coord_tokenizer: CoordTokenizers._CoordTokenizer,
		) -> list[str]:
			"""Returns tokens representing the target."""
			pass

		@classmethod
		def attribute_key(cls) -> str:
			return TargetTokenizers.key

	@serializable_dataclass(frozen=True, kw_only=True)
	class Unlabeled(_TargetTokenizer):
		"""Targets are simply listed as coord tokens.

		- `post`: Whether all coords include an integral following delimiter token
		"""

		post: bool = serializable_field(default=False)

		# inherit docstring
		def to_tokens(  # noqa: D102
			self,
			targets: Sequence[Coord],
			coord_tokenizer: CoordTokenizers._CoordTokenizer,
		) -> list[str]:
			return list(
				flatten(
					[
						[
							*coord_tokenizer.to_tokens(target),
							*empty_sequence_if_attr_false(
								[VOCAB.TARGET_POST],
								self,
								"post",
							),
						]
						for target in targets
					],
				),
			)

		# inherit docstring
		def is_valid(self, do_except: bool = False) -> bool:  # noqa: D102
			# No invalid instances possible within data member type hint bounds
			return True


class StepSizes(__TokenizerElementNamespace):
	"""Namespace for `_StepSize` subclass hierarchy used by `MazeTokenizerModular`."""

	key = "step_size"

	@serializable_dataclass(frozen=True, kw_only=True)
	class _StepSize(_TokenizerElement, abc.ABC):
		"""Specifies which coords in `maze.solution` are used to represent the path."""

		@classmethod
		def attribute_key(cls) -> str:
			return StepSizes.key

		@abc.abstractmethod  # TODO: make this a static/class method, allowing ForksAndStraightaways to skip object construction at every call
		def _step_single_indices(self, maze: SolvedMaze) -> list[int]:
			"""Returns the indices of `maze.solution` corresponding to the steps to be tokenized."""
			raise NotImplementedError(
				"Subclasses must implement `StepSize.step_indices.",
			)

		def step_start_end_indices(self, maze: SolvedMaze) -> list[tuple[int, int]]:
			"""Returns steps as tuples of starting and ending positions for each step."""
			indices: list[int] = self._step_single_indices(maze)
			# TODO: RUF007 Prefer `itertools.pairwise()` over `zip()` when iterating over successive pairs
			return [
				(start, end)
				for start, end in zip(indices[:-1], indices[1:], strict=False)  # noqa: RUF007
			]

		def is_valid(self, do_except: bool = False) -> bool:
			# No invalid instances possible within data member type hint bounds
			return True

	@serializable_dataclass(frozen=True, kw_only=True)
	class Singles(_StepSize):
		"""Every coord in `maze.solution` is represented.

		Legacy tokenizers all use this behavior.
		"""

		def _step_single_indices(self, maze: SolvedMaze) -> list[int]:
			"""Returns the indices of `maze.solution` corresponding to the steps to be tokenized."""
			return list(range(maze.solution.shape[0]))

	@serializable_dataclass(frozen=True, kw_only=True)
	@mark_as_unsupported(_unsupported_is_invalid)
	class Straightaways(_StepSize):
		"""Only coords where the path turns are represented in the path.

		I.e., the path is represented as a sequence of straightaways,
		specified by the coords at the turns.
		"""

		def _step_single_indices(self, maze: SolvedMaze) -> list[int]:
			"""Returns the indices of `maze.solution` corresponding to the steps to be tokenized."""
			last_turn_coord: Coord = maze.solution[0, ...]
			indices: list[int] = [0]
			for i, coord in enumerate(maze.solution):
				if coord[0] != last_turn_coord[0] and coord[1] != last_turn_coord[1]:
					indices.append(i - 1)
					last_turn_coord = maze.solution[i - 1, ...]
			indices.append(i)
			return indices

	@serializable_dataclass(frozen=True, kw_only=True)
	class Forks(_StepSize):
		"""Only coords at forks, where the path has >=2 options for the next step are included.

		Excludes the option of backtracking.
		The starting and ending coords are always included.
		"""

		def _step_single_indices(self, maze: SolvedMaze) -> list[int]:
			"""Returns the indices of `maze.solution` corresponding to the steps to be tokenized."""
			return maze.get_solution_forking_points(always_include_endpoints=True)[0]

	@serializable_dataclass(frozen=True, kw_only=True)
	@mark_as_unsupported(_unsupported_is_invalid)
	class ForksAndStraightaways(_StepSize):
		"""Includes the union of the coords included by `Forks` and `Straightaways`.

		See documentation for those classes for details.
		"""

		def _step_single_indices(self, maze: SolvedMaze) -> list[int]:
			"""Returns the indices of `maze.solution` corresponding to the steps to be tokenized."""
			return list(
				np.unique(
					np.concatenate(
						(
							StepSizes.Straightaways()._step_single_indices(maze),
							StepSizes.Forks()._step_single_indices(maze),
						),
					),
				),
			)


class StepTokenizers(__TokenizerElementNamespace):
	"""Namespace for `_StepTokenizer` subclass hierarchy used by `MazeTokenizerModular`."""

	key = "step_tokenizers"

	@serializable_dataclass(frozen=True, kw_only=True)
	class _StepTokenizer(_TokenizerElement, abc.ABC):
		"""Specifies how a single step (as specified by an instance of `_StepSize`) is tokenized."""

		@classmethod
		def attribute_key(cls) -> str:
			return StepTokenizers.key

		@abc.abstractmethod
		def to_tokens(
			self,
			maze: SolvedMaze,
			start_index: int,
			end_index: int,
			**kwargs,
		) -> list[str]:
			"""Tokenizes a single step in the solution.

			# Parameters
			- `maze`: Maze to be tokenized
			- `start_index`: The index of the Coord in `maze.solution` at which the current step starts
			- `end_index`: The index of the Coord in `maze.solution` at which the current step ends
			"""
			raise NotImplementedError(
				"Subclasses must implement `StepTokenizer.to_tokens.",
			)

		def is_valid(self, do_except: bool = False) -> bool:
			# No invalid instances possible within data member type hint bounds
			return True

	@serializable_dataclass(frozen=True, kw_only=True)
	class Coord(_StepTokenizer):
		"""A direct tokenization of the end position coord represents the step."""

		# inherit docstring
		def to_tokens(  # noqa: D102
			self,
			maze: SolvedMaze,
			start_index: int,
			end_index: int,
			coord_tokenizer: CoordTokenizers._CoordTokenizer,
		) -> list[str]:
			return coord_tokenizer.to_tokens(maze.solution[end_index, ...])

	@serializable_dataclass(frozen=True, kw_only=True)
	class Cardinal(_StepTokenizer):
		"""A step is tokenized with a cardinal direction token.

		It is the direction of the step from the starting position along the solution.
		"""

		# inherit docstring
		def to_tokens(  # noqa: D102
			self,
			maze: SolvedMaze,
			start_index: int,
			end_index: int,
			**kwargs,
		) -> list[str]:
			return [
				get_cardinal_direction(maze.solution[start_index : start_index + 2]),
			]

	@serializable_dataclass(frozen=True, kw_only=True)
	class Relative(_StepTokenizer):
		"""Tokenizes a solution step using relative first-person directions (right, left, forward, etc.).

		To simplify the indeterminacy, at the start of a solution the "agent" solving the maze is assumed to be facing NORTH.
		Similarly to `Cardinal`, the direction is that of the step from the starting position.
		"""

		# inherit docstring
		def to_tokens(  # noqa: D102
			self,
			maze: SolvedMaze,
			start_index: int,
			end_index: int,
			**kwargs,
		) -> list[str]:
			if start_index == 0:
				start = maze.solution[0]
				previous = start + np.array([1, 0])
				return [
					get_relative_direction(
						np.concatenate(
							(
								np.expand_dims(previous, 0),
								maze.solution[start_index : start_index + 2],
							),
							axis=0,
						),
					),
				]
			return [
				get_relative_direction(
					maze.solution[start_index - 1 : start_index + 2],
				),
			]

	@serializable_dataclass(frozen=True, kw_only=True)
	class Distance(_StepTokenizer):
		"""A count of the number of individual steps from the starting point to the end point.

		Contains no information about directionality, only the distance traveled in the step.
		`Distance` must be combined with at least one other `_StepTokenizer` in a `StepTokenizerPermutation`.
		This constraint is enforced in `_PathTokenizer.is_valid`.
		"""

		# inherit docstring
		def to_tokens(  # noqa: D102
			self,
			maze: SolvedMaze,
			start_index: int,
			end_index: int,
			**kwargs,
		) -> list[str]:
			d: int = end_index - start_index
			return [getattr(VOCAB, f"I_{d:03}")]

	"""
	`StepTokenizerPermutation`
	A sequence of unique `_StepTokenizer`s.
	This type exists mostly just for the clarity and convenience of `_PathTokenizer` code.
	"""
	StepTokenizerPermutation: type = (
		tuple[_StepTokenizer]
		| tuple[_StepTokenizer, _StepTokenizer]
		| tuple[_StepTokenizer, _StepTokenizer, _StepTokenizer]
		| tuple[_StepTokenizer, _StepTokenizer, _StepTokenizer, _StepTokenizer]
	)


class PathTokenizers(__TokenizerElementNamespace):
	"""Namespace for `_PathTokenizer` subclass hierarchy used by `MazeTokenizerModular`."""

	key = "path_tokenizer"

	@serializable_dataclass(frozen=True, kw_only=True)
	class _PathTokenizer(_TokenizerElement, abc.ABC):
		"""Superclass of tokenizers for maze solution paths."""

		@abc.abstractmethod
		def to_tokens(
			self,
			maze: SolvedMaze,
			coord_tokenizer: CoordTokenizers._CoordTokenizer,
		) -> list[str]:
			"""Returns tokens representing the solution path."""
			pass

		@classmethod
		def attribute_key(cls) -> str:
			return PathTokenizers.key

	@serializable_dataclass(frozen=True, kw_only=True)
	class StepSequence(_PathTokenizer, abc.ABC):
		"""Any `PathTokenizer` where the tokenization may be assembled from token subsequences, each of which represents a step along the path.

		Allows for a sequence of leading and trailing tokens which don't fit the step pattern.

		# Parameters
		- `step_size`: Selects the size of a single step in the sequence
		- `step_tokenizers`: Selects the combination and permutation of tokens
		- `pre`: Whether all steps include an integral preceding delimiter token
		- `intra`: Whether all steps include a delimiter token after each individual `_StepTokenizer` tokenization.
		- `post`: Whether all steps include an integral following delimiter token
		"""

		step_size: StepSizes._StepSize = serializable_field(
			default=StepSizes.Singles(),
			loading_fn=lambda x: _load_tokenizer_element(x, StepSizes),
		)
		step_tokenizers: StepTokenizers.StepTokenizerPermutation = serializable_field(
			default=(StepTokenizers.Coord(),),
			serialization_fn=lambda x: [y.serialize() for y in x],
			loading_fn=lambda x: tuple(x[StepTokenizers.key]),
		)
		pre: bool = serializable_field(default=False)
		intra: bool = serializable_field(default=False)
		post: bool = serializable_field(default=False)

		# inherit docstring
		def to_tokens(  # noqa: D102
			self,
			maze: SolvedMaze,
			coord_tokenizer: CoordTokenizers._CoordTokenizer,
		) -> list[str]:
			return [
				*self._leading_tokens(maze, coord_tokenizer),
				*flatten(
					[
						self._single_step_tokens(maze, start, end, coord_tokenizer)
						for start, end in self.step_size.step_start_end_indices(maze)
					],
				),
				*self._trailing_tokens(maze, coord_tokenizer),
			]

		def _single_step_tokens(
			self,
			maze: SolvedMaze,
			i: int,
			j: int,
			coord_tokenizer: CoordTokenizers._CoordTokenizer,
		) -> list[str]:
			"""Returns the token sequence representing a single step along the path."""
			step_rep_tokens: list[list[str]] = [
				step_tokenizer.to_tokens(maze, i, j, coord_tokenizer=coord_tokenizer)
				for step_tokenizer in self.step_tokenizers
			]
			if self.intra:
				step_rep_tokens_and_intra: list[str] = [None] * (
					len(step_rep_tokens) * 2
				)
				step_rep_tokens_and_intra[::2] = step_rep_tokens
				step_rep_tokens_and_intra[1::2] = [VOCAB.PATH_INTRA] * len(
					step_rep_tokens,
				)
				step_rep_tokens = list(flatten(step_rep_tokens_and_intra))
			all_tokens: list[str] = [
				*empty_sequence_if_attr_false((VOCAB.PATH_PRE,), self, "pre"),
				*flatten(step_rep_tokens),
				*empty_sequence_if_attr_false((VOCAB.PATH_POST,), self, "post"),
			]
			return all_tokens

		def _leading_tokens(
			self,
			maze: SolvedMaze,
			coord_tokenizer: CoordTokenizers._CoordTokenizer,
		) -> list[str]:
			"""Returns tokens preceding those from the sequence from `_single_step_tokens`.

			Since the for loop in `to_tokens` iterates `len(path)-1` times, a fencepost problem exists with `StepTokenizers.Coord`.
			<PATH_START> should NOT be included.
			"""
			if StepTokenizers.Coord() in self.step_tokenizers:
				return [
					*empty_sequence_if_attr_false((VOCAB.PATH_PRE,), self, "pre"),
					*coord_tokenizer.to_tokens(maze.solution[0, ...]),
					*empty_sequence_if_attr_false((VOCAB.PATH_INTRA,), self, "intra"),
				]
			return []

		def _trailing_tokens(
			self,
			c: Coord,
			coord_tokenizer: CoordTokenizers._CoordTokenizer,
		) -> list[str]:
			"""Returns tokens following those from the sequence from `_single_step_tokens`.

			<PATH_END> should NOT be included.
			"""
			return []

		# inherits docstring
		def is_valid(self, do_except: bool = False) -> bool:  # noqa: D102
			output: bool

			if len(set(self.step_tokenizers)) != len(self.step_tokenizers):
				# Uninteresting: repeated elements are not useful
				output = False
			else:
				# we do noqa for the comment if false
				if len(self.step_tokenizers) == 1 and isinstance(
					self.step_tokenizers[0],
					StepTokenizers.Distance,
				):
					# Untrainable: `Distance` alone cannot encode a path. >=1 `StepTokenizer` which indicates direction/location is required.
					output = False
				else:
					output = True

			if not output and do_except:
				raise ValueError(
					"PathTokenizer must contain at least one `StepTokenizer` which indicates direction/location, or it will be untrainable.",
				)

			return output


class PromptSequencers(__TokenizerElementNamespace):
	"""Namespace for `_PromptSequencer` subclass hierarchy used by `MazeTokenizerModular`."""

	key = "prompt_sequencer"

	@serializable_dataclass(frozen=True, kw_only=True)
	class _PromptSequencer(_TokenizerElement, abc.ABC):
		"""Sequences token regions into a complete maze tokenization.

		# Parameters
		- `coord_tokenizer`: Tokenizer element which tokenizes a single `Coord` aka maze position.
		- `adj_list_tokenizer`: Tokenizer element which tokenizes the adjacency list of a `LatticeMaze`.
		Uses `coord_tokenizer` to tokenize coords if needed in other `TokenizerElement`s.
		"""

		coord_tokenizer: CoordTokenizers._CoordTokenizer = serializable_field(
			default=CoordTokenizers.UT(),
			loading_fn=lambda x: _load_tokenizer_element(x, CoordTokenizers),
		)
		adj_list_tokenizer: AdjListTokenizers._AdjListTokenizer = serializable_field(
			default=AdjListTokenizers.AdjListCoord(),
			loading_fn=lambda x: _load_tokenizer_element(x, AdjListTokenizers),
		)

		@classmethod
		def attribute_key(cls) -> str:
			return PromptSequencers.key

		@staticmethod
		def _trim_if_unsolved_maze(
			untrimmed: list[str],
			is_untargeted: bool = False,
			is_unsolved: bool = False,
		) -> list[str]:
			"""Trims a full `SolvedMaze` prompt if the maze data reflects an unsolved or untargeted maze.

			# Development
			This implementation should function for `AOTP`, `AOP`, and other concrete classes using any subsequence of AOTP.
			It is not located in `token_utils.py` because it may need to be overridden in more exotic `PromptSequencer` subclasses.
			"""
			if is_untargeted:
				return tokens_between(
					untrimmed,
					VOCAB.ADJLIST_START,
					VOCAB.ADJLIST_END,
					include_start=True,
					include_end=True,
				)
			if is_unsolved:
				if VOCAB.TARGET_END in untrimmed:
					return tokens_between(
						untrimmed,
						VOCAB.ADJLIST_START,
						VOCAB.TARGET_END,
						include_start=True,
						include_end=True,
					)
				else:
					return tokens_between(
						untrimmed,
						VOCAB.ADJLIST_START,
						VOCAB.ORIGIN_END,
						include_start=True,
						include_end=True,
					)
			return untrimmed

		def to_tokens(
			self,
			maze: LatticeMaze,
			*args,
			**kwargs,
		) -> list[str]:
			"""Returns a complete list of tokens for a given set of maze elements."""
			untrimmed: list[str] = self._sequence_tokens(
				*self._get_prompt_regions(maze),
			)
			return self._trim_if_unsolved_maze(
				untrimmed,
				not hasattr(maze, "start_pos"),
				not hasattr(maze, "solution"),
			)

		def _get_prompt_regions(
			self,
			maze: LatticeMaze,
			*args,
			**kwargs,
		) -> list[list[str]]:
			"""Gets the prompt regions of a maze in a fixed sequence.

			This method is NOT responsible for including/excluding any prompt regions.
			Always return according to the API described under Returns.
			This implementation is expected to be suitable for most `PromptSequencer` subclasses.
			Subclasses may override this method if needed for special behavior.

			# Returns
			- [0]: list[str] Adjacency list tokens
			- [1]: list[str] Origin tokens
			- [2]: list[str] Target tokens
			- [3]: list[str] Path tokens

			# `None`-valued Args
			If one or more of `origin`, `target`, or `path` are `None`, that indicates that an unsolved or untargeted maze is being tokenized.
			To ensure unpackability in `_sequence_tokens`, these `None` values are substituted for empty iterables.
			"""
			origin: Coord | None = getattr(maze, "start_pos", None)
			target: list[Coord] | None = [
				getattr(maze, "end_pos", None),
			]  # TargetTokenizer requires target: Sequence[Coord]

			return [
				(
					self.adj_list_tokenizer.to_tokens(
						maze,
						coord_tokenizer=self.coord_tokenizer,
					)
					if hasattr(self, "adj_list_tokenizer")
					else []
				),
				self.coord_tokenizer.to_tokens(origin) if origin is not None else [],
				(
					self.target_tokenizer.to_tokens(
						target,
						coord_tokenizer=self.coord_tokenizer,
					)
					if target[0] is not None and hasattr(self, "target_tokenizer")
					else []
				),
				(
					self.path_tokenizer.to_tokens(
						maze,
						coord_tokenizer=self.coord_tokenizer,
					)
					if hasattr(maze, "solution") and hasattr(self, "path_tokenizer")
					else []
				),
			]

		@abc.abstractmethod
		def _sequence_tokens(
			self,
			adj_list: list[str],
			origin: list[str] | None,
			target: list[str] | None,
			path: list[str] | None,
		) -> list[str]:
			"""Sequences token regions into a complete prompt.

			Includes any boundary tokens in `constants.SPECIAL_TOKENS` such as <ADJLIST_START>, <ORIGIN_END>, etc.

			# Parameters
			- `adj_list`: Tokens representing the adjacency list
			- `origin`: Tokens representing the origin
			- `target`: Tokens representing the target
			- `path`: Tokens representing the path
			"""
			pass

		def is_valid(self, do_except: bool = False) -> bool:
			# No invalid instances possible within data member type hint bounds
			return True

	@serializable_dataclass(frozen=True, kw_only=True)
	class AOTP(_PromptSequencer):
		"""Sequences a prompt as [adjacency list, origin, target, path].

		# Parameters
		- `target_tokenizer`: Tokenizer element which tokenizes the target(s) of a `TargetedLatticeMaze`.
		Uses `coord_tokenizer` to tokenize coords if that is part of the design of that `TargetTokenizer`.
		- `path_tokenizer`: Tokenizer element which tokenizes the solution path of a `SolvedMaze`.
		Uses `coord_tokenizer` to tokenize coords if that is part of the design of that `PathTokenizer`.

		"""

		target_tokenizer: TargetTokenizers._TargetTokenizer = serializable_field(
			default=TargetTokenizers.Unlabeled(),
			loading_fn=lambda x: _load_tokenizer_element(x, TargetTokenizers),
		)
		path_tokenizer: PathTokenizers._PathTokenizer = serializable_field(
			default=PathTokenizers.StepSequence(),
			loading_fn=lambda x: _load_tokenizer_element(x, PathTokenizers),
		)

		def _sequence_tokens(
			self,
			adj_list: list[str],
			origin: list[str],
			target: list[str],
			path: list[str],
		) -> list[str]:
			return [
				VOCAB.ADJLIST_START,
				*adj_list,
				VOCAB.ADJLIST_END,
				VOCAB.ORIGIN_START,
				*origin,
				VOCAB.ORIGIN_END,
				VOCAB.TARGET_START,
				*target,
				VOCAB.TARGET_END,
				VOCAB.PATH_START,
				*path,
				VOCAB.PATH_END,
			]

	@serializable_dataclass(frozen=True, kw_only=True)
	class AOP(_PromptSequencer):
		"""Sequences a prompt as [adjacency list, origin, path].

		Still includes "<TARGET_START>" and "<TARGET_END>" tokens, but no representation of the target itself.

		# Parameters
		- `path_tokenizer`: Tokenizer element which tokenizes the solution path of a `SolvedMaze`.
		Uses `coord_tokenizer` to tokenize coords if that is part of the design of that `PathTokenizer`.
		"""

		path_tokenizer: PathTokenizers._PathTokenizer = serializable_field(
			default=PathTokenizers.StepSequence(),
			loading_fn=lambda x: _load_tokenizer_element(x, PathTokenizers),
		)

		def _sequence_tokens(
			self,
			adj_list: list[str],
			origin: list[str],
			# explicitly no target in this tokenizer
			target: list[str],
			path: list[str],
		) -> list[str]:
			return [
				VOCAB.ADJLIST_START,
				*adj_list,
				VOCAB.ADJLIST_END,
				VOCAB.ORIGIN_START,
				*origin,
				VOCAB.ORIGIN_END,
				VOCAB.TARGET_START,
				VOCAB.TARGET_END,
				VOCAB.PATH_START,
				*path,
				VOCAB.PATH_END,
			]

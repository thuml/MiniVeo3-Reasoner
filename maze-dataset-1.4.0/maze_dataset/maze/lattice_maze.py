"""Implements `LatticeMaze`, and the `TargetedLatticeMaze` and `SolvedMaze` subclasses.

also includes basic utilities, including converting to/from ascii and pixel representations.
"""

import typing
import warnings
from dataclasses import dataclass
from itertools import chain

import numpy as np
from jaxtyping import Bool, Int, Int8, Shaped
from muutils.json_serialize.serializable_dataclass import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)
from muutils.misc import isinstance_by_type_name, list_split

from maze_dataset.constants import (
	NEIGHBORS_MASK,
	SPECIAL_TOKENS,
	ConnectionList,
	Coord,
	CoordArray,
	CoordList,
	CoordTup,
)
from maze_dataset.token_utils import (
	TokenizerDeprecationWarning,
	connection_list_to_adj_list,
	get_adj_list_tokens,
	get_origin_tokens,
	get_path_tokens,
	get_target_tokens,
)

if typing.TYPE_CHECKING:
	from maze_dataset.tokenization import (
		MazeTokenizer,
		MazeTokenizerModular,
		TokenizationMode,
	)

RGB = tuple[int, int, int]
"rgb tuple of values 0-255"

PixelGrid = Int[np.ndarray, "x y rgb"]
"rgb grid of pixels"
BinaryPixelGrid = Bool[np.ndarray, "x y"]
"boolean grid of pixels"

DIM_2: int = 2
"2 dimensions"


class NoValidEndpointException(Exception):  # noqa: N818
	"""Raised when no valid start or end positions are found in a maze."""

	pass


def _fill_edges_with_walls(connection_list: ConnectionList) -> ConnectionList:
	"""fill the last elements of the connections lists as false for each dim"""
	for dim in range(connection_list.shape[0]):
		# last row for down
		if dim == 0:
			connection_list[dim, -1, :] = False
		# last column for right
		elif dim == 1:
			connection_list[dim, :, -1] = False
		else:
			err_msg: str = f"only 2d lattices supported. got {dim=}"
			raise NotImplementedError(err_msg)
	return connection_list


def color_in_pixel_grid(pixel_grid: PixelGrid, color: RGB) -> bool:
	"""check if a color is in a pixel grid"""
	for row in pixel_grid:
		for pixel in row:
			if np.all(pixel == color):
				return True
	return False


@dataclass(frozen=True)
class PixelColors:
	"standard colors for pixel grids"

	WALL: RGB = (0, 0, 0)
	OPEN: RGB = (255, 255, 255)
	START: RGB = (0, 255, 0)
	END: RGB = (255, 0, 0)
	PATH: RGB = (0, 0, 255)


@dataclass(frozen=True)
class AsciiChars:
	"standard ascii characters for mazes"

	WALL: str = "#"
	OPEN: str = " "
	START: str = "S"
	END: str = "E"
	PATH: str = "X"


ASCII_PIXEL_PAIRINGS: dict[str, RGB] = {
	AsciiChars.WALL: PixelColors.WALL,
	AsciiChars.OPEN: PixelColors.OPEN,
	AsciiChars.START: PixelColors.START,
	AsciiChars.END: PixelColors.END,
	AsciiChars.PATH: PixelColors.PATH,
}
"map ascii characters to pixel colors"


@serializable_dataclass(
	frozen=True,
	kw_only=True,
	properties_to_serialize=["lattice_dim", "generation_meta"],
)
class LatticeMaze(SerializableDataclass):
	"""lattice maze (nodes on a lattice, connections only to neighboring nodes)

	Connection List represents which nodes (N) are connected in each direction.

	First and second elements represent rightward and downward connections,
	respectively.

	Example:
		Connection list:
			[
				[ # down
					[F T],
					[F F]
				],
				[ # right
					[T F],
					[T F]
				]
			]

		Nodes with connections
			N T N F
			F	T
			N T N F
			F	F

		Graph:
			N - N
				|
			N - N

	Note: the bottom row connections going down, and the
	right-hand connections going right, will always be False.

	"""

	connection_list: ConnectionList
	generation_meta: dict | None = serializable_field(default=None, compare=False)

	lattice_dim = property(lambda self: self.connection_list.shape[0])
	grid_shape = property(lambda self: self.connection_list.shape[1:])
	n_connections = property(lambda self: self.connection_list.sum())

	@property
	def grid_n(self) -> int:
		"grid size as int, raises `AssertionError` if not square"
		assert self.grid_shape[0] == self.grid_shape[1], "only square mazes supported"
		return self.grid_shape[0]

	# ============================================================
	# basic methods
	# ============================================================

	def __eq__(self, other: object) -> bool:
		"equality check calls super"
		return super().__eq__(other)

	@staticmethod
	def heuristic(a: CoordTup, b: CoordTup) -> float:
		"""return manhattan distance between two points"""
		return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

	def __hash__(self) -> int:
		"""hash the connection list by converting connection list to bytes"""
		return hash(self.connection_list.tobytes())

	def nodes_connected(self, a: Coord, b: Coord, /) -> bool:
		"""returns whether two nodes are connected"""
		delta: Coord = b - a
		if np.abs(delta).sum() != 1:
			# return false if not even adjacent
			return False
		else:
			# test for wall
			dim: int = int(np.argmax(np.abs(delta)))
			clist_node: Coord = a if (delta.sum() > 0) else b
			return self.connection_list[dim, clist_node[0], clist_node[1]]

	def is_valid_path(self, path: CoordArray, empty_is_valid: bool = False) -> bool:
		"""check if a path is valid"""
		# check path is not empty
		if len(path) == 0:
			return empty_is_valid

		# check all coords in bounds of maze
		if not np.all((path >= 0) & (path < self.grid_shape)):
			return False

		# check all nodes connected
		for i in range(len(path) - 1):
			if not self.nodes_connected(path[i], path[i + 1]):
				return False
		return True

	def coord_degrees(self) -> Int8[np.ndarray, "row col"]:
		"""Returns an array with the connectivity degree of each coord.

		I.e., how many neighbors each coord has.
		"""
		int_conn: Int8[np.ndarray, "lattice_dim=2 row col"] = (
			self.connection_list.astype(np.int8)
		)
		degrees: Int8[np.ndarray, "row col"] = np.sum(
			int_conn,
			axis=0,
		)  # Connections to east and south
		degrees[:, 1:] += int_conn[1, :, :-1]  # Connections to west
		degrees[1:, :] += int_conn[0, :-1, :]  # Connections to north
		return degrees

	def get_coord_neighbors(self, c: Coord | CoordTup) -> CoordArray:
		"""Returns an array of the neighboring, connected coords of `c`."""
		c = np.array(c)  # type: ignore[assignment]
		neighbors: list[Coord] = [
			neighbor
			for neighbor in (c + NEIGHBORS_MASK)
			if (
				(0 <= neighbor[0] < self.grid_shape[0])  # in x bounds
				and (0 <= neighbor[1] < self.grid_shape[1])  # in y bounds
				and self.nodes_connected(c, neighbor)  # connected
			)
		]

		output: CoordArray = np.array(neighbors)
		if len(neighbors) > 0:
			assert output.shape == (
				len(neighbors),
				2,
			), (
				f"invalid shape: {output.shape}, expected ({len(neighbors)}, 2))\n{c = }\n{neighbors = }\n{self.as_ascii()}"
			)
		return output

	def gen_connected_component_from(self, c: Coord) -> CoordArray:
		"""return the connected component from a given coordinate"""
		# Stack for DFS
		stack: list[Coord] = [c]

		# Set to store visited nodes
		visited: set[CoordTup] = set()

		while stack:
			current_node: Coord = stack.pop()
			# this is fine since we know current_node is a coord and thus of length 2
			visited.add(tuple(current_node))  # type: ignore[arg-type]

			# Get the neighbors of the current node
			neighbors = self.get_coord_neighbors(current_node)

			# Iterate over neighbors
			for neighbor in neighbors:
				if tuple(neighbor) not in visited:
					stack.append(neighbor)

		return np.array(list(visited))

	def find_shortest_path(
		self,
		c_start: CoordTup | Coord,
		c_end: CoordTup | Coord,
	) -> CoordArray:
		"""find the shortest path between two coordinates, using A*"""
		c_start = tuple(c_start)  # type: ignore[assignment]
		c_end = tuple(c_end)  # type: ignore[assignment]

		g_score: dict[CoordTup, float] = (
			dict()
		)  # cost of cheapest path to node from start currently known
		f_score: dict[CoordTup, float] = {
			c_start: 0.0,
		}  # estimated total cost of path thru a node: f_score[c] := g_score[c] + heuristic(c, c_end)

		# init
		g_score[c_start] = 0.0
		g_score[c_start] = self.heuristic(c_start, c_end)

		closed_vtx: set[CoordTup] = set()  # nodes already evaluated
		# nodes to be evaluated
		# we need a set of the tuples, dont place the ints in the set
		open_vtx: set[CoordTup] = set([c_start])  # noqa: C405
		source: dict[CoordTup, CoordTup] = (
			dict()
		)  # node immediately preceding each node in the path (currently known shortest path)

		while open_vtx:
			# get lowest f_score node
			# mypy cant tell that c is of length 2
			c_current: CoordTup = min(open_vtx, key=lambda c: f_score[tuple(c)])  # type: ignore[index]
			# f_current: float = f_score[c_current]

			# check if goal is reached
			if c_end == c_current:
				path: list[CoordTup] = [c_current]
				p_current: CoordTup = c_current
				while p_current in source:
					p_current = source[p_current]
					path.append(p_current)
				# ----------------------------------------------------------------------
				# this is the only return statement
				return np.array(path[::-1])
				# ----------------------------------------------------------------------

			# close current node
			closed_vtx.add(c_current)
			open_vtx.remove(c_current)

			# update g_score of neighbors
			_np_neighbor: Coord
			for _np_neighbor in self.get_coord_neighbors(c_current):
				neighbor: CoordTup = tuple(_np_neighbor)

				if neighbor in closed_vtx:
					# already checked
					continue
				g_temp: float = g_score[c_current] + 1  # always 1 for maze neighbors

				if neighbor not in open_vtx:
					# found new vtx, so add
					open_vtx.add(neighbor)

				elif g_temp >= g_score[neighbor]:
					# if already knew about this one, but current g_score is worse, skip
					continue

				# store g_score and source
				source[neighbor] = c_current
				g_score[neighbor] = g_temp
				f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, c_end)

		raise ValueError(
			"A solution could not be found!",
			f"{c_start = }, {c_end = }",
			self.as_ascii(),
		)

	def get_nodes(self) -> CoordArray:
		"""return a list of all nodes in the maze"""
		rows: Int[np.ndarray, "x y"]
		cols: Int[np.ndarray, "x y"]
		rows, cols = np.meshgrid(
			range(self.grid_shape[0]),
			range(self.grid_shape[1]),
			indexing="ij",
		)
		nodes: CoordArray = np.vstack((rows.ravel(), cols.ravel())).T
		return nodes

	def get_connected_component(self) -> CoordArray:
		"""get the largest (and assumed only nonsingular) connected component of the maze

		TODO: other connected components?
		"""
		if (self.generation_meta is None) or (
			self.generation_meta.get("fully_connected", False)
		):
			# for fully connected case, pick any two positions
			return self.get_nodes()
		else:
			# if metadata provided, use visited cells
			visited_cells: set[CoordTup] | None = self.generation_meta.get(
				"visited_cells",
				None,
			)
			if visited_cells is None:
				# TODO: dynamically generate visited_cells?
				err_msg: str = f"a maze which is not marked as fully connected must have a visited_cells field in its generation_meta: {self.generation_meta}\n{self}\n{self.as_ascii()}"
				raise ValueError(
					err_msg,
				)
			visited_cells_np: Int[np.ndarray, "N 2"] = np.array(list(visited_cells))
			return visited_cells_np

	@typing.overload
	def generate_random_path(
		self,
		allowed_start: CoordList | None = None,
		allowed_end: CoordList | None = None,
		deadend_start: bool = False,
		deadend_end: bool = False,
		endpoints_not_equal: bool = False,
		except_on_no_valid_endpoint: typing.Literal[True] = True,
	) -> CoordArray: ...
	@typing.overload
	def generate_random_path(
		self,
		allowed_start: CoordList | None = None,
		allowed_end: CoordList | None = None,
		deadend_start: bool = False,
		deadend_end: bool = False,
		endpoints_not_equal: bool = False,
		except_on_no_valid_endpoint: typing.Literal[False] = False,
	) -> CoordArray | None: ...
	def generate_random_path(  # noqa: C901
		self,
		allowed_start: CoordList | None = None,
		allowed_end: CoordList | None = None,
		deadend_start: bool = False,
		deadend_end: bool = False,
		endpoints_not_equal: bool = False,
		except_on_no_valid_endpoint: bool = True,
	) -> CoordArray | None:
		"""return a path between randomly chosen start and end nodes within the connected component

		Note that setting special conditions on start and end positions might cause the same position to be selected as both start and end.

		# Parameters:
		- `allowed_start : CoordList | None`
			a list of allowed start positions. If `None`, any position in the connected component is allowed
			(defaults to `None`)
		- `allowed_end : CoordList | None`
			a list of allowed end positions. If `None`, any position in the connected component is allowed
			(defaults to `None`)
		- `deadend_start : bool`
			whether to ***force*** the start position to be a deadend (defaults to `False`)
			(defaults to `False`)
		- `deadend_end : bool`
			whether to ***force*** the end position to be a deadend (defaults to `False`)
			(defaults to `False`)
		- `endpoints_not_equal : bool`
			whether to ensure tha the start and end point are not the same
			(defaults to `False`)
		- `except_on_no_valid_endpoint : bool`
			whether to raise an error if no valid start or end positions are found
			if this is `False`, the function might return `None` and this must be handled by the caller
			(defaults to `True`)

		# Returns:
		- `CoordArray`
			a path between the selected start and end positions

		# Raises:
		- `NoValidEndpointException` : if no valid start or end positions are found, and `except_on_no_valid_endpoint` is `True`
		"""
		# we can't create a "path" in a single-node maze
		assert self.grid_shape[0] > 1 and self.grid_shape[1] > 1, (  # noqa: PT018
			f"can't create path in single-node maze: {self.as_ascii()}"
		)

		# get connected component
		connected_component: CoordArray = self.get_connected_component()

		# initialize start and end positions
		positions: Int[np.int8, "2 2"]

		# if no special conditions on start and end positions
		if (allowed_start, allowed_end, deadend_start, deadend_end) == (
			None,
			None,
			False,
			False,
		):
			try:
				positions = connected_component[  # type: ignore[assignment]
					np.random.choice(
						len(connected_component),
						size=2,
						replace=False,
					)
				]
			except ValueError as e:
				if except_on_no_valid_endpoint:
					err_msg: str = f"No valid start or end positions found because we could not sample from {connected_component = }"
					raise NoValidEndpointException(
						err_msg,
					) from e
				return None

			return self.find_shortest_path(positions[0], positions[1])  # type: ignore[index]

		# handle special conditions
		connected_component_set: set[CoordTup] = set(map(tuple, connected_component))
		# copy connected component set
		allowed_start_set: set[CoordTup] = connected_component_set.copy()
		allowed_end_set: set[CoordTup] = connected_component_set.copy()

		# filter by explicitly allowed start and end positions
		# '# type: ignore[assignment]' here because the returned tuple can be of any length
		if allowed_start is not None:
			allowed_start_set = set(map(tuple, allowed_start)) & connected_component_set  # type: ignore[assignment]

		if allowed_end is not None:
			allowed_end_set = set(map(tuple, allowed_end)) & connected_component_set  # type: ignore[assignment]

		# filter by forcing deadends
		if deadend_start:
			allowed_start_set = set(
				filter(
					lambda x: len(self.get_coord_neighbors(x)) == 1,
					allowed_start_set,
				),
			)

		if deadend_end:
			allowed_end_set = set(
				filter(
					lambda x: len(self.get_coord_neighbors(x)) == 1,
					allowed_end_set,
				),
			)

		# check we have valid positions
		if len(allowed_start_set) == 0 or len(allowed_end_set) == 0:
			if except_on_no_valid_endpoint:
				err_msg = f"No valid start (or end?) positions found: {allowed_start_set = }, {allowed_end_set = }"
				raise NoValidEndpointException(
					err_msg,
				)
			return None

		# randomly select start and end positions
		try:
			# ignore assignment here since `tuple()` returns a tuple of any length, but we know it will be ok
			start_pos: CoordTup = tuple(  # type: ignore[assignment]
				list(allowed_start_set)[np.random.randint(0, len(allowed_start_set))],
			)
			if endpoints_not_equal:
				# remove start position from end positions
				allowed_end_set.discard(start_pos)
			end_pos: CoordTup = tuple(  # type: ignore[assignment]
				list(allowed_end_set)[np.random.randint(0, len(allowed_end_set))],
			)
		except ValueError as e:
			if except_on_no_valid_endpoint:
				err_msg = f"No valid start or end positions found, maybe can't find an endpoint after we removed the start point: {allowed_start_set = }, {allowed_end_set = }"
				raise NoValidEndpointException(
					err_msg,
				) from e
			return None

		return self.find_shortest_path(start_pos, end_pos)

	# ============================================================
	# to and from adjacency list
	# ============================================================
	def as_adj_list(
		self,
		shuffle_d0: bool = True,
		shuffle_d1: bool = True,
	) -> Int8[np.ndarray, "conn start_end coord"]:
		"""return the maze as an adjacency list, wraps `maze_dataset.token_utils.connection_list_to_adj_list`"""
		return connection_list_to_adj_list(self.connection_list, shuffle_d0, shuffle_d1)

	@classmethod
	def from_adj_list(
		cls,
		adj_list: Int8[np.ndarray, "conn start_end coord"],
	) -> "LatticeMaze":
		"""create a LatticeMaze from a list of connections

		> [!NOTE]
		> This has only been tested for square mazes. Might need to change some things if rectangular mazes are needed.
		"""
		# this is where it would probably break for rectangular mazes
		grid_n: int = adj_list.max() + 1

		connection_list: ConnectionList = np.zeros(
			(2, grid_n, grid_n),
			dtype=np.bool_,
		)

		for c_start, c_end in adj_list:
			# check that exactly 1 coordinate matches
			if (c_start == c_end).sum() != 1:
				raise ValueError("invalid connection")

			# get the direction
			d: int = (c_start != c_end).argmax()

			x: int
			y: int
			# pick whichever has the lesser value in the direction `d`
			if c_start[d] < c_end[d]:
				x, y = c_start
			else:
				x, y = c_end

			connection_list[d, x, y] = True

		return LatticeMaze(
			connection_list=connection_list,
		)

	def as_adj_list_tokens(self) -> list[str | CoordTup]:
		"""(deprecated!) turn the maze into adjacency list tokens, use `MazeTokenizerModular` instead"""
		warnings.warn(
			"`LatticeMaze.as_adj_list_tokens` will be removed from the public API in a future release.",
			TokenizerDeprecationWarning,
		)
		return [
			SPECIAL_TOKENS.ADJLIST_START,
			*chain.from_iterable(  # type: ignore[list-item]
				[
					[
						tuple(c_s),
						SPECIAL_TOKENS.CONNECTOR,
						tuple(c_e),
						SPECIAL_TOKENS.ADJACENCY_ENDLINE,
					]
					for c_s, c_e in self.as_adj_list()
				],
			),
			SPECIAL_TOKENS.ADJLIST_END,
		]

	def _as_adj_list_tokens(self) -> list[str | CoordTup]:
		return [
			SPECIAL_TOKENS.ADJLIST_START,
			*chain.from_iterable(  # type: ignore[list-item]
				[
					[
						tuple(c_s),
						SPECIAL_TOKENS.CONNECTOR,
						tuple(c_e),
						SPECIAL_TOKENS.ADJACENCY_ENDLINE,
					]
					for c_s, c_e in self.as_adj_list()
				],
			),
			SPECIAL_TOKENS.ADJLIST_END,
		]

	def _as_coords_and_special_AOTP(self) -> list[CoordTup | str]:
		"""turn the maze into adjacency list, origin, target, and solution -- keep coords as tuples"""
		output: list[CoordTup | str] = self._as_adj_list_tokens()
		# if getattr(self, "start_pos", None) is not None:
		if isinstance(self, TargetedLatticeMaze):
			output += self._get_start_pos_tokens()
		if isinstance(self, TargetedLatticeMaze):
			output += self._get_end_pos_tokens()
		if isinstance(self, SolvedMaze):
			output += self._get_solution_tokens()
		return output

	def _as_tokens(
		self,
		maze_tokenizer: "MazeTokenizer | TokenizationMode",
	) -> list[str]:
		# type ignores here fine since we check the instance
		if isinstance_by_type_name(maze_tokenizer, "TokenizationMode"):
			maze_tokenizer = maze_tokenizer.to_legacy_tokenizer()  # type: ignore[union-attr]
		if (
			isinstance_by_type_name(maze_tokenizer, "MazeTokenizer")
			and maze_tokenizer.is_AOTP()  # type: ignore[union-attr]
		):
			coords_raw: list[CoordTup | str] = self._as_coords_and_special_AOTP()
			coords_processed: list[str] = maze_tokenizer.coords_to_strings(  # type: ignore[union-attr]
				coords=coords_raw,
				when_noncoord="include",
			)
			return coords_processed
		else:
			err_msg: str = f"Unsupported tokenizer type: {maze_tokenizer}"
			raise NotImplementedError(err_msg)

	def as_tokens(
		self,
		maze_tokenizer: "MazeTokenizer | TokenizationMode | MazeTokenizerModular",
	) -> list[str]:
		"""serialize maze and solution to tokens"""
		if isinstance_by_type_name(maze_tokenizer, "MazeTokenizerModular"):
			return maze_tokenizer.to_tokens(self)  # type: ignore[union-attr]
		else:
			return self._as_tokens(maze_tokenizer)  # type: ignore[union-attr,arg-type]

	@classmethod
	def _from_tokens_AOTP(
		cls,
		tokens: list[str],
		maze_tokenizer: "MazeTokenizer | MazeTokenizerModular",
	) -> "LatticeMaze | TargetedLatticeMaze | SolvedMaze":
		"""create a LatticeMaze from a list of tokens"""
		# figure out what input format
		# ========================================
		if tokens[0] == SPECIAL_TOKENS.ADJLIST_START:
			adj_list_tokens = get_adj_list_tokens(tokens)
		else:
			# If we're not getting a "complete" tokenized maze, assume it's just a the adjacency list tokens
			adj_list_tokens = tokens
			warnings.warn(
				"Assuming input is just adjacency list tokens, no special tokens found",
			)

		# process edges for adjacency list
		# ========================================
		edges: list[list[str]] = list_split(
			adj_list_tokens,
			SPECIAL_TOKENS.ADJACENCY_ENDLINE,
		)

		coordinates: list[tuple[CoordTup, CoordTup]] = list()
		for e in edges:
			# skip last endline
			if len(e) != 0:
				# convert to coords, split start and end
				e_coords: list[str | CoordTup] = maze_tokenizer.strings_to_coords(
					e,
					when_noncoord="include",
				)
				# this assertion depends on the tokenizer having exactly one token for the connector
				# which is also why we "include" above
				# the connector token is discarded below
				assert len(e_coords) == 3, f"invalid edge: {e = } {e_coords = }"  # noqa: PLR2004
				assert e_coords[1] == SPECIAL_TOKENS.CONNECTOR, (
					f"invalid edge: {e = } {e_coords = }"
				)
				e_coords_first: CoordTup = e_coords[0]  # type: ignore[assignment]
				e_coords_last: CoordTup = e_coords[-1]  # type: ignore[assignment]
				coordinates.append((e_coords_first, e_coords_last))

		assert all(len(c) == DIM_2 for c in coordinates), (
			f"invalid coordinates: {coordinates = }"
		)
		adj_list: Int8[np.ndarray, "conn start_end coord"] = np.array(coordinates)
		assert tuple(adj_list.shape) == (
			len(coordinates),
			2,
			2,
		), f"invalid adj_list: {adj_list.shape = } {coordinates = }"

		output_maze: LatticeMaze = cls.from_adj_list(adj_list)

		# add start and end positions
		# ========================================
		is_targeted: bool = False
		if all(
			x in tokens
			for x in (
				SPECIAL_TOKENS.ORIGIN_START,
				SPECIAL_TOKENS.ORIGIN_END,
				SPECIAL_TOKENS.TARGET_START,
				SPECIAL_TOKENS.TARGET_END,
			)
		):
			start_pos_list: list[CoordTup] = maze_tokenizer.strings_to_coords(
				get_origin_tokens(tokens),
				when_noncoord="error",
			)
			end_pos_list: list[CoordTup] = maze_tokenizer.strings_to_coords(
				get_target_tokens(tokens),
				when_noncoord="error",
			)
			assert len(start_pos_list) == 1, (
				f"invalid start_pos_list: {start_pos_list = }"
			)
			assert len(end_pos_list) == 1, f"invalid end_pos_list: {end_pos_list = }"

			start_pos: CoordTup = start_pos_list[0]
			end_pos: CoordTup = end_pos_list[0]

			output_maze = TargetedLatticeMaze.from_lattice_maze(
				lattice_maze=output_maze,
				start_pos=start_pos,
				end_pos=end_pos,
			)

			is_targeted = True

		if all(
			x in tokens for x in (SPECIAL_TOKENS.PATH_START, SPECIAL_TOKENS.PATH_END)
		):
			assert is_targeted, "maze must be targeted to have a solution"
			solution: list[CoordTup] = maze_tokenizer.strings_to_coords(
				get_path_tokens(tokens, trim_end=True),
				when_noncoord="error",
			)
			output_maze = SolvedMaze.from_targeted_lattice_maze(
				# HACK: I think this is fine, but im not sure
				targeted_lattice_maze=output_maze,  # type: ignore[arg-type]
				solution=solution,
			)

		return output_maze

	# TODO: any way to get return type hinting working for this?
	@classmethod
	def from_tokens(
		cls,
		tokens: list[str],
		maze_tokenizer: "MazeTokenizer | TokenizationMode | MazeTokenizerModular",
	) -> "LatticeMaze | TargetedLatticeMaze | SolvedMaze":
		"""Constructs a maze from a tokenization.

		Only legacy tokenizers and their `MazeTokenizerModular` analogs are supported.
		"""
		# HACK: type ignores here fine since we check the instance
		if isinstance_by_type_name(maze_tokenizer, "TokenizationMode"):
			maze_tokenizer = maze_tokenizer.to_legacy_tokenizer()  # type: ignore[union-attr]
		if (
			isinstance_by_type_name(maze_tokenizer, "MazeTokenizerModular")
			and not maze_tokenizer.is_legacy_equivalent()  # type: ignore[union-attr]
		):
			err_msg: str = f"Only legacy tokenizers and their exact `MazeTokenizerModular` analogs supported, not {maze_tokenizer}."
			raise NotImplementedError(
				err_msg,
			)

		if isinstance(tokens, str):
			tokens = tokens.split()

		if maze_tokenizer.is_AOTP():  # type: ignore[union-attr]
			return cls._from_tokens_AOTP(tokens, maze_tokenizer)  # type: ignore[arg-type]
		else:
			raise NotImplementedError("only AOTP tokenization is supported")

	# ============================================================
	# to and from pixels
	# ============================================================
	def _as_pixels_bw(self) -> BinaryPixelGrid:
		assert self.lattice_dim == DIM_2, "only 2D mazes are supported"
		# Create an empty pixel grid with walls
		pixel_grid: Int[np.ndarray, "x y"] = np.full(
			(self.grid_shape[0] * 2 + 1, self.grid_shape[1] * 2 + 1),
			False,
			dtype=np.bool_,
		)

		# Set white nodes
		pixel_grid[1::2, 1::2] = True

		# Set white connections (downward)
		for i, row in enumerate(self.connection_list[0]):
			for j, connected in enumerate(row):
				if connected:
					pixel_grid[i * 2 + 2, j * 2 + 1] = True

		# Set white connections (rightward)
		for i, row in enumerate(self.connection_list[1]):
			for j, connected in enumerate(row):
				if connected:
					pixel_grid[i * 2 + 1, j * 2 + 2] = True

		return pixel_grid

	def as_pixels(
		self,
		show_endpoints: bool = True,
		show_solution: bool = True,
	) -> PixelGrid:
		"""convert the maze to a pixel grid

		- useful as a simpler way of plotting the maze than the more complex `MazePlot`
		- the same underlying representation as `as_ascii` but as an image
		- used in `RasterizedMazeDataset`, which mimics the mazes in https://github.com/aks2203/easy-to-hard-data
		"""
		# HACK: lots of `# type: ignore[attr-defined]` here since its defined for any `LatticeMaze`
		# but solution, start_pos, end_pos not always defined
		# but its fine since we explicitly check the type
		if show_solution and not show_endpoints:
			raise ValueError("show_solution=True requires show_endpoints=True")
		# convert original bool pixel grid to RGB
		pixel_grid_bw: BinaryPixelGrid = self._as_pixels_bw()
		pixel_grid: PixelGrid = np.full(
			(*pixel_grid_bw.shape, 3),
			PixelColors.WALL,
			dtype=np.uint8,
		)
		pixel_grid[pixel_grid_bw == True] = PixelColors.OPEN  # noqa: E712

		if self.__class__ == LatticeMaze:
			return pixel_grid

		# set endpoints for TargetedLatticeMaze
		if self.__class__ == TargetedLatticeMaze:
			if show_endpoints:
				pixel_grid[self.start_pos[0] * 2 + 1, self.start_pos[1] * 2 + 1] = (  # type: ignore[attr-defined]
					PixelColors.START
				)
				pixel_grid[self.end_pos[0] * 2 + 1, self.end_pos[1] * 2 + 1] = (  # type: ignore[attr-defined]
					PixelColors.END
				)
			return pixel_grid

		# set solution -- we only reach this part if `self.__class__ == SolvedMaze`
		if show_solution:
			for coord in self.solution:  # type: ignore[attr-defined]
				pixel_grid[coord[0] * 2 + 1, coord[1] * 2 + 1] = PixelColors.PATH

			# set pixels between coords
			for index, coord in enumerate(self.solution[:-1]):  # type: ignore[attr-defined]
				next_coord = self.solution[index + 1]  # type: ignore[attr-defined]
				# check they are adjacent using norm
				assert np.linalg.norm(np.array(coord) - np.array(next_coord)) == 1, (
					f"Coords {coord} and {next_coord} are not adjacent"
				)
				# set pixel between them
				pixel_grid[
					coord[0] * 2 + 1 + next_coord[0] - coord[0],
					coord[1] * 2 + 1 + next_coord[1] - coord[1],
				] = PixelColors.PATH

			# set endpoints (again, since path would overwrite them)
			pixel_grid[self.start_pos[0] * 2 + 1, self.start_pos[1] * 2 + 1] = (  # type: ignore[attr-defined]
				PixelColors.START
			)
			pixel_grid[self.end_pos[0] * 2 + 1, self.end_pos[1] * 2 + 1] = (  # type: ignore[attr-defined]
				PixelColors.END
			)

		return pixel_grid

	@classmethod
	def _from_pixel_grid_bw(
		cls,
		pixel_grid: BinaryPixelGrid,
	) -> tuple[ConnectionList, tuple[int, int]]:
		grid_shape: tuple[int, int] = (
			pixel_grid.shape[0] // 2,
			pixel_grid.shape[1] // 2,
		)
		connection_list: ConnectionList = np.zeros((2, *grid_shape), dtype=np.bool_)

		# Extract downward connections
		connection_list[0] = pixel_grid[2::2, 1::2]

		# Extract rightward connections
		connection_list[1] = pixel_grid[1::2, 2::2]

		return connection_list, grid_shape

	@classmethod
	def _from_pixel_grid_with_positions(
		cls,
		pixel_grid: PixelGrid | BinaryPixelGrid,
		marked_positions: dict[str, RGB],
	) -> tuple[ConnectionList, tuple[int, int], dict[str, CoordArray]]:
		# Convert RGB pixel grid to Bool pixel grid
		# error: Incompatible types in assignment (expression has type
		# "numpy.bool[builtins.bool] | ndarray[tuple[int, ...], dtype[numpy.bool[builtins.bool]]]",
		# variable has type "ndarray[Any, Any]")  [assignment]
		pixel_grid_bw: BinaryPixelGrid = ~np.all(  # type: ignore[assignment]
			pixel_grid == PixelColors.WALL,
			axis=-1,
		)
		connection_list: ConnectionList
		grid_shape: tuple[int, int]
		connection_list, grid_shape = cls._from_pixel_grid_bw(pixel_grid_bw)

		# Find any marked positions
		out_positions: dict[str, CoordArray] = dict()
		for key, color in marked_positions.items():
			pos_temp: Int[np.ndarray, "x y"] = np.argwhere(
				np.all(pixel_grid == color, axis=-1),
			)
			pos_save: list[CoordTup] = list()
			for pos in pos_temp:
				# if it is a coordinate and not connection (transform position, %2==1)
				if pos[0] % 2 == 1 and pos[1] % 2 == 1:
					pos_save.append((pos[0] // 2, pos[1] // 2))

			out_positions[key] = np.array(pos_save)

		return connection_list, grid_shape, out_positions

	@classmethod
	def from_pixels(
		cls,
		pixel_grid: PixelGrid,
	) -> "LatticeMaze":
		"""create a LatticeMaze from a pixel grid. reverse of `as_pixels`

		# Raises:
		- `ValueError` : if the pixel grid cannot be cast to a `LatticeMaze` -- it's probably a `TargetedLatticeMaze` or `SolvedMaze`
		"""
		connection_list: ConnectionList
		grid_shape: tuple[int, int]

		# if a binary pixel grid, return regular LatticeMaze
		if len(pixel_grid.shape) == 2:  # noqa: PLR2004
			connection_list, grid_shape = cls._from_pixel_grid_bw(pixel_grid)
			return LatticeMaze(connection_list=connection_list)

		# otherwise, detect and check it's valid
		cls_detected: typing.Type[LatticeMaze] = detect_pixels_type(pixel_grid)
		if cls not in cls_detected.__mro__:
			err_msg: str = f"Pixel grid cannot be cast to {cls.__name__ = }, detected type {cls_detected.__name__ = }"
			raise ValueError(
				err_msg,
			)

		(
			connection_list,
			grid_shape,
			marked_pos,
		) = cls._from_pixel_grid_with_positions(
			pixel_grid=pixel_grid,
			marked_positions=dict(
				start=PixelColors.START,
				end=PixelColors.END,
				solution=PixelColors.PATH,
			),
		)
		# if we wanted a LatticeMaze, return it
		if cls == LatticeMaze:
			return LatticeMaze(connection_list=connection_list)

		# otherwise, keep going
		temp_maze: LatticeMaze = LatticeMaze(connection_list=connection_list)

		# start and end pos
		start_pos_arr, end_pos_arr = marked_pos["start"], marked_pos["end"]
		assert start_pos_arr.shape == (
			1,
			2,
		), (
			f"start_pos_arr {start_pos_arr} has shape {start_pos_arr.shape}, expected shape (1, 2) -- a single coordinate"
		)
		assert end_pos_arr.shape == (
			1,
			2,
		), (
			f"end_pos_arr {end_pos_arr} has shape {end_pos_arr.shape}, expected shape (1, 2) -- a single coordinate"
		)

		start_pos: Coord = start_pos_arr[0]
		end_pos: Coord = end_pos_arr[0]

		# return a TargetedLatticeMaze if that's what we wanted
		if cls == TargetedLatticeMaze:
			return TargetedLatticeMaze(
				connection_list=connection_list,
				start_pos=start_pos,
				end_pos=end_pos,
			)

		# raw solution, only contains path elements and not start or end
		solution_raw: CoordArray = marked_pos["solution"]
		if len(solution_raw.shape) == 2:  # noqa: PLR2004
			assert solution_raw.shape[1] == 2, (  # noqa: PLR2004
				f"solution {solution_raw} has shape {solution_raw.shape}, expected shape (n, 2)"
			)
		elif solution_raw.shape == (0,):
			# the solution and end should be immediately adjacent
			assert np.sum(np.abs(start_pos - end_pos)) == 1, (
				f"start_pos {start_pos} and end_pos {end_pos} are not adjacent, but no solution was given"
			)

		# order the solution, by creating a list from the start to the end
		# add end pos, since we will iterate over all these starting from the start pos
		solution_raw_list: list[CoordTup] = [tuple(c) for c in solution_raw] + [
			tuple(end_pos),
		]
		# solution starts with start point
		solution: list[CoordTup] = [tuple(start_pos)]
		while solution[-1] != tuple(end_pos):
			# use `get_coord_neighbors` to find connected neighbors
			neighbors: CoordArray = temp_maze.get_coord_neighbors(solution[-1])
			# TODO: make this less ugly
			assert (len(neighbors.shape) == 2) and (neighbors.shape[1] == 2), (  # noqa: PT018, PLR2004
				f"neighbors {neighbors} has shape {neighbors.shape}, expected shape (n, 2)\n{neighbors = }\n{solution = }\n{solution_raw = }\n{temp_maze.as_ascii()}"
			)
			# neighbors = neighbors[:, [1, 0]]
			# filter out neighbors that are not in the raw solution
			neighbors_filtered: CoordArray = np.array(
				[
					coord
					for coord in neighbors
					if (
						tuple(coord) in solution_raw_list
						and tuple(coord) not in solution
					)
				],
			)
			# assert only one element is left, and then add it to the solution
			assert neighbors_filtered.shape == (
				1,
				2,
			), (
				f"neighbors_filtered has shape {neighbors_filtered.shape}, expected shape (1, 2)\n{neighbors = }\n{neighbors_filtered = }\n{solution = }\n{solution_raw_list = }\n{temp_maze.as_ascii()}"
			)
			solution.append(tuple(neighbors_filtered[0]))

		# assert the solution is complete
		assert solution[0] == tuple(start_pos), (
			f"solution {solution} does not start at start_pos {start_pos}"
		)
		assert solution[-1] == tuple(end_pos), (
			f"solution {solution} does not end at end_pos {end_pos}"
		)

		return cls(
			connection_list=np.array(connection_list),
			solution=np.array(solution),  # type: ignore[call-arg]
		)

	# ============================================================
	# to and from ASCII
	# ============================================================
	def _as_ascii_grid(self) -> Shaped[np.ndarray, "x y"]:
		# Get the pixel grid using to_pixels().
		pixel_grid: Bool[np.ndarray, "x y"] = self._as_pixels_bw()

		# Replace pixel values with ASCII characters.
		ascii_grid: Shaped[np.ndarray, "x y"] = np.full(
			pixel_grid.shape,
			AsciiChars.WALL,
			dtype=str,
		)
		ascii_grid[pixel_grid == True] = AsciiChars.OPEN  # noqa: E712

		return ascii_grid

	def as_ascii(
		self,
		show_endpoints: bool = True,
		show_solution: bool = True,
	) -> str:
		"""return an ASCII grid of the maze

		useful for debugging in the terminal, or as it's own format

		can be reversed with `LatticeMaze.from_ascii()`
		"""
		ascii_grid: Shaped[np.ndarray, "x y"] = self._as_ascii_grid()
		pixel_grid: PixelGrid = self.as_pixels(
			show_endpoints=show_endpoints,
			show_solution=show_solution,
		)

		chars_replace: tuple = tuple()
		if show_endpoints:
			chars_replace += (AsciiChars.START, AsciiChars.END)
		if show_solution:
			chars_replace += (AsciiChars.PATH,)

		for ascii_char, pixel_color in ASCII_PIXEL_PAIRINGS.items():
			if ascii_char in chars_replace:
				ascii_grid[(pixel_grid == pixel_color).all(axis=-1)] = ascii_char

		return "\n".join("".join(row) for row in ascii_grid)

	@classmethod
	def from_ascii(cls, ascii_str: str) -> "LatticeMaze":
		"get a `LatticeMaze` from an ASCII representation (reverses `LaticeMaze.as_ascii`)"
		lines: list[str] = ascii_str.strip().split("\n")
		lines = [line.strip() for line in lines]
		ascii_grid: Shaped[np.ndarray, "x y"] = np.array(
			[list(line) for line in lines],
			dtype=str,
		)
		pixel_grid: PixelGrid = np.zeros((*ascii_grid.shape, 3), dtype=np.uint8)

		for ascii_char, pixel_color in ASCII_PIXEL_PAIRINGS.items():
			pixel_grid[ascii_grid == ascii_char] = pixel_color

		return cls.from_pixels(pixel_grid)


# type ignore here even though theyre all frozen
# maybe `SerializeableDataclass` itself is not frozen, but thats an ABC
# error: Cannot inherit frozen dataclass from a non-frozen one  [misc]
@serializable_dataclass(frozen=True, kw_only=True)
class TargetedLatticeMaze(LatticeMaze):  # type: ignore[misc]
	"""A LatticeMaze with a start and end position"""

	# this jank is so that SolvedMaze can inherit from this class without needing arguments for start_pos and end_pos
	# type ignore here because even though its a kw-only dataclass,
	# mypy doesn't like that non-default arguments are after default arguments
	start_pos: Coord = serializable_field(  # type: ignore[misc]
		assert_type=False,
	)
	end_pos: Coord = serializable_field(  # type: ignore[misc]
		assert_type=False,
	)

	def __post_init__(self) -> None:
		"post init converts start and end pos to numpy arrays, checks they exist and are in bounds"
		# make things numpy arrays (very jank to override frozen dataclass)
		self.__dict__["start_pos"] = np.array(self.start_pos)
		self.__dict__["end_pos"] = np.array(self.end_pos)
		assert self.start_pos is not None
		assert self.end_pos is not None
		# check that start and end are in bounds
		if (
			self.start_pos[0] >= self.grid_shape[0]
			or self.start_pos[1] >= self.grid_shape[1]
		):
			err_msg: str = f"start_pos {self.start_pos} is out of bounds for grid shape {self.grid_shape}"
			raise ValueError(
				err_msg,
			)
		if (
			self.end_pos[0] >= self.grid_shape[0]
			or self.end_pos[1] >= self.grid_shape[1]
		):
			err_msg = f"end_pos {self.end_pos = } is out of bounds for grid shape {self.grid_shape = }"
			raise ValueError(
				err_msg,
			)

	def __eq__(self, other: object) -> bool:
		"check equality, calls parent class equality check"
		return super().__eq__(other)

	def __hash__(self) -> int:
		"hash the `TargetedLatticeMaze` by hashing a tuple of the connection list and start and end positions as bytes"
		return hash(
			(
				self.connection_list.tobytes(),
				self.start_pos.tobytes(),
				self.end_pos.tobytes(),
			)
		)

	def _get_start_pos_tokens(self) -> list[str | CoordTup]:
		return [
			SPECIAL_TOKENS.ORIGIN_START,
			tuple(self.start_pos),
			SPECIAL_TOKENS.ORIGIN_END,
		]

	def get_start_pos_tokens(self) -> list[str | CoordTup]:
		"(deprecated!) return the start position as a list of tokens"
		warnings.warn(
			"`TargetedLatticeMaze.get_start_pos_tokens` will be removed from the public API in a future release.",
			TokenizerDeprecationWarning,
		)
		return self._get_start_pos_tokens()

	def _get_end_pos_tokens(self) -> list[str | CoordTup]:
		return [
			SPECIAL_TOKENS.TARGET_START,
			tuple(self.end_pos),
			SPECIAL_TOKENS.TARGET_END,
		]

	def get_end_pos_tokens(self) -> list[str | CoordTup]:
		"(deprecated!) return the end position as a list of tokens"
		warnings.warn(
			"`TargetedLatticeMaze.get_end_pos_tokens` will be removed from the public API in a future release.",
			TokenizerDeprecationWarning,
		)
		return self._get_end_pos_tokens()

	@classmethod
	def from_lattice_maze(
		cls,
		lattice_maze: LatticeMaze,
		start_pos: Coord | CoordTup,
		end_pos: Coord | CoordTup,
	) -> "TargetedLatticeMaze":
		"get a `TargetedLatticeMaze` from a `LatticeMaze` by specifying start and end positions"
		return cls(
			connection_list=lattice_maze.connection_list,
			start_pos=np.array(start_pos),
			end_pos=np.array(end_pos),
			generation_meta=lattice_maze.generation_meta,
		)


@serializable_dataclass(frozen=True, kw_only=True)
class SolvedMaze(TargetedLatticeMaze):  # type: ignore[misc]
	"""Stores a maze and a solution"""

	solution: CoordArray = serializable_field(  # type: ignore[misc]
		assert_type=False,
	)

	def __init__(
		self,
		connection_list: ConnectionList,
		solution: CoordArray,
		generation_meta: dict | None = None,
		start_pos: Coord | None = None,
		end_pos: Coord | None = None,
		allow_invalid: bool = False,
	) -> None:
		"""Create a SolvedMaze from a connection list and a solution

		> DOCS: better documentation for this init method
		"""
		# figure out the solution
		solution_valid: bool = False
		if solution is not None:
			solution = np.array(solution)
			# note that a path length of 1 here is valid, since the start and end pos could be the same
			if (solution.shape[0] > 0) and (solution.shape[1] == 2):  # noqa: PLR2004
				solution_valid = True

		if not solution_valid and not allow_invalid:
			err_msg: str = f"invalid solution: {solution.shape = } {solution = } {solution_valid = } {allow_invalid = }"
			raise ValueError(
				err_msg,
				f"{connection_list = }",
			)

		# init the TargetedLatticeMaze
		super().__init__(
			connection_list=connection_list,
			generation_meta=generation_meta,
			# TODO: the argument type is stricter than the expected type but it still fails?
			# error: Argument "start_pos" to "__init__" of "TargetedLatticeMaze" has incompatible type
			# "ndarray[tuple[int, ...], dtype[Any]] | None"; expected "ndarray[Any, Any]"  [arg-type]
			start_pos=np.array(solution[0]) if solution_valid else None,  # type: ignore[arg-type]
			end_pos=np.array(solution[-1]) if solution_valid else None,  # type: ignore[arg-type]
		)

		self.__dict__["solution"] = solution

		# adjust the endpoints
		if not allow_invalid:
			if start_pos is not None:
				assert np.array_equal(np.array(start_pos), self.start_pos), (
					f"when trying to create a SolvedMaze, the given start_pos does not match the one in the solution: given={start_pos}, solution={self.start_pos}"
				)
			if end_pos is not None:
				assert np.array_equal(np.array(end_pos), self.end_pos), (
					f"when trying to create a SolvedMaze, the given end_pos does not match the one in the solution: given={end_pos}, solution={self.end_pos}"
				)
			# TODO: assert the path does not backtrack, walk through walls, etc?

	def __eq__(self, other: object) -> bool:
		"check equality, calls parent class equality check"
		return super().__eq__(other)

	def __hash__(self) -> int:
		"hash the `SolvedMaze` by hashing a tuple of the connection list and solution arrays as bytes"
		return hash((self.connection_list.tobytes(), self.solution.tobytes()))

	def _get_solution_tokens(self) -> list[str | CoordTup]:
		return [
			SPECIAL_TOKENS.PATH_START,
			*[tuple(c) for c in self.solution],
			SPECIAL_TOKENS.PATH_END,
		]

	def get_solution_tokens(self) -> list[str | CoordTup]:
		"(deprecated!) return the solution as a list of tokens"
		warnings.warn(
			"`LatticeMaze.get_solution_tokens` is deprecated.",
			TokenizerDeprecationWarning,
		)
		return self._get_solution_tokens()

	# for backwards compatibility
	@property
	def maze(self) -> LatticeMaze:
		"(deprecated!) return the maze without the solution"
		warnings.warn(
			"`maze` is deprecated, SolvedMaze now inherits from LatticeMaze.",
			DeprecationWarning,
		)
		return LatticeMaze(connection_list=self.connection_list)

	# type ignore here since we're overriding a method with a different signature
	@classmethod
	def from_lattice_maze(  # type: ignore[override]
		cls,
		lattice_maze: LatticeMaze,
		solution: list[CoordTup] | CoordArray,
	) -> "SolvedMaze":
		"get a `SolvedMaze` from a `LatticeMaze` by specifying a solution"
		return cls(
			connection_list=lattice_maze.connection_list,
			solution=np.array(solution),
			generation_meta=lattice_maze.generation_meta,
		)

	@classmethod
	def from_targeted_lattice_maze(
		cls,
		targeted_lattice_maze: TargetedLatticeMaze,
		solution: list[CoordTup] | CoordArray | None = None,
	) -> "SolvedMaze":
		"""solves the given targeted lattice maze and returns a SolvedMaze"""
		if solution is None:
			solution = targeted_lattice_maze.find_shortest_path(
				targeted_lattice_maze.start_pos,
				targeted_lattice_maze.end_pos,
			)
		return cls(
			connection_list=targeted_lattice_maze.connection_list,
			solution=np.array(solution),
			generation_meta=targeted_lattice_maze.generation_meta,
		)

	def get_solution_forking_points(
		self,
		always_include_endpoints: bool = False,
	) -> tuple[list[int], CoordArray]:
		"""coordinates and their indicies from the solution where a fork is present

		- if the start point is not a dead end, this counts as a fork
		- if the end point is not a dead end, this counts as a fork
		"""
		output_idxs: list[int] = list()
		output_coords: list[CoordTup] = list()

		for idx, coord in enumerate(self.solution):
			# more than one choice for first coord, or more than 2 for any other
			# since the previous coord doesn't count as a choice
			is_endpoint: bool = idx == 0 or idx == self.solution.shape[0] - 1
			theshold: int = 1 if is_endpoint else 2
			if self.get_coord_neighbors(coord).shape[0] > theshold or (
				is_endpoint and always_include_endpoints
			):
				output_idxs.append(idx)
				output_coords.append(coord)

		return output_idxs, np.array(output_coords)

	def get_solution_path_following_points(self) -> tuple[list[int], CoordArray]:
		"""coordinates from the solution where there is only a single (non-backtracking) point to move to

		returns the complement of `get_solution_forking_points` from the path
		"""
		forks_idxs, _ = self.get_solution_forking_points()
		# HACK: idk why type ignore here
		return (  # type: ignore[return-value]
			np.delete(np.arange(self.solution.shape[0]), forks_idxs, axis=0),
			np.delete(self.solution, forks_idxs, axis=0),
		)


def detect_pixels_type(data: PixelGrid) -> typing.Type[LatticeMaze]:
	"""Detects the type of pixels data by checking for the presence of start and end pixels"""
	if color_in_pixel_grid(data, PixelColors.START) or color_in_pixel_grid(
		data,
		PixelColors.END,
	):
		if color_in_pixel_grid(data, PixelColors.PATH):
			return SolvedMaze
		else:
			return TargetedLatticeMaze
	else:
		return LatticeMaze


def _remove_isolated_cells(
	image: Int[np.ndarray, "RGB x y"],
) -> Int[np.ndarray, "RGB x y"]:
	"""Removes isolated cells from an image. An isolated cell is a cell that is surrounded by walls on all sides."""
	# Create a binary mask where True represents walls
	wall_mask = np.all(image == PixelColors.WALL, axis=-1)

	# Pad the wall mask to handle edge cases
	padded_wall_mask = np.pad(
		wall_mask,
		((1, 1), (1, 1)),
		mode="constant",
		constant_values=True,
	)

	# Check neighbors in all four directions
	isolated_mask = (
		padded_wall_mask[1:-1, 2:]  # right
		& padded_wall_mask[1:-1, :-2]  # left
		& padded_wall_mask[2:, 1:-1]  # down
		& padded_wall_mask[:-2, 1:-1]  # up
	)

	# Combine with non-wall mask to only affect open cells
	non_wall_mask = ~wall_mask
	isolated_mask = isolated_mask & non_wall_mask

	# Create the output image
	output_image = image.copy()
	output_image[isolated_mask] = PixelColors.WALL

	return output_image


_RIC_PADS: dict = {
	"left": ((1, 0), (0, 0)),
	"right": ((0, 1), (0, 0)),
	"up": ((0, 0), (1, 0)),
	"down": ((0, 0), (0, 1)),
}

# Define slices for each direction
_RIC_SLICES: dict = {
	"left": (slice(1, None), slice(None, None)),
	"right": (slice(None, -1), slice(None, None)),
	"up": (slice(None, None), slice(1, None)),
	"down": (slice(None, None), slice(None, -1)),
}


# TODO: figure out why this function doesnt work, or maybe just get rid of it
# def _remove_isolated_cells_old(
#     image: Int[np.ndarray, "RGB x y"],
# ) -> Int[np.ndarray, "RGB x y"]:
#     """
#     Removes isolated cells from an image. An isolated cell is a cell that is surrounded by walls on all sides.
#     """
#     warnings.warn("this functin doesn't work and I have no idea why!!!")
#     masks: dict[str, np.ndarray] = {
#         d: np.all(
#             np.pad(
#                 image[_RIC_SLICES[d][0], _RIC_SLICES[d][1], :] == PixelColors.WALL,
#                 np.array((*_RIC_PADS[d], (0, 0)), dtype=np.int8),
#                 mode="constant",
#                 constant_values=True,
#             ),
#             axis=2,
#         )
#         for d in _RIC_SLICES.keys()
#     }

#     # Create a mask for non-wall cells
#     mask_non_wall = np.all(image != PixelColors.WALL, axis=2)

#     # print(f"{mask_non_wall.shape = }")
#     # print(f"{ {k: masks[k].shape for k in masks.keys()} = }")

#     # print(f"{mask_non_wall = }")
#     # print(f"{masks['down'] = }")

#     # Combine the masks
#     mask = mask_non_wall & masks["left"] & masks["right"] & masks["up"] & masks["down"]

#     # Apply the mask
#     output_image = np.where(
#         np.stack([mask] * 3, axis=-1),
#         PixelColors.WALL,
#         image,
#     )

#     return output_image

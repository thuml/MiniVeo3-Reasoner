"""generation functions have signature `(grid_shape: Coord, **kwargs) -> LatticeMaze` and are methods in `LatticeMazeGenerators`"""

import random
import warnings
from typing import Callable, Concatenate, ParamSpec

import numpy as np
from jaxtyping import Bool

from maze_dataset.constants import CoordArray, CoordTup
from maze_dataset.generation.seed import GLOBAL_SEED
from maze_dataset.maze import ConnectionList, Coord, LatticeMaze, SolvedMaze
from maze_dataset.maze.lattice_maze import NEIGHBORS_MASK, _fill_edges_with_walls

_NUMPY_RNG: np.random.Generator = np.random.default_rng(GLOBAL_SEED)
random.seed(GLOBAL_SEED)


def _random_start_coord(
	grid_shape: Coord,
	start_coord: Coord | CoordTup | None,
) -> Coord:
	"picking a random start coord within the bounds of `grid_shape` if none is provided"
	start_coord_: Coord
	if start_coord is None:
		start_coord_ = np.random.randint(
			0,  # lower bound
			np.maximum(grid_shape - 1, 1),  # upper bound (at least 1)
			size=len(grid_shape),  # dimensionality
		)
	else:
		start_coord_ = np.array(start_coord)

	return start_coord_


def get_neighbors_in_bounds(
	coord: Coord,
	grid_shape: Coord,
) -> CoordArray:
	"get all neighbors of a coordinate that are within the bounds of the grid"
	# get all neighbors
	neighbors: CoordArray = coord + NEIGHBORS_MASK

	# filter neighbors by being within grid bounds
	neighbors_in_bounds: CoordArray = neighbors[
		(neighbors >= 0).all(axis=1) & (neighbors < grid_shape).all(axis=1)
	]

	return neighbors_in_bounds


class LatticeMazeGenerators:
	"""namespace for lattice maze generation algorithms

	examples of generated mazes can be found here:
	https://understanding-search.github.io/maze-dataset/examples/maze_examples.html
	"""

	@staticmethod
	def gen_dfs(
		grid_shape: Coord | CoordTup,
		*,
		lattice_dim: int = 2,
		accessible_cells: float | None = None,
		max_tree_depth: float | None = None,
		do_forks: bool = True,
		randomized_stack: bool = False,
		start_coord: Coord | None = None,
	) -> LatticeMaze:
		"""generate a lattice maze using depth first search, iterative

		# Arguments
		- `grid_shape: Coord`: the shape of the grid
		- `lattice_dim: int`: the dimension of the lattice
			(default: `2`)
		- `accessible_cells: int | float |None`: the number of accessible cells in the maze. If `None`, defaults to the total number of cells in the grid. if a float, asserts it is <= 1 and treats it as a proportion of **total cells**
			(default: `None`)
		- `max_tree_depth: int | float | None`: the maximum depth of the tree. If `None`, defaults to `2 * accessible_cells`. if a float, asserts it is <= 1 and treats it as a proportion of the **sum of the grid shape**
			(default: `None`)
		- `do_forks: bool`: whether to allow forks in the maze. If `False`, the maze will be have no forks and will be a simple hallway.
		- `start_coord: Coord | None`: the starting coordinate of the generation algorithm. If `None`, defaults to a random coordinate.

		# algorithm
		1. Choose the initial cell, mark it as visited and push it to the stack
		2. While the stack is not empty
			1. Pop a cell from the stack and make it a current cell
			2. If the current cell has any neighbours which have not been visited
				1. Push the current cell to the stack
				2. Choose one of the unvisited neighbours
				3. Remove the wall between the current cell and the chosen cell
				4. Mark the chosen cell as visited and push it to the stack
		"""
		# Default values if no constraints have been passed
		grid_shape_: Coord = np.array(grid_shape)
		n_total_cells: int = int(np.prod(grid_shape_))

		n_accessible_cells: int
		if accessible_cells is None:
			n_accessible_cells = n_total_cells
		elif isinstance(accessible_cells, float):
			assert accessible_cells <= 1, (
				f"accessible_cells must be an int (count) or a float in the range [0, 1] (proportion), got {accessible_cells}"
			)

			n_accessible_cells = int(accessible_cells * n_total_cells)
		else:
			assert isinstance(accessible_cells, int)
			n_accessible_cells = accessible_cells

		if max_tree_depth is None:
			max_tree_depth = (
				2 * n_total_cells
			)  # We define max tree depth counting from the start coord in two directions. Therefore we divide by two in the if clause for neighboring sites later and multiply by two here.
		elif isinstance(max_tree_depth, float):
			assert max_tree_depth <= 1, (
				f"max_tree_depth must be an int (count) or a float in the range [0, 1] (proportion), got {max_tree_depth}"
			)

			max_tree_depth = int(max_tree_depth * np.sum(grid_shape_))

		# choose a random start coord
		start_coord = _random_start_coord(grid_shape_, start_coord)

		# initialize the maze with no connections
		connection_list: ConnectionList = np.zeros(
			(lattice_dim, grid_shape_[0], grid_shape_[1]),
			dtype=np.bool_,
		)

		# initialize the stack with the target coord
		visited_cells: set[tuple[int, int]] = set()
		visited_cells.add(tuple(start_coord))  # this wasnt a bug after all lol
		stack: list[Coord] = [start_coord]

		# initialize tree_depth_counter
		current_tree_depth: int = 1

		# loop until the stack is empty or n_connected_cells is reached
		while stack and (len(visited_cells) < n_accessible_cells):
			# get the current coord from the stack
			current_coord: Coord
			if randomized_stack:
				current_coord = stack.pop(random.randint(0, len(stack) - 1))
			else:
				current_coord = stack.pop()

			# filter neighbors by being within grid bounds and being unvisited
			unvisited_neighbors_deltas: list[tuple[Coord, Coord]] = [
				(neighbor, delta)
				for neighbor, delta in zip(
					current_coord + NEIGHBORS_MASK,
					NEIGHBORS_MASK,
					strict=False,
				)
				if (
					(tuple(neighbor) not in visited_cells)
					and (0 <= neighbor[0] < grid_shape_[0])
					and (0 <= neighbor[1] < grid_shape_[1])
				)
			]

			# don't continue if max_tree_depth/2 is already reached (divide by 2 because we can branch to multiple directions)
			if unvisited_neighbors_deltas and (
				current_tree_depth <= max_tree_depth / 2
			):
				# if we want a maze without forks, simply don't add the current coord back to the stack
				if do_forks and (len(unvisited_neighbors_deltas) > 1):
					stack.append(current_coord)

				# choose one of the unvisited neighbors
				chosen_neighbor, delta = random.choice(unvisited_neighbors_deltas)

				# add connection
				dim: int = int(np.argmax(np.abs(delta)))
				# if positive, down/right from current coord
				# if negative, up/left from current coord (down/right from neighbor)
				clist_node: Coord = (
					current_coord if (delta.sum() > 0) else chosen_neighbor
				)
				connection_list[dim, clist_node[0], clist_node[1]] = True

				# add to visited cells and stack
				visited_cells.add(tuple(chosen_neighbor))
				stack.append(chosen_neighbor)

				# Update current tree depth
				current_tree_depth += 1
			else:
				current_tree_depth -= 1

		return LatticeMaze(
			connection_list=connection_list,
			generation_meta=dict(
				func_name="gen_dfs",
				grid_shape=grid_shape_,
				start_coord=start_coord,
				n_accessible_cells=int(n_accessible_cells),
				max_tree_depth=int(max_tree_depth),
				# oh my god this took so long to track down. its almost 5am and I've spent like 2 hours on this bug
				# it was checking that len(visited_cells) == n_accessible_cells, but this means that the maze is
				# treated as fully connected even when it is most certainly not, causing solving the maze to break
				fully_connected=bool(len(visited_cells) == n_total_cells),
				visited_cells={tuple(int(x) for x in coord) for coord in visited_cells},
			),
		)

	@staticmethod
	def gen_prim(
		grid_shape: Coord | CoordTup,
		lattice_dim: int = 2,
		accessible_cells: float | None = None,
		max_tree_depth: float | None = None,
		do_forks: bool = True,
		start_coord: Coord | None = None,
	) -> LatticeMaze:
		"(broken!) generate a lattice maze using Prim's algorithm"
		warnings.warn(
			"gen_prim does not correctly implement prim's algorithm, see issue: https://github.com/understanding-search/maze-dataset/issues/12",
		)
		return LatticeMazeGenerators.gen_dfs(
			grid_shape=grid_shape,
			lattice_dim=lattice_dim,
			accessible_cells=accessible_cells,
			max_tree_depth=max_tree_depth,
			do_forks=do_forks,
			start_coord=start_coord,
			randomized_stack=True,
		)

	@staticmethod
	def gen_wilson(
		grid_shape: Coord | CoordTup,
		**kwargs,
	) -> LatticeMaze:
		"""Generate a lattice maze using Wilson's algorithm.

		# Algorithm
		Wilson's algorithm generates an unbiased (random) maze
		sampled from the uniform distribution over all mazes, using loop-erased random walks. The generated maze is
		acyclic and all cells are part of a unique connected space.
		https://en.wikipedia.org/wiki/Maze_generation_algorithm#Wilson's_algorithm
		"""
		assert not kwargs, (
			f"gen_wilson does not take any additional arguments, got {kwargs = }"
		)

		grid_shape_: Coord = np.array(grid_shape)

		# Initialize grid and visited cells
		connection_list: ConnectionList = np.zeros((2, *grid_shape_), dtype=np.bool_)
		visited: Bool[np.ndarray, "x y"] = np.zeros(grid_shape_, dtype=np.bool_)

		# Choose a random cell and mark it as visited
		start_coord: Coord = _random_start_coord(grid_shape_, None)
		visited[start_coord[0], start_coord[1]] = True
		del start_coord

		while not visited.all():
			# Perform loop-erased random walk from another random cell

			# Choose walk_start only from unvisited cells
			unvisited_coords: CoordArray = np.column_stack(np.where(~visited))
			walk_start: Coord = unvisited_coords[
				np.random.choice(unvisited_coords.shape[0])
			]

			# Perform the random walk
			path: list[Coord] = [walk_start]
			current: Coord = walk_start

			# exit the loop once the current path hits a visited cell
			while not visited[current[0], current[1]]:
				# find a valid neighbor (one always exists on a lattice)
				neighbors: CoordArray = get_neighbors_in_bounds(current, grid_shape_)
				next_cell: Coord = neighbors[np.random.choice(neighbors.shape[0])]

				# Check for loop
				loop_exit: int | None = None
				for i, p in enumerate(path):
					if np.array_equal(next_cell, p):
						loop_exit = i
						break

				# erase the loop, or continue the walk
				if loop_exit is not None:
					# this removes everything after and including the loop start
					path = path[: loop_exit + 1]
					# reset current cell to end of path
					current = path[-1]
				else:
					path.append(next_cell)
					current = next_cell

			# Add the path to the maze
			for i in range(len(path) - 1):
				c_1: Coord = path[i]
				c_2: Coord = path[i + 1]

				# find the dimension of the connection
				delta: Coord = c_2 - c_1
				dim: int = int(np.argmax(np.abs(delta)))

				# if positive, down/right from current coord
				# if negative, up/left from current coord (down/right from neighbor)
				clist_node: Coord = c_1 if (delta.sum() > 0) else c_2
				connection_list[dim, clist_node[0], clist_node[1]] = True
				visited[c_1[0], c_1[1]] = True
				# we dont add c_2 because the last c_2 will have already been visited

		return LatticeMaze(
			connection_list=connection_list,
			generation_meta=dict(
				func_name="gen_wilson",
				grid_shape=grid_shape_,
				fully_connected=True,
			),
		)

	@staticmethod
	def gen_percolation(
		grid_shape: Coord | CoordTup,
		p: float = 0.4,
		lattice_dim: int = 2,
		start_coord: Coord | None = None,
	) -> LatticeMaze:
		"""generate a lattice maze using simple percolation

		note that p in the range (0.4, 0.7) gives the most interesting mazes

		# Arguments
		- `grid_shape: Coord`: the shape of the grid
		- `lattice_dim: int`: the dimension of the lattice (default: `2`)
		- `p: float`: the probability of a cell being accessible (default: `0.5`)
		- `start_coord: Coord | None`: the starting coordinate for the connected component (default: `None` will give a random start)
		"""
		assert p >= 0 and p <= 1, f"p must be between 0 and 1, got {p}"  # noqa: PT018
		grid_shape_: Coord = np.array(grid_shape)

		start_coord = _random_start_coord(grid_shape_, start_coord)

		connection_list: ConnectionList = np.random.rand(lattice_dim, *grid_shape_) < p

		connection_list = _fill_edges_with_walls(connection_list)

		output: LatticeMaze = LatticeMaze(
			connection_list=connection_list,
			generation_meta=dict(
				func_name="gen_percolation",
				grid_shape=grid_shape_,
				percolation_p=p,
				start_coord=start_coord,
			),
		)

		# generation_meta is sometimes None, but not here since we just made it a dict above
		output.generation_meta["visited_cells"] = output.gen_connected_component_from(  # type: ignore[index]
			start_coord,
		)

		return output

	@staticmethod
	def gen_dfs_percolation(
		grid_shape: Coord | CoordTup,
		p: float = 0.4,
		lattice_dim: int = 2,
		accessible_cells: int | None = None,
		max_tree_depth: int | None = None,
		start_coord: Coord | None = None,
	) -> LatticeMaze:
		"""dfs and then percolation (adds cycles)"""
		grid_shape_: Coord = np.array(grid_shape)
		start_coord = _random_start_coord(grid_shape_, start_coord)

		# generate initial maze via dfs
		maze: LatticeMaze = LatticeMazeGenerators.gen_dfs(
			grid_shape=grid_shape_,
			lattice_dim=lattice_dim,
			accessible_cells=accessible_cells,
			max_tree_depth=max_tree_depth,
			start_coord=start_coord,
		)

		# percolate
		connection_list_perc: np.ndarray = (
			np.random.rand(*maze.connection_list.shape) < p
		)
		connection_list_perc = _fill_edges_with_walls(connection_list_perc)

		maze.__dict__["connection_list"] = np.logical_or(
			maze.connection_list,
			connection_list_perc,
		)

		# generation_meta is sometimes None, but not here since we just made it a dict above
		maze.generation_meta["func_name"] = "gen_dfs_percolation"  # type: ignore[index]
		maze.generation_meta["percolation_p"] = p  # type: ignore[index]
		maze.generation_meta["visited_cells"] = maze.gen_connected_component_from(  # type: ignore[index]
			start_coord,
		)

		return maze

	@staticmethod
	def gen_kruskal(
		grid_shape: "Coord | CoordTup",
		lattice_dim: int = 2,
		start_coord: "Coord | None" = None,
	) -> "LatticeMaze":
		"""Generate a maze using Kruskal's algorithm.

		This function generates a random spanning tree over a grid using Kruskal's algorithm.
		Each cell is treated as a node, and all valid adjacent edges are listed and processed
		in random order. An edge is added (i.e. its passage carved) only if it connects two cells
		that are not already connected. The resulting maze is a perfect maze (i.e. a spanning tree)
		without cycles.

		https://en.wikipedia.org/wiki/Kruskal's_algorithm

		# Parameters:
		- `grid_shape : Coord | CoordTup`
			The shape of the maze grid (for example, `(n_rows, n_cols)`).
		- `lattice_dim : int`
			The lattice dimension (default is `2`).
		- `start_coord : Coord | None`
			Optionally, specify a starting coordinate. If `None`, a random coordinate will be chosen.
		- `**kwargs`
			Additional keyword arguments (currently unused).

		# Returns:
		- `LatticeMaze`
			A maze represented by a connection list, generated as a spanning tree using Kruskal's algorithm.

		# Usage:
		```python
		maze = gen_kruskal((10, 10))
		```
		"""
		assert lattice_dim == 2, (  # noqa: PLR2004
			"Kruskal's algorithm is only implemented for 2D lattices."
		)
		# Convert grid_shape to a tuple of ints
		grid_shape_: CoordTup = tuple(int(x) for x in grid_shape)  # type: ignore[assignment]
		n_rows, n_cols = grid_shape_

		# Initialize union-find data structure.
		parent: dict[tuple[int, int], tuple[int, int]] = {}

		def find(cell: tuple[int, int]) -> tuple[int, int]:
			while parent[cell] != cell:
				parent[cell] = parent[parent[cell]]
				cell = parent[cell]
			return cell

		def union(cell1: tuple[int, int], cell2: tuple[int, int]) -> None:
			root1 = find(cell1)
			root2 = find(cell2)
			parent[root2] = root1

		# Initialize each cell as its own set.
		for i in range(n_rows):
			for j in range(n_cols):
				parent[(i, j)] = (i, j)

		# List all possible edges.
		# For vertical edges (i.e. connecting a cell to its right neighbor):
		edges: list[tuple[tuple[int, int], tuple[int, int], int]] = []
		for i in range(n_rows):
			for j in range(n_cols - 1):
				edges.append(((i, j), (i, j + 1), 1))
		# For horizontal edges (i.e. connecting a cell to its bottom neighbor):
		for i in range(n_rows - 1):
			for j in range(n_cols):
				edges.append(((i, j), (i + 1, j), 0))

		# Shuffle the list of edges.
		import random

		random.shuffle(edges)

		# Initialize connection_list with no connections.
		# connection_list[0] stores downward connections (from cell (i,j) to (i+1,j)).
		# connection_list[1] stores rightward connections (from cell (i,j) to (i,j+1)).
		import numpy as np

		connection_list = np.zeros((2, n_rows, n_cols), dtype=bool)

		# Process each edge; if it connects two different trees, union them and carve the passage.
		for cell1, cell2, direction in edges:
			if find(cell1) != find(cell2):
				union(cell1, cell2)
				if direction == 0:
					# Horizontal edge: connection is stored in connection_list[0] at cell1.
					connection_list[0, cell1[0], cell1[1]] = True
				else:
					# Vertical edge: connection is stored in connection_list[1] at cell1.
					connection_list[1, cell1[0], cell1[1]] = True

		if start_coord is None:
			start_coord = tuple(np.random.randint(0, n) for n in grid_shape_)  # type: ignore[assignment]

		generation_meta: dict = dict(
			func_name="gen_kruskal",
			grid_shape=grid_shape_,
			start_coord=start_coord,
			algorithm="kruskal",
			fully_connected=True,
		)
		return LatticeMaze(
			connection_list=connection_list, generation_meta=generation_meta
		)

	@staticmethod
	def gen_recursive_division(
		grid_shape: "Coord | CoordTup",
		lattice_dim: int = 2,
		start_coord: "Coord | None" = None,
	) -> "LatticeMaze":
		"""Generate a maze using the recursive division algorithm.

		This function generates a maze by recursively dividing the grid with walls and carving a single
		passage through each wall. The algorithm begins with a fully connected grid (i.e. every pair of adjacent
		cells is connected) and then removes connections along a chosen division line—leaving one gap as a passage.
		The resulting maze is a perfect maze, meaning there is exactly one path between any two cells.

		# Parameters:
		- `grid_shape : Coord | CoordTup`
			The shape of the maze grid (e.g., `(n_rows, n_cols)`).
		- `lattice_dim : int`
			The lattice dimension (default is `2`).
		- `start_coord : Coord | None`
			Optionally, specify a starting coordinate. If `None`, a random coordinate is chosen.
		- `**kwargs`
			Additional keyword arguments (currently unused).

		# Returns:
		- `LatticeMaze`
			A maze represented by a connection list, generated using recursive division.

		# Usage:
		```python
		maze = gen_recursive_division((10, 10))
		```
		"""
		assert lattice_dim == 2, (  # noqa: PLR2004
			"Recursive division algorithm is only implemented for 2D lattices."
		)
		# Convert grid_shape to a tuple of ints.
		grid_shape_: CoordTup = tuple(int(x) for x in grid_shape)  # type: ignore[assignment]
		n_rows, n_cols = grid_shape_

		# Initialize connection_list as a fully connected grid.
		# For horizontal connections: for each cell (i,j) with i in [0, n_rows-2], set connection to True.
		# For vertical connections: for each cell (i,j) with j in [0, n_cols-2], set connection to True.
		connection_list = np.zeros((2, n_rows, n_cols), dtype=bool)
		connection_list[0, : n_rows - 1, :] = True
		connection_list[1, :, : n_cols - 1] = True

		def divide(x: int, y: int, width: int, height: int) -> None:
			"""Recursively divide the region starting at (x, y) with the given width and height.

			Removes connections along the chosen division line except for one randomly chosen gap.
			"""
			if width < 2 or height < 2:  # noqa: PLR2004
				return

			if width > height:
				# Vertical division.
				wall_col = random.randint(x + 1, x + width - 1)
				gap_row = random.randint(y, y + height - 1)
				for row in range(y, y + height):
					if row == gap_row:
						continue
					# Remove the vertical connection between (row, wall_col-1) and (row, wall_col).
					if wall_col - 1 < n_cols - 1:
						connection_list[1, row, wall_col - 1] = False
				# Recurse on the left and right subregions.
				divide(x, y, wall_col - x, height)
				divide(wall_col, y, x + width - wall_col, height)
			else:
				# Horizontal division.
				wall_row = random.randint(y + 1, y + height - 1)
				gap_col = random.randint(x, x + width - 1)
				for col in range(x, x + width):
					if col == gap_col:
						continue
					# Remove the horizontal connection between (wall_row-1, col) and (wall_row, col).
					if wall_row - 1 < n_rows - 1:
						connection_list[0, wall_row - 1, col] = False
				# Recurse on the top and bottom subregions.
				divide(x, y, width, wall_row - y)
				divide(x, wall_row, width, y + height - wall_row)

		# Begin the division on the full grid.
		divide(0, 0, n_cols, n_rows)

		if start_coord is None:
			start_coord = tuple(np.random.randint(0, n) for n in grid_shape_)  # type: ignore[assignment]

		generation_meta: dict = dict(
			func_name="gen_recursive_division",
			grid_shape=grid_shape_,
			start_coord=start_coord,
			algorithm="recursive_division",
			fully_connected=True,
		)
		return LatticeMaze(
			connection_list=connection_list, generation_meta=generation_meta
		)


P_GeneratorKwargs = ParamSpec("P_GeneratorKwargs")
MazeGeneratorFunc = Callable[
	Concatenate[Coord | CoordTup, P_GeneratorKwargs],
	LatticeMaze,
]


# cant automatically populate this because it messes with pickling :(
GENERATORS_MAP: dict[str, MazeGeneratorFunc] = {
	"gen_dfs": LatticeMazeGenerators.gen_dfs,
	# TYPING: error: Dict entry 1 has incompatible type
	# "str": "Callable[[ndarray[Any, Any] | tuple[int, int], KwArg(Any)], LatticeMaze]";
	# expected "str": "Callable[[ndarray[Any, Any] | tuple[int, int], Any], LatticeMaze]"  [dict-item]
	# gen_wilson takes no kwargs and we check that the kwargs are empty
	# but mypy doesnt like this, `Any` != `KwArg(Any)`
	"gen_wilson": LatticeMazeGenerators.gen_wilson,  # type: ignore[dict-item]
	"gen_percolation": LatticeMazeGenerators.gen_percolation,
	"gen_dfs_percolation": LatticeMazeGenerators.gen_dfs_percolation,
	"gen_prim": LatticeMazeGenerators.gen_prim,
	"gen_kruskal": LatticeMazeGenerators.gen_kruskal,
	"gen_recursive_division": LatticeMazeGenerators.gen_recursive_division,
}
"mapping of generator names to generator functions, useful for loading `MazeDatasetConfig`"

_GENERATORS_PERCOLATED: list[str] = [
	"gen_percolation",
	"gen_dfs_percolation",
]
"""list of generator names that generate percolated mazes
we use this to figure out the expected success rate, since depending on the endpoint kwargs this might fail
this variable is primarily used in `MazeDatasetConfig._to_ps_array` and `MazeDatasetConfig._from_ps_array`
"""


# TODO: we should deprecate this, always get a dataset when you want a maze with a solution
def get_maze_with_solution(
	gen_name: str,
	grid_shape: Coord | CoordTup,
	maze_ctor_kwargs: dict | None = None,
) -> SolvedMaze:
	"helper function to get a maze already with a solution"
	if maze_ctor_kwargs is None:
		maze_ctor_kwargs = dict()
	# TYPING: error: Too few arguments  [call-arg]
	# not sure why this is happening -- doesnt recognize the kwargs?
	maze: LatticeMaze = GENERATORS_MAP[gen_name](grid_shape, **maze_ctor_kwargs)  # type: ignore[call-arg]
	solution: CoordArray = np.array(maze.generate_random_path())
	return SolvedMaze.from_lattice_maze(lattice_maze=maze, solution=solution)

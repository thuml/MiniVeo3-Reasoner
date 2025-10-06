r"""`LatticeMaze` and the classes like `SolvedMaze` that inherit from it, along with a variety of helper functions"

This package utilizes a simple, efficient representation of mazes. Using an adjacency list to represent mazes would lead to a poor lookup time of whether any given connection exists, whilst using a dense adjacency matrix would waste memory by failing to exploit the structure (e.g., only 4 of the diagonals would be filled in).
Instead, we describe mazes with the following simple representation: for a $d$-dimensional lattice with $r$ rows and $c$ columns, we initialize a boolean array $A = \{0, 1\}^{d \times r \times c}$, which we refer to in the code as a `connection_list`. The value at $A[0,i,j]$ determines whether a downward connection exists from node $[i,j]$ to $[i+1, j]$. Likewise, the value at $A[1,i,j]$ determines whether a rightwards connection to $[i, j+1]$ exists. Thus, we avoid duplication of data about the existence of connections, at the cost of requiring additional care with indexing when looking for a connection upwards or to the left. Note that this setup allows for a periodic lattice.

To produce solutions to mazes, two points are selected uniformly at random without replacement from the connected component of the maze, and the $A^*$ algorithm is applied to find the shortest path between them.

Parallelization is implemented via the `multiprocessing` module in the Python standard library, and parallel generation can be controlled via keyword arguments to the `MazeDataset.from_config()` function.
"""

from maze_dataset.maze.lattice_maze import (
	AsciiChars,
	ConnectionList,
	Coord,
	CoordArray,
	LatticeMaze,
	PixelColors,
	SolvedMaze,
	TargetedLatticeMaze,
)

__all__ = [
	# submodules
	"lattice_maze",
	# imports
	"SolvedMaze",
	"TargetedLatticeMaze",
	"LatticeMaze",
	"ConnectionList",
	"AsciiChars",
	"Coord",
	"CoordArray",
	"PixelColors",
]

"""generation functions have signature `(grid_shape: Coord, **kwargs) -> LatticeMaze` and are methods in `LatticeMazeGenerators`

`DEFAULT_GENERATORS` is a list of generator name, generator kwargs pairs used in tests and demos

you can add your own maze generators by:
- adding a static method implementing your generation function to `LatticeMazeGenerators`, with the signature `(grid_shape: Coord, **kwargs) -> LatticeMaze` and adding the `(name, func)` pair to `GENERATORS_MAP`
- using the `@register_maze_generator` decorator on your generation function. However, this is only for testing purposes where modifying the original package is not possible.

If you implement a new generation function, please make a pull request!
https://github.com/understanding-search/maze-dataset/pulls
"""

from maze_dataset.generation.generators import (
	_NUMPY_RNG,
	GENERATORS_MAP,
	LatticeMazeGenerators,
	get_maze_with_solution,
)

__all__ = [
	# submodules
	"default_generators",
	"generators",
	"registrationseed",
	# imports
	"LatticeMazeGenerators",
	"GENERATORS_MAP",
	"get_maze_with_solution",
	"_NUMPY_RNG",
]

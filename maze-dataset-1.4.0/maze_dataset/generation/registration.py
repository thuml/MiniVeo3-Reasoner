"""Registration system for custom maze generators"""

import inspect
from typing import TypeVar, Union, get_args, get_origin

from maze_dataset.constants import Coord, CoordTup
from maze_dataset.generation.generators import (
	GENERATORS_MAP,
	LatticeMazeGenerators,
	MazeGeneratorFunc,
)
from maze_dataset.maze import LatticeMaze

F_MazeGeneratorFunc = TypeVar("F_MazeGeneratorFunc", bound=MazeGeneratorFunc)


class MazeGeneratorRegistrationError(TypeError):
	"""error for maze generator registration issues"""

	pass


def _check_grid_shape_annotation(
	annotation: Union[type, str],
	func_name: str,
) -> None:
	"""Check if the annotation for grid_shape is valid

	# Parameters:
	- `annotation` : Union[type, str] the type annotation of the grid_shape parameter

	# Raises:
	- `MazeGeneratorRegistrationError` if the annotation is not compatible
	"""
	annotation_str: str = str(annotation)

	if not any(
		[
			annotation == Coord,
			annotation == CoordTup,
			annotation_str.lower() == "tuple[int, int]",
			# TODO: these are pretty loose checks, would be better to make this more robust
			"ndarray" in annotation_str,
			"Int8" in annotation_str,
		]
	):
		err_msg = (
			f"Maze generator function '{func_name}' first parameter 'grid_shape' "
			f"must be typed as 'Coord | CoordTup' or compatible type, "
			f"got {annotation = }, {annotation_str = }"
		)
		raise MazeGeneratorRegistrationError(err_msg)


def validate_MazeGeneratorFunc(
	func: F_MazeGeneratorFunc,
) -> None:
	"""validate the signature of a maze generator function

	return `None` if valid, otherwise raises `MazeGeneratorRegistrationError`
	(which is a subclass of `TypeError`)

	# Parameters:
	- `func : MazeGeneratorFunc` function to validate

	# Raises:
	- `MazeGeneratorRegistrationError` : type error describing the issue with the function signature
	"""
	func_name: str = func.__name__
	sig: inspect.Signature = inspect.signature(func)
	params: list[str] = list(sig.parameters.keys())

	if not params or params[0] != "grid_shape":
		err_msg = (
			f"Maze generator function '{func_name}' must have 'grid_shape' "
			"as its first parameter. Please ensure the function signature starts with 'grid_shape: Coord | CoordTup'."
			f"{params = }"
		)
		raise MazeGeneratorRegistrationError(err_msg)

	# Check first parameter type annotation if present
	first_param_annotation = sig.parameters["grid_shape"].annotation
	if first_param_annotation == inspect.Parameter.empty:
		err_msg = (
			f"Maze generator function '{func_name}' must have a type annotation for 'grid_shape'. "
			"Please add `grid_shape: Coord | CoordTup` to the function signature."
		)

	# Check if it's a Union type
	if get_origin(first_param_annotation) in (Union, type(Union[int, str])):
		args: tuple = get_args(first_param_annotation)
		# Check all of the args look like Coord or CoordTup
		for arg in args:
			_check_grid_shape_annotation(arg, func_name)
	else:
		# Check if the annotation is a single type
		_check_grid_shape_annotation(first_param_annotation, func_name)

	# Check return type annotation - must be present and correct
	if sig.return_annotation == inspect.Signature.empty:
		err_msg = (
			f"Maze generator function '{func_name}' must have a return type annotation "
			"of LatticeMaze. Please add `-> LatticeMaze` to the function signature."
		)
		raise MazeGeneratorRegistrationError(err_msg)

	if sig.return_annotation != LatticeMaze:
		err_msg = (
			f"Maze generator function '{func_name}' must return LatticeMaze, "
			f"got return type annotation: {sig.return_annotation}. "
			"Please ensure the function returns a valid LatticeMaze instance."
		)
		raise MazeGeneratorRegistrationError(err_msg)


def register_maze_generator(func: F_MazeGeneratorFunc) -> F_MazeGeneratorFunc:
	"""Decorator to register a custom maze generator function.

	This decorator allows users to register their own maze generation functions
	without modifying the core library code. The registered function will be:
	1. Added to `GENERATORS_MAP`
	2. Added as a static method to `LatticeMazeGenerators` (for compatibility)

	# NOTE:
	In general, you should avoid using this decorator! instead, just add your function
	to the GENERATORS_MAP dictionary directly and to `LatticeMazeGenerators` as a static method.
	If you add a new function, please make a pull request!
	https://github.com/understanding-search/maze-dataset/pulls

	# Usage:
	```python
	@register_maze_generator
	def gen_my_custom(
		grid_shape: Coord | CoordTup,
		**kwargs, # this can be anything you like
	) -> LatticeMaze:
		# Your custom maze generation logic here
		connection_list = ...  # Create your maze structure

		# Important: If your maze is not fully connected, you must include
		# `visited_cells` in `generation_meta`, or mark it as `fully_connected=True`
		return LatticeMaze(
			connection_list=connection_list,
			generation_meta=dict(
				func_name="gen_my_custom",
				grid_shape=np.array(grid_shape),
				fully_connected=True,  # or provide visited_cells
			),
		)
	```

	note that to properly create **or load** a maze dataset with your custom generator,
	you must be importing the file in which you register the generator

	# Returns:
	The decorated function, unchanged.
	"""
	# Validate function signature
	validate_MazeGeneratorFunc(func)

	# Check if name already exists
	func_name: str = func.__name__
	if func_name in GENERATORS_MAP:
		err_msg = (
			f"Generator with name '{func_name}' already exists in GENERATORS_MAP. "
			"Please choose a different name."
		)
		raise ValueError(err_msg)

	# Register the function
	GENERATORS_MAP[func_name] = func

	# Also add as a static method to LatticeMazeGenerators
	setattr(LatticeMazeGenerators, func_name, staticmethod(func))

	return func

"""Plot a maze as SVG with rounded corners."""

from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np

# Known color map (excluding walls).
COLOR_MAP: dict[tuple[int, int, int], str] = {
	(255, 255, 255): "#f0f0f0",
	(0, 255, 0): "#4caf50",
	(255, 0, 0): "#f44336",
	(0, 0, 255): "#2196f3",
}

WALL_COLOR_HEX: str = "#222"  # (0,0,0) in hex
WALL_RGB: tuple[int, int, int] = (0, 0, 0)

# Offsets in the order [top, right, bottom, left]
_NEIGHBORS: np.ndarray = np.array(
	[
		[-1, 0],  # top
		[0, +1],  # right
		[+1, 0],  # bottom
		[0, -1],  # left
	],
	dtype=int,
)


def is_wall(y: int, x: int, grid: np.ndarray) -> bool:
	"""True if (y, x) is out of bounds or has the wall color."""
	h, w, _ = grid.shape
	if not (0 <= y < h and 0 <= x < w):
		return True
	return bool((grid[y, x] == WALL_RGB).all())


def create_tile_path(
	origin: tuple[float, float],
	tile_size: float,
	corner_radius: float,
	edges: tuple[bool, bool, bool, bool],
) -> str:
	"""Generate an SVG path for a tile at `origin` with side length `tile_size`.

	`edges` is (top, right, bottom, left) booleans, where True means that edge
	borders a wall/outside. If both edges meeting at a corner are True and
	corner_radius>0, we draw a rounded corner; else it's a sharp corner.

	Corner order (clockwise):
		c0 = top-left
		c1 = top-right
		c2 = bottom-right
		c3 = bottom-left

	edges = (top, right, bottom, left).
		corner c0 is formed by edges top + left => edges[0] & edges[3]
		corner c1 => top + right => edges[0] & edges[1]
		corner c2 => right + bottom => edges[1] & edges[2]
		corner c3 => bottom + left => edges[2] & edges[3]
	"""
	x0, y0 = origin
	top, right, bottom, left = edges

	# A corner is "exposed" if both adjoining edges are True
	c0_exposed: bool = top and left  # top-left
	c1_exposed: bool = top and right  # top-right
	c2_exposed: bool = right and bottom  # bottom-right
	c3_exposed: bool = bottom and left  # bottom-left

	# If corner_radius=0, arcs become straight lines.
	r: float = corner_radius

	# We'll construct the path in a standard top-left -> top-right -> bottom-right -> bottom-left order.
	path_cmds = []
	# Move to top-left corner, possibly offset if c0 is exposed
	# (meaning both top and left edges are external).
	start_x = x0 + (r if c0_exposed else 0)
	start_y = y0
	path_cmds.append(f"M {start_x},{start_y}")

	# === TOP edge to top-right corner
	end_x = x0 + tile_size - (r if c1_exposed else 0)
	end_y = y0
	path_cmds.append(f"L {end_x},{end_y}")
	# Arc if c1_exposed
	if c1_exposed and r > 0:
		path_cmds.append(f"A {r} {r} 0 0 1 {x0 + tile_size},{y0 + r}")

	# === RIGHT edge to bottom-right corner
	path_cmds.append(f"L {x0 + tile_size},{y0 + tile_size - (r if c2_exposed else 0)}")
	if c2_exposed and r > 0:
		path_cmds.append(f"A {r} {r} 0 0 1 {x0 + tile_size - r},{y0 + tile_size}")

	# === BOTTOM edge to bottom-left corner
	path_cmds.append(f"L {x0 + (r if c3_exposed else 0)},{y0 + tile_size}")
	if c3_exposed and r > 0:
		path_cmds.append(f"A {r} {r} 0 0 1 {x0},{y0 + tile_size - r}")

	# === LEFT edge back up to top-left corner
	path_cmds.append(f"L {x0},{y0 + (r if c0_exposed else 0)}")
	if c0_exposed and r > 0:
		path_cmds.append(f"A {r} {r} 0 0 1 {x0 + r},{y0}")

	path_cmds.append("Z")
	return " ".join(path_cmds)


def plot_svg_fancy(
	pixel_grid: np.ndarray,
	size: int = 40,
	corner_radius: float = 8.0,
	bounding_corner_radius: float = 20.0,
) -> str:
	"""plot the output of SolvedMaze(...).as_pixels() as a nice svg

	Create an SVG with:
	- A single rounded-square background (walls).
	- Each non-wall cell is drawn via create_tile_path, with corner_radius controlling
		whether corners are rounded. (Set corner_radius=0 for squares.)

	# Parameters:
	- `pixel_grid : np.ndarray`
		3D array of shape (h, w, 3) with RGB values
	- `size : int`
		Size (in px) of each grid cell
	- `corner_radius : float`
		Radius for rounding corners of each tile (0 => squares)
	- `bounding_corner_radius : float`
		Radius for rounding the outer bounding rectangle

	# Returns:
	`str`: A pretty-printed SVG string
	"""
	h, w, _ = pixel_grid.shape

	# Create the root <svg>
	svg = Element(
		"svg",
		xmlns="http://www.w3.org/2000/svg",
		width=str(w * size),
		height=str(h * size),
		viewBox=f"0 0 {w * size} {h * size}",
	)

	# Single rounded-square background for the walls
	SubElement(
		svg,
		"rect",
		{
			"x": "0",
			"y": "0",
			"width": str(w * size),
			"height": str(h * size),
			"fill": WALL_COLOR_HEX,
			"rx": str(bounding_corner_radius),
			"ry": str(bounding_corner_radius),
		},
	)

	for yy in range(h):
		for xx in range(w):
			rgb_tuple = tuple(pixel_grid[yy, xx])
			if rgb_tuple == WALL_RGB:
				# It's a wall => skip (already covered by background)
				continue

			fill_color: str | None = COLOR_MAP.get(rgb_tuple, None)  # noqa: SIM910
			if fill_color is None:
				# Unknown color => skip or handle differently
				continue

			# Check which edges are "external" => next cell is wall
			# edges in the order (top, right, bottom, left)
			edges_bool = [
				is_wall(yy + dy, xx + dx, pixel_grid) for (dy, dx) in _NEIGHBORS
			]

			d_path = create_tile_path(
				origin=(xx * size, yy * size),
				tile_size=size,
				corner_radius=corner_radius,
				edges=tuple(edges_bool),  # type: ignore[arg-type]
			)

			SubElement(
				svg,
				"path",
				{
					"d": d_path,
					"fill": fill_color,
					"stroke": "none",
				},
			)

	raw_svg = tostring(svg, encoding="unicode")
	# we are in charge of the svg so it's safe to decode
	return parseString(raw_svg).toprettyxml(indent="  ")  # noqa: S318

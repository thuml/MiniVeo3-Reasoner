"""provides `MazePlot`, which has many tools for plotting mazes with multiple paths, colored nodes, and more"""

from __future__ import annotations  # for type hinting self as return value

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Bool, Float

from maze_dataset.constants import Coord, CoordArray, CoordList
from maze_dataset.maze import (
	LatticeMaze,
	SolvedMaze,
	TargetedLatticeMaze,
)

LARGE_NEGATIVE_NUMBER: float = -1e10


@dataclass(kw_only=True)
class PathFormat:
	"""formatting options for path plot"""

	label: str | None = None
	fmt: str = "o"
	color: str | None = None
	cmap: str | None = None
	line_width: float | None = None
	quiver_kwargs: dict | None = None

	def combine(self, other: PathFormat) -> PathFormat:
		"""combine with other PathFormat object, overwriting attributes with non-None values.

		returns a modified copy of self.
		"""
		output: PathFormat = deepcopy(self)
		for key, value in other.__dict__.items():
			if key == "path":
				err_msg: str = f"Cannot overwrite path attribute! {self = }, {other = }"
				raise ValueError(
					err_msg,
				)
			if value is not None:
				setattr(output, key, value)

		return output


# styled path
@dataclass
class StyledPath(PathFormat):
	"a `StyledPath` is a `PathFormat` with a specific path"

	path: CoordArray


DEFAULT_FORMATS: dict[str, PathFormat] = {
	"true": PathFormat(
		label="true path",
		fmt="--",
		color="red",
		line_width=2.5,
		quiver_kwargs=None,
	),
	"predicted": PathFormat(
		label=None,
		fmt=":",
		color=None,
		line_width=2,
		quiver_kwargs={"width": 0.015},
	),
}


def process_path_input(
	path: CoordList | CoordArray | StyledPath,
	_default_key: str,
	path_fmt: PathFormat | None = None,
	**kwargs,
) -> StyledPath:
	"convert a path, which might be a list or array of coords, into a `StyledPath`"
	styled_path: StyledPath
	if isinstance(path, StyledPath):
		styled_path = path
	elif isinstance(path, np.ndarray):
		styled_path = StyledPath(path=path)
		# add default formatting
		styled_path = styled_path.combine(DEFAULT_FORMATS[_default_key])
	elif isinstance(path, list):
		styled_path = StyledPath(path=np.array(path))
		# add default formatting
		styled_path = styled_path.combine(DEFAULT_FORMATS[_default_key])
	else:
		err_msg: str = (
			f"Expected CoordList, CoordArray or StyledPath, got {type(path)}: {path}"
		)
		raise TypeError(
			err_msg,
		)

	# add formatting from path_fmt
	if path_fmt is not None:
		styled_path = styled_path.combine(path_fmt)

	# add formatting from kwargs
	for key, value in kwargs.items():
		setattr(styled_path, key, value)

	return styled_path


DEFAULT_PREDICTED_PATH_COLORS: list[str] = [
	"tab:orange",
	"tab:olive",
	"sienna",
	"mediumseagreen",
	"tab:purple",
	"slategrey",
]


class MazePlot:
	"""Class for displaying mazes and paths"""

	def __init__(self, maze: LatticeMaze, unit_length: int = 14) -> None:
		"""UNIT_LENGTH: Set ratio between node size and wall thickness in image.

		Wall thickness is fixed to 1px
		A "unit" consists of a single node and the right and lower connection/wall.
		Example: ul = 14 yields 13:1 ratio between node size and wall thickness
		"""
		self.unit_length: int = unit_length
		self.maze: LatticeMaze = maze
		self.true_path: StyledPath | None = None
		self.predicted_paths: list[StyledPath] = []
		self.node_values: Float[np.ndarray, "grid_n grid_n"] = None
		self.custom_node_value_flag: bool = False
		self.node_color_map: str = "Blues"
		self.target_token_coord: Coord = None
		self.preceding_tokens_coords: CoordArray = None
		self.colormap_center: float | None = None
		self.cbar_ax = None
		self.marked_coords: list[tuple[Coord, dict]] = list()

		self.marker_kwargs_current: dict = dict(
			marker="s",
			color="green",
			ms=12,
		)
		self.marker_kwargs_next: dict = dict(
			marker="P",
			color="green",
			ms=12,
		)

		if isinstance(maze, SolvedMaze):
			self.add_true_path(maze.solution)
		else:
			if isinstance(maze, TargetedLatticeMaze):
				self.add_true_path(SolvedMaze.from_targeted_lattice_maze(maze).solution)
    
		import matplotlib.image as mpimg
		self.icon_orig=mpimg.imread('icons/orig.png')
		self.icon_flag=mpimg.imread('icons/flag.png')
		self.icon_star=mpimg.imread('icons/star.png')
		self.icon_trophy=mpimg.imread('icons/trophy.png')
	@property
	def solved_maze(self) -> SolvedMaze:
		"get the underlying `SolvedMaze` object"
		if self.true_path is None:
			raise ValueError(
				"Cannot return SolvedMaze object without true path. Add true path with add_true_path method.",
			)
		return SolvedMaze.from_lattice_maze(
			lattice_maze=self.maze,
			solution=self.true_path.path,
		)

	def add_true_path(
		self,
		path: CoordList | CoordArray | StyledPath,
		path_fmt: PathFormat | None = None,
		**kwargs,
	) -> MazePlot:
		"add a true path to the maze with optional formatting"
		self.true_path = process_path_input(
			path=path,
			_default_key="true",
			path_fmt=path_fmt,
			**kwargs,
		)

		return self

	def add_predicted_path(
		self,
		path: CoordList | CoordArray | StyledPath,
		path_fmt: PathFormat | None = None,
		**kwargs,
	) -> MazePlot:
		"""Recieve predicted path and formatting preferences from input and save in predicted_path list.

		Default formatting depends on nuber of paths already saved in predicted path list.
		"""
		styled_path: StyledPath = process_path_input(
			path=path,
			_default_key="predicted",
			path_fmt=path_fmt,
			**kwargs,
		)

		# set default label and color if not specified
		if styled_path.label is None:
			styled_path.label = f"predicted path {len(self.predicted_paths) + 1}"

		if styled_path.color is None:
			color_num: int = len(self.predicted_paths) % len(
				DEFAULT_PREDICTED_PATH_COLORS,
			)
			styled_path.color = DEFAULT_PREDICTED_PATH_COLORS[color_num]

		self.predicted_paths.append(styled_path)
		return self

	def add_multiple_paths(
		self,
		path_list: Sequence[CoordList | CoordArray | StyledPath],
	) -> MazePlot:
		"""Function for adding multiple paths to MazePlot at once.

		> DOCS: what are the two ways?
		This can be done in two ways:
		1. Passing a list of
		"""
		for path in path_list:
			self.add_predicted_path(path)
		return self

	def add_node_values(
		self,
		node_values: Float[np.ndarray, "grid_n grid_n"],
		color_map: str = "Blues",
		target_token_coord: Coord | None = None,
		preceeding_tokens_coords: CoordArray = None,
		colormap_center: float | None = None,
		colormap_max: float | None = None,
		hide_colorbar: bool = False,
	) -> MazePlot:
		"""add node values to the maze for visualization as a heatmap

		> DOCS: what are these arguments?

		# Parameters:
		- `node_values : Float[np.ndarray, &quot;grid_n grid_n&quot;]`
		- `color_map : str`
			(defaults to `"Blues"`)
		- `target_token_coord : Coord | None`
			(defaults to `None`)
		- `preceeding_tokens_coords : CoordArray`
			(defaults to `None`)
		- `colormap_center : float | None`
			(defaults to `None`)
		- `colormap_max : float | None`
			(defaults to `None`)
		- `hide_colorbar : bool`
			(defaults to `False`)

		# Returns:
		- `MazePlot`
		"""
		assert node_values.shape == self.maze.grid_shape, (
			"Please pass node values of the same sape as LatticeMaze.grid_shape"
		)
		# assert np.min(node_values) >= 0, "Please pass non-negative node values only."

		self.node_values = node_values
		# Set flag for choosing cmap while plotting maze
		self.custom_node_value_flag = True
		# Retrieve Max node value for plotting, +1e-10 to avoid division by zero
		self.node_color_map = color_map
		self.colormap_center = colormap_center
		self.colormap_max = colormap_max
		self.hide_colorbar = hide_colorbar

		if target_token_coord is not None:
			self.marked_coords.append((target_token_coord, self.marker_kwargs_next))
		if preceeding_tokens_coords is not None:
			for coord in preceeding_tokens_coords:
				self.marked_coords.append((coord, self.marker_kwargs_current))
		return self

	# Add a continuous route plot
	def plot_continuous(
		self,
		frames_num=81,
		fig_size_pixel=(832,480),
  		dpi: int = 100,
		title: str = "",
		fig_ax: tuple | None = None,
		plain: bool = False,
  
	):
		plot_len=len(self.true_path.path)
		frames=[]

		save_truepath=list(self.true_path.path)
		path_continuous=[]
		frames_add=frames_num-plot_len
		if frames_add<0:
			frames_add=0

		def split_into_parts(n, k):
			"""将n分成k个尽可能均匀的部分"""
			base = n // k
			remainder = n % k
			return [base + 1 if i < remainder else base for i in range(k)]
		intervals=split_into_parts(frames_add,plot_len-1)
  
		for i in range(plot_len-1):
			path_continuous.append(self.true_path.path[i])
			intersects=intervals[i]
   
			for j in range(intersects):
				inter_x=(self.true_path.path[i+1][0]-self.true_path.path[i][0])/(intersects+1)*(j+1)+self.true_path.path[i][0]
				inter_y=(self.true_path.path[i+1][1]-self.true_path.path[i][1])/(intersects+1)*(j+1)+self.true_path.path[i][1]
				path_continuous.append((inter_x,inter_y))
		path_continuous.append(self.true_path.path[-1])

		

		for i in range(len(path_continuous)):
			
			# set up figure
			if fig_ax is None:
				self.frame = plt.figure(figsize=(fig_size_pixel[0]/dpi,fig_size_pixel[1]/dpi), dpi=dpi)
				self.ax = self.frame.add_subplot(1, 1, 1)
			else:
				self.frame, self.ax = fig_ax

			# plot maze
			self._plot_maze()

			# Plot labels
			if not plain:
				tick_arr = np.arange(self.maze.grid_shape[0])
				self.ax.set_xticks(self.unit_length * (tick_arr + 0.5), tick_arr)
				self.ax.set_yticks(self.unit_length * (tick_arr + 0.5), tick_arr)
				self.ax.set_xlabel("col")
				self.ax.set_ylabel("row")
				self.ax.set_title(title)
			else:
				self.ax.set_xticks([])
				self.ax.set_yticks([])
				self.ax.set_xlabel("")
				self.ax.set_ylabel("")
				self.ax.axis("off")
    
			adapt_path=StyledPath(path=np.array(path_continuous[:i+1]))
			adapt_path=adapt_path.combine(DEFAULT_FORMATS["true"])
			if len(adapt_path.path)>0:
				self._plot_path(adapt_path,path_continuous[-1],self.unit_length*0.8)
			for coord, kwargs in self.marked_coords:
				self._place_marked_coords([coord], **kwargs)
			
			frames.append(self.frame)
			plt.close(self.frame)
   
		return frames

	def plot(
		self,
		dpi: int = 100,
		title: str = "",
		fig_ax: tuple | None = None,
		plain: bool = False,
	) -> MazePlot:
		"""Plot the maze and paths."""
		# set up figure
		if fig_ax is None:
			self.fig = plt.figure(dpi=dpi)
			self.ax = self.fig.add_subplot(1, 1, 1)
		else:
			self.fig, self.ax = fig_ax

		# plot maze
		self._plot_maze()

		# Plot labels
		if not plain:
			tick_arr = np.arange(self.maze.grid_shape[0])
			self.ax.set_xticks(self.unit_length * (tick_arr + 0.5), tick_arr)
			self.ax.set_yticks(self.unit_length * (tick_arr + 0.5), tick_arr)
			self.ax.set_xlabel("col")
			self.ax.set_ylabel("row")
			self.ax.set_title(title)
		else:
			self.ax.set_xticks([])
			self.ax.set_yticks([])
			self.ax.set_xlabel("")
			self.ax.set_ylabel("")
			self.ax.axis("off")

		# plot paths
  
		adapt_path=StyledPath(self.true_path)
		save_path=list(self.true_path.path)
		print(self.true_path)
		# adapt_path.path.path=adapt_path.path.path[:2]
		
		if self.true_path is not None:
			self._plot_path(adapt_path.path,save_path[-1])
		#for path in self.predicted_paths:
		#	self._plot_path(path)

		# plot markers
		for coord, kwargs in self.marked_coords:
			self._place_marked_coords([coord], **kwargs)

		return self

	def _rowcol_to_coord(self, point: Coord) -> np.ndarray:
		"""Transform Point from MazeTransformer (row, column) notation to matplotlib default (x, y) notation where x is the horizontal axis."""
		point = np.array([point[1], point[0]])
		return self.unit_length * (point + 0.5)

	def mark_coords(self, coords: CoordArray | list[Coord], **kwargs) -> MazePlot:
		"""Mark coordinates on the maze with a marker.

		default marker is a blue "+":
		`dict(marker="+", color="blue")`
		"""
		kwargs = {
			**dict(marker="+", color="blue"),
			**kwargs,
		}
		for coord in coords:
			self.marked_coords.append((coord, kwargs))

		return self

	def _place_marked_coords(
		self,
		coords: CoordArray | list[Coord],
		**kwargs,
	) -> MazePlot:
		coords_tp = np.array([self._rowcol_to_coord(coord) for coord in coords])
		self.ax.plot(coords_tp[:, 0], coords_tp[:, 1], **kwargs)

		return self

	def _plot_maze(self) -> None:  # noqa: C901, PLR0912
		"""Define Colormap and plot maze.

		Colormap: x is -inf: black
		else: use colormap
		"""
		img = self._lattice_maze_to_img()

		# if no node_values have been passed (no colormap)
		if self.custom_node_value_flag is False:
			self.ax.imshow(img, cmap="gray", vmin=-1, vmax=1,zorder=1)

		else:
			assert self.node_values is not None, "Please pass node values."
			assert not np.isnan(self.node_values).any(), (
				"Please pass node values, they cannot be nan."
			)

			vals_min: float = np.nanmin(self.node_values)
			vals_max: float = np.nanmax(self.node_values)
			# if both are negative or both are positive, set max/min to 0
			if vals_max < 0.0:
				vals_max = 0.0
			elif vals_min > 0.0:
				vals_min = 0.0

			# adjust vals_max, in case you need consistent colorbar across multiple plots
			vals_max = self.colormap_max or vals_max

			self.ax.set_xlim(vals_min,vals_max)
			self.ax.set_ylim(vals_min,vals_max)

			# create colormap
			cmap = mpl.colormaps[self.node_color_map]
			# TODO: this is a hack, we make the walls black (while still allowing negative values) by setting the nan color to black
			cmap.set_bad(color="black")

			if self.colormap_center is not None:
				if not (vals_min < self.colormap_center < vals_max):
					if vals_min == self.colormap_center:
						vals_min -= 1e-10
					elif vals_max == self.colormap_center:
						vals_max += 1e-10
					else:
						err_msg: str = f"Please pass colormap_center value between {vals_min} and {vals_max}"
						raise ValueError(
							err_msg,
						)

				norm = mpl.colors.TwoSlopeNorm(
					vmin=vals_min,
					vcenter=self.colormap_center,
					vmax=vals_max,
				)
				_plotted = self.ax.imshow(img, cmap=cmap, norm=norm)
			else:
				_plotted = self.ax.imshow(img, cmap=cmap, vmin=vals_min, vmax=vals_max)

			# Add colorbar based on the condition of self.hide_colorbar
			if not self.hide_colorbar:
				ticks = np.linspace(vals_min, vals_max, 5)

				if (vals_min < 0.0 < vals_max) and (0.0 not in ticks):
					ticks = np.insert(ticks, np.searchsorted(ticks, 0.0), 0.0)

				if (
					self.colormap_center is not None
					and self.colormap_center not in ticks
					and vals_min < self.colormap_center < vals_max
				):
					ticks = np.insert(
						ticks,
						np.searchsorted(ticks, self.colormap_center),
						self.colormap_center,
					)

				cbar = plt.colorbar(
					_plotted,
					ticks=ticks,
					ax=self.ax,
					cax=self.cbar_ax,
				)
				self.cbar_ax = cbar.ax

		# make the boundaries of the image thicker (walls look weird without this)
		for axis in ["top", "bottom", "left", "right"]:
			self.ax.spines[axis].set_linewidth(2)

	def _lattice_maze_to_img(
		self,
		connection_val_scale: float = 0.93,
	) -> Bool[np.ndarray, "row col"]:
		"""Build an image to visualise the maze.

		Each "unit" consists of a node and the right and lower adjacent wall/connection. Its area is ul * ul.
		- Nodes have area: (ul-1) * (ul-1) and value 1 by default
			- take node_value if passed via .add_node_values()
		- Walls have area: 1 * (ul-1) and value -1
		- Connections have area: 1 * (ul-1); color and value 0.93 by default
			- take node_value if passed via .add_node_values()

		Axes definition:
		(0,0)     col
		----|----------->
			|
		row |
			|
			v

		Returns a matrix of side length (ul) * n + 1 where n is the number of nodes.
		"""
		# TODO: this is a hack, but if you add 1 always then non-node valued plots have their walls dissapear. if you dont add 1, you get ugly colors between nodes when they are colored
		node_bdry_hack: int
		connection_list_processed: Float[np.ndarray, "dim row col"]
		# Set node and connection values
		if self.node_values is None:
			scaled_node_values = np.ones(self.maze.grid_shape)
			connection_values = scaled_node_values * connection_val_scale
			node_bdry_hack = 0
			# TODO: hack
			# invert connection list
			connection_list_processed = np.logical_not(self.maze.connection_list)
		else:
			# TODO: hack
			scaled_node_values = self.node_values
			# connection_values = scaled_node_values
			connection_values = np.full_like(scaled_node_values, np.nan)
			node_bdry_hack = 1
			connection_list_processed = self.maze.connection_list

		# Create background image (all pixels set to -1, walls everywhere)
		img: Float[np.ndarray, "row col"] = -np.ones(
			(
				self.maze.grid_shape[0] * self.unit_length + 1,
				self.maze.grid_shape[1] * self.unit_length + 1,
			),
			dtype=float,
		)

		# Draw nodes and connections by iterating through lattice
		for row in range(self.maze.grid_shape[0]):
			for col in range(self.maze.grid_shape[1]):
				# Draw node
				img[
					row * self.unit_length + 1 : (row + 1) * self.unit_length
					+ node_bdry_hack,
					col * self.unit_length + 1 : (col + 1) * self.unit_length
					+ node_bdry_hack,
				] = scaled_node_values[row, col]

				# Down connection
				if not connection_list_processed[0, row, col]:
					img[
						(row + 1) * self.unit_length,
						col * self.unit_length + 1 : (col + 1) * self.unit_length,
					] = connection_values[row, col]

				# Right connection
				if not connection_list_processed[1, row, col]:
					img[
						row * self.unit_length + 1 : (row + 1) * self.unit_length,
						(col + 1) * self.unit_length,
					] = connection_values[row, col]

		return img

	def _plot_path(self, path_format: PathFormat, target=None, grid_size=None) -> None:
		if len(path_format.path) == 0:
			warnings.warn(f"Empty path, skipping plotting\n{path_format = }")
			return
		p_transformed = np.array(
			[self._rowcol_to_coord(coord) for coord in path_format.path],
		)
		if target is not None:
			target_transformed=self._rowcol_to_coord(target)

		need_route = False
  
		if need_route == False:
			pass
		else:
			if path_format.quiver_kwargs is not None:
				try:
					x: np.ndarray = p_transformed[:, 0]
					y: np.ndarray = p_transformed[:, 1]
				except Exception as e:
					err_msg: str = f"Error in plotting quiver path:\n{path_format = }\n{p_transformed = }\n{e}"
					raise ValueError(
						err_msg,
					) from e

				# Generate colors from the colormap
				if path_format.cmap is not None:
					n = len(x) - 1  # Number of arrows
					cmap = plt.get_cmap(path_format.cmap)
					colors = [cmap(i / n) for i in range(n)]
				else:
					colors = path_format.color

				self.ax.quiver(
					x[:-1],
					y[:-1],
					x[1:] - x[:-1],
					y[1:] - y[:-1],
					scale_units="xy",
					angles="xy",
					scale=1,
					color=colors,
					**path_format.quiver_kwargs,
				)
			else:
				self.ax.plot(
					p_transformed[:, 0],
					p_transformed[:, 1],
					path_format.fmt,
					lw=path_format.line_width,
					color=path_format.color,
					label=path_format.label,
				)
		# mark endpoints
		# self.ax.plot(
		# 	[p_transformed[0][0]],
		# 	[p_transformed[0][1]],
		# 	"o",
		# 	color=path_format.color,
		# 	ms=10,
		# )
		if grid_size is None:
			return

		def calc_extent(pos):
			return list([pos[0]-grid_size/2,pos[0]+grid_size/2,pos[1]-grid_size/2,pos[1]+grid_size/2])
		# rewrite logic
  
		tar_pos=target_transformed
		cur_pos=[p_transformed[-1][0],p_transformed[-1][1]]
		orig_pos=[p_transformed[0][0],p_transformed[0][1]]
  
		self.ax.imshow(self.icon_orig,extent=[orig_pos[0]-grid_size/2,orig_pos[0]+grid_size/2,orig_pos[1]-grid_size/2,orig_pos[1]+grid_size/2],zorder=2)
		# self.ax.imshow(self.icon_orig,extent=calc_extent(orig_pos))

		def arr_eq(arr1,arr2):
			return arr1[0]==arr2[0] and arr1[1]==arr2[1]

		if arr_eq(cur_pos,orig_pos):
			# self.ax.imshow(self.icon_flag,extent=calc_extent(tar_pos))
   			self.ax.imshow(self.icon_flag,extent=[tar_pos[0]-grid_size/2,tar_pos[0]+grid_size/2,tar_pos[1]-grid_size/2,tar_pos[1]+grid_size/2],zorder=2)
		elif arr_eq(cur_pos,tar_pos):
			# self.ax.imshow(self.icon_trophy,extent=calc_extent(tar_pos))
   			self.ax.imshow(self.icon_trophy,extent=[tar_pos[0]-grid_size/2,tar_pos[0]+grid_size/2,tar_pos[1]-grid_size/2,tar_pos[1]+grid_size/2],zorder=2)
		else:
			# self.ax.imshow(self.icon_flag,extent=calc_extent(tar_pos))
   			self.ax.imshow(self.icon_flag,extent=[tar_pos[0]-grid_size/2,tar_pos[0]+grid_size/2,tar_pos[1]-grid_size/2,tar_pos[1]+grid_size/2],zorder=2)
			# self.ax.imshow(self.icon_star,extent=calc_extent(cur_pos))
   			self.ax.imshow(self.icon_star,extent=[cur_pos[0]-grid_size/2,cur_pos[0]+grid_size/2,cur_pos[1]-grid_size/2,cur_pos[1]+grid_size/2],zorder=2)
			
  
		# if target is not None:
		# 	self.ax.plot(
		# 		[target_transformed[0]],
		# 		[target_transformed[1]],
		# 		"x",
		# 		color=path_format.color,
		# 		ms=10,
		# 		zorder=1
		# 	)
		
		# else:
		# 	self.ax.plot(
		# 		[p_transformed[-1][0]],
		# 		[p_transformed[-1][1]],
		# 		"x",
		# 		color=path_format.color,
		# 		ms=10,
		# 		zorder=1
		# 	)
		self.ax.plot(
			[p_transformed[-1][0]],
			[p_transformed[-1][1]],
			"*",
			color='gold' if p_transformed[-1][0]==target_transformed[0] and p_transformed[-1][1]==target_transformed[1] else 'blue',
			ms=10,
			zorder=1
		) ## this is to keep the show correct idk why 

	def to_ascii(
		self,
		show_endpoints: bool = True,
		show_solution: bool = True,
	) -> str:
		"wrapper for `self.solved_maze.as_ascii()`, shows the path if we have `self.true_path`"
		if self.true_path:
			return self.solved_maze.as_ascii(
				show_endpoints=show_endpoints,
				show_solution=show_solution,
			)
		else:
			return self.maze.as_ascii(show_endpoints=show_endpoints)

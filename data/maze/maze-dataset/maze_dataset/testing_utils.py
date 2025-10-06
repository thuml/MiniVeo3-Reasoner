"""Shared utilities for tests only.

Do not import into any module outside of the tests directory
"""

import itertools
from typing import Final, NamedTuple, Sequence

import frozendict
import numpy as np

from maze_dataset import (
	CoordArray,
	LatticeMaze,
	LatticeMazeGenerators,
	MazeDataset,
	MazeDatasetConfig,
	SolvedMaze,
	TargetedLatticeMaze,
)
from maze_dataset.tokenization import (
	MazeTokenizer,
	MazeTokenizerModular,
	TokenizationMode,
)

GRID_N: Final[int] = 5
N_MAZES: Final[int] = 5
CFG: Final[MazeDatasetConfig] = MazeDatasetConfig(
	name="test",
	grid_n=GRID_N,
	n_mazes=N_MAZES,
	maze_ctor=LatticeMazeGenerators.gen_dfs,
)
MAZE_DATASET: Final[MazeDataset] = MazeDataset.from_config(
	CFG,
	do_download=False,
	load_local=False,
	do_generate=True,
	save_local=False,
	verbose=True,
	gen_parallel=False,
)
LATTICE_MAZES: Final[tuple[LatticeMaze, ...]] = tuple(
	LatticeMazeGenerators.gen_dfs(np.array([GRID_N, GRID_N])) for _ in range(N_MAZES)
)
_PATHS = tuple(maze.generate_random_path() for maze in LATTICE_MAZES)
TARGETED_MAZES: Final[tuple[TargetedLatticeMaze, ...]] = tuple(
	TargetedLatticeMaze.from_lattice_maze(maze, path[0], path[-1])
	for maze, path in zip(LATTICE_MAZES, _PATHS, strict=False)
)
# MIXED_MAZES alternates the maze types, so you can slice a contiguous subset and still get all types
MIXED_MAZES: Final[tuple[LatticeMaze | TargetedLatticeMaze | SolvedMaze, ...]] = tuple(
	x
	for x in itertools.chain.from_iterable(
		itertools.zip_longest(MAZE_DATASET.mazes, TARGETED_MAZES, LATTICE_MAZES),
	)
)


class MANUAL_MAZE(NamedTuple):  # noqa: N801
	"""A named tuple for manual maze definitions"""

	tokens: str
	ascii: Sequence[str]
	straightaway_footprints: CoordArray


ASCII_MAZES: Final[frozendict.frozendict[str, MANUAL_MAZE]] = frozendict.frozendict(
	small_3x3=MANUAL_MAZE(
		tokens="<ADJLIST_START> (2,0) <--> (2,1) ; (0,0) <--> (0,1) ; (0,0) <--> (1,0) ; (0,2) <--> (1,2) ; (1,0) <--> (2,0) ; (0,2) <--> (0,1) ; (2,2) <--> (2,1) ; (1,1) <--> (2,1) ; <ADJLIST_END> <ORIGIN_START> (0,0) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (0,0) (1,0) (2,0) (2,1) <PATH_END>",
		ascii=(
			"#######",
			"#S    #",
			"#X### #",
			"#X# # #",
			"#X# ###",
			"#XXE  #",
			"#######",
		),
		straightaway_footprints=np.array(
			[
				[0, 0],
				[2, 0],
				[2, 1],
			],
		),
	),
	big_10x10=MANUAL_MAZE(
		tokens="<ADJLIST_START> (8,2) <--> (8,3) ; (3,7) <--> (3,6) ; (6,7) <--> (6,8) ; (4,6) <--> (5,6) ; (9,5) <--> (9,4) ; (3,3) <--> (3,4) ; (5,1) <--> (4,1) ; (2,6) <--> (2,7) ; (8,5) <--> (8,4) ; (1,9) <--> (2,9) ; (4,1) <--> (4,2) ; (0,8) <--> (0,7) ; (5,4) <--> (5,3) ; (6,3) <--> (6,4) ; (5,0) <--> (4,0) ; (5,3) <--> (5,2) ; (3,1) <--> (2,1) ; (9,1) <--> (9,0) ; (3,5) <--> (3,6) ; (5,5) <--> (6,5) ; (7,1) <--> (7,2) ; (0,1) <--> (1,1) ; (7,8) <--> (8,8) ; (3,9) <--> (4,9) ; (4,6) <--> (4,7) ; (0,6) <--> (0,7) ; (3,4) <--> (3,5) ; (6,0) <--> (5,0) ; (7,7) <--> (7,6) ; (1,6) <--> (0,6) ; (6,1) <--> (6,0) ; (8,6) <--> (8,7) ; (9,9) <--> (9,8) ; (1,8) <--> (1,9) ; (2,1) <--> (2,2) ; (9,2) <--> (9,3) ; (5,9) <--> (6,9) ; (3,2) <--> (2,2) ; (0,8) <--> (0,9) ; (5,6) <--> (5,7) ; (2,3) <--> (2,4) ; (4,5) <--> (4,4) ; (8,9) <--> (8,8) ; (9,6) <--> (8,6) ; (3,7) <--> (3,8) ; (8,0) <--> (7,0) ; (6,1) <--> (6,2) ; (0,1) <--> (0,0) ; (7,3) <--> (7,4) ; (9,4) <--> (9,3) ; (9,6) <--> (9,5) ; (8,7) <--> (7,7) ; (5,2) <--> (5,1) ; (0,0) <--> (1,0) ; (7,2) <--> (7,3) ; (2,5) <--> (2,6) ; (4,9) <--> (5,9) ; (5,5) <--> (5,4) ; (5,6) <--> (6,6) ; (7,8) <--> (7,9) ; (1,7) <--> (2,7) ; (4,6) <--> (4,5) ; (1,1) <--> (1,2) ; (3,1) <--> (3,0) ; (1,5) <--> (1,6) ; (8,3) <--> (8,4) ; (9,9) <--> (8,9) ; (8,5) <--> (7,5) ; (1,4) <--> (2,4) ; (3,0) <--> (4,0) ; (3,3) <--> (4,3) ; (6,9) <--> (6,8) ; (1,0) <--> (2,0) ; (6,0) <--> (7,0) ; (8,0) <--> (9,0) ; (2,3) <--> (2,2) ; (2,8) <--> (3,8) ; (5,7) <--> (6,7) ; (1,3) <--> (0,3) ; (9,7) <--> (9,8) ; (7,5) <--> (7,4) ; (1,8) <--> (2,8) ; (6,5) <--> (6,4) ; (0,2) <--> (1,2) ; (0,7) <--> (1,7) ; (0,3) <--> (0,2) ; (4,3) <--> (4,2) ; (5,8) <--> (4,8) ; (9,1) <--> (8,1) ; (9,2) <--> (8,2) ; (1,3) <--> (1,4) ; (2,9) <--> (3,9) ; (4,8) <--> (4,7) ; (0,5) <--> (0,4) ; (8,1) <--> (7,1) ; (0,3) <--> (0,4) ; (9,7) <--> (9,6) ; (7,6) <--> (6,6) ; (1,5) <--> (0,5) ; <ADJLIST_END> <ORIGIN_START> (6,2) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (6,2) (6,1) (6,0) (5,0) (4,0) (3,0) (3,1) (2,1) <PATH_END>",
		ascii=(
			"#####################",
			"#   #       #       #",
			"# # # # ### # # #####",
			"# #   #   #   # #   #",
			"# ####### ##### # # #",
			"# #E      #     # # #",
			"###X# ########### # #",
			"#XXX# #           # #",
			"#X##### ########### #",
			"#X#     #         # #",
			"#X# ######### ### # #",
			"#X#         #   # # #",
			"#X######### # # ### #",
			"#XXXXS#     # #     #",
			"# ########### #######",
			"# #         #   #   #",
			"# # ####### ### # ###",
			"# # #       #   #   #",
			"# # # ####### ##### #",
			"#   #               #",
			"#####################",
		),
		straightaway_footprints=np.array(
			[
				[6, 2],
				[6, 0],
				[3, 0],
				[3, 1],
				[2, 1],
			],
		),
	),
	longer_10x10=MANUAL_MAZE(
		tokens="<ADJLIST_START> (8,2) <--> (8,3) ; (3,7) <--> (3,6) ; (6,7) <--> (6,8) ; (4,6) <--> (5,6) ; (9,5) <--> (9,4) ; (3,3) <--> (3,4) ; (5,1) <--> (4,1) ; (2,6) <--> (2,7) ; (8,5) <--> (8,4) ; (1,9) <--> (2,9) ; (4,1) <--> (4,2) ; (0,8) <--> (0,7) ; (5,4) <--> (5,3) ; (6,3) <--> (6,4) ; (5,0) <--> (4,0) ; (5,3) <--> (5,2) ; (3,1) <--> (2,1) ; (9,1) <--> (9,0) ; (3,5) <--> (3,6) ; (5,5) <--> (6,5) ; (7,1) <--> (7,2) ; (0,1) <--> (1,1) ; (7,8) <--> (8,8) ; (3,9) <--> (4,9) ; (4,6) <--> (4,7) ; (0,6) <--> (0,7) ; (3,4) <--> (3,5) ; (6,0) <--> (5,0) ; (7,7) <--> (7,6) ; (1,6) <--> (0,6) ; (6,1) <--> (6,0) ; (8,6) <--> (8,7) ; (9,9) <--> (9,8) ; (1,8) <--> (1,9) ; (2,1) <--> (2,2) ; (9,2) <--> (9,3) ; (5,9) <--> (6,9) ; (3,2) <--> (2,2) ; (0,8) <--> (0,9) ; (5,6) <--> (5,7) ; (2,3) <--> (2,4) ; (4,5) <--> (4,4) ; (8,9) <--> (8,8) ; (9,6) <--> (8,6) ; (3,7) <--> (3,8) ; (8,0) <--> (7,0) ; (6,1) <--> (6,2) ; (0,1) <--> (0,0) ; (7,3) <--> (7,4) ; (9,4) <--> (9,3) ; (9,6) <--> (9,5) ; (8,7) <--> (7,7) ; (5,2) <--> (5,1) ; (0,0) <--> (1,0) ; (7,2) <--> (7,3) ; (2,5) <--> (2,6) ; (4,9) <--> (5,9) ; (5,5) <--> (5,4) ; (5,6) <--> (6,6) ; (7,8) <--> (7,9) ; (1,7) <--> (2,7) ; (4,6) <--> (4,5) ; (1,1) <--> (1,2) ; (3,1) <--> (3,0) ; (1,5) <--> (1,6) ; (8,3) <--> (8,4) ; (9,9) <--> (8,9) ; (8,5) <--> (7,5) ; (1,4) <--> (2,4) ; (3,0) <--> (4,0) ; (3,3) <--> (4,3) ; (6,9) <--> (6,8) ; (1,0) <--> (2,0) ; (6,0) <--> (7,0) ; (8,0) <--> (9,0) ; (2,3) <--> (2,2) ; (2,8) <--> (3,8) ; (5,7) <--> (6,7) ; (1,3) <--> (0,3) ; (9,7) <--> (9,8) ; (7,5) <--> (7,4) ; (1,8) <--> (2,8) ; (6,5) <--> (6,4) ; (0,2) <--> (1,2) ; (0,7) <--> (1,7) ; (0,3) <--> (0,2) ; (4,3) <--> (4,2) ; (5,8) <--> (4,8) ; (9,1) <--> (8,1) ; (9,2) <--> (8,2) ; (1,3) <--> (1,4) ; (2,9) <--> (3,9) ; (4,8) <--> (4,7) ; (0,5) <--> (0,4) ; (8,1) <--> (7,1) ; (0,3) <--> (0,4) ; (9,7) <--> (9,6) ; (7,6) <--> (6,6) ; (1,5) <--> (0,5) ; <ADJLIST_END> <ORIGIN_START> (6,2) <ORIGIN_END> <TARGET_START> (2,1) <TARGET_END> <PATH_START> (6,2) (6,1) (6,0) (5,0) (4,0) (3,0) (3,1) (2,1) (2,2) (2,3) (2,4) (1,4) (1,3) (0,3) (0,4) (0,5) (1,5) (1,6) (0,6) (0,7) (0,8) <PATH_END>",
		ascii=(
			"#####################",
			"#   #  XXXXX#XXXXE  #",
			"# # # #X###X#X# #####",
			"# #   #XXX#XXX# #   #",
			"# #######X##### # # #",
			"# #XXXXXXX#     # # #",
			"###X# ########### # #",
			"#XXX# #           # #",
			"#X##### ########### #",
			"#X#     #         # #",
			"#X# ######### ### # #",
			"#X#         #   # # #",
			"#X######### # # ### #",
			"#XXXXS#     # #     #",
			"# ########### #######",
			"# #         #   #   #",
			"# # ####### ### # ###",
			"# # #       #   #   #",
			"# # # ####### ##### #",
			"#   #               #",
			"#####################",
		),
		straightaway_footprints=np.array(
			[
				[6, 2],
				[6, 0],
				[3, 0],
				[3, 1],
				[2, 1],
				[2, 4],
				[1, 4],
				[1, 3],
				[0, 3],
				[0, 5],
				[1, 5],
				[1, 6],
				[0, 6],
				[0, 8],
			],
		),
	),
)

# A list of legacy `MazeTokenizer`s and their `MazeTokenizerModular` equivalents.
# Used for unit tests where both versions are supported
LEGACY_AND_EQUIVALENT_TOKENIZERS: list[MazeTokenizer | MazeTokenizerModular] = [
	*[
		MazeTokenizer(tokenization_mode=tok_mode, max_grid_size=20)
		for tok_mode in TokenizationMode
	],
	*[MazeTokenizerModular.from_legacy(tok_mode) for tok_mode in TokenizationMode],
]

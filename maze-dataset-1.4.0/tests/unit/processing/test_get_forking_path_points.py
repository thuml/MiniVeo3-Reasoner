import numpy as np
import pytest

from maze_dataset import CoordArray, SolvedMaze

_MANUAL_EXAMPLES: dict[str, dict[str, SolvedMaze | list[int] | CoordArray]] = {
	"small_5x5": dict(
		maze=SolvedMaze.from_ascii(
			"""
			###########
			#    XXX# #
			# ###X#X# #
			#   #X#S  #
			#####X#####
			#XXXXX#EXX#
			#X### ###X#
			#X#     #X#
			#X#######X#
			#XXXXXXXXX#
			###########
		""",
		),
		fork_idxs=[0, 2, 4],
		fork_coords=[
			[1, 3],
			[0, 2],
			[2, 2],
		],
		follow_idxs=[1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
		follow_coords=[
			[0, 3],
			[1, 2],
			[2, 1],
			[2, 0],
			[3, 0],
			[4, 0],
			[4, 1],
			[4, 2],
			[4, 3],
			[4, 4],
			[3, 4],
			[2, 4],
			[2, 3],
		],
	),
	"medium_10x10": dict(
		maze=SolvedMaze.from_ascii(
			"""
			#####################
			#         # #       #
			# ##### # # # ### ###
			# #     #   # # #   #
			# # ######### # ### #
			# #       #   #   # #
			# ####### # ### ### #
			#   #   # #     #   #
			### # # # ##### # ###
			#E# # # # #     # # #
			#X# # # # # ##### # #
			#X  # # #   #     # #
			#X##### ########### #
			#XXXXXXX#  XXXXX#XXX#
			#######X###X###X#X#X#
			#S    #XXXXX# #XXX#X#
			#X########### #####X#
			#X#XXX#XXXXXXXXX#XXX#
			#X#X#X#X#######X#X###
			#XXX#XXX      #XXX  #
			#####################
		""",
		),
		fork_idxs=[0, 7, 11, 14, 18, 24, 28, 32],
		fork_coords=[[7, 0], [9, 3], [8, 6], [9, 8], [6, 9], [6, 5], [6, 3], [5, 0]],
		follow_idxs=[
			1,
			2,
			3,
			4,
			5,
			6,
			8,
			9,
			10,
			12,
			13,
			15,
			16,
			17,
			19,
			20,
			21,
			22,
			23,
			25,
			26,
			27,
			29,
			30,
			31,
			33,
		],
		follow_coords=[
			[8, 0],
			[9, 0],
			[9, 1],
			[8, 1],
			[8, 2],
			[9, 2],
			[8, 3],
			[8, 4],
			[8, 5],
			[8, 7],
			[9, 7],
			[8, 8],
			[8, 9],
			[7, 9],
			[6, 8],
			[7, 8],
			[7, 7],
			[6, 7],
			[6, 6],
			[7, 5],
			[7, 4],
			[7, 3],
			[6, 2],
			[6, 1],
			[6, 0],
			[4, 0],
		],
	),
}


@pytest.mark.parametrize("example_name", _MANUAL_EXAMPLES.keys())
def test_fork_and_following_points(example_name):
	example = _MANUAL_EXAMPLES[example_name]
	maze = example["maze"]

	# compute
	fork_idxs, fork_coords = maze.get_solution_forking_points()
	follow_idxs, follow_coords = maze.get_solution_path_following_points()

	# check that entire solution is covered
	fork_coords_set = set(map(tuple, fork_coords))
	follow_coords_set = set(map(tuple, follow_coords))
	for s_idx, s_coord_ in enumerate(maze.solution):
		s_coord = tuple(s_coord_)
		# exclusive or
		assert (s_idx in fork_idxs) != (s_idx in follow_idxs)
		assert (s_coord in fork_coords_set) != (s_coord in follow_coords_set)

	# assert equal
	assert np.all(fork_idxs == np.array(example["fork_idxs"]))
	assert np.all(fork_coords == np.array(example["fork_coords"]))
	assert np.all(follow_idxs == np.array(example["follow_idxs"]))
	assert np.all(follow_coords == np.array(example["follow_coords"]))

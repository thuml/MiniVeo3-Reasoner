import itertools
import random
import re
from collections import Counter
from itertools import product
from typing import Iterable, Sequence

import frozendict
import numpy as np
import pytest
from jaxtyping import Int
from muutils.misc import flatten

from maze_dataset import (
	VOCAB,
	ConnectionArray,
	Coord,
	CoordArray,
	CoordTup,
	LatticeMaze,
	MazeDataset,
	MazeDatasetConfig,
	SolvedMaze,
)
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.generation.seed import GLOBAL_SEED
from maze_dataset.plotting.print_tokens import color_maze_tokens_AOTP
from maze_dataset.testing_utils import (
	ASCII_MAZES,
	LEGACY_AND_EQUIVALENT_TOKENIZERS,
	MANUAL_MAZE,
	MAZE_DATASET,
	MIXED_MAZES,
)
from maze_dataset.token_utils import (
	connection_list_to_adj_list,
	equal_except_adj_list_sequence,
)
from maze_dataset.tokenization import (
	AdjListTokenizers,
	CoordTokenizers,
	EdgeGroupings,
	EdgePermuters,
	EdgeSubsets,
	MazeTokenizer,
	MazeTokenizerModular,
	PathTokenizers,
	PromptSequencers,
	StepSizes,
	StepTokenizers,
	TargetTokenizers,
	TokenizationMode,
	_TokenizerElement,
)
from maze_dataset.tokenization.modular.all_instances import all_instances
from maze_dataset.utils import lattice_max_degrees, manhattan_distance

# Use for test fuzzing when there are too many possible tokenizers
NUM_TOKENIZERS_TO_TEST = 100


@pytest.mark.parametrize(
	("tok_mode", "max_grid_size"),
	list(
		product(
			[
				TokenizationMode.AOTP_UT_rasterized,
				TokenizationMode.AOTP_UT_uniform,
				TokenizationMode.AOTP_CTT_indexed,
			],
			[None, 3, 100],
		),
	),
)
def test_tokenizer_serialization(tok_mode: TokenizationMode, max_grid_size: int | None):
	tokenizer: MazeTokenizer = MazeTokenizer(
		tokenization_mode=tok_mode,
		max_grid_size=max_grid_size,
	)

	serialized: dict = tokenizer.serialize()
	print(serialized)
	tokenizer_loaded: MazeTokenizer = MazeTokenizer.load(serialized)

	assert tokenizer == tokenizer_loaded


def test_tokenizer():
	cfg: MazeDatasetConfig = MazeDatasetConfig(
		name="test",
		grid_n=5,
		n_mazes=3,
		maze_ctor=LatticeMazeGenerators.gen_dfs,
	)
	# to create a dataset, just call MazeDataset.from_config
	dataset: MazeDataset = MazeDataset.from_config(
		cfg,
		do_download=False,
		load_local=False,
		do_generate=True,
		save_local=False,
		verbose=True,
		gen_parallel=False,
	)

	for mode in (
		TokenizationMode.AOTP_UT_rasterized,
		TokenizationMode.AOTP_UT_uniform,
		TokenizationMode.AOTP_CTT_indexed,
	):
		tokenizer: MazeTokenizer = MazeTokenizer(
			tokenization_mode=mode,
			max_grid_size=100,
		)

		assert tokenizer.name == f"maze_tokenizer-{mode.name}-g{100}"

		if mode == TokenizationMode.AOTP_CTT_indexed:
			assert tokenizer.node_strings_map is not None
			assert 100 < tokenizer.vocab_size < 200
		elif mode in (
			TokenizationMode.AOTP_UT_rasterized,
			TokenizationMode.AOTP_UT_uniform,
		):
			assert tokenizer.node_strings_map is None
			assert tokenizer.vocab_size > 10000

		assert isinstance(tokenizer.token_arr, Iterable)
		assert all(isinstance(token, str) for token in tokenizer.token_arr)
		assert len(tokenizer.token_arr) == tokenizer.vocab_size

		print(tokenizer.summary())

		for maze in dataset:
			# clear the cache here so we test if it works fine on the next loop
			tokenizer.clear_cache()

			maze_tok = maze.as_tokens(maze_tokenizer=tokenizer)

			maze_encoded = tokenizer.encode(maze_tok)
			maze_decoded = tokenizer.decode(maze_encoded)

			assert maze_tok == maze_decoded

			# you can view the tokens directly
			print("\nRaw tokens:\n")
			print(" ".join(maze_tok))

			maze_recovered = SolvedMaze.from_tokens(maze_tok, maze_tokenizer=tokenizer)

			assert (maze.connection_list == maze_recovered.connection_list).all()

			# or color and print them in various formats
			print("\nColored tokens, raw html:\n")
			print(color_maze_tokens_AOTP(maze_tok, fmt="html"))
			print("\nColored tokens, raw latex:\n")
			print(color_maze_tokens_AOTP(maze_tok, fmt="latex"))
			print("\nColored tokens, terminal:\n")
			print(color_maze_tokens_AOTP(maze_tok, fmt="terminal"))


@pytest.mark.parametrize(
	("maze_ascii", "tokenizer", "tokens"),
	[
		pytest.param(
			ASCII_MAZES[maze_ascii_key][1],  # maze_ascii
			tokenizer,  # tok_mode
			ASCII_MAZES[maze_ascii_key][0],  # tokens
			id=f"{tokenizer.name}_{maze_ascii_key}",
		)
		for maze_ascii_key, tokenizer in product(
			["small_3x3", "big_10x10"],
			LEGACY_AND_EQUIVALENT_TOKENIZERS,
		)
	],
)
def test_maze_to_tokens_roundtrip(
	maze_ascii: list[str],
	tokenizer: MazeTokenizer | MazeTokenizerModular,
	tokens: str,
):
	if not tokenizer.is_UT():
		# The hardcoded `tokens` assumes a UT tokenizer.
		# Here we modify `tokens` to match what a `AOTP_CTT_indexed` tokenizer would produce.
		tokens = re.sub(r"\(([0-9]),([0-9])\)", r"(\1 , \2)", tokens)
		tokens = re.sub(r"\(([0-9]+ ,)", r"( \1", tokens)
		tokens = re.sub(r"(, [0-9]+)\)", r"\1 )", tokens)
	tokens_original_split: list[str] = tokens.split()

	# join into a single string, and get a maze out
	ascii_str: str = "\n".join(maze_ascii)
	maze: SolvedMaze = SolvedMaze.from_ascii(ascii_str)

	# maze as tokens
	tokens_from_maze: list[str] = maze.as_tokens(tokenizer)

	# maze round trip
	maze_roundtrip: SolvedMaze = SolvedMaze.from_tokens(tokens_from_maze, tokenizer)
	tokens_roundtrip: list[str] = maze_roundtrip.as_tokens(tokenizer)

	# check that the mazes and tokens are all equivalent
	assert maze == maze_roundtrip
	assert equal_except_adj_list_sequence(tokens_original_split, tokens_from_maze)
	assert equal_except_adj_list_sequence(tokens_original_split, tokens_roundtrip)


@pytest.mark.parametrize(
	("tok_mode", "max_grid_size", "result"),
	[
		pytest.param(
			tok_mode,
			max_grid_size,
			MazeTokenizer(tokenization_mode=tok_mode, max_grid_size=max_grid_size),
			id=f"{tok_mode}-{max_grid_size}",
		)
		for tok_mode, max_grid_size in [
			(TokenizationMode.AOTP_CTT_indexed, None),
			(TokenizationMode.AOTP_UT_rasterized, None),
			(TokenizationMode.AOTP_UT_uniform, None),
			(TokenizationMode.AOTP_CTT_indexed, 5),
		]
	],
)
def test_to_legacy_tokenizer(
	tok_mode: TokenizationMode,
	max_grid_size: int | None,
	result: MazeTokenizer,
):
	assert tok_mode.to_legacy_tokenizer(max_grid_size) == result


# MazeTokenizerModular tests
# =====================

# Backwards compatibility tests
# =============================


@pytest.mark.parametrize(
	("maze", "legacy_tokenizer"),
	[
		pytest.param(maze[0], tok_spec, id=f"{tok_spec.value}-maze{maze[1]}")
		for maze, tok_spec in itertools.product(
			[(maze, i) for i, maze in enumerate(MIXED_MAZES)],
			[tok_mode for tok_mode in TokenizationMode],  # noqa: C416
		)
	],
)
def test_to_tokens_backwards_compatible(
	maze: SolvedMaze,
	legacy_tokenizer: TokenizationMode,
):
	tokenizer: MazeTokenizerModular = MazeTokenizerModular.from_legacy(legacy_tokenizer)
	toks: list[str] = maze.as_tokens(tokenizer)
	toks2: list[str] = tokenizer.to_tokens(maze)
	toks_legacy: list[str] = maze.as_tokens(legacy_tokenizer)

	try:
		assert equal_except_adj_list_sequence(toks, toks_legacy)
		assert equal_except_adj_list_sequence(toks2, toks_legacy)
	except AssertionError as e:
		msg: str = (
			"Tokens from `as_tokens` and `to_tokens` should be equal to tokens from `as_tokens` with the legacy tokenizer.\n"
			f"{len(toks) = }, {len(toks2) = }, {len(toks_legacy) = }\n"
			f"{toks = }\n{toks2 = }\n{toks_legacy = }"
		)
		raise AssertionError(msg) from e


@pytest.mark.parametrize(
	("coords", "legacy_tok_mode"),
	[
		pytest.param(
			coords,
			tok_mode,
			id=f"{tok_mode.value}-coords(type={type(coords[0])},len={len(coords)})",
		)
		for tok_mode, coords in itertools.product(
			[tok_mode for tok_mode in TokenizationMode],  # noqa: C416
			[
				*[[maze.start_pos] for maze in MAZE_DATASET.mazes[:2]],
				[maze.start_pos for maze in MAZE_DATASET.mazes],
				*[[tuple(maze.start_pos)] for maze in MAZE_DATASET.mazes[:2]],
				[tuple(maze.start_pos) for maze in MAZE_DATASET.mazes],
			],
		)
	],
)
def test_coords_to_strings_backwards_compatible(
	coords: list[Coord, CoordTup],
	legacy_tok_mode: TokenizationMode,
):
	tokenizer: MazeTokenizerModular = MazeTokenizerModular.from_legacy(legacy_tok_mode)
	legacy_tokenizer = MazeTokenizer(tokenization_mode=legacy_tok_mode)
	strings: list[str] = tokenizer.coords_to_strings(coords)
	strings_legacy: list[str] = legacy_tokenizer.coords_to_strings(coords)
	assert strings == strings_legacy


@pytest.mark.parametrize(
	("maze", "tok_mode"),
	[
		pytest.param(maze[0], tok_spec, id=f"{tok_spec.value}-maze{maze[1]}")
		for maze, tok_spec in itertools.product(
			[(maze, i) for i, maze in enumerate(MIXED_MAZES)],
			[tok_mode for tok_mode in TokenizationMode],  # noqa: C416
		)
	],
)
def test_from_tokens_backwards_compatible(
	maze: LatticeMaze,
	tok_mode: TokenizationMode,
):
	tokenizer = MazeTokenizerModular.from_legacy(tok_mode)
	toks = maze.as_tokens(tok_mode)
	# Equality test of `as_tokens` output done in a separate unit test
	maze_legacy: LatticeMaze = LatticeMaze.from_tokens(toks, tok_mode)
	maze: LatticeMaze = LatticeMaze.from_tokens(toks, tokenizer)
	assert maze == maze_legacy


# General functionality tests
# ===========================


@pytest.mark.parametrize(
	("el", "result"),
	[
		pytest.param(elem, result, id=elem.name)
		for elem, result in [
			(CoordTokenizers.CTT(), True),
			(CoordTokenizers.CTT(intra=True), True),
			(CoordTokenizers.UT(), True),
			(AdjListTokenizers.AdjListCoord(), True),
			(AdjListTokenizers.AdjListCoord(post=True), True),
			(TargetTokenizers.Unlabeled(post=True), True),
			(PathTokenizers.StepSequence(), True),
			(
				PathTokenizers.StepSequence(step_tokenizers=(StepTokenizers.Coord(),)),
				True,
			),
			(
				PathTokenizers.StepSequence(
					step_tokenizers=(
						StepTokenizers.Coord(),
						StepTokenizers.Coord(),
					),
				),
				False,
			),
			(PromptSequencers.AOP(), True),
			(PromptSequencers.AOP(path_tokenizer=PathTokenizers.StepSequence()), True),
			(
				PromptSequencers.AOP(
					path_tokenizer=PathTokenizers.StepSequence(
						step_tokenizers=(StepTokenizers.Coord(),),
					),
				),
				True,
			),
			(
				PromptSequencers.AOP(
					path_tokenizer=PathTokenizers.StepSequence(
						step_tokenizers=(
							StepTokenizers.Coord(),
							StepTokenizers.Coord(),
						),
					),
				),
				True,
			),
		]
	],
)
def test_tokenizer_element_is_valid(el: _TokenizerElement, result: bool):
	assert el.is_valid() == result


@pytest.mark.parametrize(
	("tokenizer", "result"),
	[
		pytest.param(tokenizer, result, id=str(tokenizer))
		for tokenizer, result in [
			(MazeTokenizerModular(), True),
			(MazeTokenizerModular.from_legacy(TokenizationMode.AOTP_CTT_indexed), True),
			(MazeTokenizerModular(prompt_sequencer=PromptSequencers.AOP()), False),
		]
	],
)
def test_is_legacy_equivalent(tokenizer: MazeTokenizerModular, result: bool):
	assert tokenizer.is_legacy_equivalent() == result


def _helper_test_path_tokenizers(
	pt: PathTokenizers._PathTokenizer,
	maze: SolvedMaze,
	footprint_inds: Sequence[int],
):
	ct: CoordTokenizers._CoordTokenizer = CoordTokenizers.UT()
	path_toks: list[str] = pt.to_tokens(maze, ct)
	path_toks_set: set[str] = set(path_toks)
	footprint_inds: Int[np.ndarray, " footprint_index"] = np.array(footprint_inds)
	footprints: Int[np.ndarray, "footprint_index row_col=2"] = maze.solution[
		footprint_inds
	]
	if StepTokenizers.Coord() in pt.step_tokenizers:
		non_steps: set[CoordTup] = {tuple(c) for c in maze.solution} - {
			tuple(c) for c in footprints
		}
		assert all(ct.to_tokens(coord)[0] in path_toks_set for coord in footprints)
		assert all(ct.to_tokens(coord)[0] not in path_toks_set for coord in non_steps)
	if StepTokenizers.Distance() in pt.step_tokenizers:
		distances: list[int] = footprint_inds[1:] - footprint_inds[:-1]
		assert (
			len(
				Counter(getattr(VOCAB, f"I_{d:03}") for d in distances)
				- Counter(path_toks),
			)
			== 0
		)
	if StepTokenizers.Cardinal() in pt.step_tokenizers:
		c = Counter(path_toks)
		assert (
			c[VOCAB.PATH_NORTH]
			+ c[VOCAB.PATH_SOUTH]
			+ c[VOCAB.PATH_EAST]
			+ c[VOCAB.PATH_WEST]
			== len(footprint_inds) - 1
		)
	if StepTokenizers.Relative() in pt.step_tokenizers:
		c = Counter(path_toks)
		assert (
			c[VOCAB.PATH_LEFT]
			+ c[VOCAB.PATH_RIGHT]
			+ c[VOCAB.PATH_FORWARD]
			+ c[VOCAB.PATH_BACKWARD]
			== len(footprint_inds) - 1
		)


@pytest.mark.parametrize(
	("pt", "manual_maze"),
	[
		pytest.param(tokenizer, maze_kv[1], id=f"{tokenizer.name}-{maze_kv[0]}")
		for maze_kv, tokenizer in itertools.product(
			ASCII_MAZES.items(),
			random.sample(
				list(
					all_instances(
						PathTokenizers._PathTokenizer,
						{_TokenizerElement: lambda x: x.is_valid()},
					),
				),
				NUM_TOKENIZERS_TO_TEST,
			),
		)
	],
)
def test_path_tokenizers(pt: PathTokenizers._PathTokenizer, manual_maze: MANUAL_MAZE):
	solved_maze: SolvedMaze = SolvedMaze.from_ascii("\n".join(manual_maze.ascii))
	match type(pt.step_size):
		case StepSizes.Singles:
			footprint_inds = range(solved_maze.solution.shape[0])
		case StepSizes.Straightaways:
			swy_coordtup_set: set[CoordTup] = {
				tuple(c) for c in manual_maze.straightaway_footprints
			}
			footprint_inds: list[int] = [
				i
				for i, c in enumerate(solved_maze.solution)
				if tuple(c) in swy_coordtup_set
			]
		case StepSizes.Forks:
			footprint_inds = solved_maze.get_solution_forking_points(
				always_include_endpoints=True,
			)[0]
		case StepSizes.ForksAndStraightaways:
			swy_step_inds: list[int] = StepSizes.Straightaways()._step_single_indices(
				solved_maze,
			)
			footprint_inds: Int[np.ndarray, " footprint_index"] = np.concatenate(
				(
					solved_maze.get_solution_forking_points(
						always_include_endpoints=True,
					)[0],
					swy_step_inds,
				),
			)
			footprint_inds, _ = np.unique(footprint_inds, axis=0, return_index=True)
	_helper_test_path_tokenizers(
		pt,
		solved_maze,
		footprint_inds,
	)


@pytest.mark.parametrize(
	("ep", "maze"),
	[
		pytest.param(tokenizer, maze, id=f"{tokenizer.name}-maze[{i}]")
		for (i, maze), tokenizer in itertools.product(
			enumerate(MIXED_MAZES[:6]),
			all_instances(
				EdgePermuters._EdgePermuter,
				frozendict.frozendict({_TokenizerElement: lambda x: x.is_valid()}),
			),
		)
	],
)
def test_edge_permuters(ep: EdgePermuters._EdgePermuter, maze: LatticeMaze):
	edges: ConnectionArray = connection_list_to_adj_list(
		maze.connection_list,
		shuffle_d0=False,
		shuffle_d1=False,
	)
	edges_copy: ConnectionArray = connection_list_to_adj_list(
		maze.connection_list,
		shuffle_d0=False,
		shuffle_d1=False,
	)
	assert np.array_equal(edges, edges_copy)
	old_shape = edges.shape
	permuted: ConnectionArray = ep._permute(edges)
	match ep:
		case EdgePermuters.RandomCoords():
			assert permuted.shape == old_shape
			assert edges is permuted
			i = 0
			while np.array_equal(permuted, edges_copy) and i < 2:
				# Permute again in case for small mazes the random selection happened to not change anything
				permuted: ConnectionArray = ep._permute(permuted)
				i += 1
			assert not np.array_equal(permuted, edges_copy)
		case EdgePermuters.BothCoords():
			new_shape = old_shape[0] * 2, *old_shape[1:]
			n = old_shape[0]
			assert permuted.shape == new_shape
			assert np.array_equal(permuted[:n, ...], edges_copy)
			assert np.array_equal(permuted[:n, 0, :], permuted[n:, 1, :])
			assert np.array_equal(permuted[:n, 1, :], permuted[n:, 0, :])
			assert edges is not permuted


@pytest.mark.parametrize(
	("es", "maze"),
	[
		pytest.param(tokenizer, maze, id=f"{tokenizer.name}-maze[{i}]")
		for (i, maze), tokenizer in itertools.product(
			enumerate(MIXED_MAZES[:6]),
			all_instances(
				EdgeSubsets._EdgeSubset,
				frozendict.frozendict({_TokenizerElement: lambda x: x.is_valid()}),
			),
		)
	],
)
def test_edge_subsets(es: EdgeSubsets._EdgeSubset, maze: LatticeMaze):
	edges: ConnectionArray = es._get_edges(maze)
	n: int = maze.grid_n
	match type(es):
		case EdgeSubsets.AllLatticeEdges:
			assert_shape: tuple = (2 * n * (n - 1), 2, 2)
		case EdgeSubsets.ConnectionEdges:
			if not es.walls:
				assert_shape: tuple = (np.count_nonzero(maze.connection_list), 2, 2)
			else:
				assert_shape: tuple = (
					2 * n * (n - 1) - np.count_nonzero(maze.connection_list),
					2,
					2,
				)
	assert edges.dtype == np.int8
	assert assert_shape == tuple(edges.shape)
	assert assert_shape == tuple(
		np.unique(edges, axis=0).shape,
	)  # All edges are unique (swapping leading/trailing coords is considered different)
	assert np.array_equal(
		manhattan_distance(edges),
		np.array([1] * assert_shape[0], dtype=np.int8),
	)


@pytest.mark.parametrize(
	("tok_elem", "es", "maze"),
	[
		# we do a little accessing private members here
		pytest.param(tok_elem, es, maze, id=f"{tok_elem.name}-{es.name}-maze[{i}]")
		for (i, maze), tok_elem, es in itertools.product(
			enumerate(MIXED_MAZES[:6]),
			all_instances(
				EdgeGroupings._EdgeGrouping,
				frozendict.frozendict(
					{
						_TokenizerElement: lambda x: x.is_valid(),
						# Add a condition to prune the range space that doesn't affect functionality being tested
						EdgeGroupings.ByLeadingCoord: lambda x: x.intra
						and x.connection_token_ordinal == 1,
					},
				),
			),
			all_instances(
				EdgeSubsets._EdgeSubset,
				frozendict.frozendict({_TokenizerElement: lambda x: x.is_valid()}),
			),
		)
	],
)
def test_edge_groupings(
	tok_elem: EdgeGroupings._EdgeGrouping,
	es: EdgeSubsets._EdgeSubset,
	maze: LatticeMaze,
):
	# we do a little more accessing private members here
	edges: ConnectionArray = es._get_edges(maze)
	# n: int = maze.grid_n
	groups: Sequence[ConnectionArray] = tok_elem._group_edges(edges)

	assert all(
		not np.any(np.diff(g[:, 0], axis=0)) for g in groups
	)  # Asserts that the leading coord is the same for all edges within each group
	match type(tok_elem):
		case EdgeGroupings.Ungrouped:
			assert_shape = edges.shape[0], 1, 2, 2
			assert tuple(groups.shape) == assert_shape
		case EdgeGroupings.ByLeadingCoord:
			assert len(groups) == np.unique(edges[:, 0, :], axis=0).shape[0]
			assert sum(g.shape[0] for g in groups) == edges.shape[0]
			# trailing_coords: list[CoordArray] = [g[:, 1, :] for g in groups]
			# vector_diffs is the position vector difference between the trailing coords of each group
			# These are stacked into a single array since we don't care about maintaining group separation
			vector_diffs: CoordArray = np.stack(
				list(flatten([np.diff(g[:, 1, :], axis=0) for g in groups], 1)),
			)
			if tok_elem.shuffle_group:
				allowed_diffs = {(1, -1), (1, 1), (0, 2), (2, 0)}
				# The set of all 2D vectors between any 2 coords adjacent to a central coord
				allowed_diffs = allowed_diffs.union(
					{(-d[0], -d[1]) for d in allowed_diffs},
				)
			else:
				# If vector_diffs are lexicographically sorted, these are the only possible values. Any other value indicates an error in sorting
				allowed_diffs = {(1, -1), (1, 1), (0, 2), (2, 0)}
			assert all(
				tuple(diff) in allowed_diffs for diff in np.unique(vector_diffs, axis=0)
			)


random.seed(GLOBAL_SEED)


@pytest.mark.parametrize(
	("tok_elem", "maze"),
	[
		pytest.param(tok_elem, maze, id=f"{tok_elem.name}-maze[{i}]")
		for (i, maze), tok_elem in itertools.product(
			enumerate(MAZE_DATASET),
			random.sample(
				list(
					all_instances(
						# yes we access a private member
						AdjListTokenizers._AdjListTokenizer,
						{
							_TokenizerElement: lambda x: x.is_valid(),
						},
					),
				),
				100,
			),
		)
	],
)
# too many branches and "too complex" but whatever
def test_adjlist_tokenizers(  # noqa: PLR0912,C901
	tok_elem: AdjListTokenizers._AdjListTokenizer,
	maze: LatticeMaze,
):
	toks: list[str] = tok_elem.to_tokens(maze, CoordTokenizers.UT())
	tok_counter: Counter = Counter(toks)
	n: int = maze.grid_n
	edge_count: int = 1  # To be updated in match/case blocks
	group_count: int = 1  # To be updated in match/case blocks

	match tok_elem.edge_subset:
		case EdgeSubsets.AllLatticeEdges():
			edge_count *= n * (n - 1) * 2
		case EdgeSubsets.ConnectionEdges(walls=False):
			edge_count *= np.count_nonzero(maze.connection_list)
		case EdgeSubsets.ConnectionEdges(walls=True):
			edge_count *= n * (n - 1) * 2 - np.count_nonzero(maze.connection_list)
		case _:
			msg: str = f"`match` case missing for {tok_elem.edge_subset = }"
			raise NotImplementedError(msg)

	match tok_elem.edge_permuter:
		case EdgePermuters.BothCoords():
			edge_count *= 2
			if tok_elem.edge_subset == EdgeSubsets.ConnectionEdges(walls=True):
				group_count *= np.count_nonzero(
					lattice_max_degrees(n) - maze.coord_degrees() > 0,
				)  # All coords with 1 adjacent wall, not counting outer boundaries
			else:
				group_count *= np.count_nonzero(
					maze.coord_degrees() > 0,
				)  # All coords with >0 connections
		case EdgePermuters.RandomCoords() | EdgePermuters.SortedCoords():
			edge_count *= 1
			group_count = None  # Group count is stochastic

	match type(tok_elem.edge_grouping):
		case EdgeGroupings.Ungrouped:
			group_count = edge_count  # Override all above cases
		case EdgeGroupings.ByLeadingCoord:
			if group_count is not None:
				group_count *= 1
			if tok_elem.edge_grouping.intra:
				assert tok_counter[VOCAB.ADJLIST_INTRA] == edge_count
		case _:
			msg: str = f"`match` case missing for {tok_elem.edge_grouping = }"
			raise NotImplementedError(msg)

	match type(tok_elem):
		case AdjListTokenizers.AdjListCoord:
			pass
		case AdjListTokenizers.AdjListCardinal:
			assert (
				tok_counter[VOCAB.PATH_NORTH]
				+ tok_counter[VOCAB.PATH_SOUTH]
				+ tok_counter[VOCAB.PATH_EAST]
				+ tok_counter[VOCAB.PATH_WEST]
				== edge_count
			)

	if group_count is not None:
		if tok_elem.pre:
			assert tok_counter[VOCAB.ADJLIST_PRE] == group_count
		if tok_elem.post:
			assert tok_counter[VOCAB.ADJACENCY_ENDLINE] == group_count

	assert tok_counter[VOCAB.CONNECTOR] + tok_counter[VOCAB.ADJLIST_WALL] == edge_count


@pytest.mark.parametrize(
	("tok_elem", "valid"),
	[
		pytest.param(
			tok_elem,
			valid,
			id=f"{tok_elem!r}",
		)
		for tok_elem, valid in (
			[
				(StepSizes.ForksAndStraightaways(), False),
				(StepSizes.Straightaways(), False),
				(StepSizes.Forks(), True),
				(AdjListTokenizers.AdjListCoord(), True),
				(AdjListTokenizers.AdjListCoord(pre=True), False),
				(AdjListTokenizers.AdjListCardinal(), True),
				(AdjListTokenizers.AdjListCardinal(pre=True), False),
				(EdgeGroupings.Ungrouped(), True),
				(EdgeGroupings.ByLeadingCoord(), False),
				(EdgeGroupings.ByLeadingCoord(connection_token_ordinal=0), False),
			]
		)
	],
)
def test_unsupported_elements(tok_elem: _TokenizerElement, valid: bool):
	assert tok_elem.is_valid() == valid

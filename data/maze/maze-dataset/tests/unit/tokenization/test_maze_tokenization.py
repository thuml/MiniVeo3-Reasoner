import pytest

from maze_dataset import (
	LatticeMazeGenerators,
	MazeDataset,
	MazeDatasetConfig,
	SolvedMaze,
)
from maze_dataset.testing_utils import LEGACY_AND_EQUIVALENT_TOKENIZERS
from maze_dataset.tokenization import MazeTokenizer, MazeTokenizerModular


@pytest.mark.parametrize(
	"tokenizer",
	[
		pytest.param(tokenizer, id=tokenizer.name)
		for tokenizer in LEGACY_AND_EQUIVALENT_TOKENIZERS
	],
)
def test_tokenization_roundtrip(tokenizer: MazeTokenizer | MazeTokenizerModular):
	dataset: MazeDataset = MazeDataset.from_config(
		MazeDatasetConfig(
			name="test",
			grid_n=5,
			n_mazes=5,
			maze_ctor=LatticeMazeGenerators.gen_dfs,
		),
		allow_generation_metadata_filter_mismatch=True,
	)

	dataset_tokenized: list[list[str]] = dataset.as_tokens(tokenizer)
	# dataset_tokenized_joined: list[str] = dataset.as_tokens(
	#     tokenizer, join_tokens_individual_maze=True
	# )

	# TODO: can't test that these match because order in adjacency list is random

	dataset_tokenized_individual: list[list[str]] = [
		maze.as_tokens(tokenizer) for maze in dataset.mazes
	]

	# we can't type hint easily that from_tokens usually returns a SolvedMaze
	mazes_roundtrip: list[SolvedMaze] = [
		SolvedMaze.from_tokens(  # type: ignore[misc]
			tokens=maze_tokens,
			maze_tokenizer=tokenizer,
		)
		for maze_tokens in dataset_tokenized
	]

	mazes_roundtrip_individual: list[SolvedMaze] = [
		SolvedMaze.from_tokens(  # type: ignore[misc]
			tokens=maze_tokens,
			maze_tokenizer=tokenizer,
		)
		for maze_tokens in dataset_tokenized_individual
	]

	# NOTE: can't test the tokenization explicitly because order in adjacency list is random
	# test both tokenized as a whole and tokenized individually
	# for maze_tok, maze_tok_indiv in zip(dataset_tokenized, dataset_tokenized_individual):
	#     assert all(
	#         x == y
	#         for x, y in zip(maze_tok, maze_tok_indiv)
	#     ), f"maze_tok: {' '.join(maze_tok)}, maze_tok_indiv: {' '.join(maze_tok_indiv)}"

	# test roundtrip
	for maze, maze_rt, maze_rt_indiv in zip(
		dataset.mazes,
		mazes_roundtrip,
		mazes_roundtrip_individual,
		strict=False,
	):
		assert maze == maze_rt, f"maze: {maze}, maze_rt: {maze_rt}"
		assert maze == maze_rt_indiv, f"maze: {maze}, maze_rt_indiv: {maze_rt_indiv}"

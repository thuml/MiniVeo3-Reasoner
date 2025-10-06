import itertools
from typing import Callable

import frozendict
import numpy as np
import pytest
from jaxtyping import Int

from maze_dataset import LatticeMaze
from maze_dataset.constants import VOCAB, Connection, ConnectionArray
from maze_dataset.generation import _NUMPY_RNG
from maze_dataset.testing_utils import GRID_N, MAZE_DATASET
from maze_dataset.token_utils import (
	_coord_to_strings_UT,
	coords_to_strings,
	equal_except_adj_list_sequence,
	get_adj_list_tokens,
	get_origin_tokens,
	get_path_tokens,
	get_relative_direction,
	get_target_tokens,
	is_connection,
	strings_to_coords,
	tokens_between,
)
from maze_dataset.tokenization import (
	PathTokenizers,
	StepTokenizers,
	get_tokens_up_to_path_start,
)
from maze_dataset.tokenization.modular.all_instances import FiniteValued, all_instances
from maze_dataset.utils import (
	lattice_connection_array,
	manhattan_distance,
)

MAZE_TOKENS: tuple[list[str], str] = (
	[
		"<ADJLIST_START>",
		"(0,1)",
		"<-->",
		"(1,1)",
		";",
		"(1,0)",
		"<-->",
		"(1,1)",
		";",
		"(0,1)",
		"<-->",
		"(0,0)",
		";",
		"<ADJLIST_END>",
		"<ORIGIN_START>",
		"(1,0)",
		"<ORIGIN_END>",
		"<TARGET_START>",
		"(1,1)",
		"<TARGET_END>",
		"<PATH_START>",
		"(1,0)",
		"(1,1)",
		"<PATH_END>",
	],
	"AOTP_UT",
)
MAZE_TOKENS_AOTP_CTT_indexed: tuple[list[str], str] = (
	[
		"<ADJLIST_START>",
		"(",
		"0",
		",",
		"1",
		")",
		"<-->",
		"(",
		"1",
		",",
		"1",
		")",
		";",
		"(",
		"1",
		",",
		"0",
		")",
		"<-->",
		"(",
		"1",
		",",
		"1",
		")",
		";",
		"(",
		"0",
		",",
		"1",
		")",
		"<-->",
		"(",
		"0",
		",",
		"0",
		")",
		";",
		"<ADJLIST_END>",
		"<ORIGIN_START>",
		"(",
		"1",
		",",
		"0",
		")",
		"<ORIGIN_END>",
		"<TARGET_START>",
		"(",
		"1",
		",",
		"1",
		")",
		"<TARGET_END>",
		"<PATH_START>",
		"(",
		"1",
		",",
		"0",
		")",
		"(",
		"1",
		",",
		"1",
		")",
		"<PATH_END>",
	],
	"AOTP_CTT_indexed",
)
TEST_TOKEN_LISTS: list[tuple[list[str], str]] = [
	MAZE_TOKENS,
	MAZE_TOKENS_AOTP_CTT_indexed,
]


@pytest.mark.parametrize(
	("toks", "tokenizer_name"),
	[
		pytest.param(
			token_list[0],
			token_list[1],
			id=f"{token_list[1]}",
		)
		for token_list in TEST_TOKEN_LISTS
	],
)
def test_tokens_between(toks: list[str], tokenizer_name: str):
	result = tokens_between(toks, "<PATH_START>", "<PATH_END>")
	match tokenizer_name:
		case "AOTP_UT":
			assert result == ["(1,0)", "(1,1)"]
		case "AOTP_CTT_indexed":
			assert result == ["(", "1", ",", "0", ")", "(", "1", ",", "1", ")"]

	# Normal case
	tokens = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
	start_value = "quick"
	end_value = "over"
	assert tokens_between(tokens, start_value, end_value) == ["brown", "fox", "jumps"]

	# Including start and end values
	assert tokens_between(tokens, start_value, end_value, True, True) == [
		"quick",
		"brown",
		"fox",
		"jumps",
		"over",
	]

	# When start_value or end_value is not unique and except_when_tokens_not_unique is True
	with pytest.raises(ValueError):  # noqa: PT011
		tokens_between(tokens, "the", "dog", False, False, True)

	# When start_value or end_value is not unique and except_when_tokens_not_unique is False
	assert tokens_between(tokens, "the", "dog", False, False, False) == [
		"quick",
		"brown",
		"fox",
		"jumps",
		"over",
		"the",
		"lazy",
	]

	# Empty tokens list
	with pytest.raises(ValueError):  # noqa: PT011
		tokens_between([], "start", "end")

	# start_value and end_value are the same
	with pytest.raises(ValueError):  # noqa: PT011
		tokens_between(tokens, "fox", "fox")

	# start_value or end_value not in the tokens list
	with pytest.raises(ValueError):  # noqa: PT011
		tokens_between(tokens, "start", "end")

	# start_value comes after end_value in the tokens list
	with pytest.raises(AssertionError):
		tokens_between(tokens, "over", "quick")

	# start_value and end_value are at the beginning and end of the tokens list, respectively
	assert tokens_between(tokens, "the", "dog", True, True) == tokens

	# Single element in the tokens list, which is the same as start_value and end_value
	with pytest.raises(ValueError):  # noqa: PT011
		tokens_between(["fox"], "fox", "fox", True, True)


@pytest.mark.parametrize(
	("toks", "tokenizer_name"),
	[
		pytest.param(
			token_list[0],
			token_list[1],
			id=f"{token_list[1]}",
		)
		for token_list in TEST_TOKEN_LISTS
	],
)
def test_tokens_between_out_of_order(toks: list[str], tokenizer_name: str):
	assert tokenizer_name
	with pytest.raises(AssertionError):
		tokens_between(toks, "<PATH_END>", "<PATH_START>")


@pytest.mark.parametrize(
	("toks", "tokenizer_name"),
	[
		pytest.param(
			token_list[0],
			token_list[1],
			id=f"{token_list[1]}",
		)
		for token_list in TEST_TOKEN_LISTS
	],
)
def test_get_adj_list_tokens(toks: list[str], tokenizer_name: str):
	result = get_adj_list_tokens(toks)
	match tokenizer_name:
		case "AOTP_UT":
			expected = [
				"(0,1)",
				"<-->",
				"(1,1)",
				";",
				"(1,0)",
				"<-->",
				"(1,1)",
				";",
				"(0,1)",
				"<-->",
				"(0,0)",
				";",
			]
		case "AOTP_CTT_indexed":
			expected = [
				"(",
				"0",
				",",
				"1",
				")",
				"<-->",
				"(",
				"1",
				",",
				"1",
				")",
				";",
				"(",
				"1",
				",",
				"0",
				")",
				"<-->",
				"(",
				"1",
				",",
				"1",
				")",
				";",
				"(",
				"0",
				",",
				"1",
				")",
				"<-->",
				"(",
				"0",
				",",
				"0",
				")",
				";",
			]
	assert result == expected


@pytest.mark.parametrize(
	("toks", "tokenizer_name"),
	[
		pytest.param(
			token_list[0],
			token_list[1],
			id=f"{token_list[1]}",
		)
		for token_list in TEST_TOKEN_LISTS
	],
)
def test_get_path_tokens(toks: list[str], tokenizer_name: str):
	result_notrim = get_path_tokens(toks)
	result_trim = get_path_tokens(toks, trim_end=True)
	match tokenizer_name:
		case "AOTP_UT":
			assert result_notrim == ["<PATH_START>", "(1,0)", "(1,1)", "<PATH_END>"]
			assert result_trim == ["(1,0)", "(1,1)"]
		case "AOTP_CTT_indexed":
			assert result_notrim == [
				"<PATH_START>",
				"(",
				"1",
				",",
				"0",
				")",
				"(",
				"1",
				",",
				"1",
				")",
				"<PATH_END>",
			]
			assert result_trim == ["(", "1", ",", "0", ")", "(", "1", ",", "1", ")"]


@pytest.mark.parametrize(
	("toks", "tokenizer_name"),
	[
		pytest.param(
			token_list[0],
			token_list[1],
			id=f"{token_list[1]}",
		)
		for token_list in TEST_TOKEN_LISTS
	],
)
def test_get_origin_tokens(toks: list[str], tokenizer_name: str):
	result = get_origin_tokens(toks)
	match tokenizer_name:
		case "AOTP_UT":
			assert result == ["(1,0)"]
		case "AOTP_CTT_indexed":
			assert result == ["(", "1", ",", "0", ")"]


@pytest.mark.parametrize(
	("toks", "tokenizer_name"),
	[
		pytest.param(
			token_list[0],
			token_list[1],
			id=f"{token_list[1]}",
		)
		for token_list in TEST_TOKEN_LISTS
	],
)
def test_get_target_tokens(toks: list[str], tokenizer_name: str):
	result = get_target_tokens(toks)
	match tokenizer_name:
		case "AOTP_UT":
			assert result == ["(1,1)"]
		case "AOTP_CTT_indexed":
			assert result == ["(", "1", ",", "1", ")"]


@pytest.mark.parametrize(
	("toks", "tokenizer_name"),
	[
		pytest.param(
			token_list[0],
			token_list[1],
			id=f"{token_list[1]}",
		)
		for token_list in [MAZE_TOKENS]
	],
)
def test_get_tokens_up_to_path_start_including_start(
	toks: list[str],
	tokenizer_name: str,
):
	# Dont test on `MAZE_TOKENS_AOTP_CTT_indexed` because this function doesn't support `AOTP_CTT_indexed` when `include_start_coord=True`.
	result = get_tokens_up_to_path_start(toks, include_start_coord=True)
	match tokenizer_name:
		case "AOTP_UT":
			expected = [
				"<ADJLIST_START>",
				"(0,1)",
				"<-->",
				"(1,1)",
				";",
				"(1,0)",
				"<-->",
				"(1,1)",
				";",
				"(0,1)",
				"<-->",
				"(0,0)",
				";",
				"<ADJLIST_END>",
				"<ORIGIN_START>",
				"(1,0)",
				"<ORIGIN_END>",
				"<TARGET_START>",
				"(1,1)",
				"<TARGET_END>",
				"<PATH_START>",
				"(1,0)",
			]
		case "AOTP_CTT_indexed":
			expected = [
				"<ADJLIST_START>",
				"(",
				"0",
				",",
				"1",
				")",
				"<-->",
				"(",
				"1",
				",",
				"1",
				")",
				";",
				"(",
				"1",
				",",
				"0",
				")",
				"<-->",
				"(",
				"1",
				",",
				"1",
				")",
				";",
				"(",
				"0",
				",",
				"1",
				")",
				"<-->",
				"(",
				"0",
				",",
				"0",
				")",
				";",
				"<ADJLIST_END>",
				"<ORIGIN_START>",
				"(",
				"1",
				",",
				"0",
				")",
				"<ORIGIN_END>",
				"<TARGET_START>",
				"(",
				"1",
				",",
				"1",
				")",
				"<TARGET_END>",
				"<PATH_START>",
				"(",
				"1",
				",",
				"0",
				")",
			]
	assert result == expected


@pytest.mark.parametrize(
	("toks", "tokenizer_name"),
	[
		pytest.param(
			token_list[0],
			token_list[1],
			id=f"{token_list[1]}",
		)
		for token_list in TEST_TOKEN_LISTS
	],
)
def test_get_tokens_up_to_path_start_excluding_start(
	toks: list[str],
	tokenizer_name: str,
):
	result = get_tokens_up_to_path_start(toks, include_start_coord=False)
	match tokenizer_name:
		case "AOTP_UT":
			expected = [
				"<ADJLIST_START>",
				"(0,1)",
				"<-->",
				"(1,1)",
				";",
				"(1,0)",
				"<-->",
				"(1,1)",
				";",
				"(0,1)",
				"<-->",
				"(0,0)",
				";",
				"<ADJLIST_END>",
				"<ORIGIN_START>",
				"(1,0)",
				"<ORIGIN_END>",
				"<TARGET_START>",
				"(1,1)",
				"<TARGET_END>",
				"<PATH_START>",
			]
		case "AOTP_CTT_indexed":
			expected = [
				"<ADJLIST_START>",
				"(",
				"0",
				",",
				"1",
				")",
				"<-->",
				"(",
				"1",
				",",
				"1",
				")",
				";",
				"(",
				"1",
				",",
				"0",
				")",
				"<-->",
				"(",
				"1",
				",",
				"1",
				")",
				";",
				"(",
				"0",
				",",
				"1",
				")",
				"<-->",
				"(",
				"0",
				",",
				"0",
				")",
				";",
				"<ADJLIST_END>",
				"<ORIGIN_START>",
				"(",
				"1",
				",",
				"0",
				")",
				"<ORIGIN_END>",
				"<TARGET_START>",
				"(",
				"1",
				",",
				"1",
				")",
				"<TARGET_END>",
				"<PATH_START>",
			]
	assert result == expected


@pytest.mark.parametrize(
	("toks", "tokenizer_name"),
	[
		pytest.param(
			token_list[0],
			token_list[1],
			id=f"{token_list[1]}",
		)
		for token_list in TEST_TOKEN_LISTS
	],
)
def test_strings_to_coords(toks: list[str], tokenizer_name: str):
	assert tokenizer_name
	adj_list = get_adj_list_tokens(toks)
	skipped = strings_to_coords(adj_list, when_noncoord="skip")
	included = strings_to_coords(adj_list, when_noncoord="include")

	assert skipped == [
		(0, 1),
		(1, 1),
		(1, 0),
		(1, 1),
		(0, 1),
		(0, 0),
	]

	assert included == [
		(0, 1),
		"<-->",
		(1, 1),
		";",
		(1, 0),
		"<-->",
		(1, 1),
		";",
		(0, 1),
		"<-->",
		(0, 0),
		";",
	]

	with pytest.raises(ValueError):  # noqa: PT011
		strings_to_coords(adj_list, when_noncoord="error")

	assert strings_to_coords("(1,2) <ADJLIST_START> (5,6)") == [(1, 2), (5, 6)]
	assert strings_to_coords("(1,2) <ADJLIST_START> (5,6)", when_noncoord="skip") == [
		(1, 2),
		(5, 6),
	]
	assert strings_to_coords(
		"(1,2) <ADJLIST_START> (5,6)",
		when_noncoord="include",
	) == [(1, 2), "<ADJLIST_START>", (5, 6)]
	with pytest.raises(ValueError):  # noqa: PT011
		strings_to_coords("(1,2) <ADJLIST_START> (5,6)", when_noncoord="error")


@pytest.mark.parametrize(
	("toks", "tokenizer_name"),
	[
		pytest.param(
			token_list[0],
			token_list[1],
			id=f"{token_list[1]}",
		)
		for token_list in TEST_TOKEN_LISTS
	],
)
def test_coords_to_strings(toks: list[str], tokenizer_name: str):
	assert tokenizer_name
	adj_list = get_adj_list_tokens(toks)
	# config = MazeDatasetConfig(name="test", grid_n=2, n_mazes=1)
	coords = strings_to_coords(adj_list, when_noncoord="include")

	skipped = coords_to_strings(
		coords,
		coord_to_strings_func=_coord_to_strings_UT,
		when_noncoord="skip",
	)
	included = coords_to_strings(
		coords,
		coord_to_strings_func=_coord_to_strings_UT,
		when_noncoord="include",
	)

	assert skipped == [
		"(0,1)",
		"(1,1)",
		"(1,0)",
		"(1,1)",
		"(0,1)",
		"(0,0)",
	]

	assert included == [
		"(0,1)",
		"<-->",
		"(1,1)",
		";",
		"(1,0)",
		"<-->",
		"(1,1)",
		";",
		"(0,1)",
		"<-->",
		"(0,0)",
		";",
	]

	with pytest.raises(ValueError):  # noqa: PT011
		coords_to_strings(
			coords,
			coord_to_strings_func=_coord_to_strings_UT,
			when_noncoord="error",
		)


def test_equal_except_adj_list_sequence():
	assert equal_except_adj_list_sequence(MAZE_TOKENS[0], MAZE_TOKENS[0])
	assert not equal_except_adj_list_sequence(
		MAZE_TOKENS[0],
		MAZE_TOKENS_AOTP_CTT_indexed[0],
	)
	assert equal_except_adj_list_sequence(
		[
			"<ADJLIST_START>",
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
		],
		[
			"<ADJLIST_START>",
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
		],
	)
	assert equal_except_adj_list_sequence(
		[
			"<ADJLIST_START>",
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
		],
		[
			"<ADJLIST_START>",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
		],
	)
	assert equal_except_adj_list_sequence(
		[
			"<ADJLIST_START>",
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
		],
		[
			"<ADJLIST_START>",
			"(1,1)",
			"<-->",
			"(0,1)",
			";",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
		],
	)
	assert not equal_except_adj_list_sequence(
		[
			"<ADJLIST_START>",
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
		],
		[
			"<ADJLIST_START>",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,1)",
			"(1,0)",
			"<PATH_END>",
		],
	)
	assert not equal_except_adj_list_sequence(
		[
			"<ADJLIST_START>",
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
		],
		[
			"<ADJLIST_START>",
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
			"<PATH_END>",
		],
	)
	assert not equal_except_adj_list_sequence(
		[
			"<ADJLIST_START>",
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"<ADJLIST_END>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
		],
		[
			"<ADJLIST_START>",
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
		],
	)
	assert not equal_except_adj_list_sequence(
		[
			"<ADJLIST_START>",
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
		],
		[
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
		],
	)
	with pytest.raises(ValueError):  # noqa: PT011
		equal_except_adj_list_sequence(
			[
				"(0,1)",
				"<-->",
				"(1,1)",
				";",
				"(1,0)",
				"<-->",
				"(1,1)",
				";",
				"(0,1)",
				"<-->",
				"(0,0)",
				";",
				"<ADJLIST_END>",
				"<ORIGIN_START>",
				"(1,0)",
				"<ORIGIN_END>",
				"<TARGET_START>",
				"(1,1)",
				"<TARGET_END>",
				"<PATH_START>",
				"(1,0)",
				"(1,1)",
				"<PATH_END>",
			],
			[
				"(0,1)",
				"<-->",
				"(1,1)",
				";",
				"(1,0)",
				"<-->",
				"(1,1)",
				";",
				"(0,1)",
				"<-->",
				"(0,0)",
				";",
				"<ADJLIST_END>",
				"<ORIGIN_START>",
				"(1,0)",
				"<ORIGIN_END>",
				"<TARGET_START>",
				"(1,1)",
				"<TARGET_END>",
				"<PATH_START>",
				"(1,0)",
				"(1,1)",
				"<PATH_END>",
			],
		)
	with pytest.raises(ValueError):  # noqa: PT011
		equal_except_adj_list_sequence(
			[
				"<ADJLIST_START>",
				"(0,1)",
				"<-->",
				"(1,1)",
				";",
				"(1,0)",
				"<-->",
				"(1,1)",
				";",
				"(0,1)",
				"<-->",
				"(0,0)",
				";",
				"<ORIGIN_START>",
				"(1,0)",
				"<ORIGIN_END>",
				"<TARGET_START>",
				"(1,1)",
				"<TARGET_END>",
				"<PATH_START>",
				"(1,0)",
				"(1,1)",
				"<PATH_END>",
			],
			[
				"<ADJLIST_START>",
				"(0,1)",
				"<-->",
				"(1,1)",
				";",
				"(1,0)",
				"<-->",
				"(1,1)",
				";",
				"(0,1)",
				"<-->",
				"(0,0)",
				";",
				"<ORIGIN_START>",
				"(1,0)",
				"<ORIGIN_END>",
				"<TARGET_START>",
				"(1,1)",
				"<TARGET_END>",
				"<PATH_START>",
				"(1,0)",
				"(1,1)",
				"<PATH_END>",
			],
		)
	assert not equal_except_adj_list_sequence(
		[
			"<ADJLIST_START>",
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
		],
		[
			"<ADJLIST_START>",
			"(0,1)",
			"<-->",
			"(1,1)",
			";",
			"(1,0)",
			"<-->",
			"(1,1)",
			";",
			"(0,1)",
			"<-->",
			"(0,0)",
			";",
			"<ORIGIN_START>",
			"(1,0)",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(1,1)",
			"<TARGET_END>",
			"<PATH_START>",
			"(1,0)",
			"(1,1)",
			"<PATH_END>",
		],
	)

	# CTT
	assert equal_except_adj_list_sequence(
		[
			"<ADJLIST_START>",
			"(",
			"0",
			",",
			"1",
			")",
			"<-->",
			"(",
			"1",
			",",
			"1",
			")",
			";",
			"(",
			"1",
			",",
			"0",
			")",
			"<-->",
			"(",
			"1",
			",",
			"1",
			")",
			";",
			"(",
			"0",
			",",
			"1",
			")",
			"<-->",
			"(",
			"0",
			",",
			"0",
			")",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(",
			"1",
			",",
			"0",
			")",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(",
			"1",
			",",
			"1",
			")",
			"<TARGET_END>",
			"<PATH_START>",
			"(",
			"1",
			",",
			"0",
			")",
			"(",
			"1",
			",",
			"1",
			")",
			"<PATH_END>",
		],
		[
			"<ADJLIST_START>",
			"(",
			"0",
			",",
			"1",
			")",
			"<-->",
			"(",
			"1",
			",",
			"1",
			")",
			";",
			"(",
			"1",
			",",
			"0",
			")",
			"<-->",
			"(",
			"1",
			",",
			"1",
			")",
			";",
			"(",
			"0",
			",",
			"1",
			")",
			"<-->",
			"(",
			"0",
			",",
			"0",
			")",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(",
			"1",
			",",
			"0",
			")",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(",
			"1",
			",",
			"1",
			")",
			"<TARGET_END>",
			"<PATH_START>",
			"(",
			"1",
			",",
			"0",
			")",
			"(",
			"1",
			",",
			"1",
			")",
			"<PATH_END>",
		],
	)
	assert equal_except_adj_list_sequence(
		[
			"<ADJLIST_START>",
			"(",
			"0",
			",",
			"1",
			")",
			"<-->",
			"(",
			"1",
			",",
			"1",
			")",
			";",
			"(",
			"1",
			",",
			"0",
			")",
			"<-->",
			"(",
			"1",
			",",
			"1",
			")",
			";",
			"(",
			"0",
			",",
			"1",
			")",
			"<-->",
			"(",
			"0",
			",",
			"0",
			")",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(",
			"1",
			",",
			"0",
			")",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(",
			"1",
			",",
			"1",
			")",
			"<TARGET_END>",
			"<PATH_START>",
			"(",
			"1",
			",",
			"0",
			")",
			"(",
			"1",
			",",
			"1",
			")",
			"<PATH_END>",
		],
		[
			"<ADJLIST_START>",
			"(",
			"1",
			",",
			"1",
			")",
			"<-->",
			"(",
			"0",
			",",
			"1",
			")",
			";",
			"(",
			"1",
			",",
			"0",
			")",
			"<-->",
			"(",
			"1",
			",",
			"1",
			")",
			";",
			"(",
			"0",
			",",
			"1",
			")",
			"<-->",
			"(",
			"0",
			",",
			"0",
			")",
			";",
			"<ADJLIST_END>",
			"<ORIGIN_START>",
			"(",
			"1",
			",",
			"0",
			")",
			"<ORIGIN_END>",
			"<TARGET_START>",
			"(",
			"1",
			",",
			"1",
			")",
			"<TARGET_END>",
			"<PATH_START>",
			"(",
			"1",
			",",
			"0",
			")",
			"(",
			"1",
			",",
			"1",
			")",
			"<PATH_END>",
		],
	)
	# This inactive test demonstrates the lack of robustness of the function for comparing source `LatticeMaze` objects.
	# See function documentation for details.
	# assert not equal_except_adj_list_sequence(
	#     "<ADJLIST_START> ( 0 , 1 ) <--> ( 1 , 1 ) ; ( 1 , 0 ) <--> ( 1 , 1 ) ; ( 0 , 1 ) <--> ( 0 , 0 ) ; <ADJLIST_END> <ORIGIN_START> ( 1 , 0 ) <ORIGIN_END> <TARGET_START> ( 1 , 1 ) <TARGET_END> <PATH_START> ( 1 , 0 ) ( 1 , 1 ) <PATH_END>".split(),
	#     "<ADJLIST_START> ( 1 , 0 ) <--> ( 1 , 1 ) ; ( 1 , 0 ) <--> ( 1 , 1 ) ; ( 0 , 1 ) <--> ( 0 , 0 ) ; <ADJLIST_END> <ORIGIN_START> ( 1 , 0 ) <ORIGIN_END> <TARGET_START> ( 1 , 1 ) <TARGET_END> <PATH_START> ( 1 , 0 ) ( 1 , 1 ) <PATH_END>".split()
	# )


# @mivanit: this was really difficult to understand
@pytest.mark.parametrize(
	("type_", "validation_funcs", "assertion"),
	[
		pytest.param(
			type_,
			vfs,
			assertion,
			id=f"{i}-{type_.__name__}",
		)
		for i, (type_, vfs, assertion) in enumerate(
			[
				(
					# type
					PathTokenizers._PathTokenizer,
					# validation_funcs
					dict(),
					# assertion
					lambda x: PathTokenizers.StepSequence(
						step_tokenizers=(StepTokenizers.Distance(),),
					)
					in x,
				),
				(
					# type
					PathTokenizers._PathTokenizer,
					# validation_funcs
					{PathTokenizers._PathTokenizer: lambda x: x.is_valid()},
					# assertion
					lambda x: PathTokenizers.StepSequence(
						step_tokenizers=(StepTokenizers.Distance(),),
					)
					not in x
					and PathTokenizers.StepSequence(
						step_tokenizers=(
							StepTokenizers.Coord(),
							StepTokenizers.Coord(),
						),
					)
					not in x,
				),
			],
		)
	],
)
def test_all_instances2(
	type_: FiniteValued,
	validation_funcs: frozendict.frozendict[
		FiniteValued,
		Callable[[FiniteValued], bool],
	],
	assertion: Callable[[list[FiniteValued]], bool],
):
	assert assertion(all_instances(type_, validation_funcs))


@pytest.mark.parametrize(
	("coords", "result"),
	[
		pytest.param(
			np.array(coords),
			res,
			id=f"{coords}",
		)
		for coords, res in (
			[
				([[0, 0], [0, 1], [1, 1]], VOCAB.PATH_RIGHT),
				([[0, 0], [1, 0], [1, 1]], VOCAB.PATH_LEFT),
				([[0, 0], [0, 1], [0, 2]], VOCAB.PATH_FORWARD),
				([[0, 0], [0, 1], [0, 0]], VOCAB.PATH_BACKWARD),
				([[0, 0], [0, 1], [0, 1]], VOCAB.PATH_STAY),
				([[1, 1], [0, 1], [0, 0]], VOCAB.PATH_LEFT),
				([[1, 1], [1, 0], [0, 0]], VOCAB.PATH_RIGHT),
				([[0, 2], [0, 1], [0, 0]], VOCAB.PATH_FORWARD),
				([[0, 0], [0, 1], [0, 0]], VOCAB.PATH_BACKWARD),
				([[0, 1], [0, 1], [0, 0]], ValueError),
				([[0, 1], [1, 1], [0, 0]], ValueError),
				([[1, 0], [1, 1], [0, 0]], ValueError),
				([[0, 1], [0, 2], [0, 0]], ValueError),
				([[0, 1], [0, 0], [0, 0]], VOCAB.PATH_STAY),
				([[1, 1], [0, 0], [0, 1]], ValueError),
				([[1, 1], [0, 0], [1, 0]], ValueError),
				([[0, 2], [0, 0], [0, 1]], ValueError),
				([[0, 0], [0, 0], [0, 1]], ValueError),
				([[0, 1], [0, 0], [0, 1]], VOCAB.PATH_BACKWARD),
				([[-1, 0], [0, 0], [1, 0]], VOCAB.PATH_FORWARD),
				([[-1, 0], [0, 0], [0, 1]], VOCAB.PATH_LEFT),
				([[-1, 0], [0, 0], [-1, 0]], VOCAB.PATH_BACKWARD),
				([[-1, 0], [0, 0], [0, -1]], VOCAB.PATH_RIGHT),
				([[-1, 0], [0, 0], [1, 0], [2, 0]], ValueError),
				([[-1, 0], [0, 0]], ValueError),
				([[-1, 0, 0], [0, 0, 0]], ValueError),
			]
		)
	],
)
def test_get_relative_direction(
	coords: Int[np.ndarray, "prev_cur_next=3 axis=2"],
	result: str | type[Exception],
):
	if isinstance(result, type) and issubclass(result, Exception):
		with pytest.raises(result):
			get_relative_direction(coords)
		return
	assert get_relative_direction(coords) == result


@pytest.mark.parametrize(
	("edges", "result"),
	[
		pytest.param(
			edges,
			res,
			id=f"{edges}",
		)
		for edges, res in (
			[
				(np.array([[0, 0], [0, 1]]), 1),
				(np.array([[1, 0], [0, 1]]), 2),
				(np.array([[-1, 0], [0, 1]]), 2),
				(np.array([[0, 0], [5, 3]]), 8),
				(
					np.array(
						[
							[[0, 0], [0, 1]],
							[[1, 0], [0, 1]],
							[[-1, 0], [0, 1]],
							[[0, 0], [5, 3]],
						],
					),
					[1, 2, 2, 8],
				),
				(np.array([[[0, 0], [5, 3]]]), [8]),
			]
		)
	],
)
def test_manhattan_distance(
	edges: ConnectionArray | Connection,
	result: Int[np.ndarray, " edges"] | Int[np.ndarray, ""] | type[Exception],
):
	if isinstance(result, type) and issubclass(result, Exception):
		with pytest.raises(result):
			manhattan_distance(edges)
		return
	assert np.array_equal(manhattan_distance(edges), np.array(result, dtype=np.int8))


@pytest.mark.parametrize(
	"n",
	[pytest.param(n) for n in [2, 3, 5, 20]],
)
def test_lattice_connection_arrray(n):
	edges = lattice_connection_array(n)
	assert tuple(edges.shape) == (2 * n * (n - 1), 2, 2)
	assert np.all(np.sum(edges[:, 1], axis=1) > np.sum(edges[:, 0], axis=1))
	assert tuple(np.unique(edges, axis=0).shape) == (2 * n * (n - 1), 2, 2)


@pytest.mark.parametrize(
	("edges", "maze"),
	[
		pytest.param(
			edges(),
			maze,
			id=f"edges[{i}]; maze[{j}]",
		)
		for (i, edges), (j, maze) in itertools.product(
			enumerate(
				[
					lambda: lattice_connection_array(GRID_N),
					lambda: np.flip(lattice_connection_array(GRID_N), axis=1),
					lambda: lattice_connection_array(GRID_N - 1),
					lambda: _NUMPY_RNG.choice(
						lattice_connection_array(GRID_N),
						2 * GRID_N,
						axis=0,
					),
					lambda: _NUMPY_RNG.choice(
						lattice_connection_array(GRID_N),
						1,
						axis=0,
					),
				],
			),
			enumerate(MAZE_DATASET.mazes),
		)
	],
)
def test_is_connection(edges: ConnectionArray, maze: LatticeMaze):
	output = is_connection(edges, maze.connection_list)
	sorted_edges = np.sort(edges, axis=1)
	edge_direction = (
		(sorted_edges[:, 1, :] - sorted_edges[:, 0, :])[:, 0] == 0
	).astype(np.int8)
	assert np.array_equal(
		output,
		maze.connection_list[
			edge_direction,
			sorted_edges[:, 0, 0],
			sorted_edges[:, 0, 1],
		],
	)

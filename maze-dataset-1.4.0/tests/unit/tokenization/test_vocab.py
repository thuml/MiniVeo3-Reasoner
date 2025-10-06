import pytest

from maze_dataset.constants import (
	SPECIAL_TOKENS,
	VOCAB,
	VOCAB_LIST,
	VOCAB_TOKEN_TO_INDEX,
)


def test_special_tokens_base():
	# Test the getitem method
	assert SPECIAL_TOKENS["ADJLIST_START"] == "<ADJLIST_START>"

	with pytest.raises(KeyError):
		SPECIAL_TOKENS["NON_EXISTENT_KEY"]

	# Test the len method
	assert len(SPECIAL_TOKENS) == 11

	# Test the contains method
	assert "ADJLIST_START" in SPECIAL_TOKENS
	assert "NON_EXISTENT_KEY" not in SPECIAL_TOKENS

	# Test the values method
	assert "<ADJLIST_START>" in SPECIAL_TOKENS.values()

	# Test the items method
	assert ("ADJLIST_START", "<ADJLIST_START>") in SPECIAL_TOKENS.items()

	# Test the keys method
	assert "ADJLIST_START" in SPECIAL_TOKENS


def test_vocab():
	assert len(VOCAB) == 4096
	# due to typing issue with VOCAB being instance of a dynamic dataclass
	assert VOCAB.CTT_10 == "10"  # type: ignore[attr-defined]
	assert VOCAB_LIST[0] == "<ADJLIST_START>"
	assert VOCAB_LIST[706] == "&"
	assert VOCAB_TOKEN_TO_INDEX["<UNK>"] == 19
	assert VOCAB_TOKEN_TO_INDEX["0"] == 320
	assert VOCAB_TOKEN_TO_INDEX["-1"] == 703
	assert VOCAB_TOKEN_TO_INDEX["(0,0)"] == 1596

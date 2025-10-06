"""legacy system for checking a `ModularMazeTokenizer` is valid -- compare its hash to a table of known hashes

this has been superseded by the fst system

"""

import hashlib
from pathlib import Path

import numpy as np
from jaxtyping import UInt32

# NOTE: these all need to match!

AllTokenizersHashBitLength = 32
"bit length of the hashes of all tokenizers, must match `AllTokenizersHashDtype` and `AllTokenizersHashesArray`"

AllTokenizersHashDtype = np.uint32
"numpy data type of the hashes of all tokenizers, must match `AllTokenizersHashBitLength` and `AllTokenizersHashesArray`"

AllTokenizersHashesArray = UInt32[np.ndarray, " n_tokens"]
"jaxtyping type of the hashes of all tokenizers, must match `AllTokenizersHashBitLength` and `AllTokenizersHashDtype`"


def _hash_tokenizer_name(s: str) -> int:
	h64: int = int.from_bytes(
		hashlib.shake_256(s.encode("utf-8")).digest(64),
		byteorder="big",
	)
	return (h64 >> 32) ^ (h64 & 0xFFFFFFFF)


_ALL_TOKENIZER_HASHES: AllTokenizersHashesArray
"private array of all tokenizer hashes"
_TOKENIZER_HASHES_PATH: Path = Path(__file__).parent / "MazeTokenizerModular_hashes.npz"
"path to where we expect the hashes file -- in the same dir as this file, by default. change with `set_tokenizer_hashes_path`"


def set_tokenizer_hashes_path(path: Path) -> None:
	"""set path to tokenizer hashes, and reload the hashes if needed

	the hashes are expected to be stored in and read from `_TOKENIZER_HASHES_PATH`,
	which by default is `Path(__file__).parent / "MazeTokenizerModular_hashes.npz"` or in this file's directory.

	However, this might not always work, so we provide a way to change this.
	"""
	global _TOKENIZER_HASHES_PATH, _ALL_TOKENIZER_HASHES  # noqa: PLW0603

	path = Path(path)
	if path.is_dir():
		path = path / "MazeTokenizerModular_hashes.npz"

	if not path.is_file():
		err_msg: str = f"could not find maze tokenizer hashes file at: {path}"
		raise FileNotFoundError(err_msg)

	if _TOKENIZER_HASHES_PATH.absolute() != path.absolute():
		# reload if they aren't equal
		_TOKENIZER_HASHES_PATH = path
		_ALL_TOKENIZER_HASHES = _load_tokenizer_hashes()
	else:
		# always set to new path
		_TOKENIZER_HASHES_PATH = path


def _load_tokenizer_hashes() -> AllTokenizersHashesArray:
	"""Loads the sorted list of `all_tokenizers.get_all_tokenizers()` hashes from disk."""
	global _TOKENIZER_HASHES_PATH  # noqa: PLW0602
	try:
		path: Path = _TOKENIZER_HASHES_PATH
		return np.load(path)["hashes"]
	except FileNotFoundError as e:
		err_msg: str = (
			"Tokenizers hashes cannot be loaded. To fix this, run"
			"\n`python -m maze-dataset.tokenization.save_hashes` which will save the hashes to"
			"\n`data/MazeTokenizerModular_hashes.npz`"
			"\nrelative to the current working directory -- this is where the code looks for them."
		)
		raise FileNotFoundError(err_msg) from e


def get_all_tokenizer_hashes() -> AllTokenizersHashesArray:
	"""returns all the tokenizer hashes in an `AllTokenizersHashesDtype` array, setting global variable if needed"""
	# naughty use of globals
	global _ALL_TOKENIZER_HASHES  # noqa: PLW0603
	try:
		got_tokenizers: bool = len(_ALL_TOKENIZER_HASHES) > 0
		if got_tokenizers:
			return _ALL_TOKENIZER_HASHES
		else:
			_ALL_TOKENIZER_HASHES = _load_tokenizer_hashes()
	except NameError:
		_ALL_TOKENIZER_HASHES = _load_tokenizer_hashes()

	return _ALL_TOKENIZER_HASHES

"implements the actual `MazeTokenizerModular` class"

import base64
import warnings
from functools import cached_property
from typing import (
	Iterable,
	Literal,
	Sequence,
	overload,
)

from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)
from muutils.misc import flatten
from muutils.misc.sequence import WhenMissing

# from maze_dataset import SolvedMaze
from maze_dataset.constants import (
	VOCAB,
	VOCAB_LIST,
	VOCAB_TOKEN_TO_INDEX,
	Coord,
	CoordTup,
)
from maze_dataset.maze.lattice_maze import LatticeMaze
from maze_dataset.token_utils import (
	TokenizerPendingDeprecationWarning,
	strings_to_coords,
)
from maze_dataset.tokenization.common import TokenError
from maze_dataset.tokenization.maze_tokenizer_legacy import (
	MazeTokenizer,
	TokenizationMode,
)
from maze_dataset.tokenization.modular.element_base import (
	_load_tokenizer_element,
	_TokenizerElement,
)
from maze_dataset.tokenization.modular.elements import CoordTokenizers, PromptSequencers
from maze_dataset.tokenization.modular.fst_load import check_tokenizer_in_fst
from maze_dataset.tokenization.modular.hashing import (
	_hash_tokenizer_name,
)


@serializable_dataclass(
	frozen=True,
	kw_only=True,
	properties_to_serialize=["tokenizer_element_tree_concrete", "name"],
)
class MazeTokenizerModular(SerializableDataclass):
	"""Tokenizer for mazes

	# Parameters
	- `prompt_sequencer`: Tokenizer element which assembles token regions (adjacency list, origin, target, path) into a complete prompt.

	# Development
	- To ensure backwards compatibility, the default constructor must always return a tokenizer equivalent to the legacy `TokenizationMode.AOTP_UT_Uniform`.
	- Furthermore, the mapping reflected in `from_legacy` must also be maintained.
	- Updates to `MazeTokenizerModular` or the `_TokenizerElement` hierarchy must maintain that behavior.
	"""

	prompt_sequencer: PromptSequencers._PromptSequencer = serializable_field(
		default=PromptSequencers.AOTP(),
		loading_fn=lambda x: _load_tokenizer_element(x, PromptSequencers),
	)

	def hash_int(self) -> int:
		"return integer hash using blake2b"
		return _hash_tokenizer_name(self.name)

	def __hash__(self) -> int:
		"Stable hash to identify unique `MazeTokenizerModular` instances. uses name"
		return self.hash_int()

	def hash_b64(self, n_bytes: int = 8) -> str:
		"""filename-safe base64 encoding of the hash"""
		# Use modulus to ensure the integer fits within n_bytes * 8 bits
		hash_mod: int = self.hash_int() % (1 << (n_bytes * 8))

		encoded = base64.b64encode(
			hash_mod.to_bytes(n_bytes, byteorder="big"),
			altchars=b"-_",
		).decode()

		# Remove any padding equals signs
		return encoded.rstrip("=")

	# Information Querying Methods

	@cached_property
	def tokenizer_elements(self) -> list[_TokenizerElement]:
		"returns a list of all the elements of this tokenizer"
		return [self.prompt_sequencer, *self.prompt_sequencer.tokenizer_elements()]

	def tokenizer_element_tree(self, abstract: bool = False) -> str:
		"""Returns a string representation of the tree of tokenizer elements contained in `self`.

		# Parameters
		- `abstract: bool`: Whether to print the name of the abstract base class or the concrete class for each `_TokenizerElement` instance.
		"""
		return "\n".join(
			[
				type(self).__name__,
				self.prompt_sequencer.tokenizer_element_tree(
					abstract=abstract,
					depth=1,
				),
			],
		)

	@property
	def tokenizer_element_tree_concrete(self) -> str:
		"""Property wrapper for `tokenizer_element_tree` so that it can be used in `properties_to_serialize`."""
		return self.tokenizer_element_tree()

	def tokenizer_element_dict(self) -> dict:
		"""Nested dictionary of the internal `TokenizerElement`s."""
		return {type(self).__name__: self.prompt_sequencer.tokenizer_element_dict()}

	@property
	def name(self) -> str:
		"""Serializes MazeTokenizer into a key for encoding in zanj"""
		return "-".join([type(self).__name__, self.prompt_sequencer.name])  # noqa: FLY002

	def summary(self) -> dict[str, str]:
		"""Single-level dictionary of the internal `TokenizerElement`s."""
		return {
			# "prompt_sequencer": self.prompt_sequencer.name,
			**{elem.attribute_key(): elem.name for elem in self.tokenizer_elements},
		}

	@staticmethod
	def _type_check(obj: any) -> None:
		"""Helper method for `has_element`"""
		if not (
			isinstance(obj, _TokenizerElement)
			or (isinstance(obj, type) and issubclass(obj, _TokenizerElement))
		):
			err_msg: str = f"{obj} is not a `_TokenizerElement` instance or subclass."
			raise TypeError(err_msg)

	def _has_element_singular(
		self,
		el: type[_TokenizerElement] | _TokenizerElement,
	) -> bool:
		"""Helper method for `has_element`"""
		self._type_check(el)
		if isinstance(el, type):
			return any(isinstance(e, el) for e in self.tokenizer_elements)
		else:
			return el in self.tokenizer_elements

	def has_element(
		self,
		*elements: Sequence[type[_TokenizerElement] | _TokenizerElement],
	) -> bool:
		"""Returns True if the `MazeTokenizerModular` instance contains ALL of the items specified in `elements`.

		Querying with a partial subset of `_TokenizerElement` fields is not currently supported.
		To do such a query, assemble multiple calls to `has_elements`.

		# Parameters
		- `elements`: Singleton or iterable of `_TokenizerElement` instances or classes.
		If an instance is provided, then comparison is done via instance equality.
		If a class is provided, then comparison isdone via `isinstance`. I.e., any instance of that class is accepted.
		"""
		if len(elements) == 1 and isinstance(elements[0], Iterable):
			elements = elements[0]
		return all(self._has_element_singular(e) for e in elements)

	def is_valid(self, do_except: bool = False) -> bool:
		"""Returns `True` if `self` is a valid tokenizer.

		Evaluates the validity of all of `self.tokenizer_elements` according to each one's method.
		"""
		return all(el.is_valid(do_except=do_except) for el in self.tokenizer_elements)

	def is_legacy_equivalent(self) -> bool:
		"""Returns if `self` has identical stringification behavior as any legacy `MazeTokenizer`."""
		return any(
			self == MazeTokenizerModular.from_legacy(tok_mode)
			for tok_mode in TokenizationMode
		)

	def is_tested_tokenizer(self, do_except: bool = False) -> bool:
		"""Returns if the tokenizer is returned by `all_tokenizers.get_all_tokenizers`, the set of tested and reliable tokenizers.

		uses an fst on the `name` attributes of all the tokenizers

		if `do_assert` is `True`, raises an `AssertionError` if the tokenizer is not tested.
		"""
		is_valid: bool = self.is_valid(do_except=do_except)
		in_tested_fst: bool = check_tokenizer_in_fst(self.name, do_except=do_except)

		if do_except:
			assert is_valid, "self.is_valid returns False"
			return True
		else:
			return in_tested_fst and is_valid

	def is_AOTP(self) -> bool:
		"is this tokenizer an AOTP tokenizer? AOTP = Adjacency list, Origin, Target, Path"
		return self.has_element(PromptSequencers.AOTP)

	def is_UT(self) -> bool:
		"is this tokenizer a UT tokenizer? UT = Unique Token (for each coord)"
		return self.has_element(CoordTokenizers.UT)

	# Alternate Constructors
	# ======================

	@classmethod
	def from_legacy(
		cls,
		legacy_maze_tokenizer: MazeTokenizer | TokenizationMode,
	) -> "MazeTokenizerModular":
		"""Maps a legacy `MazeTokenizer` or `TokenizationMode` to its equivalent `MazeTokenizerModular` instance."""
		if isinstance(legacy_maze_tokenizer, MazeTokenizer):
			legacy_maze_tokenizer = legacy_maze_tokenizer.tokenization_mode
		return {
			TokenizationMode.AOTP_UT_uniform: MazeTokenizerModular(),
			TokenizationMode.AOTP_UT_rasterized: MazeTokenizerModular(),
			TokenizationMode.AOTP_CTT_indexed: MazeTokenizerModular(
				prompt_sequencer=PromptSequencers.AOTP(
					coord_tokenizer=CoordTokenizers.CTT(),
				),
			),
		}[legacy_maze_tokenizer]

	# Simple properties
	# =================
	@classmethod
	def from_tokens(
		cls,
		tokens: str | list[str],
	) -> "MazeTokenizerModular":
		"""Infers most `MazeTokenizerModular` parameters from a full sequence of tokens."""
		raise NotImplementedError(
			"Recovering tokenizer objects from MazeTokenizerModular-produced strings is not supported",
		)

	@property
	def token_arr(self) -> list[str] | None:
		"""map from index to token"""
		return VOCAB_LIST

	@property
	def tokenizer_map(self) -> dict[str, int]:
		"""map from token to index"""
		return VOCAB_TOKEN_TO_INDEX

	@property
	def vocab_size(self) -> int:
		"""Number of tokens in the static vocab"""
		return len(VOCAB_LIST)

	@property
	def n_tokens(self) -> int:
		"get the number of tokens in the vocabulary (deprecated)"
		err_msg: str = "`MazeTokenizerModular.n_tokens` has been removed. Use `len(maze_dataset.VOCAB_LIST)` instead."
		raise NameError(err_msg)

	@property
	def padding_token_index(self) -> int:
		"get the index of the padding token"
		return VOCAB_TOKEN_TO_INDEX[VOCAB.PADDING]

	# conversion functions
	# ============================================================

	def to_tokens(
		self,
		maze: LatticeMaze,
	) -> list[str]:
		"""Converts maze into a list of tokens."""
		return self.prompt_sequencer.to_tokens(maze)

	def coords_to_strings(self, coords: list[CoordTup | Coord]) -> list[str]:
		"calls self.prompt_sequencer.coord_tokenizer.to_tokens(c) for each c in coords"
		return list(
			flatten(
				[self.prompt_sequencer.coord_tokenizer.to_tokens(c) for c in coords],
			),
		)

	# TODO: unclear why we need to use `noqa: N805` here since its a classmethod
	# maybe we need to hit every overload with `@classmethod`?
	@overload
	def strings_to_coords(
		cls,  # noqa: N805
		text: str | list[str],
		when_noncoord: Literal["skip"] = "skip",
	) -> list[CoordTup]: ...
	@overload
	def strings_to_coords(
		cls,  # noqa: N805
		text: str | list[str],
		when_noncoord: Literal["error"] = "error",
	) -> list[CoordTup]: ...
	@overload
	def strings_to_coords(
		cls,  # noqa: N805
		text: str | list[str],
		when_noncoord: Literal["include"] = "include",
	) -> list[str | CoordTup]: ...
	@classmethod
	def strings_to_coords(
		cls,
		text: str | list[str],
		when_noncoord: WhenMissing = "skip",
	) -> list[str | CoordTup]:
		"wrapper for maze_dataset.token_utils.strings_to_coords"
		warnings.warn(
			"`MazeTokenizerModular.strings_to_coords` only supports legacy UT strings.",
			TokenizerPendingDeprecationWarning,
		)
		return strings_to_coords(text=text, when_noncoord=when_noncoord)

	@staticmethod
	def encode(text: str | list[str]) -> list[int]:
		"""encode a string or list of strings into a list of tokens"""
		try:
			if isinstance(text, str):
				text = text.split()
			return [VOCAB_TOKEN_TO_INDEX[token] for token in text]
		except KeyError as e:
			err_msg: str = f"Token {e} not found in `VOCAB`."
			raise TokenError(err_msg) from e

	@staticmethod
	def decode(
		token_ids: Sequence[int],
		joined_tokens: bool = False,
	) -> list[str] | str:
		"""decode a list of tokens into a string or list of strings"""
		try:
			output: list[str] = [VOCAB_LIST[token_id] for token_id in token_ids]
		except IndexError as e:
			err_msg: str = f"Token index '{e}' not found in `VOCAB`."
			raise TokenError(err_msg) from e
		if joined_tokens:
			return " ".join(output)
		else:
			return output

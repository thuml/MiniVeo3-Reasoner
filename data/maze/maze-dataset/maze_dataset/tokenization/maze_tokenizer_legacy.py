"""legacy tokenizer which uses a `TokenizationMode` enum and a `MazeTokenizer` class

> [!CAUTION]
> `MazeTokenizerModular` is the new standard for tokenization. This class is no longer recommended
> for use, but will remain for compatibility with existing code.

"""

import warnings
from enum import Enum
from functools import cached_property
from typing import (
	Callable,
	Iterable,
	Literal,
	Mapping,
	Sequence,
	overload,
)

import numpy as np
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)
from muutils.kappa import Kappa
from muutils.misc.sequence import WhenMissing

# from maze_dataset import SolvedMaze
from maze_dataset.constants import (
	SPECIAL_TOKENS,
	CoordTup,
)
from maze_dataset.token_utils import (
	TokenizerPendingDeprecationWarning,
	_coord_to_strings_indexed,
	_coord_to_strings_UT,
	coords_to_strings,
	strings_to_coords,
)
from maze_dataset.tokenization.common import TokenError
from maze_dataset.utils import corner_first_ndindex


class TokenizationMode(Enum):
	"""legacy tokenization modes

	> [!CAUTION]
	> Legacy mode of tokenization. will still be around in future releases, but is no longer recommended for use.
	> Use `MazeTokenizerModular` instead.

	# Abbreviations:
	- `AOTP`: Ajacency list, Origin, Target, Path
	- `UT`: Unique Token (for each coordiate)
	- `CTT`: Coordinate Tuple Tokens (each coordinate is tokenized as a tuple of integers)

	# Modes:
	- `AOTP_UT_rasterized`: the "classic" mode: assigning tokens to each coordinate is done via rasterization
		example: for a 3x3 maze, token order is `(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)`
	- `AOTP_UT_uniform`: new mode, where a 3x3 tokenization scheme and 5x5 tokenizations scheme are compatible
		uses `corner_first_ndindex` function to order the tokens
	- `AOTP_CTT_indexed`: each coordinate is a tuple of integers
	"""

	AOTP_UT_rasterized = "AOTP_UT_rasterized"
	AOTP_UT_uniform = "AOTP_UT_uniform"
	AOTP_CTT_indexed = "AOTP_CTT_indexed"

	def to_legacy_tokenizer(self, max_grid_size: int | None = None) -> "MazeTokenizer":
		"convert the mode to a legacy `MazeTokenizer` object given a `max_grid_size`"
		return MazeTokenizer(tokenization_mode=self, max_grid_size=max_grid_size)


_NDINDEX_FUNC_MAP: dict[
	TokenizationMode,
	Callable[[int], Iterable[tuple[int, ...]]],
] = {
	TokenizationMode.AOTP_UT_rasterized: lambda n: list(np.ndindex(n, n)),
	TokenizationMode.AOTP_UT_uniform: lambda n: corner_first_ndindex(n, 2),
}


def is_UT(tokenization_mode: TokenizationMode) -> bool:
	"returns true if a tokenization mode is a UT mode: UT = Unique Token (for each coordinate)"
	return tokenization_mode in (
		TokenizationMode.AOTP_UT_rasterized,
		TokenizationMode.AOTP_UT_uniform,
	)


def get_tokens_up_to_path_start(
	tokens: list[str],
	include_start_coord: bool = True,
	tokenization_mode: TokenizationMode = TokenizationMode.AOTP_UT_uniform,
) -> list[str]:
	"""get tokens up to the path start token

	# Parameters:
	- `tokens : list[str]`
	- `include_start_coord : bool`
		(defaults to `True`)
	- `tokenization_mode : TokenizationMode`
		(defaults to `TokenizationMode.AOTP_UT_uniform`)

	# Returns:
	- `list[str]` subsequence of `tokens` up to the path start token

	# Raises:
	- `ValueError` : if `tokenization_mode` is invalid
	"""
	warnings.warn(
		"`maze_tokenizer.get_tokens_up_to_path_start` will be deprecated for a `MazeTokenizerModular`-compatible function in a future release.",
		TokenizerPendingDeprecationWarning,
	)
	path_start_idx: int = tokens.index(SPECIAL_TOKENS.PATH_START) + 1
	if include_start_coord:
		if is_UT(tokenization_mode):
			return tokens[: path_start_idx + 1]
		elif tokenization_mode == TokenizationMode.AOTP_CTT_indexed:
			return tokens[: path_start_idx + 5]
		else:
			err_msg: str = f"Invalid tokenization mode: {tokenization_mode}"
			raise ValueError(err_msg)
	else:
		return tokens[:path_start_idx]


_MAZETOKENIZER_PROPERTIES_TO_SERIALIZE: list[str] = [
	"name",
	"max_grid_size",
	"token_arr",
	"tokenizer_map",
	"vocab_size",
	"padding_token_index",
]


@serializable_dataclass(
	properties_to_serialize=_MAZETOKENIZER_PROPERTIES_TO_SERIALIZE,
	kw_only=True,
)
class MazeTokenizer(SerializableDataclass):
	"""LEGACY Tokenizer for mazes

	> [!CAUTION]
	> `MazeTokenizerModular` is the new standard for tokenization. This class is no longer recommended
	> for use, but will remain for compatibility with existing code.

	# Parameters:
	- `tokenization_mode: TokenizationMode`
		mode of tokenization. required.
	- `max_grid_size: int | None`
		maximum grid size. required for actually turning text tokens to numerical tokens, but not for moving between coordinates/mazes and text

	# Properties
	- `name: str`
		auto-generated name of the tokenizer from mode and size

	## Conditional Properties

	- `node_strings_map: Mapping[CoordTup, str]`
		map from node to string. This returns a `muutils.kappa.Kappa` object which you can use like a dictionary. returns `None` if not a `UT` mode

	these all return `None` if `max_grid_size` is `None`.
	Prepend `_` to the name to get a guaranteed type, and cause an exception if `max_grid_size` is `None`

	- `token_arr: list[str]`
		list of tokens, in order of their indices in the vocabulary
	- `tokenizer_map: Mapping[str, int]`
		map from token to index
	- `vocab_size: int`
		size of the vocabulary
	- `padding_token_index: int`
		index of the padding token

	# Methods
	- `coords_to_strings(coords: list[CoordTup]) -> list[str]`
		convert a list of coordinates to a list of tokens. Optionally except, skip, or ignore non-coordinates
	- `strings_to_coords(strings: list[str]) -> list[CoordTup]`
		convert a list of tokens to a list of coordinates. Optionally except, skip, or ignore non-coordinates

	"""

	# parameters
	# ============================================================

	tokenization_mode: TokenizationMode = serializable_field(
		default=TokenizationMode.AOTP_UT_uniform,
		serialization_fn=lambda x: x.value,
		loading_fn=lambda x: TokenizationMode[x["tokenization_mode"]],
	)

	max_grid_size: int | None = serializable_field(default=None)

	# properties
	# ============================================================

	@property
	def name(self) -> str:
		"""auto-generated name of the tokenizer from mode and size"""
		max_grid_size_str: str = (
			f"-g{self.max_grid_size}" if self.max_grid_size is not None else ""
		)
		return f"maze_tokenizer-{self.tokenization_mode.value}{max_grid_size_str}"

	@cached_property
	def _node_strings_map(self) -> Mapping[CoordTup, list[str]]:
		"""map a coordinate to a token"""
		if self.tokenization_mode in (
			TokenizationMode.AOTP_UT_rasterized,
			TokenizationMode.AOTP_UT_uniform,
		):
			return Kappa(_coord_to_strings_UT)
		elif self.tokenization_mode == TokenizationMode.AOTP_CTT_indexed:
			return Kappa(_coord_to_strings_indexed)
		else:
			err_msg: str = f"Invalid tokenization mode {self.tokenization_mode}, expected one of {TokenizationMode.__members__}"
			raise ValueError(err_msg)

	@cached_property
	def node_strings_map(self) -> Mapping[CoordTup, list[str]] | None:
		"""map a coordinate to a token"""
		if self.tokenization_mode in (
			TokenizationMode.AOTP_UT_rasterized,
			TokenizationMode.AOTP_UT_uniform,
		):
			return None
		else:
			return self._node_strings_map

	# conditional properties (on max_grid_size existing)
	# ------------------------------------------------------------

	@cached_property
	def _token_arr(self) -> list[str]:
		"""map from index to token"""
		if self.max_grid_size is None:
			err_msg: str = f"max_grid_size must be specified to use token_arr property: {self.max_grid_size = }"
			raise ValueError(err_msg)

		output: list[str] = list(SPECIAL_TOKENS.values())

		if self.tokenization_mode in (
			TokenizationMode.AOTP_UT_rasterized,
			TokenizationMode.AOTP_UT_uniform,
		):
			output.extend(
				[
					self._node_strings_map[coord][0]
					for coord in _NDINDEX_FUNC_MAP[self.tokenization_mode](
						self.max_grid_size,
					)
				],
			)
		elif self.tokenization_mode == TokenizationMode.AOTP_CTT_indexed:
			# TODO: this is hacky, but we don't want to modify the original SPECIAL_TOKENS since that will break old models
			output.extend(
				[
					"(",
					",",
					")",  # new special chars
					*map(str, range(self.max_grid_size)),  # numbers
				],
			)
		else:
			err_msg: str = (
				f"Invalid tokenization mode {self.tokenization_mode}, expected one of {TokenizationMode.__members__}",
			)
			raise ValueError(err_msg)

		return output

	@cached_property
	def token_arr(self) -> list[str] | None:
		"get the token array if the max_grid_size is specified"
		if self.max_grid_size is None:
			return None
		return self._token_arr

	@cached_property
	def _tokenizer_map(self) -> dict[str, int]:
		"""map from token to index"""
		return {token: i for i, token in enumerate(self._token_arr)}

	@cached_property
	def tokenizer_map(self) -> dict[str, int] | None:
		"get the tokenizer map if the max_grid_size is specified"
		if self.max_grid_size is None:
			return None
		return self._tokenizer_map

	@property
	def _vocab_size(self) -> int:
		return len(self._token_arr)

	@property
	def vocab_size(self) -> int | None:
		"get the size of the vocabulary if the max_grid_size is specified"
		if self.max_grid_size is None:
			return None
		return self._vocab_size

	@property
	def _n_tokens(self) -> int:
		# TODO: deprecate
		return self._vocab_size

	@property
	def n_tokens(self) -> int | None:
		"get the number of tokens if the max_grid_size is specified"
		if self.max_grid_size is None:
			return None
		return self._n_tokens

	@cached_property
	def _padding_token_index(self) -> int:
		return self.tokenizer_map[SPECIAL_TOKENS.PADDING]

	@cached_property
	def padding_token_index(self) -> int | None:
		"get the index of the padding token if it exists"
		if self.max_grid_size is None:
			return None
		return self._padding_token_index

	# conversion functions
	# ============================================================

	@overload
	def coords_to_strings(
		self,
		coords: list[str | CoordTup],
		when_noncoord: Literal["include", "skip"] = "skip",
	) -> list[str]: ...
	@overload
	def coords_to_strings(
		self,
		coords: list[CoordTup],
		when_noncoord: Literal["error"] = "error",
	) -> list[str]: ...
	def coords_to_strings(
		self,
		coords: list[CoordTup],
		when_noncoord: WhenMissing = "skip",
	) -> list[str]:
		"""map a list of coordinate tuples (and maybe other tokens) to strings

		wraps `maze_dataset.token_utils.coords_to_strings` with either
		`_coord_to_strings_UT` or `_coord_to_strings_indexed` depending on the tokenization mode
		"""
		if self.tokenization_mode in (
			TokenizationMode.AOTP_UT_rasterized,
			TokenizationMode.AOTP_UT_uniform,
		):
			return coords_to_strings(
				coords=coords,
				coord_to_strings_func=_coord_to_strings_UT,
				when_noncoord=when_noncoord,
			)
		elif self.tokenization_mode == TokenizationMode.AOTP_CTT_indexed:
			return coords_to_strings(
				coords=coords,
				coord_to_strings_func=_coord_to_strings_indexed,
				when_noncoord=when_noncoord,
			)
		else:
			err_msg: str = f"Invalid tokenization mode {self.tokenization_mode}, expected one of {TokenizationMode.__members__}"
			raise ValueError(err_msg)

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
		"wrapper for `maze_dataset.token_utils.strings_to_coords`"
		return strings_to_coords(text=text, when_noncoord=when_noncoord)

	def encode(self, text: str | list[str]) -> list[int]:
		"""encode a string or list of strings into a list of tokens"""
		try:
			if isinstance(text, str):
				text = text.split()
			return [self.tokenizer_map[token] for token in text]
		except KeyError as e:
			err_msg: str = (
				f"Token {e} not found in vocabulary of {self}:\n{self.token_arr}"
			)
			raise TokenError(err_msg) from e

	def decode(
		self,
		tokens: Sequence[int],
		joined_tokens: bool = False,
	) -> list[str] | str:
		"""decode a list of tokens into a string or list of strings"""
		try:
			output: list[str] = [self.token_arr[token] for token in tokens]
		except IndexError as e:
			err_msg: str = (
				f"Token index '{e}' not found in vocabulary of length {self.vocab_size}"
			)
			raise TokenError(err_msg) from e
		if joined_tokens:
			return " ".join(output)
		else:
			return output

	# UT-only coordinate stuff
	# ============================================================

	@cached_property
	def coordinate_tokens_coords(self) -> dict[CoordTup, int]:
		"map of coordiante tuples to their token ids, only valid for UT"
		# print(f"{self.tokenization_mode = }")
		if not self.is_UT():
			err_msg: str = f"coordinate_tokens_coords is only valid for UT tokenization modes, got {self.tokenization_mode = }"
			raise ValueError(err_msg)

		if self.max_grid_size is None:
			err_msg: str = f"max_grid_size must be specified to use coordinate_tokens: {self.max_grid_size = }"
			raise ValueError(err_msg)

		raw_converted: list[CoordTup | str] = self.strings_to_coords(
			self.token_arr,
			when_noncoord="include",
		)

		# filter out non-coordinates
		return {
			coord: i
			for i, coord in enumerate(raw_converted)
			if not isinstance(coord, str)
		}

	@cached_property
	def coordinate_tokens_ids(self) -> dict[str, int]:
		"map of coordinate tokens to their token ids, only valid for UT"
		# checks performed in call
		output: dict[str, int] = dict()

		for coord, index in self.coordinate_tokens_coords.items():
			_for_key: list[str] = self.coords_to_strings([coord])
			assert len(_for_key) == 1
			output[_for_key[0]] = index

		return output

	# other
	# ============================================================

	def summary(self) -> dict:
		"""returns a summary of the tokenization mode"""
		return {
			"tokenization_mode": self.tokenization_mode.value,
			"max_grid_size": self.max_grid_size,
			"vocab_size": self.vocab_size,
		}

	def is_AOTP(self) -> bool:
		"""returns true if a tokenization mode is Adjacency list, Origin, Target, Path"""
		return self.tokenization_mode in (
			TokenizationMode.AOTP_UT_rasterized,
			TokenizationMode.AOTP_UT_uniform,
			TokenizationMode.AOTP_CTT_indexed,
		)

	def is_UT(self) -> bool:
		"returns true if a tokenization mode is a UT mode: UT = Unique Token (for each coordinate)"
		return is_UT(self.tokenization_mode)

	def clear_cache(self) -> None:
		"""clears all cached properties"""
		# delete the properties only if they exist
		for name, prop in self.__class__.__dict__.items():
			if isinstance(prop, cached_property):
				# if the property exists, delete it
				try:  # noqa: SIM105
					delattr(self, name)
				except AttributeError:
					pass

"""constants and type hints used accross the package"""

import warnings
from dataclasses import dataclass, field, make_dataclass
from typing import Iterator

import numpy as np
from jaxtyping import Bool, Int8

from maze_dataset.utils import corner_first_ndindex

# various type hints for coordinates, connections, etc.

Coord = Int8[np.ndarray, "row_col=2"]
"single coordinate as array"

CoordTup = tuple[int, int]
"single coordinate as tuple"

CoordArray = Int8[np.ndarray, "coord row_col=2"]
"array of coordinates"

CoordList = list[CoordTup]
"list of tuple coordinates"

Connection = Int8[np.ndarray, "coord=2 row_col=2"]
"single connection (pair of coords) as array"

ConnectionList = Bool[np.ndarray, "lattice_dim=2 row col"]
"internal representation used in `LatticeMaze`"

ConnectionArray = Int8[np.ndarray, "edges leading_trailing_coord=2 row_col=2"]
"n_edges * 2 * 2 array of connections, like an adjacency list"


class SpecialTokensError(Exception):
	"(unused!) errors related to special tokens"

	pass


_SPECIAL_TOKENS_ABBREVIATIONS: dict[str, str] = {
	"<ADJLIST_START>": "<A_S>",
	"<ADJLIST_END>": "<A_E>",
	"<TARGET_START>": "<T_S>",
	"<TARGET_END>": "<T_E>",
	"<ORIGIN_START>": "<O_S>",
	"<ORIGIN_END>": "<O_E>",
	"<PATH_START>": "<P_S>",
	"<PATH_END>": "<P_E>",
	"<-->": "<-->",
	";": ";",
	"<PADDING>": "<PAD>",
}
"map abbreviations for (some) special tokens"


@dataclass(frozen=True)
class _SPECIAL_TOKENS_BASE:  # noqa: N801
	"special dataclass used for handling special tokens"

	ADJLIST_START: str = "<ADJLIST_START>"
	ADJLIST_END: str = "<ADJLIST_END>"
	TARGET_START: str = "<TARGET_START>"
	TARGET_END: str = "<TARGET_END>"
	ORIGIN_START: str = "<ORIGIN_START>"
	ORIGIN_END: str = "<ORIGIN_END>"
	PATH_START: str = "<PATH_START>"
	PATH_END: str = "<PATH_END>"
	CONNECTOR: str = "<-->"
	ADJACENCY_ENDLINE: str = ";"
	PADDING: str = "<PADDING>"

	def __getitem__(self, key: str) -> str:
		key_upper: str = key.upper()

		if not isinstance(key, str):
			err_msg: str = f"key must be str, not {type(key)}"
			raise TypeError(err_msg)

		# error checking for old lowercase format
		if key != key_upper:
			warnings.warn(
				f"Accessing special token '{key}' without uppercase. this is deprecated and will be removed in the future.",
				DeprecationWarning,
			)
			key = key_upper

		# `ADJLIST` used to be `adj_list`, changed to match actual token content
		if key_upper not in self.keys():
			key_upper_modified: str = key_upper.replace("ADJ_LIST", "ADJLIST")
			if key_upper_modified in self.keys():
				warnings.warn(
					f"Accessing '{key}' in old format, should use {key_upper_modified}. this is deprecated and will be removed in the future.",
					DeprecationWarning,
				)
				return getattr(self, key_upper_modified)
			else:
				err_msg: str = f"invalid special token '{key}'"
				raise KeyError(err_msg)

		# normal return
		return getattr(self, key.upper())

	def get_abbrev(self, key: str) -> str:
		return _SPECIAL_TOKENS_ABBREVIATIONS[self[key]]

	def __iter__(self) -> Iterator[str]:
		return iter(self.__dict__.keys())

	def __len__(self) -> int:
		return len(self.__dict__.keys())

	def __contains__(self, key: str) -> bool:
		return key in self.__dict__

	def values(self) -> Iterator[str]:
		return self.__dict__.values()

	def items(self) -> Iterator[tuple[str, str]]:
		return self.__dict__.items()

	def keys(self) -> Iterator[str]:
		return self.__dict__.keys()


SPECIAL_TOKENS: _SPECIAL_TOKENS_BASE = _SPECIAL_TOKENS_BASE()
"special tokens"


DIRECTIONS_MAP: Int8[np.ndarray, "direction axes"] = np.array(
	[
		[0, 1],  # down
		[0, -1],  # up
		[1, 1],  # right
		[1, -1],  # left
	],
)
"down, up, right, left directions for when inside a `ConnectionList`"


NEIGHBORS_MASK: Int8[np.ndarray, "coord point"] = np.array(
	[
		[0, 1],  # down
		[0, -1],  # up
		[1, 0],  # right
		[-1, 0],  # left
	],
)
"down, up, right, left as vectors"


# last element of the tuple is actually a Field[str], but mypy complains
_VOCAB_FIELDS: list[tuple[str, type[str], str]] = [
	# *[(k, str, field(default=v)) for k, v in SPECIAL_TOKENS.items()],
	("COORD_PRE", str, field(default="(")),
	("COORD_INTRA", str, field(default=",")),
	("COORD_POST", str, field(default=")")),
	("TARGET_INTRA", str, field(default="=")),
	("TARGET_POST", str, field(default="||")),
	("PATH_INTRA", str, field(default=":")),
	("PATH_POST", str, field(default="THEN")),
	("NEGATIVE", str, field(default="-")),
	("UNKNOWN", str, field(default="<UNK>")),
	*[
		(f"TARGET_{a}", str, field(default=f"TARGET_{a}"))
		for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	],
	("TARGET_NORTH", str, field(default="TARGET_NORTH")),
	("TARGET_SOUTH", str, field(default="TARGET_SOUTH")),
	("TARGET_EAST", str, field(default="TARGET_EAST")),
	("TARGET_WEST", str, field(default="TARGET_WEST")),
	("TARGET_NORTHEAST", str, field(default="TARGET_NORTHEAST")),
	("TARGET_NORTHWEST", str, field(default="TARGET_NORTHWEST")),
	("TARGET_SOUTHEAST", str, field(default="TARGET_SOUTHEAST")),
	("TARGET_SOUTHWEST", str, field(default="TARGET_SOUTHWEST")),
	("TARGET_CENTER", str, field(default="TARGET_CENTER")),
	("PATH_NORTH", str, field(default="NORTH")),
	("PATH_SOUTH", str, field(default="SOUTH")),
	("PATH_EAST", str, field(default="EAST")),
	("PATH_WEST", str, field(default="WEST")),
	("PATH_FORWARD", str, field(default="FORWARD")),
	("PATH_BACKWARD", str, field(default="BACKWARD")),
	("PATH_LEFT", str, field(default="LEFT")),
	("PATH_RIGHT", str, field(default="RIGHT")),
	("PATH_STAY", str, field(default="STAY")),
	*[
		(f"I_{i:03}", str, field(default=f"+{i}")) for i in range(256)
	],  # General purpose positive int tokens. Used by `StepTokenizers.Distance`.
	*[
		(f"CTT_{i}", str, field(default=f"{i}")) for i in range(128)
	],  # Coord tuple tokens
	*[
		(f"I_N{-i:03}", str, field(default=f"{i}")) for i in range(-256, 0)
	],  # General purpose negative int tokens
	("PATH_PRE", str, field(default="STEP")),
	("ADJLIST_PRE", str, field(default="ADJ_GROUP")),
	("ADJLIST_INTRA", str, field(default="&")),
	("ADJLIST_WALL", str, field(default="<XX>")),
	*[(f"RESERVE_{i}", str, field(default=f"<RESERVE_{i}>")) for i in range(708, 1596)],
	*[
		(f"UT_{x:02}_{y:02}", str, field(default=f"({x},{y})"))
		for x, y in corner_first_ndindex(50)
	],
]
"fields for the `MazeTokenizerModular` style combined vocab"

_VOCAB_BASE: type = make_dataclass(
	"_VOCAB_BASE",
	fields=_VOCAB_FIELDS,
	bases=(_SPECIAL_TOKENS_BASE,),
	frozen=True,
)
"combined vocab class, private"
# TODO: edit __getitem__ to add warning for accessing a RESERVE token

# HACK: mypy doesn't recognize the fields in this dataclass
VOCAB: _VOCAB_BASE = _VOCAB_BASE()  # type: ignore
"public access to universal vocabulary for `MazeTokenizerModular`"
VOCAB_LIST: list[str] = list(VOCAB.values())
"list of `VOCAB` tokens, in order"
VOCAB_TOKEN_TO_INDEX: dict[str, int] = {token: i for i, token in enumerate(VOCAB_LIST)}
"map of `VOCAB` tokens to their indices"

# CARDINAL_MAP: Maps tuple(coord1 - coord0) : cardinal direction
CARDINAL_MAP: dict[tuple[int, int], str] = {
	(-1, 0): VOCAB.PATH_NORTH,
	(1, 0): VOCAB.PATH_SOUTH,
	(0, -1): VOCAB.PATH_WEST,
	(0, 1): VOCAB.PATH_EAST,
}
"map of cardinal directions to appropriate tokens"

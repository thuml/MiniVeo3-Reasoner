"""Contains `get_all_tokenizers()` and supporting limited-use functions.

# `get_all_tokenizers()`
returns a comprehensive collection of all valid `MazeTokenizerModular` objects.
This is an overwhelming majority subset of the set of all possible `MazeTokenizerModular` objects.
Other tokenizers not contained in `get_all_tokenizers()` may be possible to construct, but they are untested and not guaranteed to work.
This collection is in a separate module since it is expensive to compute and will grow more expensive as features are added to `MazeTokenizerModular`.

## Use Cases
In general, uses for this module are limited to development of the library and specific research studying many tokenization behaviors.
- Unit testing:
  - Tokenizers to use in unit tests are sampled from `get_all_tokenizers()`
- Large-scale tokenizer research:
  - Specific research training models on many tokenization behaviors can use `get_all_tokenizers()` as the maximally inclusive collection
  - `get_all_tokenizers()` may be subsequently filtered using `MazeTokenizerModular.has_element`
For other uses, it's likely that the computational expense can be avoided by using
- `maze_tokenizer.get_all_tokenizer_hashes()` for membership checks
- `utils.all_instances` for generating smaller subsets of `MazeTokenizerModular` or `_TokenizerElement` objects

# `EVERY_TEST_TOKENIZERS`
A collection of the tokenizers which should always be included in unit tests when test fuzzing is used.
This collection should be expanded as specific tokenizers become canonical or popular.
"""

import functools
import multiprocessing
import random
from functools import cache
from pathlib import Path
from typing import Callable

import frozendict
import numpy as np
from muutils.spinner import NoOpContextManager, SpinnerContext
from tqdm import tqdm

from maze_dataset.tokenization import (
	CoordTokenizers,
	MazeTokenizerModular,
	PromptSequencers,
	StepTokenizers,
	_TokenizerElement,
)
from maze_dataset.tokenization.modular.all_instances import FiniteValued, all_instances
from maze_dataset.tokenization.modular.hashing import (
	AllTokenizersHashBitLength,
	AllTokenizersHashDtype,
	AllTokenizersHashesArray,
)

# Always include this as the first item in the dict `validation_funcs` whenever using `all_instances` with `MazeTokenizerModular`
# TYPING: error: Type variable "maze_dataset.utils.FiniteValued" is unbound  [valid-type]
#   note: (Hint: Use "Generic[FiniteValued]" or "Protocol[FiniteValued]" base class to bind "FiniteValued" inside a class)
#   note: (Hint: Use "FiniteValued" in function signature to bind "FiniteValued" inside a function)
MAZE_TOKENIZER_MODULAR_DEFAULT_VALIDATION_FUNCS: frozendict.frozendict[
	type[FiniteValued],
	Callable[[FiniteValued], bool],
] = frozendict.frozendict(
	{
		# TYPING: Item "bool" of the upper bound "bool | IsDataclass | Enum" of type variable "FiniteValued" has no attribute "is_valid"  [union-attr]
		_TokenizerElement: lambda x: x.is_valid(),
		# Currently no need for `MazeTokenizerModular.is_valid` since that method contains no special cases not already covered by `_TokenizerElement.is_valid`
		# MazeTokenizerModular: lambda x: x.is_valid(),
		# TYPING: error: No overload variant of "set" matches argument type "FiniteValued"  [call-overload]
		#   note: Possible overload variants:
		#   note:     def [_T] set(self) -> set[_T]
		#   note:     def [_T] set(self, Iterable[_T], /) -> set[_T]
		# TYPING: error: Argument 1 to "len" has incompatible type "FiniteValued"; expected "Sized"  [arg-type]
		StepTokenizers.StepTokenizerPermutation: lambda x: len(set(x)) == len(x)
		and x != (StepTokenizers.Distance(),),
	},
)

DOWNLOAD_URL: str = "https://raw.githubusercontent.com/understanding-search/maze-dataset/main/maze_dataset/tokenization/MazeTokenizerModular_hashes.npz"


@cache
def get_all_tokenizers() -> list[MazeTokenizerModular]:
	"""Computes a complete list of all valid tokenizers.

	Warning: This is an expensive function.
	"""
	return list(
		all_instances(
			MazeTokenizerModular,
			validation_funcs=MAZE_TOKENIZER_MODULAR_DEFAULT_VALIDATION_FUNCS,
		),
	)


@cache
def get_all_tokenizers_names() -> list[str]:
	"""computes the sorted list of names of all tokenizers"""
	return sorted([tokenizer.name for tokenizer in get_all_tokenizers()])


EVERY_TEST_TOKENIZERS: list[MazeTokenizerModular] = [
	MazeTokenizerModular(),
	MazeTokenizerModular(
		prompt_sequencer=PromptSequencers.AOTP(coord_tokenizer=CoordTokenizers.CTT()),
	),
	# TODO: add more here as specific tokenizers become canonical and frequently used
]


@cache
def all_tokenizers_set() -> set[MazeTokenizerModular]:
	"""Casts `get_all_tokenizers()` to a set."""
	return set(get_all_tokenizers())


@cache
def _all_tokenizers_except_every_test_tokenizers() -> list[MazeTokenizerModular]:
	"""Returns"""
	return list(all_tokenizers_set().difference(EVERY_TEST_TOKENIZERS))


def sample_all_tokenizers(n: int) -> list[MazeTokenizerModular]:
	"""Samples `n` tokenizers from `get_all_tokenizers()`."""
	return random.sample(get_all_tokenizers(), n)


def sample_tokenizers_for_test(n: int | None) -> list[MazeTokenizerModular]:
	"""Returns a sample of size `n` of unique elements from `get_all_tokenizers()`,

	always including every element in `EVERY_TEST_TOKENIZERS`.
	"""
	if n is None:
		return get_all_tokenizers()

	if n < len(EVERY_TEST_TOKENIZERS):
		err_msg: str = f"`n` must be at least {len(EVERY_TEST_TOKENIZERS) = } such that the sample can contain `EVERY_TEST_TOKENIZERS`."
		raise ValueError(
			err_msg,
		)
	sample: list[MazeTokenizerModular] = random.sample(
		_all_tokenizers_except_every_test_tokenizers(),
		n - len(EVERY_TEST_TOKENIZERS),
	)
	sample.extend(EVERY_TEST_TOKENIZERS)
	return sample


def save_hashes(
	path: Path | None = None,
	verbose: bool = False,
	parallelize: bool | int = False,
) -> AllTokenizersHashesArray:
	"""Computes, sorts, and saves the hashes of every member of `get_all_tokenizers()`."""
	spinner = (
		functools.partial(SpinnerContext, spinner_chars="square_dot")
		if verbose
		else NoOpContextManager
	)

	# get all tokenizers
	with spinner(initial_value="getting all tokenizers...", update_interval=2.0):
		all_tokenizers = get_all_tokenizers()

	# compute hashes
	hashes_array_np64: AllTokenizersHashesArray
	if parallelize:
		n_cpus: int = (
			parallelize if int(parallelize) > 1 else multiprocessing.cpu_count()
		)
		with spinner(  # noqa: SIM117
			initial_value=f"using {n_cpus} processes to compute {len(all_tokenizers)} tokenizer hashes...",
			update_interval=2.0,
		):
			with multiprocessing.Pool(processes=n_cpus) as pool:
				hashes_list: list[int] = list(pool.map(hash, all_tokenizers))

		with spinner(initial_value="converting hashes to numpy array..."):
			hashes_array_np64 = np.array(hashes_list, dtype=np.int64)
	else:
		with spinner(
			initial_value=f"computing {len(all_tokenizers)} tokenizer hashes...",
		):
			hashes_array_np64 = np.array(
				[
					hash(obj)  # uses stable hash
					for obj in tqdm(all_tokenizers, disable=not verbose)
				],
				dtype=np.int64,
			)

	# convert to correct dtype
	hashes_array: AllTokenizersHashesArray = (
		hashes_array_np64 % (1 << AllTokenizersHashBitLength)
		if AllTokenizersHashBitLength < 64  # noqa: PLR2004
		else hashes_array_np64
	).astype(AllTokenizersHashDtype)

	# make sure there are no dupes
	with spinner(initial_value="sorting and checking for hash collisions..."):
		sorted_hashes, counts = np.unique(hashes_array, return_counts=True)
		if sorted_hashes.shape[0] != hashes_array.shape[0]:
			collisions: np.array = sorted_hashes[counts > 1]
			n_collisions: int = hashes_array.shape[0] - sorted_hashes.shape[0]
			err_msg: str = (
				f"{n_collisions} tokenizer hash collisions: {collisions}\n"
				"Report error to the developer to increase the hash size or otherwise update the tokenizer hashing size:\n"
				f"https://github.com/understanding-search/maze-dataset/issues/new?labels=bug,tokenization&title=Tokenizer+hash+collision+error&body={n_collisions}+collisions+out+of+{hashes_array.shape[0]}+total+hashes",
			)

			raise ValueError(
				err_msg,
			)

	# save and return
	with spinner(initial_value="saving hashes...", update_interval=0.5):
		if path is None:
			path = Path(__file__).parent / "MazeTokenizerModular_hashes.npz"
		np.savez_compressed(
			path,
			hashes=sorted_hashes,
		)

	return sorted_hashes

"""to check if a tokenizer is one of our "approved" ones, we store this in a fst set using `rust_fst`

this file handles the creation of this fst file, which we ship to the user

this file relies on importing `get_all_tokenizers` and thus `MazeTokenizerModular`.
as such, loading this file for validating a tokenizer is the separate `maze_dataset.tokenization.modular.fst_load`
module, since we need to be able to import that from `maze_dataset.tokenization.modular.maze_tokenizer_modular` and
we cannot circularly import

"""

import functools
import random

import tqdm
from muutils.misc.numerical import shorten_numerical_to_str
from muutils.parallel import run_maybe_parallel
from muutils.spinner import NoOpContextManager, SpinnerContext
from rust_fst import Set as FstSet  # type: ignore[import-untyped]

from maze_dataset.tokenization.modular.all_tokenizers import get_all_tokenizers
from maze_dataset.tokenization.modular.fst_load import (
	MMT_FST_PATH,
	check_tokenizer_in_fst,
	get_tokenizers_fst,
)


def _get_tokenizer_name(tokenizer) -> str:  # noqa: ANN001
	return tokenizer.name


def save_all_tokenizers_fst(
	verbose: bool = True, parallel: bool | int = False
) -> FstSet:
	"""get all the tokenizers, save an fst file at `MMT_FST_PATH` and return the set"""
	# TYPING: add a protocol or abc for both of these which is a context manager that takes the args we care about
	# probably do this in muutils
	sp: type[SpinnerContext | NoOpContextManager] = (
		SpinnerContext if verbose else NoOpContextManager
	)

	with sp(message="getting all tokenizers"):
		all_tokenizers: list = get_all_tokenizers()

	n_tokenizers: int = len(all_tokenizers)

	all_tokenizers_names: list[str] = run_maybe_parallel(
		func=_get_tokenizer_name,
		iterable=all_tokenizers,
		parallel=parallel,
		pbar=tqdm.tqdm,
		pbar_kwargs=dict(
			total=n_tokenizers, desc="get name of each tokenizer", disable=not verbose
		),
	)

	assert n_tokenizers == len(all_tokenizers_names)
	print(
		f"# got {shorten_numerical_to_str(n_tokenizers)} ({n_tokenizers}) tokenizers names"
	)

	with sp(message="sorting tokenizer names"):
		all_tokenizers_names_sorted: list[str] = sorted(all_tokenizers_names)

	# construct an fst set and save it
	# we expect it to be 1.6kb or so
	with sp(message="constructing and saving tokenizers fst set"):
		tok_set: FstSet = FstSet.from_iter(
			all_tokenizers_names_sorted,
			path=MMT_FST_PATH.as_posix(),
		)

	print(
		f"# tokenizers fst set saved to {MMT_FST_PATH}, size: {MMT_FST_PATH.stat().st_size} bytes"
	)

	return tok_set


def check_tokenizers_fst(
	verbose: bool = True,
	parallel: bool | int = False,
	n_check: int | None = None,
) -> FstSet:
	"regen all tokenizers, check they are in the pre-existing fst set"
	sp: type[SpinnerContext | NoOpContextManager] = (
		SpinnerContext if verbose else NoOpContextManager
	)

	with sp(message="getting all tokenizers from scratch"):
		all_tokenizers: list = get_all_tokenizers()

	with sp(message="load the pre-existing tokenizers fst set"):
		get_tokenizers_fst()

	n_tokenizers: int = len(all_tokenizers)

	selected_tokenizers: list
	if n_check is not None:
		selected_tokenizers = random.sample(all_tokenizers, n_check)
	else:
		selected_tokenizers = all_tokenizers

	tokenizers_names: list[str] = run_maybe_parallel(
		func=_get_tokenizer_name,
		iterable=selected_tokenizers,
		parallel=parallel,
		pbar=tqdm.tqdm,
		pbar_kwargs=dict(
			total=n_tokenizers, desc="get name of each tokenizer", disable=not verbose
		),
	)

	if n_check is None:
		assert n_tokenizers == len(tokenizers_names)
		print(
			f"# got {shorten_numerical_to_str(n_tokenizers)} ({n_tokenizers}) tokenizers names"
		)
	else:
		assert n_check == len(tokenizers_names)
		print(
			f"# selected {n_check} tokenizers to check out of {shorten_numerical_to_str(n_tokenizers)} ({n_tokenizers}) total"
		)

	check_tokenizer_in_fst__do_except = functools.partial(
		check_tokenizer_in_fst, do_except=True
	)

	run_maybe_parallel(
		func=check_tokenizer_in_fst__do_except,
		iterable=tokenizers_names,
		parallel=parallel,
		pbar=tqdm.tqdm,
		pbar_kwargs=dict(
			total=len(selected_tokenizers),
			desc="checking tokenizers in fst",
			disable=not verbose,
		),
	)

	if n_check is None:
		print("# all tokenizers are in the pre-existing fst set!")
	else:
		print(f"# all {n_check} selected tokenizers are in the pre-existing fst set!")


if __name__ == "__main__":
	import argparse

	arg_parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="save the tokenizers fst set"
	)
	arg_parser.add_argument(
		"-c",
		"--check",
		action="store_true",
		help="check that all tokenizers are in the pre-existing fst set",
	)
	arg_parser.add_argument(
		"-q",
		"--quiet",
		action="store_true",
		help="don't show spinners and progress bars",
	)
	arg_parser.add_argument(
		"-p",
		"--parallel",
		action="store",
		nargs="?",
		type=int,
		const=True,
		default=False,
		help="Control parallelization. will run in serial if nothing specified, use all cpus if flag passed without args, or number of cpus if int passed.",
	)
	arg_parser.add_argument(
		"-n",
		"--n-check",
		action="store",
		default=None,
		help="if passed, check n random tokenizers. pass an int to check that many. pass 'none' or a -1 to check all",
	)
	args: argparse.Namespace = arg_parser.parse_args()

	n_check: int | None = (
		int(args.n_check)
		if (args.n_check is not None and args.n_check.lower() != "none")
		else None
	)
	if n_check is not None and n_check < 0:
		n_check = None

	if args.check:
		check_tokenizers_fst(
			verbose=not args.quiet,
			parallel=args.parallel,
			n_check=n_check,
		)
	else:
		save_all_tokenizers_fst(verbose=not args.quiet, parallel=args.parallel)

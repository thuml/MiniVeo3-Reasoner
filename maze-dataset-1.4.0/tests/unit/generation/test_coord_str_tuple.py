import numpy as np
import pytest

from maze_dataset.token_utils import (
	_coord_to_strings_indexed,
	_coord_to_strings_UT,
	coord_str_to_coord_np,
	coord_str_to_tuple,
	coord_str_to_tuple_noneable,
	coords_to_strings,
	str_is_coord,
)


def test_coord_to_strings():
	assert _coord_to_strings_UT((1, 2)) == ["(1,2)"]
	# assert _coord_to_strings_UT((-1, 0)) == ["(-1,0)"]

	assert _coord_to_strings_indexed((1, 2)) == ["(", "1", ",", "2", ")"]
	assert _coord_to_strings_indexed((-1, 0)) == ["(", "-1", ",", "0", ")"]


# TODO: test for negative coords


def test_str_is_coord():
	assert str_is_coord("(1,2)")
	# assert str_is_coord("(-1,0)")
	assert str_is_coord("(1,2,3)")
	assert not str_is_coord("1,2")
	assert str_is_coord("(1, 2)")
	assert str_is_coord("( 1 , 2 )")
	assert not str_is_coord("(1, 2)", allow_whitespace=False)


def test_coord_str_to_tuple():
	assert coord_str_to_tuple("(1,2)") == (1, 2)
	# assert coord_str_to_tuple("(-1,0)") == (-1, 0)
	assert coord_str_to_tuple("(1,2,3)") == (1, 2, 3)
	assert coord_str_to_tuple("(1, 2)") == (1, 2)
	assert coord_str_to_tuple("( 1 , 2 )") == (1, 2)
	assert coord_str_to_tuple("(1, 2)", allow_whitespace=False) == (1, 2)


def test_coord_str_to_coord_np():
	assert (coord_str_to_coord_np("(1,2)") == np.array([1, 2])).all()
	# assert (coord_str_to_coord_np("(-1,0)") == np.array([-1, 0])).all()
	assert (coord_str_to_coord_np("(1,2,3)") == np.array([1, 2, 3])).all()
	assert (coord_str_to_coord_np("(1, 2)") == np.array([1, 2])).all()
	assert (coord_str_to_coord_np("( 1 , 2 )") == np.array([1, 2])).all()
	assert (
		coord_str_to_coord_np("(1, 2)", allow_whitespace=False) == np.array([1, 2])
	).all()


def test_coord_str_to_tuple_noneable():
	assert coord_str_to_tuple_noneable("(1,2)") == (1, 2)
	# assert coord_str_to_tuple_noneable("(-1,0)") == (-1, 0)
	assert coord_str_to_tuple_noneable("(1,2,3)") == (1, 2, 3)
	assert coord_str_to_tuple_noneable("(1, 2)") == (1, 2)
	assert coord_str_to_tuple_noneable("( 1 , 2 )") == (1, 2)
	assert coord_str_to_tuple_noneable("1,2") is None


def test_coords_to_strings():
	# TODO: resolve testing duplication in test_token_utils.py
	assert coords_to_strings(
		[(1, 2), "<ADJLIST_START>", (5, 6)],
		_coord_to_strings_UT,
	) == ["(1,2)", "(5,6)"]
	assert coords_to_strings(
		[(1, 2), "<ADJLIST_START>", (5, 6)],
		_coord_to_strings_UT,
		when_noncoord="skip",
	) == ["(1,2)", "(5,6)"]
	assert coords_to_strings(
		[(1, 2), "<ADJLIST_START>", (5, 6)],
		_coord_to_strings_UT,
		when_noncoord="include",
	) == ["(1,2)", "<ADJLIST_START>", "(5,6)"]
	with pytest.raises(ValueError):  # noqa: PT011
		# this is meant to raise an error, so type ignore
		coords_to_strings(  # type: ignore[call-overload]
			[(1, 2), "<ADJLIST_START>", (5, 6)],
			_coord_to_strings_UT,
			when_noncoord="error",
		)

	assert coords_to_strings(
		[(1, 2), "<ADJLIST_START>", (5, 6)],
		_coord_to_strings_indexed,
	) == ["(", "1", ",", "2", ")", "(", "5", ",", "6", ")"]
	assert coords_to_strings(
		[(1, 2), "<ADJLIST_START>", (5, 6)],
		_coord_to_strings_indexed,
		when_noncoord="skip",
	) == ["(", "1", ",", "2", ")", "(", "5", ",", "6", ")"]
	assert coords_to_strings(
		[(1, 2), "<ADJLIST_START>", (5, 6)],
		_coord_to_strings_indexed,
		when_noncoord="include",
	) == ["(", "1", ",", "2", ")", "<ADJLIST_START>", "(", "5", ",", "6", ")"]


def test_str_is_coord_2():
	assert str_is_coord("(1,2)")
	assert str_is_coord("( 1 , 2 )")
	assert not str_is_coord("1,2")
	assert not str_is_coord("(1,2")
	assert not str_is_coord("1,2)")
	assert not str_is_coord("(1, a)")
	assert not str_is_coord("()")


def test_coord_str_to_tuple_excepts():
	assert coord_str_to_tuple("(1,2)") == (1, 2)
	with pytest.raises(ValueError):  # noqa: PT011
		coord_str_to_tuple("(1, a)")
	with pytest.raises(ValueError):  # noqa: PT011
		coord_str_to_tuple("()")


def test_coord_str_to_tuple_noneable_2():
	assert coord_str_to_tuple_noneable("(1,2)") == (1, 2)
	assert coord_str_to_tuple_noneable("1,2") is None
	assert coord_str_to_tuple_noneable("(1,2") is None
	assert coord_str_to_tuple_noneable("1,2)") is None
	assert coord_str_to_tuple_noneable("(1, a)") is None
	assert coord_str_to_tuple_noneable("()") is None


def test_coord_to_str():
	assert _coord_to_strings_UT((1, 2)) == ["(1,2)"]
	assert _coord_to_strings_UT((10, 20)) == ["(10,20)"]
	assert _coord_to_strings_UT((0, 0)) == ["(0,0)"]
	with pytest.raises(TypeError):
		# this is meant to raise an error, so type ignore
		_coord_to_strings_UT(1)  # type: ignore[arg-type]

	assert _coord_to_strings_indexed((1, 2)) == ["(", "1", ",", "2", ")"]
	assert _coord_to_strings_indexed((10, 20)) == ["(", "10", ",", "20", ")"]
	assert _coord_to_strings_indexed((0, 0)) == ["(", "0", ",", "0", ")"]
	with pytest.raises(TypeError):
		_coord_to_strings_indexed(1)  # type: ignore[arg-type]

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from maze_dataset.utils import bool_array_from_string


def test_bool_array_from_string():
	connection_list = bool_array_from_string(
		"""
        TTF TFF TFF
        TFT FFT FFF
        """,
		shape=[2, 3, 3],
	)

	expected = np.array(
		[
			[[True, True, False], [True, False, False], [True, False, False]],
			[[True, False, True], [False, False, True], [False, False, False]],
		],
	)

	assert_array_equal(expected, connection_list)


def test_bool_array_from_string_wrong_shape():
	with pytest.raises(ValueError):  # noqa: PT011
		bool_array_from_string("TF TF TF F", shape=[2, 2, 2])


def test_bool_array_from_string_custom_symbol():
	actual = bool_array_from_string(
		"""
        x x x
        x _ x
        x x x
        """,
		shape=[3, 3],
		true_symbol="_",
	)

	expected = [[False, False, False], [False, True, False], [False, False, False]]

	assert_array_equal(expected, actual)

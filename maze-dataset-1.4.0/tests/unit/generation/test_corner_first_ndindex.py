import numpy as np

from maze_dataset.utils import corner_first_ndindex


def test_corner_first_ndindex():
	for n in range(1, 11):
		a_n = corner_first_ndindex(n)
		a_n_plus_1 = corner_first_ndindex(n + 1)
		assert np.all(a_n == a_n_plus_1[: n**2])

"""math for getting the `MazeDatasetConfig.success_fraction_estimate()` function to work

Desmos link: https://www.desmos.com/calculator/qllvhwftvy
"""

import numpy as np
from jaxtyping import Float


def sigmoid(x: float) -> float:
	r"$\sigma(x) = \frac{1}{1 + e^{-x}}$"
	return 1 / (1 + np.exp(-x))


# sigmoid_shifted = lambda x: 1 / (1 + np.exp(-1000 * (x - 0.5)))
# r"sigmoid(x)= 1 / (1 + e^{-b(x-0.5)})"

# g_poly = lambda q, a: 1 - np.abs(2 * q - 1) ** a
# r"g(q,a) = 1 - (|2q-1|)^{a}"

# f_poly = lambda q, a: q * g_poly(q, a)
# r"f(q,a) = q * g(q,a)"

# h_func = lambda q, a: f_poly(q, a) * (1 - sigmoid_shifted(q)) + (1 - f_poly(1 - q, a)) * sigmoid_shifted(q)
# r"h(q,a,b) = f(q,a) * (1-s(q,b)) + (1-f(1-q,a)) * s(q,b)"

# A_scaling = lambda q, a, w: w * g_poly(q, a)
# r"A(q) = b * g(q, a)"


def sigmoid_shifted(x: float) -> float:
	r"\sigma_s(x)= \frac{1}{1 + e^{-10^3 \cdot (x-0.5)}}"
	return 1 / (1 + np.exp(-1000 * (x - 0.5)))


def g_poly(q: float, a: float) -> float:
	r"$g(q,a) = 1 - (|2q-1|)^{a}$"
	return 1 - np.abs(2 * q - 1) ** a


def f_poly(q: float, a: float) -> float:
	r"$f(q,a) = q \cdot g(q,a)$"
	return q * g_poly(q, a)


def h_func(q: float, a: float) -> float:
	r"""$h(q,a,b) = f(q,a) \cdot (1-\sigma_s(q)) + (1-f(1-q,a)) \cdot \sigma_s(q)$"""
	return f_poly(q, a) * (1 - sigmoid_shifted(q)) + (
		1 - f_poly(1 - q, a)
	) * sigmoid_shifted(q)


def A_scaling(q: float, a: float, w: float) -> float:
	r"$A(q) = w \cdot g(q, a)$"
	return w * g_poly(q, a)


def soft_step(
	x: float | np.floating | Float[np.ndarray, " n"],
	p: float | np.floating,
	alpha: float | np.floating = 5,
	w: float | np.floating = 50,
) -> float:
	"""when p is close to 0.5 acts like the identity wrt x, but when p is close to 0 or 1, pushes x to 0 or 1 (whichever is closest)

	https://www.desmos.com/calculator/qllvhwftvy
	"""
	# TYPING: this is messed up, some of these args can be arrays but i dont remember which?
	return h_func(
		x,  # type: ignore[arg-type]
		A_scaling(p, alpha, w),  # type: ignore[arg-type]
	)


# `cfg: MazeDatasetConfig` but we can't import that because it would create a circular import
def cfg_success_predict_fn(cfg) -> float:  # noqa: ANN001
	"learned by pysr, see `estimate_dataset_fractions.ipynb` and `maze_dataset.benchmark.config_fit`"
	x = cfg._to_ps_array()
	raw_val: float = sigmoid(
		(
			(
				((sigmoid((x[1] - x[3]) ** 3) * -4.721228) - (x[3] * 1.4636494))
				* (
					x[2]
					* (
						x[4]
						+ (((x[0] + 0.048765484) ** 9.746339) + (0.8998194 ** x[1]))
					)
				)
			)
			+ (2.4524326 ** (2.9501643 - x[0]))
		)
		* (
			(
				(((0.9077277 - x[0]) * ((x[4] * 1.0520288) ** x[1])) + x[0])
				* sigmoid(x[1]) ** 3
			)
			+ -0.18268494
		),
	)
	return soft_step(
		x=raw_val,
		p=x[0],
		alpha=5,  # manually tuned
		w=10,  # manually tuned
	)

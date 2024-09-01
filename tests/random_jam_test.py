import unittest

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from jam_echo_gen.echo_gen import SPEED_OF_LIGHT, random_jam_alphas


def check_jam_alphas(alpha_c, jam_alphas, delta_m, delta_s, alpha_pad):
    num_jams = len(jam_alphas)

    jam_alphas = np.reshape(jam_alphas, [-1])

    if np.any(np.abs(jam_alphas) > 1 - alpha_pad):
        print("Padding check failed.")
        return False

    if np.any(np.abs(jam_alphas - alpha_c) < delta_m):
        print("Main-lobe separation check failed.")
        return False

    alpha_tiled = np.tile(jam_alphas, (num_jams, 1))
    delta_jam_alphas = np.abs(alpha_tiled - alpha_tiled.T) \
                       + np.diag(np.ones(num_jams)) * 100  # avoid diagonal zeros
    if np.any(delta_jam_alphas < delta_s):
        print("Separations-between-jams check failed.")
        return False

    return True


class RandomJamAlphaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plot = False

    def test_random_gen(self):
        wavelength = SPEED_OF_LIGHT / 3e9
        num_subarrays = 256
        lw = 2 / num_subarrays

        num_jams = 18
        alpha_c = np.random.rand() * 2 - 1
        delta_m = 0.05
        delta_s = 0.05
        alpha_pad = 0.01
        delta_m_lw = delta_m / lw
        delta_s_lw = delta_s / lw
        alpha_pad_lw = alpha_pad / lw

        jam_alphas = random_jam_alphas(
            alpha_c, wavelength, num_subarrays, wavelength / 2,
            num_jams, delta_m_lw, delta_s_lw, alpha_pad_lw
        )
        if jam_alphas is not None and self.plot:
            self.assertTrue(check_jam_alphas(alpha_c, jam_alphas, delta_m, delta_s, alpha_pad))

            fig: Figure
            axes: Axes
            fig, axes = plt.subplots()
            axes.vlines(alpha_c, -1, 1, colors="red", linestyles="-")
            axes.vlines(jam_alphas, -1, 1, colors="k", linestyles=":")
            axes.fill_betweenx(y=np.linspace(-1, 1, 256),
                               x1=alpha_c - delta_m,
                               x2=alpha_c + delta_m,
                               color="red",
                               alpha=0.5)
            for _alpha in jam_alphas:
                axes.fill_betweenx(y=np.linspace(-1, 1, 256),
                                   x1=_alpha - delta_s,
                                   x2=_alpha + delta_s,
                                   color="orange",
                                   alpha=0.3)
            axes.fill_betweenx(y=np.linspace(-1, 1, 256),
                               x1=-1,
                               x2=-1 + alpha_pad,
                               color="blue",
                               alpha=0.5)
            axes.fill_betweenx(y=np.linspace(-1, 1, 256),
                               x1=1,
                               x2=1 - alpha_pad,
                               color="blue",
                               alpha=0.5)
            axes.set_xlim(-1, 1)
            axes.set_xlabel(r"$\alpha$")

            plt.show()

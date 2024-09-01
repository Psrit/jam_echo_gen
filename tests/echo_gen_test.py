import typing
import unittest

import numpy as np
from scipy.signal.windows import chebwin

from jam_echo_gen.echo_gen import SPEED_OF_LIGHT, echo_gen
from jam_echo_gen.utils import db, two_pow_ceil


def n_vec_from_azel(azim: float, elev: float):
    return np.array([
        np.cos(elev) * np.sin(azim),
        np.sin(elev),
        np.cos(elev) * np.cos(azim)
    ])


def steering_vec(
        r_target: np.ndarray,
        r_ants: np.ndarray,
        k: float,
        normalize: bool = False
):
    """
    Calculate the steering vector.

    :param r_target: Position of target (shape=(3,)).
    :param r_ants: Positions of antennas (shape=(num_antennas, 3).
    :param k: The norm of wave vector, i.e. 2 pi / wavelength.
    :param normalize: Whether to normalize the steering vector.
    :return: The steering vector (shape=(num_antennas,)).

    """
    n_target = r_target / np.linalg.norm(r_target)  # shape=(3,)
    r_ant_center = np.mean(r_ants, axis=0)  # shape=(3,)
    dr_ants = r_ants - r_ant_center  # shape=(num_ants, 3)
    a_vec = np.exp(
        1j * k * dr_ants @ n_target
    )

    if normalize:
        a_vec = a_vec / np.sqrt(len(r_ants))
    return a_vec


def mvdr_adbf(a_vec: np.ndarray, sig_cov_mat: np.ndarray) -> np.ndarray:
    """
    Solve ADBF problem with analytic solution.

    :param a_vec: The steering vector (shape=(num_antennas,)).
    :param sig_cov_mat: Covariance matrix of received signal of all antennas
        (shape=(num_antennas, num_antennas)).
    :return: The optimal weight found (shape=(num_antennas,)).

    """
    inv_cov = np.linalg.inv(sig_cov_mat)

    tmp = inv_cov @ a_vec
    weight = tmp / (a_vec.conj() @ tmp)
    return weight


def est_num_jams(
        signal: np.ndarray,
        fft_len: typing.Optional[int] = None
) -> np.ndarray:
    """
    Estimate the number of incoming jams.

    :param signal: Received signal (shape=(num_antennas, num_range_gates)).
        Note that `num_range_gate` may not represent the full number of range
        gates of one pack of echo samples, since down-sampling can be performed
        by the function caller.
    :param fft_len: The number of spatial power spectrum points.
        If being None, `num_antennas` will be used.
    :return: Estimated number of incoming jams.

    """
    num_antennas = len(signal)
    win_attenuation = 60  # dB
    win = chebwin(num_antennas, at=win_attenuation)

    # Evaluate the 0-freq-leading spectrum (amp), shape=(fft_len,)
    spat_spec = np.max(
        np.abs(
            np.fft.fft(
                # perform window onto signal of each antenna
                np.einsum("i,ij->ij", win, signal),
                n=fft_len, axis=0, norm="backward"
            )  # FFT along the first axis, output shape=(fft_len, num_range_gates)
        ),  # element-wise abs
        axis=1
    )  # max along the second axis, output shape=(fft_len,)

    peak_indices = np.where(
        # is element greater than its left neighbor, ...
        (spat_spec > np.roll(spat_spec, 1)) &
        # is element greater than its right neighbor, ...
        (spat_spec > np.roll(spat_spec, -1)) &
        # and is element amp > 60dB
        (db(spat_spec, unit="voltage") > 60)
    )[0]  # indices of peak_indices's elements which satisfy the conditions,
    # i.e. have high jam powers.
    # Note that since the comparison between adjacent points is implemented with
    # `roll`, jams across the frequency boundary (i=0 and i=fft_len-1) are also
    # handled properly here.

    return len(peak_indices)


class EchoGenTest(unittest.TestCase):
    def test_invalid_case(self):
        for case in (-1, 4, 5):
            with self.assertRaisesRegex(ValueError, "Invalid case type:"):
                echo_gen(
                    case=case,
                    num_jams=4,
                    num_subarrays=128,
                    num_targets=16
                )

    def test_too_few_num_subarrays(self):
        for case in range(0, 4):
            with self.assertRaisesRegex(ValueError, "Too few sub-arrays!"):
                echo_gen(
                    case=case,
                    num_jams=4,
                    num_subarrays=1,
                    num_targets=1
                )

        for case in range(0, 4):
            echo_gen(
                case=case,
                num_jams=4,
                num_subarrays=16,
                num_targets=1
            )

    def test_0_jam_ok(self):
        for case in range(0, 3):
            signal = echo_gen(
                case=case,
                num_targets=3,
                num_jams=0,
                num_subarrays=64
            )[-2]
            self.assertEqual(np.ndim(signal), 2)
            self.assertEqual(signal.shape[0], 64)
            self.assertTrue(signal.shape[1] <= 3 * 64)

    def test_0_target_ok(self):
        for case in range(0, 3):
            signal, subarray_indices = echo_gen(
                case=case,
                num_targets=0,
                num_jams=3,
                num_subarrays=64
            )[-2:]
            self.assertEqual(np.ndim(signal), 2)
            self.assertEqual(signal.shape[0], 64)
            self.assertTrue(signal.shape[1] <= 3 * 64)

    def test_0_target_0_jam_ok(self):
        for case in range(0, 3):
            signal, subarray_indices = echo_gen(
                case=case,
                num_targets=0,
                num_jams=0,
                num_subarrays=64
            )[-2:]
            self.assertEqual(np.ndim(signal), 2)
            self.assertEqual(signal.shape[0], 64)
            self.assertTrue(signal.shape[1] <= 3 * 64)

    def test_target_pos(self):
        num_subarrays = 64
        num_targets = 5
        freq, mainlobe_azim, noise_power, signal, _ = \
            echo_gen(case=0,
                     num_targets=num_targets,
                     num_jams=3,
                     num_subarrays=num_subarrays)
        num_range_gates = np.shape(signal)[1]

        wavelength = SPEED_OF_LIGHT / freq
        dx = wavelength / 2.
        k = 2 * np.pi / wavelength
        r_arrays = np.zeros((num_subarrays, 3))
        r_arrays[:, 0] = \
            (np.arange(num_subarrays) - (num_subarrays - 1) / 2) * dx
        a0_vec = steering_vec(
            n_vec_from_azel(mainlobe_azim, 0.),
            r_arrays, k, normalize=True
        )

        cov_mat = np.cov(signal[:, int(num_range_gates / 2): num_range_gates])
        cov_mat += 100 * np.identity(num_subarrays)  # load identity matrix
        w_mvdr = mvdr_adbf(a0_vec, cov_mat)

        y_output = w_mvdr.conj() @ signal
        y_db = db(y_output, unit="voltage")
        output_noise_power = np.mean(
            np.abs(y_output[int(num_range_gates / 2):]) ** 2
        )
        range_gate_mask = np.logical_and(
            np.concatenate([
                [y_db[0] > y_db[1]],
                np.logical_and(
                    y_db[1:-1] > y_db[2:],
                    y_db[1:-1] > y_db[:-2]
                ),
                [y_db[-1] > y_db[-2]]
            ]),
            y_db > db(output_noise_power, unit="power") + 15
        )
        target_range_gates = np.arange(num_range_gates)[range_gate_mask]
        # allow an error of one target since two targets may be very close:
        self.assertTrue(abs(len(target_range_gates) - num_targets) <= 1)

    def test_jam_alphas(self):
        num_subarrays = 64
        num_targets = 5
        num_jams = 3
        freq, mainlobe_azim, noise_power, signal, _ = \
            echo_gen(case=0,
                     num_targets=num_targets,
                     num_jams=num_jams,
                     num_subarrays=num_subarrays)
        num_jams = est_num_jams(
            signal, fft_len=max(4096, two_pow_ceil(num_subarrays))
        )

        # allow an error of one jam since two jams may be very close:
        self.assertTrue(abs(num_jams - num_jams) <= 1)

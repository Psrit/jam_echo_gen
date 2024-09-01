import unittest
import warnings

import numpy as np

from jam_echo_gen.utils import db, is_real_array, join_interval, two_pow_ceil


def is_real(x):
    return np.all(np.isreal(x))


def py_db(x, unit="power"):
    """
    Pure Python implementation of dB function, which has the same signature as
    `jam_echo_gen.utils.db`.

    """
    if unit == "power" and (not is_real(x) or not np.all(x >= 0)):
        warnings.warn(
            "Input x is not pure positive but its unit is assumed to be "
            f"'power': {x}."
        )

    x = np.abs(x)
    if unit == "power":
        return 10 * np.log10(x)
    else:  # voltage
        return 20 * np.log10(x)


class IsRealArrayTest(unittest.TestCase):
    def test_is_real_array(self):
        real_array = np.random.rand(128)
        self.assertTrue(is_real_array(real_array))

        complex_array = np.random.rand(128) + 1j * np.random.rand(128)
        self.assertFalse(is_real_array(complex_array))

    def test_scalar_argument(self):
        with self.assertRaises(TypeError):
            is_real_array(1)

        for x, y in zip(
            [np.array(1), np.array(1 + 2j)],
            [True, False]
        ):
            self.assertEqual(
                is_real_array(x), y
            )


class DbTest(unittest.TestCase):
    def test_random_real_power(self):
        # array argument:
        x = np.random.rand(128)
        self.assertTrue(
            np.allclose(
                py_db(x, "power"),
                db(x, "power")
            )
        )

        # scalar argument:
        x = np.random.rand()
        self.assertTrue(
            np.allclose(
                py_db(x, "power"),
                db(x, "power")
            )
        )

    def test_random_real_voltage(self):
        # array argument:
        x = np.random.rand(128)
        self.assertTrue(
            np.allclose(
                py_db(x, "voltage"),
                db(x, "voltage")
            )
        )

        # scalar argument:
        x = np.random.rand()
        self.assertTrue(
            np.allclose(
                py_db(x, "voltage"),
                db(x, "voltage")
            )
        )

    def test_random_complex_voltage(self):
        # array argument:
        x = np.random.rand(128) + 1j * np.random.rand(128)
        self.assertTrue(
            np.allclose(
                py_db(x, "voltage"),
                db(x, "voltage")
            )
        )

        # scalar argument:
        x = np.random.rand()
        self.assertTrue(
            np.allclose(
                py_db(x, "voltage"),
                db(x, "voltage")
            )
        )


class TwoPowCeilTest(unittest.TestCase):
    def setUp(self):
        _x = 2 ** np.arange(10)  # 1, 2, 4, 8, ..., 512
        self.x = np.concatenate([
            _x,
            (_x[:-1] + _x[1:]) / 2
        ])

        # expected output of two_pow_ceil(self.x)
        self.y = np.concatenate([
            _x,
            _x[1:]
        ])

    def test_array_argument(self):
        y = two_pow_ceil(self.x)
        self.assertTrue(np.allclose(y, self.y))

    def test_scalar_argument(self):
        for _x_val, _y_val in zip(self.x, self.y):
            self.assertEqual(_y_val, two_pow_ceil(_x_val))


class JoinIntervalTest(unittest.TestCase):
    def test_join_overlapping_intervals(self):
        test_data = (
            ([(1, 2.35), (3, 4)], (2.35, 2.5), [(1, 2.5), (3, 4)]),
            ([(1, 2.35), (3, 4)], (2.35, 3), [(1, 4)]),
            ([(1, 2.35), (3, 4)], (2.35, 3.5), [(1, 4)]),
            ([(1, 2.35), (3, 4)], (2.35, 4.5), [(1, 4.5)]),
            ([(1, 2.35), (3, 4)], (1.5, 4.5), [(1, 4.5)]),
            ([(1, 2.35), (3, 4)], (-1, 4.5), [(-1, 4.5)]),
        )
        for arg0, arg1, output in test_data:
            self.assertEqual(
                join_interval(arg0, arg1),
                output
            )

    def test_join_non_overlapping_intervals(self):
        test_data = (
            ((), (0, 1), [(0, 1)]),
            ([(1, 2.35), (3, 4)], (2.36, 2.5), [(1, 2.35), (2.36, 2.5), (3, 4)]),
            ([(1, 2.35), (3, 4)], (-2, -1), [(-2, -1), (1, 2.35), (3, 4)]),
            ([(1, 2.35), (3, 4)], (9, 10), [(1, 2.35), (3, 4), (9, 10)])
        )
        for arg0, arg1, output in test_data:
            self.assertEqual(
                join_interval(arg0, arg1),
                output
            )

    def test_complicated_cases(self):
        test_data = (
            ([(-1.5, -0.9), (1, 2.35), (3, 4), (4.2, 5.0), (5.2, 9.9)],
             (0.95, 4.5),
             [(-1.5, -0.9), (0.95, 5.0), (5.2, 9.9)]),
        )
        for arg0, arg1, output in test_data:
            self.assertEqual(
                join_interval(arg0, arg1),
                output
            )

# cython: language_level=3

import logging
import numbers
import sys
import warnings

import numpy as np

cimport numpy as cnp

from libc.math cimport ceil, log2

cdef int _two_pow_ceil(double a):
    """ Return the smallest value of 2^n that is not smaller than `a`. """
    return 2 ** int(ceil(log2(a)))

cdef cnp.ndarray _two_pow_ceil_arr(cnp.ndarray arr):
    """ Return the smallest value of 2^n that is not smaller than `a`. """
    return 2 ** np.ceil(np.log2(arr)).astype(int)

cpdef two_pow_ceil(a):
    """ Return the smallest value of 2^n that is not smaller than `a`. """
    if isinstance(a, numbers.Number):
        return _two_pow_ceil(a)
    else:
        a = np.asanyarray(a)
        return _two_pow_ceil_arr(a)

cdef cnp.ndarray _array_db(cnp.ndarray x, str unit="power"):
    if unit == "power" and (not is_real_array(x) or not np.all(x >= 0)):
        warnings.warn(
            "Input x is not pure positive but its unit is assumed to be "
            f"'power': {x}."
        )
        
    x = np.abs(x)
    # This requires x to be a np.ndarray of at least 1 dimension here, since 
    # the output of np.abs(<number or 0-D array>) is always a numpy scalar
    # instead of np.ndarray (likely for np.log10, etc.).

    if unit == "power":
        return 10 * np.log10(x)
    else:  # voltage
        return 20 * np.log10(x)

cpdef db(x, str unit="power"):
    """
    Evaluate the power in decibels of input `x`.

    :param x: Input value. If being complex, abs values will firstly be
        calculated before evaluating power dBs.
    :param unit: Unit of `x`. Possible values are "power" and "voltage".
    :return: The power in decibels of `x`. If `unit` == "power", then
        returns 10*log10(abs(x)), otherwise returns 20*log10(abs(x)).

    """
    cdef bint number_input = isinstance(x, numbers.Number)
    x = np.asanyarray(x)
    if number_input:
        x = np.reshape(x, [-1])

    cdef cnp.ndarray _db_arr = _array_db(x, unit)
    if number_input:
        return _db_arr.item()
    return _db_arr

cpdef bint is_real_array(cnp.ndarray arr):
    """ Check whether an array is pure real. """
    return np.all(np.isreal(arr))

cdef cnp.ndarray awgn(cnp.ndarray signal, double snr_db, signal_power_db=None):
    """
    Add white Gaussian noise to signal.

    :param signal: Input signal (linear scale of amplitudes).
    :param snr_db: Signal-noise ratio in dB.
    :param signal_power_db: Signal power in dB.
        If being None, measure the power of `signal`.
        If being 0, snr_db is also the decibels of the reciprocal of the absolute
        noise power.

    """
    _shape = np.shape(signal)

    if signal_power_db is None:
        signal_power = np.linalg.norm(signal) ** 2 / np.size(signal)
    else:
        signal_power = 10 ** (signal_power_db / 10)

    snr = 10 ** (snr_db / 10)
    noise_power = signal_power / snr
    if is_real_array(signal):
        noise = np.sqrt(noise_power) * np.random.randn(*_shape)
    else:
        noise = np.sqrt(noise_power / 2) * (
                np.random.randn(*_shape) + 1j * np.random.randn(*_shape)
        )
    return signal + noise

cdef float _tuple_first_elem(tuple t):
    return t[0]

cpdef list join_interval(
        list interval_union,
        tuple interval
):
    if interval[0] >= interval[1]:
        raise ValueError("`interval[0]` must be less than `interval[1]`.")

    cdef list overlapped_intervals = []
    cdef list joined = []
    cdef int i
    cdef tuple _inter
    for i, _inter in enumerate(interval_union):
        if _inter[0] <= interval[0] <= _inter[1]:
            overlapped_intervals.append(i)
        elif (interval[0] <= _inter[0]) and (_inter[0] <= interval[1]):
            overlapped_intervals.append(i)
        else:
            joined.append(_inter)

    if overlapped_intervals:
        joined.append(
            (min(interval[0], interval_union[overlapped_intervals[0]][0]),
             max(interval[1], interval_union[overlapped_intervals[-1]][1]))
        )
    else:
        joined.append(interval)

    joined = sorted(joined, key=_tuple_first_elem)
    return joined

def random_from_intervals(*intervals):
    lengths = np.zeros(len(intervals))
    total_len = 0.
    for i, intv in enumerate(intervals):
        intv_len = intv[1] - intv[0]
        if intv_len < 0:
            raise ValueError(f"Invalid interval: ({intv[0]}, {intv[1]})")
        lengths[i] = intv_len
        total_len += intv[1] - intv[0]

    rand_val = np.random.rand() * sum(lengths)

    for i, intv in enumerate(intervals):
        if 0 <= rand_val < lengths[i]:
            return rand_val + intv[0]
        else:
            rand_val -= lengths[i]

LOG = logging.getLogger("ADBF benchmark")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("[%(levelname)s] %(message)s")
)
LOG.addHandler(handler)

# cython: language_level=3
# distutils: language = c++

import logging
import numbers
import sys
import warnings

import numpy as np

cimport numpy as cnp

from libc.math cimport ceil, log2
from libcpp.algorithm cimport sort
from libcpp.random cimport random_device, mt19937, uniform_real_distribution
from libcpp.vector cimport vector

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
    elif unit == "voltage":
        return 20 * np.log10(x)
    else:
        raise ValueError(f"Invalid unit: '{unit}'.")

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

cdef bint _comp_intvs_by_first_elem((double, double) t1, (double, double) t2):
    return t1[0] < t2[0]

cpdef vector[(double, double)] join_interval(
        vector[(double, double)] interval_union,
        (double, double) interval
):
    """
    Join an interval union and a new interval.

    :param interval_union: An union of intervals, each element of which
        represent a interval.
        Following requirements must be met, otherwise the behavior is undefined:
        1. The interval union must be already simplified (i.e. there are no 
        overlapping intervals);
        2. Contained intervals have been sorted according to their left bounds;
        3. For each contained intervals, the left bound must be strictly less 
        than the right bound.
    :param interval: The new interval to be joined into `interval_union`.
    :return: Joined interval union, where contained intervals have been sorted 
        according to their left bounds.

    """
    if interval[0] >= interval[1]:
        raise ValueError("`interval[0]` must be less than `interval[1]`.")

    cdef vector[int] overlapped_intervals
    cdef vector[(double, double)] joined = []
    cdef int i
    cdef (double, double) _inter
    for i in range(len(interval_union)):
        _inter = interval_union[i]
        if _inter[0] <= interval[0] <= _inter[1]:
            overlapped_intervals.push_back(i)
        elif (interval[0] <= _inter[0]) and (_inter[0] <= interval[1]):
            overlapped_intervals.push_back(i)
        else:
            joined.push_back(_inter)

    if overlapped_intervals.size():
        joined.push_back(
            (min(interval[0], interval_union[overlapped_intervals[0]][0]),
             max(interval[1], interval_union[overlapped_intervals[overlapped_intervals.size() - 1]][1]))
        )
    else:
        joined.push_back(interval)

    sort(joined.begin(), joined.end(), _comp_intvs_by_first_elem)
    return joined

cpdef double random_from_intervals(vector[(double, double)] intervals):
    cdef int num_intervals = intervals.size()
    cdef vector[double] lengths = vector[double](num_intervals, 0)
    cdef double total_len = 0.
    
    cdef int i
    cdef (double, double) intv
    cdef double intv_len
    for i in range(num_intervals):
        intv = intervals[i]
        intv_len = intv[1] - intv[0]
        if intv_len < 0:
            raise ValueError(f"Invalid interval: ({intv[0]}, {intv[1]})")
        lengths[i] = intv_len
        total_len += intv_len

    cdef:
        random_device rd
        mt19937 gen = mt19937(rd())
        uniform_real_distribution[double] dis = uniform_real_distribution[double](0., total_len)
        double rand_val = dis(gen)
    for i in range(num_intervals):
        intv = intervals[i]
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

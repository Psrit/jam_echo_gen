# cython: language_level=3
# distutils: language = c++

cimport numpy as cnp

from libcpp.vector cimport vector

cpdef two_pow_ceil(a)

cpdef db(x, str unit=?)

cdef cnp.ndarray awgn(cnp.ndarray signal, double snr_db, signal_power_db=?)

cpdef bint is_real_array(cnp.ndarray arr)
cpdef vector[(double, double)] join_interval(
        vector[(double, double)] interval_union,
        (double, double) interval
)
cpdef double random_from_intervals(vector[(double, double)] intervals)

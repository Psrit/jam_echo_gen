# cython: language_level=3

cimport numpy as cnp

cpdef two_pow_ceil(a)

cpdef db(x, str unit=?)

cdef cnp.ndarray awgn(cnp.ndarray signal, double snr_db, signal_power_db=?)

cpdef bint is_real_array(cnp.ndarray arr)
cpdef list join_interval(
        list interval_union,
        tuple interval
)

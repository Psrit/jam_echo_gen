# cython: language_level=3
# distutils: language = c++

import typing
import warnings

import numpy as np

from libcpp.random cimport random_device, mt19937, uniform_real_distribution
from libcpp.vector cimport vector

cimport numpy as cnp

from .utils cimport (
    join_interval,
    db,
    awgn,
    two_pow_ceil,
    random_from_intervals
)

SPEED_OF_LIGHT = 299792458  # m/s

cdef cnp.ndarray n_vec_from_azel(double azim, double elev):
    return np.array([
        np.cos(elev) * np.sin(azim),
        np.sin(elev),
        np.cos(elev) * np.cos(azim)
    ])

cdef cnp.ndarray steering_vec(
        cnp.ndarray r_target,
        cnp.ndarray r_ants,
        double k,
        bint normalize = False
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

cdef tuple line_array_pattern_steering_vecs(
        double wavelength,
        int num_subarrays,
        double dx,
        int num_directions = 256,
        str dimension = "angle"
):
    cdef cnp.ndarray r_arrays = np.zeros((num_subarrays, 3))
    r_arrays[:, 0] = \
        (np.arange(num_subarrays) - (num_subarrays - 1) / 2) * dx

    cdef double k = 2 * np.pi / wavelength
    cdef cnp.ndarray azims, alphas, n_vec, a_vec

    # steering vectors for calculating antenna pattern
    pattern_a_vecs = []  # shape=(num_directions, num_subarrays)
    if dimension == "angle":
        azims = np.linspace(-np.pi / 2, np.pi / 2, num_directions)
        alphas = np.sin(azims)
    elif dimension == "alpha":
        alphas = np.linspace(-1, 1, num_directions)
        azims = np.arcsin(alphas)
    else:
        raise ValueError(f"Unknown dimension: {dimension}")

    for _azim in azims:
        n_vec = n_vec_from_azel(_azim, 0)
        a_vec = steering_vec(n_vec, r_arrays, k, normalize=True)
        pattern_a_vecs.append(a_vec)
    pattern_a_vecs = np.array(pattern_a_vecs)

    if dimension == "angle":
        return azims, pattern_a_vecs
    return alphas, pattern_a_vecs

def channel_error(num_channels, amp_error_db=0, phase_error=np.deg2rad(0)):
    amp_max = 10 ** (amp_error_db / 10)
    amp_min = 10 ** (-amp_error_db / 10)
    amp_errors = np.random.rand(num_channels) * (amp_max - amp_min) + amp_min

    phase_errors = (2 * np.random.rand(num_channels) - 1) * phase_error

    channel_errors = amp_errors * np.exp(1j * phase_errors)

    return channel_errors

cdef cnp.ndarray _random_jam_alphas(
        double alpha_c,  # alpha of beam center
        int num_jams,
        double delta_m,
        double delta_s,
        double alpha_lim,
        tries=np.inf
):
    """
    Generate alpha values of incoming jams with constraints
        |alpha_i - alpha_c| > delta_m for any i in [0, 1, ..., num_jams-1]
        and
        |alpha_i - alpha_j| > delta_s for any i, j (i≠j) in [0, 1, ...,
        num_jams-1]
        and
        |alpha_i| < alpha_lim
    satisfied.

    When delta_m == delta_s == 0, the random generating process reduce to
    the unconstrained case.

    """
    _tries = max(tries, 1)
    cdef int _try_counts = 0
    cdef:
        bint failed
        cnp.ndarray alphas
        int i_jam
        vector[(double, double)] forbid_zone
        vector[double] avail_lengths, accum_avail_lengths
        double alpha_r_last, accum_avail_len, alpha_l, _avail_len
        int i_fzone, i_interval
        (double, double) _fzone_border
        random_device rd
        mt19937 gen = mt19937(rd())
        uniform_real_distribution[double] dis = uniform_real_distribution[double](0., 1.)
        double rand_val, alpha

    while _tries > 0:
        failed = False

        alphas = np.zeros(num_jams)
        forbid_zone = [(0, delta_m), (2 * alpha_lim - delta_m, 2 * alpha_lim)]

        i_jam = 0
        while i_jam < num_jams and not failed:
            # Length of available intervals
            avail_lengths = []
            accum_avail_lengths = []
            alpha_r_last = 0  # Upper border of last forbidden zone
            accum_avail_len = 0
            for i_fzone in range(forbid_zone.size()):
                _fzone_border = forbid_zone[i_fzone]
                alpha_l = _fzone_border[0]

                _avail_len = alpha_l - alpha_r_last
                avail_lengths.push_back(_avail_len)

                accum_avail_len += _avail_len
                accum_avail_lengths.push_back(accum_avail_len)

                alpha_r_last = _fzone_border[1]

            if accum_avail_len <= 0:
                failed = True

            rand_val = dis(gen) * accum_avail_len
            i_interval = 1
            for i_interval in range(1, accum_avail_lengths.size()):
                if rand_val < accum_avail_lengths[i_interval]:
                    break

            alpha = rand_val - accum_avail_lengths[i_interval - 1] \
                    + forbid_zone[i_interval - 1][1]

            alphas[i_jam] = alpha
            i_jam += 1

            forbid_zone = join_interval(
                forbid_zone,
                (max(0, alpha - delta_s),
                 min(2, alpha + delta_s))
            )

        if not failed:
            # wrap-around alphas
            alphas = alpha_c + np.array(alphas)
            # alphas in (_alpha_c + _delta_m, _alpha_c + 2 * _alpha_lim - _delta_m)
            alphas = np.where(alphas > alpha_lim, alphas - 2 * alpha_lim, alphas)
            # wrap alphas to [-alpha_lim, alpha_lim]
            return alphas

        _tries -= 1
        _try_counts += 1

        if _try_counts % 10 == 0:
            warnings.warn(
                f"We have tried {_try_counts} times for random jam "
                f"generating (required num_jams = {num_jams}.)"
            )

    # All tries failed
    return None

cdef cnp.ndarray _random_jam_alphas_approx(
        double alpha_c,  # alpha of beam center
        int num_jams,
        double delta_m,
        double delta_s,
        double alpha_lim,
        tries=np.inf
):
    """
    Generate alpha values of incoming jams with constraints
        |alpha_i - alpha_c| > delta_m for any i in [0, 1, ..., num_jams-1]
        and
        |alpha_i - alpha_j| > delta_s for any i, j (i≠j) in [0, 1, ...,
        num_jams-1]
        and
        |alpha_i| < alpha_lim
    satisfied.

    When delta_m == delta_s == 0, the random generating process reduce to
    the unconstrained case.

    Note that different from `_random_jam_alphas`, this function is just a fast
    and always-succeed implementation of an inaccurate approximation to the
    original random distribution.

    """
    alphas = []
    cdef:
        double d_subint = (2. * alpha_lim - (2. * delta_m + delta_s * (num_jams - 1))) \
                          / num_jams
        double int_left = delta_m
        int i

    for i in range(num_jams):
        alphas.append(random_from_intervals([(int_left, int_left + d_subint)]))
        int_left += d_subint + delta_s

    alphas = alpha_c + np.array(alphas)
    # alphas in (_alpha_c + _delta_m, _alpha_c + 2 * _alpha_lim - _delta_m)
    alphas = np.where(alphas > alpha_lim, alphas - 2 * alpha_lim, alphas)
    # wrap alphas to [-alpha_lim, alpha_lim]
    return alphas

cpdef cnp.ndarray random_jam_alphas(
        double alpha_c,  # alpha of beam center
        double wavelength,
        int num_subarrays,
        double dx,
        int num_jams,
        double delta_m_lw,
        double delta_s_lw,
        double alpha_pad_lw,
        tries=np.inf
):
    """
    Generate random incoming directions of jams (in alpha unit under sine
    coordinate).

    :param alpha_c: Alpha of beam center.
    :param wavelength: Wavelength of signal (in Hz).
    :param num_subarrays: The number of sub-arrays.
    :param dx: Distance between antenna units of the uniform linear antenna
        array (in meters).
    :param num_jams: The number of jams.
    :param delta_m_lw: Limit of the difference between the incoming direction
        of an arbitrary jam and `alpha_c` (in unit of `lw`, see below).
    :param delta_s_lw: Limit of the difference between the incoming directions
        of arbitrary two jams (in unit of `lw`, see below).
    :param alpha_pad_lw: Minimum difference between the incoming direction
        of an arbitrary jam and `alpha`=±1 (in unit of `lw`, see below).
    :param tries: Maximum number of tries.
    :return: Array of jam alphas. If generating fails, returns an empty array.

    Note: Unit `lw` denotes the side-lobe width. For the uniform linear antenna
    array simulated in this function, we have:
        sidelobe_width = wavelength / dx / num_subarrays
    For the usual case where dx = wavelength / 2,
        sidelobe_width = 2 / num_subarrays

    """
    cdef:
        double _lw = wavelength / dx / num_subarrays
        double _delta_m = delta_m_lw * _lw
        double _delta_s = delta_s_lw * _lw
        double _alpha_pad = alpha_pad_lw * _lw
        double _alpha_lim = max(min(1 - _alpha_pad, 1), 0)

    if 2 * _alpha_lim <= 2 * _delta_m + _delta_s * (num_jams - 1):
        min_num_subarrays = \
            (2 * delta_m_lw + delta_s_lw * (num_jams - 1)) * \
            wavelength / (2 * dx * _alpha_lim)
        warnings.warn(
            "Too few sub-arrays to generate sufficiently seperated jam "
            "azimuths. "
            f"More than {min_num_subarrays:.2f} "
            "sub-arrays are needed."
        )
        return None

    cdef cnp.ndarray alphas = _random_jam_alphas_approx(
        alpha_c, num_jams, _delta_m, _delta_s, _alpha_lim, tries=np.inf
    )
    return alphas

cdef tuple gen_line_array_echo(
        double wavelength,
        int num_subarrays,
        double dx,
        double mainlobe_azim,
        target_range_gates: typing.Union[int, np.ndarray],
        target_signal_powers: typing.Union[float, np.ndarray],
        jam_azims: typing.Union[float, np.ndarray],
        jam_powers: typing.Union[float, np.ndarray],
        int num_range_gates,
        double noise_power = 1.
):
    """
    Generate with-jam echo data for benchmarking on a uniform linear array
    (ULA) model. Azimuths and powers of incoming jams must be provided.

    :param wavelength: Wavelength in Hz, used for calculating steering vectors.
    :param num_subarrays: The number of sub-arrays.
    :param dx: Distance between sub-arrays, i.e. antenna units of the uniform
        linear antenna array (in meters).
    :param mainlobe_azim: Azimuth of the formed main-lobe (in radians).
        This determines the steering vector of DBF.
    :param target_range_gates: Range gate index(es) of the expected target(s).
    :param target_signal_powers: Power(s) of received echo signal(s) of target(s)
        (in linear unit).
    :param jam_azims: Azimuth(s) of incoming jam(s).
    :param jam_powers: Average power(s) of incoming jam(s) (in linear unit).
    :param num_range_gates: Total number of range gates.
        Note that `target_range_gates` and `jam_range_gates` (if not None) must
        in [0, num_range_gates-1].
    :param noise_power: Power of noise (in linear unit).
    :return: Tuple of:
        (   normalized DBF steering vector,
            simulated signal array (shape=(num_subarrays, num_range_gates))   ).

    """
    cdef cnp.ndarray r_arrays = np.zeros((num_subarrays, 3))
    r_arrays[:, 0] = \
        (np.arange(num_subarrays) - (num_subarrays - 1) / 2) * dx

    cdef double k = 2 * np.pi / wavelength

    # ---------- broadcast parameters ----------
    target_range_gates = np.reshape(target_range_gates, [-1])
    target_signal_powers = np.reshape(target_signal_powers, [-1])
    target_range_gates, target_signal_powers = np.broadcast_arrays(
        target_range_gates, target_signal_powers
    )

    jam_azims = np.reshape(jam_azims, [-1])
    jam_powers = np.reshape(jam_powers, [-1])
    jam_azims, jam_powers = np.broadcast_arrays(
        jam_azims, jam_powers
    )

    # ---------- power levels ----------
    signal_powers_db = db(target_signal_powers, unit="power")
    noise_power_db = db(noise_power, unit="power")
    jam_powers_db = db(jam_powers, unit="power")

    # ---------- generate target signals ----------
    cdef:
        cnp.ndarray a0_vec = steering_vec(
            n_vec_from_azel(mainlobe_azim, 0.),
            r_arrays, k, normalize=True
        )
        cnp.ndarray signals = np.zeros(
            (num_subarrays, num_range_gates), dtype=complex
        )
        int _target_range_gate
        double _target_sig_power
    for _target_range_gate, _target_sig_power in zip(
            target_range_gates, target_signal_powers
    ):
        signals[:, _target_range_gate] = (
                np.sqrt(_target_sig_power) * a0_vec * np.sqrt(num_subarrays)  # dot pulse
                * channel_error(num_subarrays)
        )

    # ---------- generate noise ----------
    if len(signal_powers_db) > 0:
        min_signal_power_db = min(signal_powers_db)
    else:  # there is no targets
        min_signal_power_db = 0
    signals = awgn(
        signals,
        snr_db=min_signal_power_db - noise_power_db,
        signal_power_db=min_signal_power_db
    )  # add white Gaussian noise

    # ---------- generate jams ----------
    # add jam signals
    num_jams = len(jam_azims)
    for i_jam in range(num_jams):
        # power of _jam is 0dBW
        _jam = awgn(signals[1], snr_db=0, signal_power_db=0) - signals[1]
        a_vec_jam = steering_vec(
            n_vec_from_azel(jam_azims[i_jam], 0.),
            r_arrays, k, normalize=False
        )
        signals = signals + (10 ** (jam_powers_db[i_jam] / 20) *
                             np.outer(a_vec_jam * channel_error(num_subarrays), _jam))
    return a0_vec, signals

cdef tuple gen_ula_echo_with_rand_jams(
        double wavelength,
        int num_subarrays,
        double dx,
        double target_azim,
        target_range_gates: typing.Union[int, typing.Iterable[int]],
        int num_range_gates,
        int num_jams,
        double delta_m_lw,
        double delta_s_lw,
        double alpha_pad_lw,
        double signal_power = 1,
        double noise_power = 1,
        double jam_power = 1
):
    """
    Generate with-jam signal data for benchmarking on a uniform linear array
    (ULA) model. Azimuths and powers of incoming jams are randomly generated.

    :param wavelength: Wavelength in Hz, used for calculating steering vectors.
    :param num_subarrays: The number of sub-arrays.
    :param dx: Distance between sub-arrays (in meters).
    :param target_azim: Azimuth of the expected target(s) (in radians).
        This determines the steering vector of DBF.
        Note that if there are multiple targets, i.e. `size(target_range_gates)` > 1,
        they are assumed to have the same azimuth.
    :param target_range_gates: Range gate index(es) of the expected target(s).
    :param num_range_gates: Total number of range gates.
        Note that all values in `target_range_gates` must in [0, num_range_gates-1].
    :param num_jams: The number of jams.
    :param delta_m_lw: The least difference between any jam's alpha and the alpha
        of target (i.e. center of main-lobe) in unit of side-lobe width under sine
        coordinate.
    :param delta_s_lw: The least difference between any two jams' alpha values
        in unit of side-lobe width under sine coordinate.
    :param alpha_pad_lw: The minimum distance between absolute values of jam
        alphas and 1, in unit of side-lobe width under sine coordinate.
    :param signal_power: Power of signal of each sub-array (in linear unit).
    :param noise_power: Power of noise (in linear unit).
    :param jam_power: Power of jam (in linear unit).
    :return: Tuple of:
        (   jam azimuths,
            jam powers in dB,
            normalized DBF steering vector,
            simulated signal array (shape=(num_subarrays, num_range_gates))   ).

    """
    # ---------- generate jams ----------
    cdef double target_alpha = np.sin(target_azim)
    cdef cnp.ndarray jam_alphas
    if num_jams > 0:
        jam_alphas = random_jam_alphas(
            target_alpha,
            wavelength,
            num_subarrays,
            dx,
            num_jams,
            delta_m_lw=delta_m_lw,
            delta_s_lw=delta_s_lw,
            alpha_pad_lw=alpha_pad_lw,
            tries=np.inf
        )
    else:
        jam_alphas = np.array([], dtype=np.float64)
    if jam_alphas is None:
        raise RuntimeError("Cannot generate jam alphas.")

    cdef cnp.ndarray jam_azims = np.arcsin(jam_alphas)

    # jam powers in dBW (value range: [jam_power_db, jam_power_db * 105%])
    cdef double jam_power_db = db(jam_power, unit="power")
    cdef cnp.ndarray jam_powers_db = \
        (np.ones(num_jams)  # (1 + np.power(10, np.random.rand(num_jams)) * 5 / 1000)
         * jam_power_db)

    # ---------- generate echos ----------
    cdef cnp.ndarray a0_vec, signals
    a0_vec, signals = gen_line_array_echo(
        wavelength, num_subarrays, dx,
        target_azim,
        target_range_gates, signal_power,
        jam_azims, jam_power,
        num_range_gates, noise_power
    )

    return jam_azims, jam_powers_db, a0_vec, signals

def echo_gen(
        int case,
        int num_subarrays,
        int num_targets,
        int num_jams,
):
    """
    Generate with-jam signal data for testing.

    :param case: Type of test case, possible values are (0, 1, 2, 3).
    :param num_subarrays: The number of sub-arrays, which cannot be too small.
    :param num_targets: The number of targets, which must >= 0.
    :param num_jams: The number of jams, which must >= 0.
    :return: Tuple of:
        (   carrier frequency (in Hz),
            mainlobe azimuth (in radians),
            noise power (in Watts),
            simulated signal array (shape=(num_working_subarrays, num_range_gates)),
            indices of working sub-arrays    ).
    Here 0 < num_working_subarrays <= num_subarrays.

    """
    cdef:
        double FREQ = 3e9
        double TARGET_AZIM_BOUND = 75
        double NOISE_POWER = 1  # Watts
        double FORMED_SNR_DB = 20
        double INPUT_JNR_DB = 40

        double wavelength = SPEED_OF_LIGHT / FREQ
        double dx = wavelength / 2
        double lobewidth = wavelength / dx / num_subarrays

        double DELTA_M_LW = 3
        double DELTA_S_LW = 3
        double ALPHA_PAD_LW = 0.05 / lobewidth

    # Only in case 3:
    cdef:
        int MIN_WORKING_ARR_COUNT = int(num_subarrays / 2)  # inclusive
        int MAX_WORKING_ARR_COUNT = num_subarrays  # exclusive

    cdef:
        double s_power_in_per_subarr = 10 ** (FORMED_SNR_DB / 10.) * \
                                       NOISE_POWER / num_subarrays
        double j_power = 10 ** (INPUT_JNR_DB / 10.) * NOISE_POWER

        double target_azim = (np.random.rand() * 2 - 1) * \
                                  np.deg2rad(TARGET_AZIM_BOUND)

    cdef:
        double min_num_subarrays = \
            (2 * DELTA_M_LW + DELTA_S_LW * (num_jams - 1)) * \
            wavelength / (2 * dx * (1 - ALPHA_PAD_LW * lobewidth))
        int min_num_subarrays_ = int(two_pow_ceil(min_num_subarrays))
    if min_num_subarrays_ == min_num_subarrays:
        min_num_subarrays_ *= 2  # make sure min_num_subarrays_ > min_num_subarrays
    if num_subarrays < min_num_subarrays_:
        raise ValueError(
            f"Too few sub-arrays! (num_subarrays "
            f"must >= {min_num_subarrays_} but = {num_subarrays})"
        )

    cdef:
        int num_range_gates
        cnp.ndarray possible_target_range_gates
    if case == 0:
        num_range_gates = 3 * num_subarrays
        possible_target_range_gates = np.arange(0, int(num_range_gates / 3))
    elif case == 1:
        num_range_gates = 3 * num_subarrays
        possible_target_range_gates = np.arange(0, num_range_gates)
    elif case == 2:
        num_range_gates = num_subarrays
        possible_target_range_gates = np.arange(0, num_range_gates)
    elif case == 3:
        num_range_gates = 3 * num_subarrays
        possible_target_range_gates = np.arange(0, int(num_range_gates / 3))
    else:
        raise ValueError(f"Invalid case type: {case}. "
                         f"Only 0, 1, 2 and 3 are allowed.")

    cdef int max_num_targets = possible_target_range_gates.size
    
    num_targets = min(num_targets, max_num_targets)
    cdef cnp.ndarray target_range_gates = np.random.choice(
        possible_target_range_gates,
        size=num_targets,
        replace=False
    )

    jam_azims, jam_powers_db, a_vec, signals = gen_ula_echo_with_rand_jams(
        wavelength=wavelength,
        num_subarrays=num_subarrays,
        dx=dx,
        target_azim=target_azim,
        target_range_gates=target_range_gates,
        num_range_gates=num_range_gates,
        num_jams=num_jams,
        delta_m_lw=DELTA_M_LW,
        delta_s_lw=DELTA_S_LW,
        alpha_pad_lw=ALPHA_PAD_LW,
        signal_power=s_power_in_per_subarr,
        noise_power=NOISE_POWER,
        jam_power=j_power
    )
    subarray_indices = np.arange(0, num_subarrays)
    if case == 3:
        num_working_indices = np.random.randint(
            MIN_WORKING_ARR_COUNT, MAX_WORKING_ARR_COUNT
        )
        subarray_indices = np.sort(
            np.random.choice(
                subarray_indices, size=num_working_indices, replace=False
            )
        )
        signals = signals[subarray_indices]

    return FREQ, target_azim, NOISE_POWER, signals, subarray_indices

import numpy as np
from scipy import special


def triangle_stf(t: np.ndarray, t0: float, hdur: float):
    """
    x = 0, where t < t0 - hdur,
    x = (t - t0 + hdur) / hdur, where t0 - hdur <= t <= t0,
    x = (t0 + hdur - t) / hdur, where t0 <= t <= t0 + hdur,
    x = 0, where t > t0 + hdur

    The result is a triangular wavelet that has a duration of 2*hdur and a
    maximum amplitude 1/hdur, and area of 1."""

    startpoint = t0 - hdur
    midpoint = t0
    endpoint = t0 + hdur

    s = np.zeros_like(t)

    # Set 0 before triangle
    s[t < startpoint] = 0

    # Set linear increase
    s[(t >= startpoint) & (t <= midpoint)] = (t[(t >= startpoint) & (t <= midpoint)] - t0 + hdur) / hdur

    # Set linear decrease
    s[(t > midpoint) & (t <= endpoint)] = (t0 + hdur - t[(t > midpoint) & (t <= endpoint)]) / hdur

    # Set 0 after triangle
    s[t > endpoint] = 0

    return s/hdur

def boxcar_stf(t: np.ndarray, t0: float, hdur: float):
    """
    x = 0, where t < t0 - hdur,
    x = 1/(hdur*2), where t0 - hdur <= t <= t0 + hdur,
    x = 0, where t > t0 + hdur

    The result is a square wavelet that has a duration of 2*hdur and a maximum
    amplitude of 1/(hdur*2); that is, the are underneath the wavelet is 1.
    """

    startpoint = t0 - hdur
    endpoint = t0 + hdur

    s = np.zeros_like(t)

    # Set 0 before square
    s[t < startpoint] = 0

    # Set 1 during square
    s[(t >= startpoint) & (t <= endpoint)] = 1/(hdur*2)

    # Set 0 after square
    s[t > endpoint] = 0

    return s


def gaussian_stf(t, t0: float = 0.0, hdur: float = 0.0, alpha=1.628):
    """
    """

    if hdur == 0.0:
        normalize=True
        hdur = 1e6
    else:
        normalize=False

    # Exponent for the Gaussian
    exponent = -((alpha * (t - t0) / hdur) ** 2)

    # Are under the Gaussen -> M0
    gaussian = alpha / (np.sqrt(np.pi) * hdur) * np.exp(exponent)

    # Numerically a 0 halfduration does not make sense, here we set it to 1e6
    # to make sure that the area under the gaussian is still 1, we integrate the
    # gaussian and normalize it to 1.
    if normalize:
        gaussian = gaussian / np.trapz(gaussian, t)

    return gaussian



def step_stf(t, t0: float = 0.0, hdur: float = 0.0, alpha=1.628):
    """Computes specfem style error function."""

    if hdur == 0.0:
        hdur = 1e6

    sigma = hdur / alpha

    return 0.5*special.erf((t-t0)/sigma)+0.5


def interp_stf(qt, t, stf):
    """
    Interpolates the source time function to the given time vector. Values
    outside of `t` are set to 0.


    Parameters
    ----------
    t : np.ndarray
        Time vector of the source time function.
    stf : np.ndarray
        Source time function.
    qt : np.ndarray
        Time vector to interpolate to.

    Returns
    -------
    np.ndarray
        Interpolated source time function.
    """

    return np.interp(qt, t, stf, left=0, right=0)


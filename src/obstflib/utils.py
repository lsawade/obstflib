import typing as tp
from copy import deepcopy
import numpy as np
from scipy import special
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth
import obsplotlib.plot as opl
import matplotlib.pyplot as plt
import obsnumpy as onp


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



def error_stf(t, t0: float = 0.0, hdur: float = 0.0, alpha=1.628):
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


def check_meta(ds):
    """Simpe print function of the dataset's meta data."""
    meta = ds.meta.stations
    pos = np.argsort(ds.meta.stations.distances)
    for i in pos:
        print(f"{meta.codes[i]}: {meta.distances[i]:.2f} dg -- {meta.attributes.arrivals.Love.min[i] - meta.attributes.arrivals.Love.max[i]:.2f} s")



def next_power_of_2(x):
    return int(1) if x == 0 else int(2**np.ceil(np.log2(x)))


def get_distances_azimuths(cmt, meta):
    """Generate distances and azimuths from the cmt and the meta dictionary."""

    # Get the cmt location
    cmt_lat = cmt.latitude
    cmt_lon = cmt.longitude

    slat = meta['latitudes']
    slon = meta['longitudes']

    distances = []
    azimuths = []
    back_azimuths = []
    for lat, lon in zip(slat, slon):
        dist, az, baz = gps2dist_azimuth(cmt_lat, cmt_lon, lat, lon)
        distances.append(dist)
        azimuths.append(az)
        back_azimuths.append(baz)


    # Make numpy arrays
    distances = np.array(distances)/111.11/1000
    azimuths = np.array(azimuths)
    back_azimuths = np.array(back_azimuths)

    return distances, azimuths, back_azimuths


def get_arrivals(cmt, meta, phase='P'):
    """Compute teleseismic arrivals for a given cmt and distances and azimuths."""

    # Make model
    model = TauPyModel(model='ak135')


    if phase == 'P':
        phaselist = ['P']
    elif phase == 'anyP':
        # With these three phases taup should return an arrival time for any epicentral distance
        phaselist = ['P', 'Pdiff', 'PKP', 'PKIKP']
    elif phase == 'S':
        phaselist = ['S']
    elif phase == 'anyS':
        # With these three phases taup should return an arrival time for any epicentral distance
        phaselist = ['S', 'ScS', 'SKS', 'Sdiff', 'SKIKS']
    elif phase == 'S':
        phaselist = ['S']
    elif phase == 'Rayleigh':
        minvel = 3.0
        maxvel = 4.2
    elif phase == 'Love':
        minvel = 3.0
        maxvel = 5.0
    else:
        raise ValueError(f'Phase {phase} not recognized')

    if not 'arrivals' in meta:
        meta['arrivals'] = {}

    # Get separate function to compute min and max surface wave windows.
    if phase in ['Rayleigh', 'Love']:

        minarrivals = get_surface_wave_arrivals(meta['distances'], minvel, ncircles=1)[:, 0]
        maxarrivals = get_surface_wave_arrivals(meta['distances'], maxvel, ncircles=1)[:, 0]

        if not phase in meta['arrivals']:
            meta['arrivals'][phase] = {}
        meta['arrivals'][phase]['min'] = minarrivals
        meta['arrivals'][phase]['max'] = maxarrivals

    else:

        # Get the arrivals
        phase_arrivals = []
        for dist in meta['distances']:

            arrivals = model.get_travel_times(source_depth_in_km=cmt.depth,
                                                distance_in_degree=dist,
                                                phase_list=phaselist)
            if len(arrivals) > 0:
                arrivals = sorted(arrivals, key=lambda x: x.time)
                phase_arrivals.append(arrivals[0].time)
            else:
                phase_arrivals.append(np.nan)

        meta['arrivals'][phase] = np.array(phase_arrivals)





def get_surface_wave_arrivals(dist_in_deg, vel, ncircles=1):
    """
    Calculate the arrival time of surface waves, based on the distance
    and velocity range (min_vel, max_vel).
    This function will calculate both minor-arc and major-arc surface
    waves. It further calcualte the surface orbit multiple times
    if you set the ncircles > 1.

    Returns the list of surface wave arrivals in time order.
    """

    earth_circle = 111.11*360.0
    dt1 = earth_circle / vel

    # 1st arrival: minor-arc arrival
    minor_dist_km = 111.11*dist_in_deg  # major-arc distance
    t_minor = minor_dist_km / vel

    # 2nd arrival: major-arc arrival
    major_dist_km = 111.11*(360.0 - dist_in_deg)
    t_major = major_dist_km / vel

    # prepare the arrival list
    arrivals = np.zeros((len(dist_in_deg), ncircles*2))

    for i in range(ncircles):

        ts = t_minor + i * dt1
        arrivals[:, 2*i] = ts

        ts = t_major + i * dt1
        arrivals[:, 2*i + 1] = ts

    return arrivals


def get_surface_wave_windows(dist_in_deg, min_vel, max_vel, ncircles=1):
    """
    Calculate the arrival time of surface waves, based on the distance
    and velocity range (min_vel, max_vel).
    This function will calculate both minor-arc and major-arc surface
    waves. It further calcualte the surface orbit multiple times
    if you set the ncircles > 1.

    Returns the list of surface wave arrivals in time order.
    """
    if min_vel > max_vel:
        min_vel, max_vel = max_vel, min_vel

    earth_circle = 111.11*360.0
    dt1 = earth_circle / max_vel
    dt2 = earth_circle / min_vel

    # 1st arrival: minor-arc arrival
    minor_dist_km = 111.11*dist_in_deg  # major-arc distance
    t_minor = [minor_dist_km / max_vel, minor_dist_km/min_vel]

    # 2nd arrival: major-arc arrival
    major_dist_km = 111.11*(360.0 - dist_in_deg)
    t_major = [major_dist_km / max_vel, major_dist_km / min_vel]

    # prepare the arrival list
    windows = []
    for i in range(ncircles):
        ts = [t_minor[0] + i * dt1, t_minor[1] + i * dt2]
        windows.append(ts)

        ts = [t_major[0] + i * dt1, t_major[1] + i * dt2]
        windows.append(ts)

    return windows


def reindex_dict(meta, idx, N_original=None, debug=False):
    outmeta = deepcopy(meta)

    if N_original is None:
        N_original = len(meta['stations'])

    for key in meta.keys():

        if debug:
            print(key, type(meta[key]))

        if isinstance(meta[key], list) and len(meta[key]) == N_original:
            if debug:
                print("--> Reindexing", key, type(meta[key]), len(meta[key]), N_original)
            outmeta[key] = [meta[key][i] for i in idx]

        elif isinstance(meta[key], np.ndarray) and len(meta[key]) == N_original:
            if debug:
                print("--> Reindexing", key, type(meta[key]), len(meta[key]), N_original)
            outmeta[key] = meta[key][idx]

            if debug:
                print("  |-> Reindexed", key, type(outmeta[key]), len(outmeta[key]))
        elif isinstance(meta[key], dict):
            if debug:
                print("--> Entering ")
            outmeta[key] = reindex_dict(meta[key], idx, N_original=N_original)

        else:
            if debug:
                print("--> Not reindexing.")

    return outmeta



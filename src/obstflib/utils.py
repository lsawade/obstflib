import typing as tp
import datetime
from copy import deepcopy
import numpy as np
from scipy import special
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth
import obsplotlib.plot as opl
import matplotlib.pyplot as plt
import obsnumpy as onp
import scipy as sp


def log(msg):
    length = 80
    length_msg = len(msg)
    length_right = (length - length_msg) - 1
    if length_right < 0:
        fill = ""
    else:
        fill = "-" * length_right
    print(f'[{datetime.datetime.now()}] {msg} {fill}', flush=True)
    
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
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def gaussn(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


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


def exp_model_func(t, A, B):
    return -np.exp(-A * (t - B)) + 1.0


def fit_exp_nonlinear(t, y, p0=None):
    
    # Define cost function
    fun = lambda x : np.sum((exp_model_func(t, x[0], x[1]) - y)**2)
    
    params = sp.optimize.minimize(fun, x0=p0, method='Nelder-Mead',
                                        options=dict(maxiter=1000)).x
    A, B = params
    return A, B

def log_model_func(t, A, B):
    return -1.0 / (1.0 + np.exp(A * (t - B))) + 1.0


def fit_logistic(t, y, p0=None):
    
    # Define cost function
    fun = lambda x : np.sum((log_model_func(t, x[0], x[1]) - y)**2)
    
    # optimize
    params = sp.optimize.minimize(fun, x0=p0, method='Nelder-Mead',
                                  options=dict(maxiter=1000)).x
    A, B = params

    return A, B

def find_elbow_point(k, curve):
    """Finds the elbow point of an L-Curve."""

    # Get number of points
    nPoints = len(curve)
    
    # Make vector with all the coordinates
    allCoord = np.vstack((k, curve)).T
    
    # First point is the first coordinate (top-left of the curve)
    firstPoint = allCoord[0]
    
    # Compute the vector pointing from first to last point
    lineVec = allCoord[-1] - allCoord[0]
    
    # Normalize the line vector
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    
    # Compute the vector from first point to all points
    vecFromFirst = allCoord - firstPoint
    
    # Compute the projection length of the projection of p onto b
    scalarProduct = np.sum(vecFromFirst * np.tile(lineVecNorm, (nPoints, 1)), axis=1)

    # Compute the vector on unit b by using the projection length
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    
    # Compute the vector from first to parallel line 
    vecToLine = vecFromFirst - vecFromFirstParallel
    
    # Compute the distance to the line
    distToLine = np.sqrt(np.sum(vecToLine**2, axis=1))
    
    # Get the index
    idxOfBestPoint = np.argmax(distToLine)
    
    return idxOfBestPoint

def find_cond_elbow_idx(t, STF, A_exp_thresh=0.0175, B_exp_thresh=5.0, 
                        A_log_thresh=0.0312, B_log_thresh=50.0, extra_long=False):
    """input is cumulative STF"""
    # Fit decay rate of the STF
    long_stf = False
    
    A_exp, B_exp = fit_exp_nonlinear(t, STF, p0=[0.04, 5.00])
    
    # Return the relevant function        
    func_exp = exp_model_func
    
    # Fit the cumulative STF to logistic decay
    A_log, B_log = fit_logistic(t, STF, p0=[0.04, 5.0])

    # Return the relevant function
    func_log = log_model_func
    
    # Check if the STF is long
    print("Coefficients", A_exp, B_exp, A_log, B_log)
    
    if ((B_exp >= B_exp_thresh and A_exp <= A_exp_thresh) and 
            (A_log <= A_log_thresh and B_log >= B_log_thresh)):
        _t = t[np.argmin(np.abs(STF - 0.95))]
        if extra_long:
            idx = np.argmin(np.abs(t - 3.*_t))
        else:
            idx = np.argmin(np.abs(t -2.*_t))
        long_stf = True
        
    elif ((A_exp <= A_exp_thresh and B_exp >= 10.0 ) and 
            (A_log <= 0.0375 and B_log >= 65.)):
        _t = t[np.argmin(np.abs(STF - 0.95))]
        idx = np.argmin(np.abs(t -2.*_t))
        long_stf = True
        
    elif ((A_exp <= 0.02 and B_exp >= 10.0) and 
            (A_log <= 0.0655 and B_log >= 55.)):
        _t = t[np.argmin(np.abs(STF - 0.95))]
        idx = np.argmin(np.abs(t -2.*_t))
        long_stf = True
        
    # Even longer STF
    elif ((A_exp <= A_exp_thresh and B_exp >= 10.0) and 
            (A_log <= 0.035 and B_log >= 65)):
        _t = t[np.argmin(np.abs(STF - 0.95))]
        idx = np.argmin(np.abs(t - 2.5*_t))
        long_stf = True
    
    # Even longer STF
    elif ((A_exp <= 0.0195 and B_exp >= 5.0) and 
            (A_log <= 0.0410 and B_log >= 49)):
        _t = t[np.argmin(np.abs(STF - 0.95))]
        idx = np.argmin(np.abs(t - 2.5*_t))
        long_stf = True
        
    else:
        idx = np.argmin(np.abs(STF - 0.95))
        long_stf = False
    
    return idx, (func_exp, A_exp, B_exp), (func_log, A_log, B_log), long_stf

def find_tmax(t, STF, **kwargs):
    
    # Compute the sampling interval
    dt = t[1] - t[0]
    
    # Find the elbow point
    _idx, _, _, long_stf = find_cond_elbow_idx(t, STF, **kwargs)
    
    # Find elbow point using the cumulative STF
    idx = find_elbow_point(t[:_idx], STF[:_idx]) 
    
    # Add the length of the STF divided by 10 as a buffer
    idx += t[idx]/10/dt
    idx = int(idx)
    
    # Get the final time
    tmax = t[idx]
    
    return tmax, long_stf


def compute_azimuthal_weights(az, weights=None, nbins=12, p=1):
    
    # Create the bins
    bins = np.arange(0, 360.1, 360/nbins)

    # Histogram
    H, _ = np.histogram(az, bins=bins, weights=weights)

    # Find which az is in which bin
    binass = np.digitize(az, bins) - 1

    # Compute weights
    w = (1/H[binass])**p

    # Normalize
    w /= np.mean(w)

    return w


def find_Tmax(tmaxs, costs, grads, Npad=150, cost_only=False):
    """Finds the elbow point of an L-Curve for the source time function problem.
    We are padding the L-curve to ensure that the elbow point on the lower side of the elbow."""
    
    _cost = np.pad(costs, (0, Npad), 'constant', constant_values=(0, 0))
    _grad = np.pad(grads, (0, Npad), 'constant', constant_values=(0, 0))
    
    # extend the duration values by Npad
    dt = tmaxs[1] - tmaxs[0]
    _tmax = np.hstack([tmaxs, tmaxs[-1] + np.arange(1, Npad+1)*dt])

    # Get elbow points
    ic = find_elbow_point(_tmax, _cost) + 1
    ig = find_elbow_point(_tmax, _grad) + 1
    
    # Choose the maximum of the grad and cost elbow
    if cost_only:
        imax = ic
    else:
        imax = np.max([ic, ig])
    
    return imax

def norm_AIC(costs, Ns, k, coeff=1.0):
    """Compute the normalized AIC for a given cost, number of samples and number of model parameters."""
    
    # Number of model parameters k
    # k = tmaxs * knots_per_second + 1
    
    # AIC + 1 too avoid log(0)
    aic = (coeff * Ns * np.log(costs+1) + 2 * k)
    
    # normalize aic (Not technically necessary, remnant from plot various AIC lines)
    aic = aic - np.min(aic)
    aic = aic / np.max(aic)
    
    return aic

def find_Tmax_AIC(tmaxs, costs, grads, knots_per_second, Ns, coeff=1.0):
    """Find tmax using the akaike information criterion. Note that this is "inverted" in
    the sense that we are trying to minimize the AIC and not maximize it as in the original
    likelihood version. See Kintner et al. 2024 """
    
    # Number of model parameters k
    k = tmaxs * knots_per_second + 1
    
    # Compute normalized AIC
    aic = norm_AIC(costs, Ns, k, coeff=coeff)
    
    # Get the index of the minimum aic
    idx = np.argmin(aic)
    
    return idx


def timeshift(s: np.ndarray, dt: float, shift: float) -> np.ndarray:
    """ shift a signal by a given time-shift in the frequency domain
    Parameters
    ----------
    s : Arraylike
        signal
    N2 : int
        length of the signal in f-domain (usually equals the next pof 2)
    dt : float
        sampling interval
    shift : float
        the time shift in seconds
    Returns
    -------
    Timeshifted signal"""

    S = np.fft.fft(s, axis=-1)

    # Omega
    phshift = np.exp(-1.0j*shift*np.fft.fftfreq(s.shape[-1], dt)*2*np.pi)
    s_out = np.real(np.fft.ifft(phshift[None, :] * S, axis=-1))
    return s_out


def reckon(lat, lon, distance, bearing):
    """ Computes new latitude and longitude from bearing and distance.

    Parameters
    ----------
    lat: in degrees
    lon: in degrees
    bearing: in degrees
    distance: in degrees

    Returns
    -------
    lat, lon


    lat1 = math.radians(52.20472)  # Current lat point converted to radians
    lon1 = math.radians(0.14056)  # Current long point converted to radians
    bearing = np.pi/2 # 90 degrees
    # lat2  52.20444 - the lat result I'm hoping for
    # lon2  0.36056 - the long result I'm hoping for.

    """

    # Convert degrees to radians for numpy
    lat1 = lat/180*np.pi
    lon1 = lon/180 * np.pi
    brng = bearing/180*np.pi
    d = distance/180*np.pi

    # Compute latitude
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d)
                     + np.cos(lat1) * np.sin(d) * np.cos(brng))

    # Compute longitude
    lon2 = lon1 + np.arctan2(np.sin(brng) * np.sin(d) * np.cos(lat1),
                             np.cos(d) - np.sin(lat1) * np.sin(lat2))

    # Convert back
    lat2 = lat2/np.pi*180
    lon2 = lon2/np.pi*180

    # Correct the longitude lattitude values
    lon2 = np.where(lon2 < -180.0, lon2+360.0, lon2)
    lon2 = np.where(lon2 > 180.0, lon2-360.0, lon2)

    return lat2, lon2
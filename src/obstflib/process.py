import numpy as np
import obsnumpy as onp
import obspy
from . import STF

# Processing of the Observed and synthetic datasets
def process_ds(ds: onp.Dataset, starttime, length_in_s, sps, step=False):

    # Taper the seismograms
    ds.taper(max_percentage=0.05, type='cosine')

    # Bandpass the seismograms
    ds.filter('bandpass', freqmin=0.005, freqmax=1/17.0, corners=3, zerophase=True)

    # Resample the seismograms
    ds.interpolate(sampling_rate=sps, starttime=starttime, npts=int((length_in_s) * sps),
                   method='weighted_average_slopes')

    # Rotate the seismograms
    ds.rotate('->RT')

    # Taper the seismograms
    ds.taper(max_percentage=0.05, type='cosine')

    # Convolve the seismograms with a step function
    if step:

        # Vector
        t = np.arange(0, length_in_s, 1/sps)

        # Create Error function
        error = STF.error(origin=obspy.UTCDateTime(0), t=t, tshift=150.0, hdur=1e-6, tc=0.0)

        # Convolve the seismograms
        ds.convolve(error.f, 150.0)

        # Taper the seismograms
        ds.taper(max_percentage=0.05, type='cosine')

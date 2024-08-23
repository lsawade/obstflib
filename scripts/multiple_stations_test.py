# %%
# convolution testing
import os
import obspy
import numpy as np
import matplotlib.pyplot as plt
import obsplotlib.plot as opl
from copy import deepcopy
import shutil
import obstflib as osl
import obsplotlib.plot as opl
import gf3d.utils
from scipy import fft
from gf3d.source import CMTSOLUTION
import obsnumpy as onp


# %%
# Files
scardec_stf_dir = './STF/FCTs_20180123_093140_GULF_OF_ALASKA'
scardec_stf_dir = './STF/FCTs_20070815_234057_NEAR_COAST_OF_PERU'
scardec_id = scardec_stf_dir.split('/')[-1]
cmt_file = os.path.join(scardec_stf_dir, 'CMTSOLUTION')

# %%
# Load scardec stf
scardec = osl.STF.scardecdir(scardec_stf_dir, 'optimal')

# %%
# Get the event record at II BFO for the Alaska event using the GFManager

# create CMT solution from scardec sdr and set half duration to 0
# cmt = CMTSOLUTION.from_sdr(
#     s=scardec.strike1, d=scardec.dip1, r=scardec.rake1, M0=scardec.M0 * 1e7,
#     origin_time=scardec.origin, depth=scardec.depth,
#     latitude=scardec.latitude, longitude=scardec.longitude)

cmt = CMTSOLUTION.read(cmt_file)
cmt.time_shift = 0.0


# %%
# Read the Green functions
from gf3d.seismograms import GFManager
gfm = GFManager(f'/lustre/orion/geo111/scratch/lsawade/STF_SUBSETS/{scardec_id}/subset.h5')
gfm.load()

# %%
# Get Green function using obspy Stream
gf = gfm.get_seismograms(cmt, raw=True)

# %%
# Get Seismograms
def get_ds(gfm: GFManager, cmt: CMTSOLUTION, raw=True):
    array, meta = gfm.get_seismograms(cmt=cmt, raw=raw, array=True)
    new_meta = onp.Meta.from_dict(meta)
    return onp.Dataset(data=array, meta=new_meta)


gf_base_ds = get_ds(gfm, cmt, raw=True)

#%%
gf_base_ds.compute_geometry(cmt.latitude, cmt.longitude)

# %%
# Download the corresponding data

# Remove data directory if it exists
# if os.path.exists(os.path.join(scardec_stf_dir, 'data')):
#     shutil.rmtree(os.path.join(scardec_stf_dir, 'data'))

networks = ",".join({_sta.split('.')[0] for _sta in gf_base_ds.meta.stations.codes})
stations = ",".join({_sta.split('.')[1] for _sta in gf_base_ds.meta.stations.codes})

# networks = ",".join({tr.stats.network for tr in gf})
# stations = ",".join({tr.stats.station for tr in gf})


# opl.download_data(
#     os.path.join(scardec_stf_dir, 'data'),
#     starttime=cmt.origin_time - 300,
#     endtime=cmt.origin_time + 3600 + 300,
#     network=networks, station=stations,
#     channel_priorities=['BH[ZNE12]', 'HH[ZNE12]', 'LH[ZNE12]'],
#     location_priorities=['00', '10', '01']
#     )

# %%
# Read the data
raw = obspy.read(os.path.join(scardec_stf_dir, 'data', 'waveforms/*.mseed'))
inv = obspy.read_inventory(os.path.join(scardec_stf_dir, 'data', 'stations/*.xml'))

# %%

# %%

def convolve(trace: np.ndarray, stf: np.ndarray, dt: float, tshift: float) -> np.ndarray:

    # For following FFTs
    N = len(trace)
    NP2 = gf3d.utils.next_power_of_2(2 * N)

    # Fourier Transform the STF
    TRACE = fft.fft(trace, n=NP2)
    STF = fft.fft(stf, n=NP2)

    # Compute correctional phase shift
    shift = -tshift
    phshift = np.exp(-1.0j * shift * np.fft.fftfreq(NP2, dt) * 2 * np.pi)

    # Return the convolution
    return np.real(fft.ifft(TRACE * STF * phshift))[:N] * dt

from obspy.geodetics import gps2dist_azimuth
# Process the reciprocal green function to first filter the resample
def process(st, starttime, length_in_s, sps, step=False,
            inv: obspy.Inventory | None = None, event_lat=None, event_lon=None):

    # Taper the seismograms
    st.taper(max_percentage=0.05, type='cosine')

    # Remove response if inventory is given given
    if inv:
        st.detrend('linear')
        st.detrend('demean')

        st.remove_response(inventory=inv, output='DISP',
                           pre_filt=(0.001, 0.005, 1.0, 1/0.5),
                           water_level=100)
        st.rotate('->ZNE', inventory=inv)

        for tr in st:
            station = inv.select(network=tr.stats.network, station=tr.stats.station)[0][0]
            tr.stats.latitude = station.latitude
            tr.stats.longitude = station.longitude

    st.filter('bandpass', freqmin=0.004, freqmax=1/17.0, corners=3, zerophase=True)
    st.interpolate(sampling_rate=sps, starttime=starttime,
                   npts=int((length_in_s) * sps))

    # Rotation needs to happen after resampling
    if event_lat and event_lon:
        print("ROTATING")
        # Compute distances and azimuths
        for tr in st:
            distance, tr.stats.azimuth, tr.stats.back_azimuth = gps2dist_azimuth(
                event_lat, event_lon, tr.stats.latitude, tr.stats.longitude)
            tr.stats.distance = distance / 1000.0 / 111.111 # convert m to degrees

        # Rotate the seismograms
        for tr in st:
            st.rotate('NE->RT')

    # Taper the seismograms
    st.taper(max_percentage=0.05, type='cosine')

    if step:
        # Convolve the seismograms with the source time functions
        for tr in st:
            error = osl.STF.error(origin=obspy.UTCDateTime(0), t=tr.times(), tshift=150.0, hdur=1e-6, tc=0.0)
            tr.data = convolve(tr.data, error.f, tr.stats.delta, 150.0)

        # Taper the seismograms
        st.taper(max_percentage=0.05, type='cosine')



tshift = 200.0
starttime = cmt.origin_time - tshift
sps = 2.0
length_in_s = 3600 + tshift

praw = raw.copy()
p_gf = gf.copy()
process(praw, starttime, length_in_s, sps, inv=inv, event_lat=cmt.latitude, event_lon=cmt.longitude)
process(p_gf, starttime, length_in_s, sps, step=True, event_lat=cmt.latitude, event_lon=cmt.longitude)

# %%
# Loading the raw traces into obspy with preprocessing
praw2 = raw.copy()

# Preprocess variables
pre_filt = (0.001, 0.002, 1, 2)

# Define sampling rate as a function of the pre_filt
sampling_rate = pre_filt[3] * 2.5

# Response output
rr_output = "DISP"
water_level = 60

praw_ds = onp.Dataset.from_raw(praw2, inv=inv, starttime=starttime-50,
                               sps=sampling_rate, length_in_s=length_in_s+100,
                               event_latitude=cmt.latitude,
                               event_longitude=cmt.longitude,
                               water_level = 60, pre_filt=pre_filt,
                               rr_output=rr_output,
                               filter=False)



# %%
# Processing of the Observed and synthetic datasets
def process_ds(ds, starttime, length_in_s, sps, step=False):

    # Taper the seismograms
    ds.taper(max_percentage=0.05, type='cosine')

    # Bandpass the seismograms
    ds.filter('bandpass', freqmin=0.004, freqmax=1/17.0, corners=3, zerophase=True)

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
        error = osl.STF.error(origin=obspy.UTCDateTime(0), t=t, tshift=150.0, hdur=1e-6, tc=0.0)

        # Convolve the seismograms
        ds.convolve(error.f, 150.0)

        # Taper the seismograms
        ds.taper(max_percentage=0.05, type='cosine')


gf_ds = gf_base_ds.copy()
obs_ds = praw_ds.copy()

tshift = 200.0
starttime = cmt.origin_time - tshift
sps = 2.0
length_in_s = 3600 + tshift

process_ds(obs_ds, starttime, length_in_s, sps, step=False)
process_ds(gf_ds, starttime, length_in_s, sps, step=True)


#%%

#%%
# Create reference datasets
gf_ds_check = onp.Dataset.from_stream(p_gf,
                                      event_latitude=cmt.latitude,
                                      event_longitude=cmt.longitude,
                                      components=['Z', 'R', 'T'])


obs_ds_check = onp.Dataset.from_stream(praw, inv=inv,
                                       event_latitude=cmt.latitude,
                                       event_longitude=cmt.longitude,
                                       components=['Z', 'R', 'T'])



#%%
def plot_check(ds1, ds2, label1='Direct', label2='Obspy',
               network = 'II', station = 'BFO', component = 'Z',
               outfile = 'atest.pdf'):


    # Get the station component on the subset
    idx = ds1.meta.stations.codes.tolist().index(f'{network}.{station}')
    idc = ds1.meta.components.index(component)
    idx_check = ds2.meta.stations.codes.tolist().index(f'{network}.{station}')
    idc_check = ds2.meta.components.index(component)




    import numpy as np
    import matplotlib.pyplot as plt



    plt.figure()
    lw = 0.25
    plt.subplot(111)
    t = np.arange(0, length_in_s, 1/sps)
    plt.plot(ds1.t, ds1.data[idx,idc, :], "-k", label=label1, lw=lw)
    plt.plot(ds2.t, ds2.data[idx_check, idc_check, :], "r-", label=label2,    lw=lw)
    plt.xlim(0, 3600)
    plt.legend(frameon=False)
    plt.legend(frameon=False)
    plt.savefig(outfile, dpi=300)


plot_check(gf_ds, gf_ds_check, outfile='atest_gfs.pdf')
plot_check(obs_ds, obs_ds_check, outfile='atest_obs.pdf')


# %%

ds_obs, ds_gf = obs_ds.intersect(gf_ds)


# %%
# Make array from stream

def stream_to_array(st: obspy.Stream, components = ['Z', 'R', 'T']):
    stations =set()
    for tr in st:

        station = f"{tr.stats.network}.{tr.stats.station}"
        stations.add(station)

    # Get stations with less than 3 components
    remove = []
    for i, station in enumerate(stations):
        _net, _sta = station.split('.')
        substream = st.select(network=_net, station=_sta)

        if len(substream) < 3 or len(substream) > 3:
            print(f"Station {station} has {len(substream)} components. Removing.")
            remove.append(station)

    # Remove those stations
    for station in remove:
        stations.remove(station)

    # Get number of stations, points and components to get array size
    Nstations = len(stations)
    Npts = len(st[0].data)
    Ncomponents = len(components)

    # Create array
    array = np.zeros((Nstations, Ncomponents, Npts))

    # Metadata
    meta = {
        'starttime': st[0].stats.starttime,
        'delta': st[0].stats.delta,
        'npts': Npts,
        'stations': list(stations),
        'latitudes': [],
        'longitudes': [],
        'components': components,
        'distances': [],
        'azimuths': [],
        'back_azimuths': []
    }

    for i, station in enumerate(stations):
        _net, _sta = station.split('.')
        substream = st.select(network=_net, station=_sta)

        if len(substream) < 3 or len(substream) > 3:
            print(f"Station {station} has {len(substream)} components. \n   --> Station selection not working properly!.")
            continue


        # Get stats from the processed traces
        meta['latitudes'].append(substream[0].stats.latitude)
        meta['longitudes'].append(substream[0].stats.longitude)
        meta['distances'].append(substream[0].stats.distance)
        meta['azimuths'].append(substream[0].stats.azimuth)
        meta['back_azimuths'].append(substream[0].stats.back_azimuth)


        for j, component in enumerate(components):
            subtr = substream.select(component=component)
            if len(subtr) > 0:
                array[i, j, :] = subtr[0].data
            else:
                print(f"Did not find component {component} for station {station}")

    # Make lists into arrays
    meta['latitudes'] = np.array(meta['latitudes'])
    meta['longitudes'] = np.array(meta['longitudes'])
    meta['distances'] = np.array(meta['distances'])
    meta['azimuths'] = np.array(meta['azimuths'])
    meta['back_azimuths'] = np.array(meta['back_azimuths'])

    return array, meta


# obs_array, obs_meta = stream_to_array(praw)
# gf__array, gf__meta = stream_to_array(p_gf)


# %%
# Create STF

# Convolve with STF
t = np.arange(0, gf_ds.meta.npts * gf_ds.meta.delta, gf_ds.meta.delta)

stf = osl.STF.triangle(
    origin=cmt.origin_time, t=t, tc=cmt.time_shift,
    tshift=tshift, hdur=cmt.hdur)

stf = deepcopy(scardec)
stf.interp(t, origin=cmt.origin_time, tshift=tshift)

plt.figure(figsize=(10,2))
stf.plot()
plt.savefig('astf.pdf', dpi=300)

# %%
# Make synthetics with the STF
syn_ds = gf_ds.copy()

syn_ds.convolve(stf.f/stf.M0, tshift)

plot_check(obs_ds, syn_ds, outfile='atest_syn.pdf', label1='Observed', label2='Synthetic')


# %%
# Now get the intersection of the datasets
ds_obs, ds_gfs = obs_ds.intersect(gf_ds)

ds_syn = ds_gfs.copy()
ds_syn.convolve(stf.f/stf.M0, tshift)

plot_check(ds_obs, ds_syn, outfile='atest.pdf',
           label1='Observed', label2='GF')

# %%

# Get all stations where the epicentral distance is large than 30 dg

def plot_check_section(dss, labels=['Observed','Synthetic'],
                       component = 'Z', mindist=30.0, maxdist=np.inf,scale=1.0,
                       outfile = 'atestsection.pdf'):

    # Get which traces to plot


    colors = ['k', 'tab:red', 'tab:blue', 'tab:orange']
    plt.figure(figsize=(10,10))

    scale = scale * 1/np.max([np.max(np.abs(ds.data)) for ds in dss])

    for i, ds in enumerate(dss):
        idx = np.where((ds.meta.stations.distances > mindist) &
                       (ds.meta.stations.distances < maxdist))[0]
        idx2 = np.argsort(ds.meta.stations.distances[idx])
        pos = idx[idx2]

        # Get which component to plot
        ic = ds.meta.components.index(component)

        t = np.arange(0, ds.meta.npts * ds.meta.delta, ds.meta.delta)

        x = np.arange(len(pos))
        plt.plot(t, scale * ds.data[pos, ic, :].T + x, c=colors[i], lw=0.25)
        plt.plot([],[], c=colors[i], label=labels[i])

    plt.yticks(x, [f"{dss[-1].meta.stations.distances[i]:.2f}" for i in pos],
            rotation=0)
    plt.legend(frameon=False, loc='upper right', ncol=5)
    plt.savefig(outfile, dpi=300)

plot_check_section([ds_obs, ds_syn], scale=30.0, outfile='atestsection.pdf')


# %%
# Note that any P is really my definition of any P arrival from taup P
# and that could be P, Pdiff, PKP

from obsnumpy.traveltime import get_arrivals

phases = ['P', 'S', 'Rayleigh', 'Love', 'anyP', 'anyS']

for phase in phases:

    get_arrivals(cmt, ds_obs, phase=phase)
    get_arrivals(cmt, ds_syn, phase=phase)
    get_arrivals(cmt, ds_gfs, phase=phase)


# %%

def check_meta(ds):
    meta = ds.meta.stations
    pos = np.argsort(ds.meta.stations.distances)
    for i in pos:
        print(f"{meta.codes[i]}: {meta.distances[i]:.2f} dg -- {meta.attributes.arrivals.Love.min[i] - meta.attributes.arrivals.Love.max[i]:.2f} s")

check_meta(ds_obs)


# %%
def select_traveltime_subset(ds, mindist=30.0, maxdist=np.inf, component='Z', phase='P'):
    """Selects subset of the array based on the distance range and component.
    And where we have a valid arrival time."""

    # Just get short forms of the relevant objects
    stations = ds.meta.stations
    arrivals =  ds.meta.stations.attributes.arrivals

    # Get distance selection station is at least mindist away and at most
    # maxdist away
    selection = (stations.distances > mindist ) & (stations.distances < maxdist)

    if phase == 'Ptrain':
        selection = selection & (~np.isnan(arrivals.anyP)) & (~np.isnan(arrivals.anyS))

    elif phase == 'P':
        selection = selection & (~np.isnan(arrivals.P))

    elif phase == 'Strain':
        selection = selection & (~np.isnan(arrivals.anyS))

        if component in ['Z', 'R']:
            selection = selection & (~np.isnan(arrivals.Rayleigh.min))
        elif component == 'T':
            selection = selection & (~np.isnan(arrivals.Love.min))
        else:
            raise ValueError("Component must be Z, R or T")

    elif phase == 'S':
        selection = selection & (~np.isnan(arrivals.P))

    elif phase == 'Rayleigh':
        selection = selection & (~np.isnan(arrivals.Rayleigh.min))
        selection = selection & (~np.isnan(arrivals.Rayleigh.max))

    elif phase == 'Love':
        selection = selection & (~np.isnan(arrivals.Love.min))
        selection = selection & (~np.isnan(arrivals.Love.max))

    elif phase == 'body':
        if component in ['Z', 'R']:
            selection = selection & (~np.isnan(arrivals.anyP)) & (~np.isnan(arrivals.Rayleigh.min))
        elif component == 'T':
            selection = selection & (~np.isnan(arrivals.anyP)) & (~np.isnan(arrivals.Love.min))

    else:
        raise ValueError("Component must be Z, R or T")

    # Get the indeces that match the selection
    idx = np.where(selection)[0]

    # Sort the subset by distance
    idx2 = np.argsort(stations.distances[idx])

    # Get final indeces
    pos = idx[idx2]

    # Return reindexed subset
    return ds.subset(stations=pos, components=component)

#%%

components = ['Z', 'R', 'T']
phases = ['P', 'S', 'Rayleigh', 'Love', 'body', 'Ptrain', 'Strain']
offset_window_dict = {
    'P': (60, 60),  # Window is from P to S
    'S': (60, 60),   # Window is from S to max v Love waves
    'Ptrain': (10, -120),  # Window is from P to S
    'Strain': (60, 60),   # Window is from S to max v Love waves
    'Rayleigh': (60, 60), # Window is max to min v Rayleigh waves
    'Love': (60, 60), # Window is max to min v Love waves
    'body': (60, 60), # Window is from P to max v Love waves
}

dy = 0.05
phase = 'body'
component = 'Z'

specific_phase = None #'body' # Set to None to plot all phases

# outdir = os.path.join(scardec_stf_dir, 'plots', 'azimuthal')
outdir = os.path.join(scardec_stf_dir, 'plots', 'azimuthal')

if not os.path.exists(outdir):
    os.makedirs(outdir)

for phase in phases:
    for component in components:

        if specific_phase:
            if phase != specific_phase:
                continue

        # Subselect the seignals based on distance phases etc.
        ds_tt = select_traveltime_subset(
            ds_syn, component=component, phase=phase, maxdist=145.0)

        # Bin the signals into azimuthal bins that are good for plotting
        ds_tt, theta_bin = osl.utils.azi_plot_bin(ds_tt, dy=dy)

        # Get the indeces of the stations in the original array, and the component
        indeces = [ds_obs.index(station) for station in ds_tt.meta.stations.codes]
        icx = ds_obs.meta.components.index(component)

        ds_obs_tt = ds_obs.subset(stations=indeces, components=component)
        osl.utils.azi_plot([ds_obs_tt, ds_tt],
                           theta_bin,
                           dy=dy, hspace=0.0,
                           phase=phase,
                           tshift=tshift,
                           window = 200.0,
                           pre_window=offset_window_dict[phase][0],
                           post_window=offset_window_dict[phase][1],
                           bottomspines=False)

        plt.savefig(os.path.join(outdir, f'{phase}_{component}.pdf'), dpi=300)
# %%
# construct_taper function

from scipy import signal

def construct_taper(npts, taper_type="tukey", alpha=0.2):
    """
    Construct taper based on npts

    :param npts: the number of points
    :param taper_type:
    :param alpha: taper width
    :return:
    """
    taper_type = taper_type.lower()
    _options = ['hann', 'boxcar', 'tukey', 'hamming']
    if taper_type not in _options:
        raise ValueError("taper type option: %s" % taper_type)
    if taper_type == "hann":
        taper = signal.windows.hann(npts)
    elif taper_type == "boxcar":
        taper = signal.windows.boxcar(npts)
    elif taper_type == "hamming":
        taper = signal.windows.hamming(npts)
    elif taper_type == "tukey":
        taper = signal.windows.tukey(npts, alpha=alpha)
    else:
        raise ValueError("Taper type not supported: %s" % taper_type)
    return taper

# %%
# Now given the selected traces we want to use the corresponding windows to taper
# the traces such that we can perform the inversion only on the relevant windows
# and not the whole trace.

phase = 'body'
component = 'Z'

# Subselect the seignals based on distance phases etc.
ds_gfs_tt = select_traveltime_subset(ds_gfs, component=component, phase=phase, maxdist=145.0)

# Get the corresponding observed traces
ds_gfs_tt, ds_obs_tt = ds_gfs_tt.intersect(ds_obs)
_, ds_syn_tt = ds_gfs_tt.intersect(ds_syn)

# Reomve components from observed arrays
ds_obs_tt = ds_obs_tt.subset(components=component)
ds_syn_tt = ds_syn_tt.subset(components=component)

#%%

plot_check_section([ds_obs_tt, ds_syn_tt], labels=['Observed', 'Green'],
                   outfile='atestsection_traveltime_select.pdf')



#%%
def tt_taper_dataset(ds, phase, tshift, taper_perc=0.5):

    outds = ds.copy()
    # Get sampling interval
    delta = outds.meta.delta
    npts = outds.meta.npts
    length_in_s = npts * delta
    t = np.arange(0,npts) * delta - tshift

    # Get the corresponding windows
    start_arrival, end_arrival = osl.utils.get_windows(outds.meta.stations.attributes.arrivals, phase=phase)

    # Get the taper based on the window
    outds.data = np.zeros_like(ds.data)

    for _i, (_start, _end) in enumerate(zip(start_arrival, end_arrival)):
        print(_end - _start)
        idx = np.where((_start <= t) & (t <= _end))[0]
        npts = len(idx)
        outds.data[_i, 0, idx] = ds.data[_i, 0, idx] * construct_taper(npts, taper_type="tukey", alpha=taper_perc)

    return outds

tds_obs_tt = tt_taper_dataset(ds_obs_tt, phase, tshift)
tds_syn_tt = tt_taper_dataset(ds_syn_tt, phase, tshift)
tds_gfs_tt = tt_taper_dataset(ds_gfs_tt, phase, tshift)

# %%

plot_check_section([tds_obs_tt, tds_syn_tt], labels=['Original', 'Tapered'],
                   scale=5.0, outfile='atestsection_taper_syn.pdf')

plot_check_section([tds_obs_tt, tds_gfs_tt], labels=['Original', 'Tapered'],
                   scale=5.0, outfile='atestsection_taper_gfs.pdf')


#%%

import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt


class STF(object):
    """Actual inversion class."""

    def __init__(self, observed, green, t, dt, critical,
                 minT: float = 0, maxT: float = 50,
                 lamb=None, type="2", maxiter: int = 100,
                 taper_type: str = "tukey"):
        """
        :param obs: observed traces
        :param G: Green's functions
        :param maxT: time after which STF is forced to zero
        :param crit: critical value for stopping the iteration
        :param dt: time sampling
        :param lamb: waterlevel for deconvolution if type 2 is chosen. Unused if
                     type is "1".
        :param type: string defining the type of landweber method. Type 1 is the
                     method using the steepest decent; type 2 is using a Newton
                     step.
        :return:
        """

        # Get data
        self.obs = observed
        self.green = green
        # self.windows = windows # Window measurements and weighting data
        self.dt = dt
        self.t = t

        # Get parameters
        self.minT = minT
        self.maxT = maxT
        self.critical = critical
        self.lamb = lamb
        self.perc = 0.05  # Newton
        self.maxiter = maxiter
        self.taper_type = taper_type
        self.type = type

        # Deconvolution method
        if self.type == "1":
            self.compute_gradient = self.compute_gradient_sd
        elif self.type == "2":
            self.compute_gradient = self.compute_gradient_newton
        else:
            raise ValueError('Type must be "1" or "2"')

        # Get informations about size and initialize src
        self.nr, self.nt = self.obs.shape
        self.src = np.zeros(self.nt)


        # Compute objective function and residual
        self.orig_syn = self.forward(self.src)
        self.syn = self.forward(self.src)
        self.res = self.residual()
        self.chi = self.misfit()
        self.chi0 = self.chi
        self.it = 1

        # Lists to be filled:
        self.src_list = []
        self.chi_list = []

    def landweber(self):
        """Perform Landweber iterative source time function inversion."""

        # Compute the first gradient
        grad, alpha = self.compute_gradient()

        # Create source time function windowing taper
        pos = np.where(np.logical_and((self.minT <= self.t),
                                      (self.t <= self.maxT)))[0]
        Nt = len(pos)
        taperright = construct_taper(len(pos), taper_type=self.taper_type,
                                alpha=0.05)
        taperleft = construct_taper(len(pos), taper_type=self.taper_type,
                                alpha=0.01)

        # Make taper one sided
        Ptime = np.zeros(self.nt)
        Ptime[pos[:-Nt//2]] = taperleft[:-Nt//2]
        Ptime[pos[-Nt//2:]] = taperright[-Nt//2:]

        # Perform iterative deconvolution (inverse problem)
        self.chip = self.chi0

        while self.chi > self.critical * self.chi0 and self.it <= self.maxiter:

            # Regularized gradient
            gradreg = grad

            if type == "1":
                srctmp = self.src + gradreg
            else:
                srctmp = self.src + self.perc * gradreg

            # Window source time function --> zero after some time T
            srctmp = srctmp * Ptime

            # Enforce positivity
            srctmp[np.where(srctmp < 0)[0]] = 0

            # Compute misfit function and gradient
            self.syn = self.forward(srctmp)
            self.res = self.residual()
            self.chi = self.misfit()
            grad, _ = self.compute_gradient()
            self.it = self.it + 1

            # Check convergence
            if self.chi > self.chip:
                print("NOT CONVERGING")
                break

            if abs(self.chi - self.chip)/self.chi < 10**(-5):
                print("CONVERGING TOO LITTLE")
                break

            # Update
            # chi / chi0
            self.chip = self.chi
            self.src = srctmp

            self.chi_list.append(self.chi)
            self.src_list.append(self.src)

        # Final misfit function
        print("Iteration: %d -- Misfit: %1.5f" % (self.it,
                                                        self.chi / self.chi0))

    def residual(self):
        """Computes the residual between the observed data
        and the synthetic data."""

        return self.obs - self.syn

    def misfit(self):
        """Computes the misfit between the observed data and
        the forward modeled synthetic data."""

        return 0.5 * np.sum((self.obs - self.syn) ** 2)

    def forward(self, src):
        """ Convolution of set of Green's functions

        :param green: Green's function
        :param src:
        :return:
        """
        # Get frequency spectrum of source time function
        SRC = fft(src)

        # Get frequency spectra of Green's functions
        GRE = fft(self.green, axis=1)

        # Convolve the two and return matrix containing the synthetic
        syn = np.real(ifft(GRE * SRC, axis=1)) * self.dt

        return syn

    def compute_gradient_newton(self):
        """ Compute Gradient using the waterlevel deconvolution which computes
        the Newton Step.

        :param resid: residual
        :param green: green's function
        :param lamb: waterlevel scaling
        :return:
        """

        # FFT of residuals and green functions
        RES = fft(self.res, axis=1)
        GRE = fft(self.green, axis=1)

        # Compute gradient (full wavelet estimation)
        num = np.sum(RES * np.conj(GRE), axis=0)
        den = np.sum(GRE * np.conj(GRE), axis=0)

        # Waterlevel
        wl = self.lamb * np.max(np.abs(den))
        pos = np.where(den < wl)
        den[pos] = wl
        grad = np.real(ifft(num / (den)))


        # Step value
        hmax = 1

        return grad, hmax

    def compute_gradient_sd(self):
        """ Compute the Gradient using the steepest decent method
        :param resid:
        :param green:
        :return:
        """

        # FFT of residuals and green functions
        RES = fft(self.res, axis=1)
        GRE = fft(self.green, axis=1)

        # Compute gradient (full wavelet estimation)
        num = np.sum(RES * np.conj(GRE), axis=0)
        den = np.sum(GRE * np.conj(GRE), axis=0)

        mod = np.abs(den)
        cond = np.max(mod)/np.min(mod)

        # print(np.max(mod), factor)

        # Relaxation parameter
        # The factor is chosen to be close to the maximum value of the denominator
        # but slightly smaller so that the gradient converges a little bit faster
        factor = np.max(mod) * 10**(-np.log10(cond)/20)
        tau = 1 / factor
        grad = tau * np.real(ifft(num))

        # Step value
        hmax = 1

        return grad, hmax



# %%


t = np.arange(0, length_in_s, 1/sps)
teststf = osl.STF.gaussian(origin=obspy.UTCDateTime(0), t=t, tshift=tshift, hdur=75, tc=100.0)

# Fake data
testdata = tds_gfs_tt.copy()
testdata.convolve(teststf.f, tshift)

stfinv1 = STF(
    np.roll(tds_obs_tt.data[:, 0, :], int((tshift= + 0.9) * sps), axis=-1),
    tds_gfs_tt.data[:, 0, :], t, 1/sps, .001,
    minT=tshift, maxT=tshift+120, lamb=1e-3, type="1", maxiter=400,
    taper_type="tukey")

# stfinv2 = STF(
#     np.roll(tds_obs_tt.data[:, 0, :], int(tshift * sps), axis=-1),
#     tds_gfs_tt.data[:, 0, :], t, 1/sps, .001,
#     minT=tshift, maxT=tshift+120, lamb=1e-3, type="2", maxiter=400,
#     taper_type="tukey")

# stfinv = STF(
#     tds_obs_tt.data[:, 0, :], tds_gfs_tt.data[:, 0, :],
#     t, 1/sps, 1e-3, minT=-10, maxT=120, lamb=1e-3, type="1",
#     maxiter=50, taper_type="tukey")

stfinv1.landweber()
# stfinv2.landweber()


# %%
t = np.arange(0, len(stfinv1.src) * 1 / sps , 1/sps)
gcmt = CMTSOLUTION.read(cmt_file)
cmt_stf = osl.STF.gaussian(origin=gcmt.origin_time, t=t, tshift=tshift, hdur=gcmt.hdur, tc=gcmt.time_shift)


norm = 1 # np.trapz(stfinv.src, t)
print(norm)
plt.figure(figsize=(4.5,2))
plt.plot(cmt_stf.t-tshift, cmt_stf.f, 'tab:red', label='GCMT')
plt.plot(stf.t-tshift, stf.f/stf.M0, 'tab:blue', label='SCARDEC')
plt.plot(stfinv1.t-tshift, stfinv1.src, 'tab:orange', label='GLAD-M35')
plt.xlim(-5, +120)
plt.xlabel('Time [s]')
plt.ylabel('Norm. Moment Rate')
plt.legend(frameon=False, fontsize='small', loc='upper left')
plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.25)
plt.savefig('stfinv.pdf', dpi=300)


# %%
# Convolve source wavelet with Green's function
syn_out = tds_gfs_tt.copy()
syn_out.convolve(stfinv1.src/norm, tshift)

syn_cmt = tds_gfs_tt.copy()
syn_cmt.convolve(cmt_stf.f, tshift)

plot_check_section([tds_obs_tt, syn_cmt, tds_syn_tt,  syn_out], labels=['Observed','CMT', 'SCARDEC', 'Inverted'],
                   outfile='atestsection_inversion.pdf')
plt.legend()

# %%
tds_gfs_tt.data[:, 0, :]
plt.figure(figsize=(10,10))
t = np.arange(-tshift/60, length_in_s/60, 1/sps/60)
scale = 10.0 * 1/np.max(np.abs(obs_array))

x = np.arange(gft.shape[0])
scale = 10.0 * 1/np.max(np.abs(obsd))
# scale = 10.0 * 1/np.max(np.abs(synt2))
plt.plot(t, scale * obsd.T * tapers.T + x, 'k', lw=0.25)
plt.plot(t, scale * synt.T * tapers.T + x, 'r', lw=0.25)
plt.plot(t, scale * synt2.T * tapers.T + x, 'b', lw=0.25)
plt.yticks(x, [f"{gft_meta['distances'][i]:.2f}" for i in range(len(x))],
           rotation=0)
plt.savefig('atest2.pdf', dpi=300)




# %%

# %%


def deconvolution(obs, green, lambd):
    """ Takes in the observed data and the green's function to obtain the
    source wavelet estimate.

    :param obs:
    :param green:
    :return:
    """

    nr, nt = obs.shape
    num = np.zeros(nt)
    den = np.zeros(nt)

    NP2 = gf3d.utils.next_power_of_2(nt)

    OBS = fft.fft(obs, axis=-1, n=NP2)
    GRE = fft.fft(green, axis=-1, n=NP2)

    num = np.sum(np.conj(GRE) * OBS, axis=0)
    den = np.sum(np.conj(GRE) * GRE, axis=0)

    # Get maximum value of denominator
    maxden = np.max(np.abs(den))

    # Waterlevel
    wl = lambd * maxden

    # Deconvolution using the waterlevel
    src = np.real(ifft(num / (den+wl).T))[:nt]

    # Compute fit to original data
    res = obs
    chi0 = 0.5 * np.sum(np.sum(res ** 2))

    syn = np.real(fft.ifft(fft.fft(green, axis=1) * fft.fft(src), axis=1))[:nt]
    res = obs - syn
    chi = 0.5 * np.sum(np.sum(res ** 2))

    return src, syn



# %%

plt.figure(figsize=(10,10))
plt.plot(stfinv.src)
plt.xlim((tshift-50)*sps, (tshift+100)*sps)
plt.savefig('stfinv.pdf', dpi=300)
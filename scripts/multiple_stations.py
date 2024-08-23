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



# %%
# Files
from gf3d.source import CMTSOLUTION
scardec_stf_dir = './STF/FCTs_20180123_093140_GULF_OF_ALASKA/'
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


# %%
# Read the Green functions
from gf3d.seismograms import GFManager
gfm = GFManager('/lustre/orion/geo111/scratch/lsawade/STF_SUBSETS/FCTs_20180123_093140_GULF_OF_ALASKA/subset.h5')
gfm.load()


# %%
# Get Seismograms
gf_array, meta = gfm.get_seismograms(cmt=cmt, raw=True, array=True)

# %%
# Download the corresponding data

# Remove data directory if it exists
if os.path.exists(os.path.join(scardec_stf_dir, 'data')):
    shutil.rmtree(os.path.join(scardec_stf_dir, 'data'))

networks = ",".join({_sta.split('.')[0] for _sta in meta['stations']})
stations = ",".join({_sta.split('.')[1] for _sta in meta['stations']})

opl.download_data(
    os.path.join(scardec_stf_dir, 'data'),
    starttime=cmt.origin_time - 300,
    endtime=cmt.origin_time + 3600 + 300,
    network=networks, station=stations,
    channel_priorities=['BH[ZNE12]', 'HH[ZNE12]', 'LH[ZNE12]'],
    location_priorities=['00', '10', '01']
    )

# %%
# Read the data
raw = obspy.read(os.path.join(scardec_stf_dir, 'data', 'waveforms/*.mseed'))
inv = obspy.read_inventory(os.path.join(scardec_stf_dir, 'data', 'stations/*.xml'))

# %%
# Process the reciprocal green function to first filter the resample
def preprocess(st, starttime, length_in_s, sps, tshift=200.0, step=False,
            inv: obspy.Inventory | None = None):

    # Taper the seismograms
    st.taper(max_percentage=0.05, type='cosine')

    # Remove response if inventory is given given
    if inv:
        st.detrend('linear')
        st.detrend('demean')

        st.remove_response(inventory=inv, output='DISP',
                           pre_filt=(0.002, 0.003, 1.0, 1/0.5),
                           water_level=100)
        st.rotate('->ZNE', inventory=inv)

    st.filter('bandpass', freqmin=0.004, freqmax=1/1.0, corners=3, zerophase=True)
    st.interpolate(sampling_rate=sps, starttime=starttime,
                   npts=int((length_in_s) * sps))

    # Filter out station with less than 3 components total
    stations =set()
    for tr in st:
        station = f"{tr.stats.network}.{tr.stats.station}"
        stations.add(station)

    for station in stations:

        _net, _sta = station.split('.')
        substream = st.select(network=_net, station=_sta)

        if len(substream) < 3:
            for _subtr in substream:
                st.remove(_subtr)



tshift = 200.0
starttime = cmt.origin_time - tshift
sps = 20.0
length_in_s = 3600 + 2 * tshift

praw = raw.copy()
preprocess(praw, starttime, length_in_s, sps, tshift, inv=inv)

# %%
# Make array from stream

def stream_to_array(st: obspy.Stream, components = ['N', 'E', 'Z'], inv: obspy.Inventory | None = None):
    stations =set()
    for tr in st:
        station = f"{tr.stats.network}.{tr.stats.station}"
        stations.add(station)

    # Array size
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
    }

    for i, station in enumerate(stations):
        _net, _sta = station.split('.')
        substream = st.select(network=_net, station=_sta)

        if inv:
            _inv = inv.select(network=_net, station=_sta)
            meta['latitudes'].append(_inv[0][0].latitude)
            meta['longitudes'].append(_inv[0][0].longitude)

        for j, component in enumerate(components):
            subtr = substream.select(component=component)
            if len(subtr) > 0:
                array[i, j, :] = subtr[0].data

    return array, meta


obs = raw.copy().select(network='II', station='BFO', component='Z')



obs_array, obs_meta = stream_to_array(praw, inv=inv)


# %%

# Check overlapping stations
istations = list(set(obs_meta['stations']).intersection(set(meta['stations'])))

def isolate_stations(array, meta, stations):
    idx = [meta['stations'].index(station) for station in stations]

    outmeta = deepcopy(meta)
    outmeta["latitudes"] = [meta["latitudes"][i] for i in idx]
    outmeta["longitudes"] = [meta["longitudes"][i] for i in idx]
    outmeta["stations"] = stations

    return array[idx], outmeta

obs_array, obs_meta = isolate_stations(obs_array, obs_meta, istations)
gf_array, gf_meta = isolate_stations(gf_array, meta, istations)

# %%
# Process array of synthetic data

def process_homemade(gf_array, meta, starttime, length_in_s, sps):

    # Taper the seismograms
    npts = gf_array.shape[-1]
    idata = gf_array.copy()
    idata = osl.taper(idata, max_percentage=0.05, sampling_rate=1/meta['delta'], type='cosine')

    # Filter the seismograms

    idata = osl.filter.bandpass(idata, freqmin=0.005, freqmax=1/17.0,
                                df=1/meta['delta'], corners=3, zerophase=True)
    print(1/meta['delta'])

    # Resample the seismograms
    npts = int((length_in_s) * sps)
    idata = osl.interpolate.interp1d(idata, old_dt=meta['delta'], old_start=meta['starttime'],
                             new_start=starttime, new_dt=1/sps, new_npts=npts,
                             itype='weighted_average_slopes', axis=-1)

    new_meta = deepcopy(meta)
    new_meta['starttime'] = starttime
    new_meta['delta'] = 1/sps
    new_meta['npts'] = int((length_in_s) * sps)

    return idata, new_meta


tshift = 150.0
starttime = cmt.origin_time - tshift
sps = 20.0
length_in_s = 3600 + tshift


pobs, pobs_meta = process_homemade(obs_array, obs_meta, starttime, length_in_s, sps)
p_gf, p_gf_meta = process_homemade(gf_array, gf_meta, starttime, length_in_s, sps)


# Convolve with error function to integrate
error = osl.STF.error(origin=cmt.origin_time,
                      t=np.arange(0, length_in_s, 1/sps),
                      tshift=tshift, hdur=1e-6, tc=0.0)
p_gf = osl.convolve(p_gf, error.f, 1/sps, tshift)
p_gf = osl.filter.bandpass(p_gf, freqmin=0.005, freqmax=1/17.0,
                                df=1/p_gf_meta['delta'], corners=3, zerophase=True)

# %%

# Convolve with STF
tshift = 150.0
starttime = cmt.origin_time - tshift
sps = 20.0
length_in_s = 3600 + tshift

t = np.arange(-tshift, length_in_s, 1/sps)
stf = deepcopy(scardec)
stf.interp(t)

plt.figure(figsize=(10,2))
stf.plot()
plt.savefig('astf.pdf', dpi=300)
# %%

# psyn = osl.convolve(p_gf, scardec.f/(scardec.M0), 1/sps, 0)
# psyn = osl.filter.lowpass(psyn, 1/17.0, sps, corners=3, zerophase=True)
psyn_meta = deepcopy(p_gf_meta)

# %%
# plt.figure(figsize=(10,10))
# t = np.arange(0, length_in_s, 1/sps)
# iBFO = p_gf_meta['stations'].index('II.BFO')

# # plt.plot(t, pobs[:,2,:].T + 0.075 * np.arange(pobs.shape[0])[None, :] * np.max(np.abs(pobs)), 'k', lw=0.25)
# plt.plot(t, p_gf[iBFO,2,:], 'r', lw=0.25)
# plt.plot(t, psyn[iBFO,2,:], 'b', lw=0.25)
# plt.savefig('atest.pdf', dpi=300)


# %%
# plt.figure(figsize=(10,10))
# t = np.arange(0, length_in_s, 1/sps)
# # plt.plot(t, pobs[:,2,:].T + 0.075 * np.arange(pobs.shape[0])[None, :] * np.max(np.abs(pobs)), 'k', lw=0.25)
# plt.plot(t, p_gf[:,2,:].T + 0.5 * np.arange(pobs.shape[0])[None, :] * np.max(np.abs(psyn)), 'r', lw=0.25)
# plt.plot(t, psyn[:,2,:].T + 0.5 * np.arange(pobs.shape[0])[None, :] * np.max(np.abs(psyn)), 'b', lw=0.25)
# plt.savefig('atest.pdf', dpi=300)



# %%

syns, smeta = osl.utils.get_dist_azi_waveforms(
    psyn, psyn_meta, cmt, phase='P', component='Z', min_dist=30.0)

dy = 0.025
syns, smeta, theta_bin = osl.utils.azi_plot_bin(syns, smeta, dy=dy)

indeces = [obs_meta['stations'].index(station) for station in smeta['stations']]
icx = obs_meta['components'].index('Z')
osl.utils.azi_plot([pobs[indeces, icx, :], syns], smeta, theta_bin, dy=dy)


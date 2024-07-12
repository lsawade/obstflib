# %%
import os
import obspy
import gf3d.utils
import numpy as np
from copy import deepcopy
from scipy import fft
import shutil
import obstflib as osl
from gf3d.source import CMTSOLUTION
from gf3d.seismograms import GFManager
from obspy import Stream
import obsnumpy as onp


# %%
# Get the event record at II BFO for the Alaska event using the GFManager
scardec_stf_dir = './STF/FCTs_20180123_093140_GULF_OF_ALASKA'
scardec_stf_dir = './STF/FCTs_20070815_234057_NEAR_COAST_OF_PERU'
scardec_id = scardec_stf_dir.split('/')[-1]
cmt_file = os.path.join(scardec_stf_dir, 'CMTSOLUTION')

# %%
# Load Scardec STF
scardec = osl.STF.scardecdir(scardec_stf_dir, 'optimal')

# Read CMTSOLUTION file
cmt = CMTSOLUTION.read(cmt_file)

# %%
# Read the Green functions
from gf3d.seismograms import GFManager
gfm = GFManager(f'/lustre/orion/geo111/scratch/lsawade/STF_SUBSETS/{scardec_id}/subset.h5')
gfm.load()

# %%
# Get the Green functions as both obspy stream
network = 'II'
station = 'BFO'
component = 'Z'
gf = gfm.get_seismograms(cmt, raw=True).select(component=component, network=network, station=station)

# %%
# Get the Green functions as numpy array
gf_array_base, meta_base = gfm.get_seismograms(cmt, array=True, raw=True)

# %%

def to_ds(array: np.ndarray, meta: dict) -> onp.Dataset:

    new_meta = onp.Meta.from_dict(meta)
    return onp.Dataset(data=array, meta=new_meta)

gf_ds_base = to_ds(gf_array_base, meta_base)

#%%

# Get trace index for subsequent comparison
idx_station_base = gf_ds_base.meta.stations.codes.index(f'{network}.{station}')
idx_component_base = gf_ds_base.meta.components.index(component)

station_slice = np.arange(idx_station_base, idx_station_base+1)
component_slice = np.arange(idx_component_base, idx_component_base+1)

station_slice = np.arange(0, len(gf_ds_base))
component_slice = np.arange(0,3)


gf_ds = gf_ds_base.subset(station_slice)

# Get the station component on the subset
idx_station = gf_ds.meta.stations.codes.index(f'{network}.{station}')
idx_component = gf_ds.meta.components.index(component)


# %%

def convolve(trace: np.ndarray, stf: np.ndarray, dt: float, tshift: float) -> np.ndarray:

    # For following FFTs
    N = len(trace)
    NP2 = gf3d.utils.next_power_of_2(2 * N)

    # Fourier Transform the STF
    TRACE = fft.fft(trace, n=NP2)
    STF = fft.fft(stf, n=NP2)


    #
    print("Convolve func")
    print(TRACE.shape)
    print(STF.shape)


    # Compute correctional phase shift
    shift = -tshift
    phshift = np.exp(-1.0j * shift * np.fft.fftfreq(NP2, dt) * 2 * np.pi)


    print(phshift.shape)

    # Return the convolution
    conv = TRACE * STF * phshift
    print("conv", conv.shape)
    return np.real(fft.ifft(conv))[:N] * dt



# Process the reciprocal green function to first filter the resample
def process(st, starttime, length_in_s, sps, step=True):

    # Taper the seismograms
    st.taper(max_percentage=0.05, type='cosine')
    st.filter('bandpass', freqmin=0.004, freqmax=1/17.0, corners=3,  zerophase=True)
    npts = int((length_in_s) * sps)
    st.interpolate(sampling_rate=sps, starttime=starttime, npts=npts,method='weighted_average_slopes')

    if step:
        # Convolve the seismograms with the source time functions
        for tr in st:

            t = tr.times()
            print("Time vec", np.min(t), np.max(t), t[1]-t[0], t.shape)
            error = osl.STF.error(origin=obspy.UTCDateTime(0), t=tr.times(), tshift=150, hdur=1e-6, tc=0.0)
            tr.data = convolve(tr.data, error.f, tr.stats.delta, tshift)

def process_homemade(ds, starttime, length_in_s, sps, step=True):

    # Get timing
    npts = int((length_in_s) * sps)

    # Taper the seismograms
    ds.taper(max_percentage=0.05, type='cosine')
    ds.filter('bandpass', freqmin=0.004, freqmax=1/17.0, corners=3,  zerophase=True)
    ds.interpolate(sampling_rate=sps, starttime=starttime, npts=npts, method='weighted_average_slopes')

    if step:
        t = np.arange(0, length_in_s, 1/sps)
        print("Time vec", np.min(t), np.max(t), t[1]-t[0], t.shape)
        error = osl.STF.error(origin=obspy.UTCDateTime(0),
                              t=t,
                              tshift=150.0, hdur=1e-6, tc=0.0)

        ds.convolve(error.f, tshift=150)



tshift = 150.0
starttime = cmt.origin_time - tshift
sps = 1.0
length_in_s = 3600 + tshift
step=True


st = gf.copy()
ds = gf_ds.copy()

process(st, starttime, length_in_s, sps, step=step)
process_homemade(ds, starttime, length_in_s, sps, step=step)



#%%

import numpy as np
import matplotlib.pyplot as plt



plt.figure()
lw = 0.5
plt.subplot(211)
t = np.arange(0, length_in_s, 1/sps)
plt.plot(gf[0].times(), gf[0].data, label='Obspy', lw=lw)
plt.plot(gf[0].times(), gf_ds.data[idx_station, idx_component, :], ":", label='Homemade',    lw=lw)
plt.xlim(0, 10800)
plt.legend(frameon=False)
plt.subplot(212)
plt.plot(st[0].times(), st[0].data, label='Obspy', lw=lw)
plt.plot(st[0].times(), ds.data[idx_station, idx_component, :], ':', label='Homemade', lw=lw)
plt.xlim(0, 10800)
plt.legend(frameon=False)
plt.savefig('atest.png', dpi=300)


# %%


# %%
# Compute asimuth and distance for all stations from the meta dictionary and the cmt location
from obspy.geodetics import gps2dist_azimuth

# Get the cmt location
cmt_lat = cmt.latitude
cmt_lon = cmt.longitude

slat = meta['latitudes']
slon = meta['longitudes']

distances = []
azimuths = []
for lat, lon in zip(slat, slon):
    dist, az, baz = gps2dist_azimuth(cmt_lat, cmt_lon, lat, lon)
    distances.append(dist)
    azimuths.append(az)

# Make numpy arrays
distances = np.array(distances)/111.11/1000
azimuths = np.array(azimuths)

# %%
# Now compute arrival times for all stations
from obspy.taup import TauPyModel

# Make model
model = TauPyModel(model='ak135')

# Get the arrivals
parrivals = []
for dist, az in zip(distances, azimuths):
    phase_list = ['P']
    arrivals = model.get_travel_times(source_depth_in_km=cmt.depth,
                                      distance_in_degree=dist,
                                      phase_list=phase_list)
    if len(arrivals) > 0 and dist > 30.0:
        parrivals.append(arrivals[0].time)
    else:
        parrivals.append(np.nan)

# %%
# Get stations where we have arrivals
parrivals = np.array(parrivals)
good = ~np.isnan(parrivals)
icx = pmeta['components'].index('Z')

# New signals
signals = data[good, icx, :]

# New azimuths
rad_azimuths = azimuths[good] * np.pi / 180

# %%


# We generated the angles. Now we want to bin the angles into 5 degree bins
# and select one angle from each bin but ignore bins that do not contain angles
def bin_angles(angles, dy: float = 0.1):
    import numpy as np

    y = np.clip(np.arange(1, -1 - dy, -dy), -1, 1)

    theta = np.arccos(y)
    x = np.sin(theta)

    # Make sure that we wave bins for both sides, positive and negative x
    x = np.concatenate([x[:-1], -x[::-1]])
    y = np.concatenate([y[:-1], y[::-1]])
    theta = np.concatenate([theta[:-1], np.pi + theta])

    # Initialize the new angles, and indeces
    new_angles = []
    new_indeces = []

    # Loop over the bins
    for i in range(len(theta) - 1):

        # Get the angles in the bin
        bin_angles = angles[(angles >= theta[i]) & (angles < theta[i + 1])]

        if len(bin_angles) == 0:
            continue

        # for later I also need the indices
        indices = np.where((angles >= theta[i]) & (angles < theta[i + 1]))

        # Choose the angle that is the closest to the center of the bin
        center = (theta[i] + theta[i + 1]) / 2
        index = np.argmin(np.abs(bin_angles - center))

        # append
        new_angles.append(bin_angles[index])
        new_indeces.append(indices[0][index])

    return (theta, x, y), (new_angles, new_indeces)

dy = 0.1
(theta_bin, x_bin, y_bin), (new_angles, new_indeces) = bin_angles(rad_azimuths, dy=dy)

# Get bin center for each angle
bin_center = (theta_bin[:-1] + theta_bin[1:]) / 2

new_signals = signals[new_indeces, :]
new_arrivals = parrivals[good][new_indeces]

slices = []
t = np.arange(0, length_in_s, 1/sps)
offset = int(60.0 * sps)
window = int(250 * sps)
for i in range(len(new_arrivals)):
    idx = int(np.argmin(np.abs(t - ( new_arrivals[i] + tshift))))
    slices.append(slice(idx-offset, idx+window-offset))

# %%

# Now I want to create a figure that creates one axes for each signal, and
# and plots the signal in the axes. But the axes should be arranged with respect to azimuth

import obsplotlib.plot as opl

# Make text monospapced
plt.rcParams["font.family"] = "monospace"

plt.figure(figsize=(11, 10))
mainax = plt.gca()
mainax.axis("off")

for i in range(len(new_angles)):

    # Height ration

    # Height of the axes as a function of stretch
    stretch_height = 2.0
    r = 0.5 / stretch_height
    height = 0.5 * dy

    # Distance of the axes from the center
    # r = 0.2
    total_width = 0.5 - r
    percentage_offset = 0.1
    width = total_width * (1 - percentage_offset)
    width_offset = total_width * percentage_offset

    # Height of the axes as a function of stretch
    # stretch_height = 2.0
    # height = stretch_height * dy * r

    # Angle of the axes
    az_true = new_angles[i]

    # Get bin index
    az_bin = bin_center[np.digitize(az_true, theta_bin) - 1]

    # Use azimuth to get x,y. Here azimuth is with respect to North and
    # clockkwise
    x = r * np.sin(az_bin) + 0.5
    y = stretch_height * r * np.cos(az_bin) + 0.5

    # plot bin edges
    for _i in theta_bin[:-1]:
        mainax.plot(
            [0.5, 0.5 + r * np.sin(_i)],
            [0.5, 0.5 + stretch_height * r * np.cos(_i)],
            c="lightgray",
            lw=0.1,
            ls="-",
            clip_on=False,
        )

    # If the azimuth is larger than pi the axis is on the left side
    axis_left = az_bin >= np.pi
    ylabel_location = "right" if axis_left else "left"
    yrotation = 0 if axis_left else 0

    # ADjust the left axes to match the width
    if axis_left:
        x = x - width - width_offset
    else:
        x = x + width_offset

    # Create extent
    extent = [x, y - height / 2, width, height]

    # Create axes
    ax = opl.axes_from_axes(
        mainax,
        9809834 + i,
        extent=extent,
    )
    # Remove all ticks and labels
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )
    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # ax.spines['left'].set_visible(False)

    # Remove axis background
    ax.patch.set_visible(False)

    # Set the ylabel


    ax.yaxis.set_label_position(ylabel_location)
    ax.plot(new_signals[i, slices[i]])
    ax.set_ylabel(
        f"{az_true*180/np.pi:.0f}",
        rotation=yrotation,
        horizontalalignment="center",
        verticalalignment="center",
        labelpad=10,
        fontsize="small",
    )
    mainax.scatter(
        x,
        y,
        s=5,
        c=az_true,
        marker="o",
        cmap="viridis",
        vmin=0,
        vmax=2 * np.pi,
        clip_on=False,
        zorder=20,
    )
    mainax.set_xlim(0, 1)
    mainax.set_ylim(0, 1)

# mainax.set_aspect("equal")
plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)


plt.savefig('source_plot.pdf', dpi=300)





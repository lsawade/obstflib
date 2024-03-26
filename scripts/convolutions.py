# %%
# convolution testing
import os
import obspy
import numpy as np
import matplotlib.pyplot as plt
import obsplotlib.plot as opl
from scipy import special
import gf3d.utils
from copy import deepcopy
from scipy import fft
import shutil

# %%
# Get Alaska STF
import obsplotlib.plot as opl

# Files
scardec_stf_dir = './STF/FCTs_20180123_093140_GULF_OF_ALASKA/'
cmt_file = os.path.join(scardec_stf_dir, 'CMTSOLUTION')

# Get Alaska STF
stf = opl.SCARDECSTF.fromdir(scardec_stf_dir, 'optimal')

# Get Alaska CMT solution
cmt = opl.CMTSOLUTION.read(cmt_file)

# This is the time vector of the STF
t = np.arange(-10, 120, 0.01)

# Interpolate the stf to the given time vector
scardec = interp_stf(t, stf.time, stf.moment_rate)

# Compute the other stfs
triangle = triangle_stf(t, cmt.cmt_time - stf.origin, cmt.hdur) * cmt.M0 / 1e7
square = square_stf(t, cmt.cmt_time - stf.origin, cmt.hdur) * cmt.M0 / 1e7
gaussian = gaussian_stf(t, cmt.cmt_time - stf.origin, cmt.hdur) * cmt.M0 / 1e7
error = step_stf(t, cmt.cmt_time - stf.origin, cmt.hdur) * cmt.M0 / 1e7


def plot_single_stf(ax, t, stf, title, *args, **kwargs):
    """
    Plots a single source time function.

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    stf : np.ndarray
        Source time function.
    title : str
        Title of the plot.
    """

    ax.plot(t, stf, *args, label=title, **kwargs)
    ax.set_xlim(np.min(t), np.max(t))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig, axes = plt.subplots(6,1, sharex=True, figsize=(8, 6))

# Plot scradec stf
plot_single_stf(axes[0], t, scardec, 'SCARDEC', color='k')
axes[0].legend(frameon=False)
axes[0].set_ylim(0, None)
plot_single_stf(axes[1], t, square, 'Square', color='k')
axes[1].legend(frameon=False)
axes[1].set_ylim(0, None)
plot_single_stf(axes[2], t, triangle, 'GCMT', color='k')
axes[2].legend(frameon=False)
axes[2].set_ylim(0, None)

plot_single_stf(axes[3], t, gaussian, 'SF3DG', color='k')
axes[3].legend(frameon=False)
axes[3].set_ylim(0, None)

plot_single_stf(axes[4], t, error, 'Error', color='k')
axes[4].legend(frameon=False)
axes[4].set_ylim(0, None)


plot_single_stf(axes[5], t, scardec, 'SCARDEC', color='tab:blue')
plot_single_stf(axes[5], t, square, 'Square', color='tab:green')
plot_single_stf(axes[5], t, triangle, 'GCMT', color='k')
plot_single_stf(axes[5], t, gaussian, 'SF3DG', color='tab:orange')
plot_single_stf(axes[5], t, np.gradient(error, t), 'dError/dt', color='tab:red',
                ls='--')
axes[5].legend(frameon=False)
axes[5].set_ylim(0, None)


plt.savefig('stf_wavelets.png', dpi=300)


# %%
# Get the event record at II BFO for the Alaska event using the GFManager
from gf3d.source import CMTSOLUTION
from gf3d.seismograms import GFManager

# Get GCMT solution and set the half duration to 0 for GF extraction
cmt_gcmt = CMTSOLUTION.read(cmt_file)
cmt_gf = CMTSOLUTION.read(cmt_file)
cmt_gf.hdur = 0
cmt_gf.time_shift = 0


# create CMT solution from scardec sdr and set half duration to 0
cmt_scardec = CMTSOLUTION.from_sdr(
    s=stf.strike1, d=stf.dip1, r=stf.rake1, M0=stf.M0 * 1e7,
    origin_time=stf.origin, latitude=stf.latitude, longitude=stf.longitude,
    depth=stf.depth_in_km)
cmt_scardec.hdur = 0

# %%
# Subset
gfm = GFManager('/lustre/orion/geo111/scratch/lsawade/STF_SUBSETS/FCTs_20180123_093140_GULF_OF_ALASKA/subset.h5')
gfm.load()

# %%
# Get Seismograms
network='II'
station='BFO'
gcmt_true = gfm.get_seismograms(cmt=cmt_gcmt).select(network=network, station=station, component='Z')
gf_gcmt_raw = gfm.get_seismograms(cmt=cmt_gf, raw=True).select(network=network, station=station, component='Z')
gf_scardec_raw = gfm.get_seismograms(cmt=cmt_scardec, raw=True).select(network=network, station=station, component='Z')


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


# Process the reciprocal green function to first filter the resample
def process(st, starttime, length_in_s, sps, tshift=200.0, step=False,
            inv: obspy.Inventory | None = None):

    # Taper the seismograms
    st.taper(max_percentage=0.05, type='cosine')

    # Remove response if inventory is given given
    if inv:
        st.detrend('linear')
        st.detrend('demean')

        st.remove_response(inventory=inv, output='DISP',
                           pre_filt=(0.001, 0.005, 1.0, 1/0.5),
                           water_level=100)

    st.filter('bandpass', freqmin=0.001, freqmax=1/17.0, corners=3, zerophase=True)
    st.interpolate(sampling_rate=sps, starttime=starttime-tshift, npts=int((length_in_s + tshift) * sps))

    if step:
        # Convolve the seismograms with the source time functions
        for tr in st:
            error = step_stf(tr.times(), tshift, 1e-6)
            tr.data = convolve(tr.data, error, tr.stats.delta, tshift)

st_gcmt_true = gcmt_true.copy()
gf_gcmt = gf_gcmt_raw.copy()
gf_scardec = gf_scardec_raw.copy()

tshift = 150.0
delta = 20.0
npts = 3600

process(st_gcmt_true, cmt_gcmt.origin_time, npts, delta, tshift=tshift, step=False)
process(gf_gcmt, cmt_gcmt.origin_time, npts, delta, tshift=tshift, step=True)
process(gf_scardec, cmt_gcmt.origin_time, npts, delta, tshift=tshift, step=True)


# Create source time functions for the seismograms
st_gcmt_triangle = deepcopy(gf_gcmt)
t = st_gcmt_triangle[0].times()
stf_triangle = triangle_stf(t, cmt_gcmt.time_shift + tshift, cmt_gcmt.hdur)

# Convolve the setup streams with the
st_gcmt_triangle[0].data = convolve(st_gcmt_triangle[0].data, stf_triangle, st_gcmt_triangle[0].stats.delta, tshift)

# %%
# Get P and S arrival for orientation

# Get stations info
idx = gfm.stations.index(station)
station_latitude = gfm.latitudes[idx]
station_longitude = gfm.longitudes[idx]

# Compute arrivals
arrivals = opl.get_arrivals(station_latitude, station_longitude, cmt_gcmt.latitude, cmt_gcmt.longitude, cmt_gcmt.depth, phase_list=['P', 'S'])

# %%
# Plot the raw, filtered and convolved seismograms

# Create subplots and figures
fig, axes = plt.subplots(3, 1, figsize=(8, 4.5))

# Plot the seismograms
ax_raw = axes[0]
ax_step = axes[1]
ax_tri = axes[2]

arrivaldict = dict(lw=0.5, alpha=0.5, color='tab:blue', zorder=-10)

# Add seismograms to the axes
opl.trace([gf_gcmt_raw[0]], ax=ax_raw, lw=0.75, alpha=1.0, plot_labels=False,
          labels=['Raw DB'], origin_time=cmt_gcmt.origin_time, limits=(0, 3600))
opl.plot_arrivals(arrivals, ax=ax_raw, scale=2, timescale=60,
                  **arrivaldict)
opl.plot_arrivals(arrivals, ax=ax_raw, scale=0.0005, timescale=60,
                  **arrivaldict)
ax_raw.spines['bottom'].set_visible(False)
ax_raw.set_xlabel('')
ax_raw.tick_params(axis='x', which='both', labelbottom=False, bottom=False)

opl.trace([gf_gcmt[0]], ax=ax_step, lw=0.75, alpha=1.0, plot_labels=False,
          labels=['Filt.&Step'], origin_time=cmt_gcmt.origin_time, limits=(0, 3600))
opl.plot_arrivals(arrivals, ax=ax_step, scale=2, timescale=60,
                  **arrivaldict)
ax_step.spines['bottom'].set_visible(False)
ax_step.set_xlabel('')
ax_step.tick_params(axis='x', which='both', labelbottom=False, bottom=False)


opl.trace([st_gcmt_triangle[0]], ax=ax_tri, lw=0.75, alpha=1.0, plot_labels=False,
          labels=['Conv. Triangle'], origin_time=cmt_gcmt.origin_time, limits=(0, 3600))
opl.plot_arrivals(arrivals, ax=ax_tri, scale=2.0, timescale=60,
                  **arrivaldict)

# Little inset for the STF
miniax = opl.axes_from_axes(ax_tri, 84932, [0.03, 0.3, 0.125, 0.5])
miniax.plot(t-tshift, stf_triangle, color='k', lw=0.5)
miniax.set_xlim(0, 70)
miniax.set_xlabel('Time [s]', fontsize='x-small')
# remove all ticks from miniax
miniax.tick_params(axis='both', which='major', bottom=True, top=False, left=False, right=False, labelbottom=True, labelleft=False, labeltop=False, labelright=False,
                   labelsize='x-small')


plt.subplots_adjust(hspace=0.0, left=0.05, right=0.95, top=0.95, bottom=0.1)
plt.savefig('processing_sequence.pdf', dpi=300)

# %%
# Get arrival to plot in the figure
phase_list = ['P', 'S', 'PcP', 'ScS', 'PP', 'SS', 'PKiKP', 'SKiKS', 'PKIKP', 'SKIKS']
arrivals = opl.get_arrivals(station_latitude, station_longitude, cmt_gcmt.latitude, cmt_gcmt.longitude, cmt_gcmt.depth, phase_list=phase_list)

# %%
# Plot the seismograms
plt.figure(figsize=(8, 2.5))

maxval = np.max([opl.stream_max(gf_gcmt), opl.stream_max(gf_scardec)])


opl.trace([gf_gcmt[0], gf_scardec[0]], labels=[f'GCMT: {cmt_gcmt.depth:.1f}km', f'SCARDEC: {cmt_scardec.depth:.1f}km'],
          plot_labels=False, lw=0.75, alpha=1.0)
plt.gca().get_legend().remove()
plt.legend(frameon=False, loc='upper right', fontsize='small',
           borderpad=0.0, borderaxespad=0.0)

# Plot annotations
opl.plot_label(plt.gca(), f'{network}.{station}.Z', fontsize='small', location=1, box=False,
               dist=0.01)
opl.plot_label(plt.gca(), f'|A|: {maxval:.4f}', fontsize='small', location=3, box=False,
               dist=0.01)

# Plot Origin line
plt.vlines((cmt_gcmt.origin_time + gfm.header['tc']).matplotlib_date,
           -.002, .002, color='k', linestyle='-',
           lw=.5, alpha=0.25, zorder=-10,)

plt.text((cmt_gcmt.origin_time + gfm.header['tc'] +5 ).matplotlib_date,
             0.002, 'Origin', fontsize='small', rotation=0, va='center', ha='left')

opl.plot_arrivals(arrivals, scale=0.002, timescale=60, ax=plt.gca(), alpha=0.25,               zorder=-10, lw=0.5, color='k', origin_time=cmt_gcmt.origin_time)

plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.3)
plt.savefig('raw_gf.pdf', dpi=300)

# %%
# Do the same for the other source time functions
st_gcmt_square = deepcopy(gf_gcmt)
st_gcmt_gaussian = deepcopy(gf_gcmt)
st_gcmt_scardec = deepcopy(gf_gcmt)
stf_square = square_stf(t, cmt_gcmt.time_shift + tshift, cmt_gcmt.hdur)
stf_gaussian = gaussian_stf(t, cmt_gcmt.time_shift + tshift, cmt_gcmt.hdur)
stf_scardec = interp_stf(t, stf.time + (stf.origin - cmt_gcmt.origin_time) + tshift, stf.moment_rate/stf.M0)

# Convolve the setup streams with the
st_gcmt_square[0].data = convolve(st_gcmt_square[0].data, stf_square, st_gcmt_square[0].stats.delta, tshift)
st_gcmt_gaussian[0].data = convolve(st_gcmt_gaussian[0].data, stf_gaussian, st_gcmt_gaussian[0].stats.delta, tshift)
st_gcmt_scardec[0].data = convolve(st_gcmt_scardec[0].data, stf_scardec, st_gcmt_scardec[0].stats.delta, tshift)


# %%
# Plotting all results
# Note that the STF do not have to be scaled since the GFs were already a resonse to the moment tensor.

fig, axes = plt.subplots(5,1, sharex=True, figsize=(6, 6))

# Plot scradec stf
xlim = (140, 230)
maxval = np.max([np.max(stf_triangle), np.max(stf_square), np.max(stf_scardec), np.max(stf_gaussian)])
plot_single_stf(axes[0], t, stf_scardec, 'SCARDEC', color='k')
axes[0].vlines(tshift, 0, maxval, color='k', linestyle='--', lw=0.5, zorder=-10)
axes[0].legend(frameon=False, loc='upper right', alignment='left', borderpad=0.0, borderaxespad=0.0)
axes[0].set_ylim(0, None)
axes[0].set_xlim(xlim)
plot_single_stf(axes[1], t, stf_square, 'Square', color='k')
axes[1].vlines(tshift, 0, maxval, color='k', linestyle='--', lw=0.5, zorder=-10)
axes[1].legend(frameon=False, loc='upper right', alignment='left', borderpad=0.0, borderaxespad=0.0)
axes[1].set_ylim(0, None)
axes[1].set_xlim(xlim)
plot_single_stf(axes[2], t, stf_triangle, 'GCMT', color='k')
axes[2].vlines(tshift, 0, maxval, color='k', linestyle='--', lw=0.5, zorder=-10)
axes[2].legend(frameon=False, loc='upper right', alignment='left', borderpad=0.0, borderaxespad=0.0)
axes[2].set_ylim(0, None)
axes[2].set_xlim(xlim)
plot_single_stf(axes[3], t, stf_gaussian, 'SF3DG', color='k')
axes[3].vlines(tshift, 0, maxval, color='k', linestyle='--', lw=0.5, zorder=-10)
axes[3].legend(frameon=False, loc='upper right', alignment='left', borderpad=0.0, borderaxespad=0.0)
axes[3].set_ylim(0, None)
axes[3].set_xlim(xlim)

plot_single_stf(axes[4], t, stf_scardec, 'SCARDEC', color='tab:blue')
plot_single_stf(axes[4], t, stf_square, 'Square', color='tab:green')
plot_single_stf(axes[4], t, stf_triangle, 'GCMT', color='k')
plot_single_stf(axes[4], t, stf_gaussian, 'SF3DG', color='tab:orange')
plt.vlines(tshift, 0, maxval, color='k', linestyle='--', lw=0.5, zorder=-10)
axes[4].legend(frameon=False, loc='upper right', alignment='left', borderpad=0.0, borderaxespad=0.0)
axes[4].set_ylim(0, None)
axes[4].set_xlim(xlim)
axes[4].set_xlabel('Time [s]')
plt.savefig('stf_wavelets.pdf', dpi=300)

# %%

from matplotlib import gridspec
xlim = (-10, 80)
ylim = (0,None)
maxval = np.max([np.max(stf_triangle), np.max(stf_square), np.max(stf_scardec), np.max(stf_gaussian)])

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(4, 2, width_ratios=[1, 2.5], height_ratios=[1, 1, 1, 1])

# First the source time functions in the left column
ax1 = fig.add_subplot(gs[0,0])
plot_single_stf(ax1, t - tshift, stf_scardec, 'SCARDEC', color='tab:blue')
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.tick_params(axis='x', which='both', labelbottom=False)

ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
plot_single_stf(ax2, t - tshift, stf_square, 'Square', color='tab:green')
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.tick_params(axis='x', which='both', labelbottom=False)

ax3 = fig.add_subplot(gs[2,0], sharex=ax1)
plot_single_stf(ax3, t - tshift, stf_triangle, 'GCMT', color='k')
ax3.set_xlim(xlim)
ax3.set_ylabel('             Moment Rate [N.m]')
ax3.set_ylim(ylim)
ax3.tick_params(axis='x', which='both', labelbottom=False)

ax4 = fig.add_subplot(gs[3,0], sharex=ax1)
plot_single_stf(ax4, t - tshift, stf_gaussian, 'SF3DG', color='tab:orange')
ax4.set_xlim(xlim)
ax4.set_ylim(ylim)
ax4.set_xlabel('Time [s]')


limits = (0, 3600)
limits = (10*60, 1000)

# Stream max =
maxval = np.max([opl.stream_max(st_gcmt_true), opl.stream_max(st_gcmt_scardec),
                 opl.stream_max(st_gcmt_triangle), opl.stream_max(st_gcmt_square),
                 opl.stream_max(st_gcmt_gaussian), opl.stream_max(st_gcmt_scardec)])

scale = 10
maxval /= scale

# Now the seismograms in the right column
ax5 = fig.add_subplot(gs[0,1])
opl.trace([st_gcmt_scardec[0],gf_gcmt[0]], labels=['SCARDEC', 'GF'], plot_labels=False,
          lw=[0.75, 0.5], alpha=[1.0, 0.25], colors=['tab:blue', 'k'], absmax=maxval,
          origin_time=cmt_gcmt.origin_time, limits=limits)
ax5.set_xlabel("")
ax5.tick_params(axis='x', which='both', labelbottom=False)

ax6 = fig.add_subplot(gs[1,1], sharex=ax5)
opl.trace([st_gcmt_square[0],gf_gcmt[0]], labels=['Square', 'GF'], plot_labels=False,
          lw=[0.75, 0.5], alpha=[1.0, 0.25], colors=['tab:green', 'k'], absmax=maxval,
          origin_time=cmt_gcmt.origin_time, limits=limits)
ax6.set_xlabel("")
ax6.tick_params(axis='x', which='both', labelbottom=False)

ax7 = fig.add_subplot(gs[2,1], sharex=ax5)
opl.trace([st_gcmt_triangle[0],gf_gcmt[0]], labels=['Triangle', 'GF'], plot_labels=False,
          lw=[0.75, 0.5], alpha=[1.0, 0.25], colors=['k', 'k'], absmax=maxval,
          origin_time=cmt_gcmt.origin_time, limits=limits)
ax7.set_xlabel("")
ax7.set_ylabel('Displacement [m]                ', rotation=270, labelpad=20)
ax7.yaxis.set_label_position("right")
ax7.tick_params(axis='x', which='both', labelbottom=False)

ax8 = fig.add_subplot(gs[3,1], sharex=ax5)
opl.trace([st_gcmt_gaussian[0], gf_gcmt[0]], labels=['Gaussian', 'GF'], plot_labels=False,
          lw=[0.75, 0.5], colors=['tab:orange', 'k'], absmax=maxval, alpha=[1.0, 0.25],
          origin_time=cmt_gcmt.origin_time, limits=limits, nooffset=False)


plt.subplots_adjust(left=0.1, right=0.925, top=0.95, bottom=0.1, wspace=0.05)
plt.savefig('comparison_figures.pdf', dpi=300)


# %%


from matplotlib import gridspec
xlim = (-10, 80)
ylim = (0,None)
maxval = np.max([np.max(stf_triangle), np.max(stf_square), np.max(stf_scardec), np.max(stf_gaussian)])

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(4, 3, width_ratios=[1, 2, 3], height_ratios=[1, 1, 1, 1])

# First the source time functions in the left column
ax1 = fig.add_subplot(gs[0,0])
plot_single_stf(ax1, t - tshift, stf_scardec, 'SCARDEC', color='tab:blue')
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.tick_params(axis='x', which='both', labelbottom=False)

ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
plot_single_stf(ax2, t - tshift, stf_square, 'Square', color='tab:green')
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.tick_params(axis='x', which='both', labelbottom=False)

ax3 = fig.add_subplot(gs[2,0], sharex=ax1)
plot_single_stf(ax3, t - tshift, stf_triangle, 'GCMT', color='k')
ax3.set_xlim(xlim)
ax3.set_ylabel('             Moment Rate [N.m]')
ax3.set_ylim(ylim)
ax3.tick_params(axis='x', which='both', labelbottom=False)

ax4 = fig.add_subplot(gs[3,0], sharex=ax1)
plot_single_stf(ax4, t - tshift, stf_gaussian, 'SF3DG', color='tab:orange')
ax4.set_xlim(xlim)
ax4.set_ylim(ylim)
ax4.set_xlabel('Time [s]')


limits = (0, 3600)
limits = (10*60, 1100)

# Stream max =
maxval = np.max([opl.stream_max(st_gcmt_true), opl.stream_max(st_gcmt_scardec),
                 opl.stream_max(st_gcmt_triangle), opl.stream_max(st_gcmt_square),
                 opl.stream_max(st_gcmt_gaussian), opl.stream_max(st_gcmt_scardec)])

scale = 10
maxval /= scale

# Now the seismograms in the right column
ax5 = fig.add_subplot(gs[0,1])
opl.trace([st_gcmt_scardec[0],gf_gcmt[0]], labels=['SCARDEC', 'GF'], plot_labels=False,
          lw=[0.75, 0.5], alpha=[1.0, 0.2], colors=['tab:blue', 'k'], absmax=maxval,
          origin_time=cmt_gcmt.origin_time, limits=limits, nooffset=True,
          legend=False)
ax5.set_xlabel("")
ax5.tick_params(axis='x', which='both', labelbottom=False)

factor = 5
scale_arrival = 0.00012
Parrivals = [a for a in arrivals if 'P' == a.name]
opl.plot_arrivals(Parrivals, scale=scale_arrival, timescale=1, ax=ax5, alpha=1.0,               zorder=-10, lw=0.5, color='k')

opl.plot_label(ax5, f'P waves ({factor}x scale)', fontsize='small', location=6, box=False,
               dist=0.01)


ax6 = fig.add_subplot(gs[1,1], sharex=ax5)
opl.trace([st_gcmt_square[0],gf_gcmt[0]], labels=['Square', 'GF'], plot_labels=False,
          lw=[0.75, 0.5], alpha=[1.0, 0.2], colors=['tab:green', 'k'], absmax=maxval,
          origin_time=cmt_gcmt.origin_time, limits=limits, nooffset=True,
          legend=False)
ax6.set_xlabel("")
ax6.tick_params(axis='x', which='both', labelbottom=False)

ax7 = fig.add_subplot(gs[2,1], sharex=ax5)
opl.trace([st_gcmt_triangle[0],gf_gcmt[0]], labels=['Triangle', 'GF'], plot_labels=False,
          lw=[0.75, 0.5], alpha=[1.0, 0.2], colors=['k', 'k'], absmax=maxval,
          origin_time=cmt_gcmt.origin_time, limits=limits, nooffset=True,
          legend=False)
ax7.set_xlabel("")
# ax7.set_ylabel('Displacement [m]                ', rotation=270, labelpad=20)
ax7.yaxis.set_label_position("right")
ax7.tick_params(axis='x', which='both', labelbottom=False)

ax8 = fig.add_subplot(gs[3,1], sharex=ax5)
opl.trace([st_gcmt_gaussian[0], gf_gcmt[0]], labels=['Gaussian', 'GF'], plot_labels=False,
          lw=[0.75, 0.5], colors=['tab:orange', 'k'], absmax=maxval, alpha=[1.0, 0.2],
          origin_time=cmt_gcmt.origin_time, limits=limits, nooffset=True,
          legend=False)


limits = (0, 3600)
limits = (1100, 35*60)

# Stream max =
maxval = np.max([opl.stream_max(st_gcmt_true), opl.stream_max(st_gcmt_scardec),
                 opl.stream_max(st_gcmt_triangle), opl.stream_max(st_gcmt_square),
                 opl.stream_max(st_gcmt_gaussian), opl.stream_max(st_gcmt_scardec)])


scale = 10/factor
maxval /= scale

legenddict = dict(frameon=False, loc='lower right', fontsize='small',
                  bbox_to_anchor=(1.0, 1.0), borderpad=0.0, borderaxespad=0.0)
# Now the seismograms in the right column
ax9 = fig.add_subplot(gs[0,2])
opl.trace([st_gcmt_scardec[0],gf_gcmt[0]], labels=['SCARDEC', 'GF'], plot_labels=False,
          lw=[0.75, 0.5], alpha=[1.0, 0.2], colors=['tab:blue', 'k'], absmax=maxval,
          origin_time=cmt_gcmt.origin_time, limits=limits, nooffset=True)
patches, handles = ax9.get_legend_handles_labels()
plt.legend(patches[:1], handles[:1], **legenddict)
ax9.set_xlabel("")
ax9.tick_params(axis='x', which='both', labelbottom=False)
opl.plot_label(ax9, 'S waves', fontsize='small', location=6, box=False,
               dist=0.01)


Sarrival = [a for a in arrivals if 'S' == a.name]
opl.plot_arrivals(Sarrival, scale=scale_arrival*factor, timescale=1, ax=ax9, alpha=1.0,
                  zorder=-1, lw=0.5, color='k')

ax10 = fig.add_subplot(gs[1,2], sharex=ax9)
opl.trace([st_gcmt_square[0],gf_gcmt[0]], labels=['Square', 'GF'], plot_labels=False,
          lw=[0.75, 0.5], alpha=[1.0, 0.2], colors=['tab:green', 'k'], absmax=maxval,
          origin_time=cmt_gcmt.origin_time, limits=limits, nooffset=True)
ax10.set_xlabel("")
ax10.tick_params(axis='x', which='both', labelbottom=False)
patches, handles = ax10.get_legend_handles_labels()
plt.legend(patches[:1], handles[:1], **legenddict)

ax11 = fig.add_subplot(gs[2,2], sharex=ax9)
opl.trace([st_gcmt_triangle[0],gf_gcmt[0]], labels=['Triangle', 'GF'], plot_labels=False,
          lw=[0.75, 0.5], alpha=[1.0, 0.2], colors=['k', 'k'], absmax=maxval,
          origin_time=cmt_gcmt.origin_time, limits=limits, nooffset=True)
ax11.set_xlabel("")
# ax11.set_ylabel('Displacement [m]                ', rotation=270, labelpad=20)
ax11.yaxis.set_label_position("right")
ax11.tick_params(axis='x', which='both', labelbottom=False)
patches, handles = ax11.get_legend_handles_labels()
plt.legend(patches[:1], handles[:1], **legenddict)


ax12 = fig.add_subplot(gs[3,2], sharex=ax9)
opl.trace([st_gcmt_gaussian[0], gf_gcmt[0]], labels=['Gaussian', 'GF'], plot_labels=False,
          lw=[0.75, 0.5], colors=['tab:orange', 'k'], absmax=maxval, alpha=[1.0, 0.2],
          origin_time=cmt_gcmt.origin_time, limits=limits, nooffset=True)
patches, handles = ax12.get_legend_handles_labels()
plt.legend(patches[:1], handles[:1], **legenddict)


plt.subplots_adjust(left=0.1, right=0.925, top=0.95, bottom=0.1, wspace=0.1)
plt.savefig('comparison_figures_P_S.pdf', dpi=300)
plt.close('all')


# %%
# Download the observed seismograms for the Alaska event

# Remove data directory if it exists
if os.path.exists(os.path.join(scardec_stf_dir, 'data')):
    shutil.rmtree(os.path.join(scardec_stf_dir, 'data'))

opl.download_data(
    os.path.join(scardec_stf_dir, 'data'),
    starttime=cmt_gcmt.origin_time - 300,
    endtime=cmt_gcmt.origin_time + 3600 + 300,
    network='II', station='BFO',
    channel_priorities=['BH[ZNE12]', 'HH[ZNE12]', 'LH[ZNE12]'],
    location_priorities=['00', '10', '01']
    )

# %%
# Load and process the waveforms
import obspy

# Load the waveforms
raw = obspy.read(os.path.join(scardec_stf_dir, 'data', 'waveforms', '*.mseed'))
inv = obspy.read_inventory(os.path.join(scardec_stf_dir, 'data', 'stations', '*.xml'))

# %%
# Process the waveforms
obs = raw.copy().select(network='II', station='BFO', component='Z')

process(obs, cmt_gcmt.origin_time , npts, delta, tshift=tshift, step=False, inv=inv)

# %%
# Plot the waveforms and the seismograms
from obspy.geodetics import gps2dist_azimuth
slat = inv.select(network=network, station=station)[0][0].latitude
slon = inv.select(network=network, station=station)[0][0].longitude

dist_in_m, az, baz = gps2dist_azimuth(cmt_gcmt.latitude, cmt_gcmt.longitude, slat, slon)

# Bandpass
bp = [17, 1000]

# %%
# obs.plot()


def traceL2(tr1, tr2, norm=True):
    if norm:
        return np.sum((tr1.data - tr2.data) ** 2) / np.sum(tr1.data**2)
    else:
        return 0.5 * np.sum((tr1.data - tr2.data) ** 2)


def diffstream(st1, st2):
    st = st1.copy()

    for tr in st:
        tr2 = st2.select(component=tr.stats.component)[0]
        tr.data = tr.data - tr2.data

    return st


# %%


headerdict = dict(
    event=cmt.eventname,
    event_time=cmt.cmt_time,
    event_latitude=cmt.latitude,
    event_longitude=cmt.longitude,
    event_depth_in_km=cmt.depth,
    station=f"{network}.{station}",
    station_latitude=slat,
    station_longitude=slon,
    station_azimuth=az,
    station_back_azimuth=baz,
    station_distance_in_degree=dist_in_m / 1000.0 / (40000 / 360.0),
    location=6,
    # fontsize='small'
)


plotdict = dict(
    ls=["-", "-"],
    lw=[0.75, 0.75],
    colors=["k", "darkgray"],
    limits=(0, 10800),
    nooffset=False,
    absmax=1.3e-4,
    origin_time=cmt.origin_time,
    plot_labels=False,
)

plotdict = dict(
    ls=["-", "-"],
    lw=[0.75, 0.75],
    # colors=["k", "r"],
    nooffset=False,
    origin_time=cmt.origin_time,
    plot_labels=False,
)

absmax=0.00075
factor=5
scale_arrival=0.00012

Plimits=(600, 1100)
Slimits=(1100, 2100)
Pabsmax=absmax/factor
Sabsmax=absmax


Ptrimdict = dict(
        starttime=cmt_gcmt.origin_time + Plimits[0],
        endtime=cmt_gcmt.origin_time + Plimits[1],)

Strimdict = dict(
        starttime=cmt_gcmt.origin_time + Slimits[0],
        endtime=cmt_gcmt.origin_time + Slimits[1],)

scardec_dict = dict(colors=['k', 'tab:blue'])
gaussian_dict = dict(colors=['k', 'tab:red'])


# %%
fig = plt.figure(figsize=(9, 4.5))

obs_plot = obs.select(component='Z').copy()
syn1_plot = st_gcmt_scardec.copy()
syn2_plot = st_gcmt_gaussian.copy()


# ----- P WAVE -----
Parrival = [a for a in arrivals if 'P' == a.name]


ax1 = plt.subplot(2, 2, 1)
opl.trace(
    [obs_plot[0], syn1_plot[0]],
    labels=["Observed", "SCARDEC"],
    limits=Plimits,
    legend=False,
    absmax=Pabsmax,
    **scardec_dict,
    **plotdict,
)
opl.plot_arrivals(Parrival, scale=scale_arrival, timescale=1, alpha=1.0,
                  zorder=-1, lw=0.5, color='k')


ax1.spines["bottom"].set_visible(False)
ax1.tick_params(bottom=False, labelbottom=False)
ax1.set_xlabel("")

obs_m = obs.copy()
syn1_m = st_gcmt_scardec.copy()

misfit = traceL2(
    obs_m[0].trim(**Ptrimdict),
    syn1_m[0].trim(**Ptrimdict),
    norm=True
)

opl.plot_label(
    ax1,
    f"L2: {misfit:4g}\nBP: {bp[0]:d}-{bp[1]:d}s",
    location=1,
    box=False,
    fontsize="small",
    dist=0.0,
)
opl.plot_label(ax1, f'P waves ({factor}x scale)', fontsize='medium',
               location=6, box=False)


ax2 = plt.subplot(2, 2, 3)
opl.trace(
    [obs_plot[0], syn2_plot[0]],
    ax=ax2,
    labels=["Observed", "Gaussian"],
    limits=Plimits,
    legend=False,
    absmax=Pabsmax,
    **gaussian_dict,
    **plotdict,
)

opl.plot_arrivals(Parrival, scale=scale_arrival, timescale=1, alpha=1.0,
                  zorder=-1, lw=0.5, color='k')

obs_m = obs.copy()
syn2_m = st_gcmt_gaussian.copy()

misfit = traceL2(
    obs_m[0].trim(**Ptrimdict),
    syn2_m[0].trim(**Ptrimdict),
    norm=True
)

opl.plot_label(
    ax2,
    f"L2: {misfit:4g}\nBP: {bp[0]:d}-{bp[1]:d}s",
    location=1,
    box=False,
    dist=0.0,
    fontsize="small",
)


# ----- S WAVE -----

Sarrival = [a for a in arrivals if 'S' == a.name]

ax3 = plt.subplot(2, 2, 2)
opl.trace(
    [obs_plot[0], syn1_plot[0]],
    labels=["Observed", "SCARDEC"],
    limits=Slimits,
    legend=False,
    absmax=Sabsmax,
    **scardec_dict,
    **plotdict,
)

opl.plot_arrivals(Sarrival, scale=scale_arrival*factor, timescale=1, alpha=1.0,
                  zorder=-1, lw=0.5, color='k')

legenddict = dict(frameon=False, loc='upper right', fontsize='small',
                  borderpad=0.0, borderaxespad=0.0)
plt.legend(**legenddict)

ax3.spines["bottom"].set_visible(False)
ax3.tick_params(bottom=False, labelbottom=False)
ax3.set_xlabel("")

obs_m = obs.copy()
syn1_m = st_gcmt_scardec.copy()

misfit = traceL2(
    obs_m[0].trim(**Strimdict),
    syn1_m[0].trim(**Strimdict),
    norm=True
)

opl.plot_label(
    ax3,
    f"L2: {misfit:4g}\nBP: {bp[0]:d}-{bp[1]:d}s",
    location=1,
    box=False,
    fontsize="small",
    dist=0.0,
)
opl.plot_label(ax3, f'S waves', fontsize='medium', location=6, box=False)


ax4 = plt.subplot(2, 2, 4)
opl.trace(
    [obs_plot[0], syn2_plot[0]],
    ax=ax4,
    labels=["Observed", "Gaussian"],
    limits=Slimits,
    absmax=Sabsmax,
    legend=False,
    **gaussian_dict,
    **plotdict,
)

opl.plot_arrivals(Sarrival, scale=scale_arrival*factor, timescale=1, alpha=1.0,
                  zorder=-1, lw=0.5, color='k')


plt.legend(**legenddict)

obs_m = obs.copy()
syn2_m = st_gcmt_gaussian.copy()

misfit = traceL2(
    obs_m[0].trim(**Strimdict),
    syn2_m[0].trim(**Strimdict),
    norm=True
)

opl.plot_label(
    ax4,
    f"L2: {misfit:4g}\nBP: {bp[0]:d}-{bp[1]:d}s",
    location=1,
    box=False,
    dist=0.0,
    fontsize="small",
)


plt.subplots_adjust(hspace=0.0, bottom=0.125, top=0.9, left=0.05, right=0.95,
                    wspace=0.1)

plt.show(block=False)
plt.savefig("stf_compare.pdf")
plt.close('all')
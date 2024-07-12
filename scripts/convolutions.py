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
import obstflib as osl
import obsplotlib.plot as opl

# %%
# Get Alaska STF

# Files
scardec_stf_dir = './STF/FCTs_20180123_093140_GULF_OF_ALASKA/'
cmt_file = os.path.join(scardec_stf_dir, 'CMTSOLUTION')

# Get Alaska STF
scardec = osl.STF.scardecdir(scardec_stf_dir, 'optimal')

# Get Alaska CMT solution
cmt = osl.CMTSOLUTION.read(cmt_file)

# This is the time vector of the STF
t = np.arange(-10, 120, 0.1)

# Interpolate the stf to the given time vector
scardec.interp(t)

# Create parameter dictionary for STF creation
stf_dict = dict(
    origin=scardec.origin,
    t=t, hdur=cmt.hdur, tc=cmt.cmt_time - scardec.origin,
    tshift=0.0, M0=cmt.M0 / 1e7)

# Create STFs
triangle = osl.STF.triangle(**stf_dict)
boxcar = osl.STF.boxcar(**stf_dict)
gaussian = osl.STF.gaussian(**stf_dict)
error = osl.STF.error(**stf_dict)
derrordt = error.gradient(new=True)

# %%

def plot_single_stf(ax, stf: osl.STF, title, *args, shift: bool = True, **kwargs):
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

    stf.plot(*args, label=title, ax=ax, shift=shift, **kwargs)
    ax.set_xlim(np.min(t), np.max(t))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig, axes = plt.subplots(6,1, sharex=True, figsize=(8, 6))

# Plot scradec stf
plot_single_stf(axes[0],scardec, 'SCARDEC', color='k')
axes[0].legend(frameon=False)
axes[0].set_ylim(0, None)
plot_single_stf(axes[1], boxcar, 'Boxcar', color='k')
axes[1].legend(frameon=False)
axes[1].set_ylim(0, None)
plot_single_stf(axes[2], triangle, 'GCMT', color='k')
axes[2].legend(frameon=False)
axes[2].set_ylim(0, None)

plot_single_stf(axes[3], gaussian, 'SF3DG', color='k')
axes[3].legend(frameon=False)
axes[3].set_ylim(0, None)

plot_single_stf(axes[4], error, 'Error', color='k')
axes[4].legend(frameon=False)
axes[4].set_ylim(0, None)


plot_single_stf(axes[5], scardec, 'SCARDEC', color='tab:blue')
plot_single_stf(axes[5], boxcar, 'Boxcar', color='tab:green')
plot_single_stf(axes[5], triangle, 'GCMT', color='k')
plot_single_stf(axes[5], gaussian, 'SF3DG', color='tab:orange')
plot_single_stf(axes[5], derrordt, 'dError/dt', color='tab:red',
                ls='-.')
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
    s=scardec.strike1, d=scardec.dip1, r=scardec.rake1, M0=scardec.M0 * 1e7,
    origin_time=scardec.origin, depth=scardec.depth,
    latitude=scardec.latitude, longitude=scardec.longitude)
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

    st.filter('bandpass', freqmin=0.004, freqmax=1/17.0, corners=3, zerophase=True)
    st.interpolate(sampling_rate=sps, starttime=starttime-tshift,
                   npts=int((length_in_s + tshift) * sps))

    if step:
        # Convolve the seismograms with the source time functions
        for tr in st:
            error = osl.STF.error(origin=obspy.UTCDateTime(0), t=tr.times(), tshift=tshift, hdur=1e-6, tc=0.0)
            tr.data = convolve(tr.data, error.f, tr.stats.delta, tshift)

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
stf_triangle = osl.STF.triangle(
    origin=cmt_gcmt.origin_time, t=t, tc=cmt_gcmt.time_shift,
    tshift=tshift, hdur=cmt_gcmt.hdur)

# Convolve the setup streams with the
st_gcmt_triangle[0].data = convolve(st_gcmt_triangle[0].data, stf_triangle.f, st_gcmt_triangle[0].stats.delta, tshift)

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
stf_triangle.plot(ax=miniax, shift=True, color='k', lw=0.5)
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
st_gcmt_boxcar = deepcopy(gf_gcmt)
st_gcmt_gaussian = deepcopy(gf_gcmt)
st_gcmt_scardec = deepcopy(gf_gcmt)
stf_boxcar = osl.STF.boxcar(origin=cmt_gcmt.origin_time, t=t, tc=cmt_gcmt.time_shift, tshift=tshift, hdur=cmt_gcmt.hdur)
stf_gaussian = osl.STF.gaussian(origin=cmt_gcmt.origin_time, t=t, tc=cmt_gcmt.time_shift, tshift=tshift, hdur=cmt_gcmt.hdur)
stf_scardec = osl.STF.scardecdir(scardec_stf_dir, 'optimal')
stf_scardec.interp(t, tshift=tshift, origin=cmt_gcmt.origin_time)
stf_scardec.f /= stf_scardec.M0

# Convolve the setup streams with the
st_gcmt_boxcar[0].data = convolve(st_gcmt_boxcar[0].data, stf_boxcar.f,
                                  st_gcmt_boxcar[0].stats.delta, tshift)
st_gcmt_gaussian[0].data = convolve(st_gcmt_gaussian[0].data, stf_gaussian.f,
                                    st_gcmt_gaussian[0].stats.delta, tshift)
st_gcmt_scardec[0].data = convolve(st_gcmt_scardec[0].data, stf_scardec.f,
                                   st_gcmt_scardec[0].stats.delta, tshift)

# %%
# Create a function that takes in a SCARDEC directory, and creates source time
# functions using either the CMTSOLUTION from the directory or a CMTSOLUTION
# from the SCARDEC focal mechanism and all possible stfs from
# the STF class, including the scaredc one, to create a tuple of a CMTSOLUTION
# and a list of source time functions.

def create_stfs_from_scardecdir(scardecdir: str, t: np.ndarray,
                                tshift: float = 0.0,
                                fmtype: str = 'optimal',
                                stftype: str = 'optimal') -> tuple:
    """
    Create a CMTSOLUTION and a list of source time functions from a SCARDEC directory.

    Parameters
    ----------
    scardecdir : str
        Path to the SCARDEC directory.
    t: np.ndarray
        Time vector to interpolate the source time functions to.
    fmtype : str
        Type of source time function to use. Choose from 'gcmt', 'optimal'
        or 'average'.
    stftype : str
        Type of source time function to use. Choose from 'optimal'
        or 'average'.

    Returns
    -------
    tuple
        A tuple of a CMTSOLUTION and a list of source time functions.


    Note, the output CMTSOLUTIONS if any scardec stftype is chose, will have
    the same half duration and time shift as the GCMT solution
    """

    # Get the CMTSOLUTION
    _cmt = osl.CMTSOLUTION.read(os.path.join(scardecdir, 'CMTSOLUTION'))


    # Get the STF
    stf = osl.STF.scardecdir(scardecdir, stftype)

    # Normalize to make equal to all other STFs
    stf.f /= stf.M0

    if fmtype in ['optimal', 'average']:

        # Create the CMTSOLUTION from the STF
        cmt = osl.CMTSOLUTION.from_sdr(
            s=stf.strike1, d=stf.dip1, r=stf.rake1, M0=stf.M0 * 1e7,
            origin_time=stf.origin, depth=stf.depth,
            latitude=stf.latitude, longitude=stf.longitude)

        # Set the half duration and time shift to the GCMT values
        cmt.hdur = _cmt.hdur
        cmt.time_shift = _cmt.time_shift

    elif fmtype == 'gcmt':
        cmt = _cmt
    else:
        raise ValueError(f'Unknown source time function type: {stftype}')

    # Interpolate the STF to the given time vector
    stf.interp(t, tshift=tshift, origin=cmt.origin_time)

    # Create parameter dictionary for STF creation
    stf_dict = dict(
        origin=cmt.origin_time,
        t=t, hdur=cmt.hdur, tc=cmt.cmt_time - stf.origin,
        tshift=tshift)

    # Create STFs
    boxcar = osl.STF.boxcar(**stf_dict, label = 'Boxcar')
    triangle = osl.STF.triangle(**stf_dict, label = 'GCMT')
    gaussian = osl.STF.gaussian(**stf_dict, label = 'SF3DG')

    return cmt, [stf, boxcar, triangle, gaussian]

tshift = 150.0
delta = 0.05
length_in_s = 3600

# Make time vector
t = np.arange(0, length_in_s + tshift, delta)

# Create the CMTSOLUTION and STFs
cmt, stfs = create_stfs_from_scardecdir(scardec_stf_dir, t, tshift=tshift, fmtype='gcmt', stftype='optimal')

# %%
# Plotting all results
# Note that the STF do not have to be scaled since the GFs were already a resonse to the moment tensor.
def plot_stfs(stfs, xlim=(140, 230)):

    N = len(stfs)
    colordict = dict(
        SCARDEC='tab:blue',
        Boxcar='tab:green',
        GCMT='k',
        SF3DG='tab:orange'
    )

    tshift = stfs[0].tshift

    fig, axes = plt.subplots(N+1,1, sharex=True, figsize=(6, 6))


    xlim = (-20 + tshift, 100 + tshift)
    maxval = np.max([np.max(stf.f) for stf in stfs])

    # Plot each stf separately
    for _i, _stf in enumerate(stfs):

        _stf.plot(label=_stf.label, shift=False, color=colordict[_stf.label],
                  ax=axes[_i])
        axes[_i].vlines(tshift, 0, maxval, color='k', linestyle='--', lw=0.5,
                        zorder=-10)
        axes[_i].legend(frameon=False, loc='upper right', alignment='left',
                        borderpad=0.0, borderaxespad=0.0)
        axes[_i].set_ylim(0, None)
        axes[_i].set_xlim(xlim)
        axes[_i].spines['top'].set_visible(False)
        axes[_i].spines['right'].set_visible(False)

    # Plot all stfs together
    for _i, _stf in enumerate(stfs):
        _stf.plot(label=_stf.label, shift=False, color=colordict[_stf.label],
                  ax=axes[N])

    plt.vlines(tshift, 0, maxval, color='k', linestyle='--', lw=0.5, zorder=-10)
    axes[N].legend(frameon=False, loc='upper right', alignment='left',
                   borderpad=0.0, borderaxespad=0.0)
    axes[N].set_ylim(0, None)
    axes[N].set_xlim(xlim)
    axes[N].set_xlabel('Time [s]')
    axes[N].spines['top'].set_visible(False)
    axes[N].spines['right'].set_visible(False)


plot_stfs(stfs, xlim=(140, 230))
plt.savefig('stf_wavelets.pdf', dpi=300)


# %% Now we have a CMTSOLUTION and a set of associated source time functions
# that can be used to create seismograms. So, we create a function that takes in
# a CMTSOLUTION and a list of source time function, extracts the Green functions
# from the database. It will take in the cmt, the stfs, and GFManager, and
# return a list of associated obspy.streams.

def create_seismograms(cmt: osl.CMTSOLUTION, stfs: list, gfm: GFManager,
                        network: str = "II", station: str = "BFO", component: str = 'Z',
                        **kwargs) -> tuple[obspy.Stream, dict[str, obspy.Stream]]:
    """
    Create seismograms from a CMTSOLUTION and a list of source time functions.

    Parameters
    ----------
    cmt : osl.CMTSOLUTION
        CMTSOLUTION object.
    stfs : list
        List of source time functions.
    gfm : GFManager
        Green function manager.
    network : str
        Network code.
    station : str
        Station code.
    component : str
        Component code.
    **kwargs : dict
        Additional keyword arguments for the process function.
    Returns
    -------
    list
        List of obspy.streams.
    """

    # Get delta length and time shift from the first stf
    tshift = stfs[0].tshift
    delta = (stfs[0].t[1] - stfs[0].t[0])
    sps = 1./delta
    length_in_s = len(stfs[0].t) / sps - tshift

    # Get the Green functions
    gf = gfm.get_seismograms(cmt.to_gf3d(), raw=True).select(network=network, station=station, component=component)

    # Process the Green function
    process(gf, cmt.origin_time, length_in_s=length_in_s, sps=sps, tshift=tshift,
            step=True)

    # Create the seismograms
    seismograms = dict()

    # Get the seismograms
    for _stf in stfs:

        print("Computing:", _stf.label, _stf.tshift)

        # Copy the stream
        seis = gf.copy()

        # # Convolve the setup streams with the
        seis[0].data = convolve(seis[0].data, _stf.f, seis[0].stats.delta, _stf.tshift)

        # Copy the stream to the dictionary
        seismograms[_stf.label] = seis.copy()

    return gf, seismograms


gf, seismograms = create_seismograms(cmt, stfs, gfm, network='II', station='BFO', component='Z')

#%%
from matplotlib import gridspec

# source time function plot parameters
stf_xlim = (-10, 80)
stf_ylim = (0,None)
stf_maxval = np.max([np.max(stf.f) for stf in stfs])


# Stream plot parameters
limits = (0, 3600)
trace_maxval = np.max([opl.stream_max(seismograms[stf.label]) for stf in stfs])
trace_scale = 2
trace_maxval /= trace_scale

# Create figure with Gridspec
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(4, 2, width_ratios=[1, 2.5], height_ratios=[1, 1, 1, 1])

colordict = dict(
    SCARDEC='tab:blue',
    Boxcar='tab:green',
    GCMT='k',
    SF3DG='tab:orange'
)

Nstf = len(stfs)

# Loop over Source time functions
for _i, _stf in enumerate(stfs):
    # First the source time functions in the left column
    if _i != 0:
        sharex_stf = ax_stf
    else:
        sharex_stf = None

    # Create STF Axes
    ax_stf = fig.add_subplot(gs[_i,0], sharex=sharex_stf)

    # Plot STF
    _stf.plot(ax=ax_stf, label=_stf.label, color=colordict[_stf.label])

    # Remove x-axis labels
    if _i != Nstf - 1:
        ax_stf.tick_params(axis='x', which='both', labelbottom=False)
    else:
        ax_stf.set_xlim(stf_xlim)
        ax_stf.set_ylim(stf_ylim)
        ax_stf.set_xlabel('Time [s]')

    # Remove top and right spines
    ax_stf.spines['top'].set_visible(False)
    ax_stf.spines['right'].set_visible(False)

    # Now the seismograms in the right column
    ax_trace = fig.add_subplot(gs[_i,1])
    # seismograms[_stf.label][0]
    opl.trace([seismograms[_stf.label][0], gf[0]],
              labels=[_stf.label, 'GF'],
              plot_labels=False, lw=[0.75, 0.5, 0.5], alpha=[1.0, 0.25, 0.25],
              colors=[colordict[_stf.label], 'k', 'b'], absmax=trace_maxval,
              origin_time=cmt.origin_time, limits=limits)
    if _i != Nstf - 1:
        ax_trace.set_xlabel("")
        ax_trace.tick_params(axis='x', which='both', labelbottom=False)
    else:
        pass

plt.subplots_adjust(left=0.1, right=0.925, top=0.95, bottom=0.1, wspace=0.05)
plt.savefig('comparison_figures.pdf', dpi=300)

# %%


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

def plot_misfit_labels(ax, trace1, trace2, **kwargs):

    # Compute normalized L2 norm
    misfit = traceL2(trace1, trace2, norm=True)

    # Compute maximum cross correlation and smaple shift
    maxcc, sampleshift = opl.X(trace1, trace2)

    # Compute timeshift
    timeshift = sampleshift * trace1.stats.delta

    opl.plot_label(ax,
        f"L2: {misfit:4g}\n",
        location=1,
        box=False,
        dist=0.0,
        **kwargs
    )

    opl.plot_label(ax,
        f"DT: {timeshift:4g}\n",
        location=2,
        box=False,
        dist=0.0,
        **kwargs
    )

    opl.plot_label(ax,
        f"CC: {maxcc:4g}\n",
        location=4,
        box=False,
        dist=0.0,
        **kwargs
    )

# %%

from matplotlib import gridspec
def plot_stf_P_S(stfs, seismograms, comp, cmt, station_latitude, station_longitude,
                 complabel='GF', misfit_label: bool = True, **kwargs):
    """
    Plot the source time functions and the seismograms for the P and S waves.

    Parameters
    ----------
    stfs : list
        List of source time functions.
    seismograms : dict
        Dictionary of seismograms.
    comp : obspy.Stream
        Trace to compare the stations to.
    arrivals : list
        List of arrivals.
    station : str
        Station code.
    cmt : osl.CMTSOLUTION
        CMTSOLUTION for event time plotting and stations locations.
    station_latitude : float
        Latitude of the station.
    station_longitude : float
        Longitude of the station.
    **kwargs : dict
        Additional keyword arguments for the obsplotlib.plot.trace function.
    """


    # source time function plot parameters
    stf_xlim = (-10, 80)
    stf_ylim = (0,None)
    stf_maxval = np.max([np.max(stf.f) for stf in stfs])

    # Stream plot parameters
    trace_maxval = np.max([opl.stream_max(seismograms[stf.label]) for stf in stfs])
    scale_arrival = 0.00010/0.001327 * trace_maxval

    # P versus S scaling and timing
    S2P = 5
    scale = 10
    Pmaxval = trace_maxval/scale
    Smaxval = trace_maxval/scale*S2P

    # Get P and S arrival for orientation
    arrivals = opl.get_arrivals(
        station_latitude, station_longitude,
        cmt_gcmt.latitude, cmt_gcmt.longitude, cmt_gcmt.depth,
        phase_list=['P', 'S'])

    # Get arricals from earlier arrival array
    Parrivals = [a for a in arrivals if 'P' == a.name]
    Sarrival = [a for a in arrivals if 'S' == a.name]

    # Get arrivals
    Ptime = Parrivals[0].time
    Stime = Sarrival[0].time

    # Get difference between P and S arrival
    SmP = Stime - Ptime

    # Empirically derived offsets
    P_start_offset = 1/6 * SmP
    P_duration = 2/3 * SmP

    S_start_offset = 160/600 * SmP
    S_duration = 9/6 * SmP
    S_duration = 6/6 * SmP

    # Adjust limits based on arrival
    Plimits = (Ptime - P_start_offset, Ptime + P_duration)
    Slimits = (Stime - S_start_offset, Stime + S_duration)
    print(Plimits)
    print(Slimits)
    print(Parrivals[0].name, Parrivals[0].time)
    print(Sarrival[0].name, Sarrival[0].time)

    # Misfit in window calculation
    Ptrimdict = dict(
            starttime=cmt_gcmt.origin_time + Plimits[0],
            endtime=cmt_gcmt.origin_time + Plimits[1],)

    Strimdict = dict(
            starttime=cmt_gcmt.origin_time + Slimits[0],
            endtime=cmt_gcmt.origin_time + Slimits[1],)


    # Create legend dictionary for the legend in the right column
    legenddict = dict(frameon=False, loc='lower left', fontsize='small',
                    bbox_to_anchor=(0.0, 1.025), borderpad=0.0, borderaxespad=0.0,
                    markerfirst=False)

    # Create figure and gridspec
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(4, 3, width_ratios=[1, 2, 3], height_ratios=[1, 1, 1, 1])

    # Colordict
    colordict = dict(
        Observed='k',
        SCARDEC='tab:blue',
        Boxcar='tab:green',
        GCMT='tab:red',
        SF3DG='tab:orange'
    )

    # Number source time function
    Nstf = len(stfs)

    # Loop over Source time functions
    for _i, _stf in enumerate(stfs):
        # First the source time functions in the left column
        if _i != 0:
            sharex_stf = ax_stf
            sharex_trace_p = ax_trace_p
            sharex_trace_s = ax_trace_s
        else:
            sharex_stf = None
            sharex_trace_p = None
            sharex_trace_s = None

        # Create STF Axes
        ax_stf = fig.add_subplot(gs[_i,0], sharex=sharex_stf)

        # Plot STF
        _stf.plot(ax=ax_stf, label=_stf.label, color=colordict[_stf.label],
                  lw=0.75)

        # Remove x-axis labels
        if _i != Nstf - 1:
            ax_stf.tick_params(axis='x', which='both', labelbottom=False)
        else:
            ax_stf.set_xlim(stf_xlim)
            ax_stf.set_ylim(stf_ylim)
            ax_stf.set_xlabel('Time [s]')

        # Remove top and right spines
        ax_stf.spines['top'].set_visible(False)
        ax_stf.spines['right'].set_visible(False)

        # Create legend for each plot
        patches, handles = ax_stf.get_legend_handles_labels()
        plt.legend(patches[:1], handles[:1], **legenddict)

        # Now the seismograms in the center column containing the P waves
        ax_trace_p = fig.add_subplot(gs[_i,1], sharex=sharex_trace_p)

        # Plot P traces
        opl.trace([seismograms[_stf.label][0], comp[0]],
                labels=[_stf.label, complabel],
                plot_labels=False, lw=[0.75, 0.5, 0.5], alpha=[1.0, 0.25, 0.25],
                colors=[colordict[_stf.label], 'k', 'b'], absmax=Pmaxval,
                origin_time=cmt.origin_time, limits=Plimits,
                legend=False, **kwargs)

        # Optionally plot misift in window
        if misfit_label:
            comp_m = comp.copy()
            synt_m = seismograms[_stf.label].copy()
            comp_m[0].trim(**Ptrimdict),
            synt_m[0].trim(**Ptrimdict),

            plot_misfit_labels(ax_trace_p, comp_m[0], synt_m[0], fontsize='x-small')

        if _i != Nstf - 1:
            ax_trace_p.set_xlabel("")
            ax_trace_p.tick_params(axis='x', which='both', labelbottom=False)
        else:
            pass


        if _i == 0:
            opl.plot_arrivals(Parrivals, scale=scale_arrival, timescale=1,
                            ax=ax_trace_p, alpha=1.0, zorder=-10, lw=0.5,
                            color='k')

            # Pwave label
            opl.plot_label(ax_trace_p, f'P waves ({S2P}x scale)', fontsize='small',
                        location=6, box=False,dist=0.025)

        # Now the seismograms in the right column containing the S waves
        ax_trace_s = fig.add_subplot(gs[_i,2], sharex=sharex_trace_s)

        # Plot S traces
        opl.trace([seismograms[_stf.label][0], comp[0]],
                labels=[_stf.label, complabel],
                plot_labels=False, lw=[0.75, 0.5, 0.5], alpha=[1.0, 0.25, 0.25],
                colors=[colordict[_stf.label], 'k', 'b'], absmax=Smaxval,
                origin_time=cmt.origin_time, limits=Slimits,
                legend=False, **kwargs)

        # Optionally plot misift in window
        if misfit_label:
            comp_m = comp.copy()
            synt_m = seismograms[_stf.label].copy()

            comp_m[0].trim(**Strimdict),
            synt_m[0].trim(**Strimdict),

            plot_misfit_labels(ax_trace_s, comp_m[0], synt_m[0], fontsize='x-small')


        if _i != Nstf-1:
            ax_trace_s.set_xlabel("")
            ax_trace_s.tick_params(axis='x', which='both', labelbottom=False)
        else:
            pass

        # Plot arrivals
        if _i == 0:
            opl.plot_arrivals(Sarrival, scale=scale_arrival*S2P, timescale=1,
                            ax=ax_trace_s, alpha=1.0, zorder=-10, lw=0.5,
                            color='k')

            # Swave label
            opl.plot_label(ax_trace_s, 'S waves', fontsize='small',
                        location=6, box=False, dist=0.025)


plot_stf_P_S(stfs, seismograms, comp=gf, cmt=cmt, station_latitude=station_latitude,
             station_longitude=station_longitude, complabel='GF',
             misfit_label=False, nooffset=True)
plt.subplots_adjust(left=0.1, right=0.925, top=0.95, bottom=0.1, wspace=0.1,
                    hspace=0.25)
plt.savefig('comparison_figures_P_S.pdf', dpi=300)
plt.close('all')


# %%
# Download the observed seismograms for the Alaska event

# Remove data directory if it exists
if os.path.exists(os.path.join(scardec_stf_dir, 'data')):
    shutil.rmtree(os.path.join(scardec_stf_dir, 'data'))

opl.download_data(
    os.path.join(scardec_stf_dir, 'data'),
    starttime=cmt.origin_time - 300,
    endtime=cmt.origin_time + 3600 + 300,
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

process(obs, cmt_gcmt.origin_time , npts, 1/delta, tshift=tshift, step=False, inv=inv)

# %%
# Plot the waveforms and the seismograms
from obspy.geodetics import gps2dist_azimuth
slat = inv.select(network=network, station=station)[0][0].latitude
slon = inv.select(network=network, station=station)[0][0].longitude

dist_in_m, az, baz = gps2dist_azimuth(cmt_gcmt.latitude, cmt_gcmt.longitude, slat, slon)

# %%

plot_stf_P_S(stfs, seismograms, comp=obs, cmt=cmt, station_latitude=slat,
             station_longitude=slon, complabel='Observed', nooffset=True,)
plt.subplots_adjust(left=0.1, right=0.925, top=0.95, bottom=0.1, wspace=0.1,
                    hspace=0.25)
plt.savefig('comparison_P_S_observed.pdf', dpi=300)
plt.close('all')

# %%
# Now let's repeat this test for the SCARDEC focal mechanism
# Load the SCARDEC focal mechanism

# Create the CMTSOLUTION and STFs
cmt, stfs = create_stfs_from_scardecdir(scardec_stf_dir, t, tshift=tshift,
                                        fmtype='optimal', stftype='optimal')

# Create the seismograms
gf, seismograms = create_seismograms(cmt, stfs, gfm, network='II', station='BFO', component='Z')

# Plot the seismograms compared to the observed data
plot_stf_P_S(stfs, seismograms, comp=obs, cmt=cmt, station_latitude=slat,
             station_longitude=slon, complabel='Observed', nooffset=True,)

plt.subplots_adjust(left=0.1, right=0.925, top=0.95, bottom=0.1, wspace=0.1,
                    hspace=0.25)
plt.savefig('comparison_P_S_observed_scardec.pdf', dpi=300)
plt.close('all')

#%%

# Create the CMTSOLUTION and STFs
cmt, stfs = create_stfs_from_scardecdir(scardec_stf_dir, t, tshift=tshift,
                                        fmtype='gcmt', stftype='optimal')

# Create the seismograms
gf, seismograms = create_seismograms(cmt, stfs, gfm, network='II', station='BFO', component='Z')

# Plot the seismograms compared to the observed data
plot_stf_P_S(stfs, seismograms, comp=obs, cmt=cmt, station_latitude=slat,
             station_longitude=slon, complabel='Observed', nooffset=True,)

plt.subplots_adjust(left=0.1, right=0.925, top=0.95, bottom=0.1, wspace=0.1,
                    hspace=0.25)
plt.savefig('comparison_P_S_observed_gcmt.pdf', dpi=300)
plt.close('all')

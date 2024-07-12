# %%
# convolution testing
import os
import glob
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
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# %%
# Files
scardec_stf_dir = './STF/FCTs_20070815_234057_NEAR_COAST_OF_PERU'
scardec_id = scardec_stf_dir.split('/')[-1]

cmt3_file = glob.glob(os.path.join(scardec_stf_dir, f'*_CMT3D+'))[0]

# %%
# Get the event record at II BFO for the Alaska event using the GFManager

cmt3 = CMTSOLUTION.read(cmt3_file)
cmt3_stf = deepcopy(cmt3)
cmt3.time_shift = 0.0


# %%
# Read the Green functions
from gf3d.seismograms import GFManager
gfm = GFManager(f'/lustre/orion/geo111/scratch/lsawade/STF_SUBSETS/{scardec_id}/subset.h5')
gfm.load()

# %%
# Get Seismograms
def get_ds(gfm: GFManager, cmt: CMTSOLUTION, raw=True):
    array, meta = gfm.get_seismograms(cmt=cmt, raw=raw, array=True)
    new_meta = onp.Meta.from_dict(meta)
    return onp.Dataset(data=array, meta=new_meta)


cmt3_ds_base = get_ds(gfm, cmt3, raw=True)

#%%
# Keep the geometry the same for plotting
cmt3_ds_base.compute_geometry(cmt3.latitude, cmt3.longitude)

# %%
# Download the corresponding data

# Base parameters
tshift = 200.0
starttime = cmt3.origin_time - tshift
sps = 2.0
length_in_s = 3600 + tshift

# Preprocess variables
pre_filt = (0.001, 0.002, 1, 2)

# Define sampling rate as a function of the pre_filt
sampling_rate = pre_filt[3] * 2.5

# Response output
rr_output = "DISP"
water_level = 60

# overwrite data if directory exists by downloading new data
overwrite = False

if not os.path.exists(os.path.join(scardec_stf_dir, 'preprocessed', 'waveforms.mseed')):

    # Make dirs
    if not os.path.exists(os.path.join(scardec_stf_dir, 'preprocessed')):
        os.makedirs(os.path.join(scardec_stf_dir, 'preprocessed'))

    # Remove data directory if it exists and overwrite is True
    if os.path.exists(os.path.join(scardec_stf_dir, 'data')):
        if overwrite:
            shutil.rmtree(os.path.join(scardec_stf_dir, 'data'))

    # If data directory does not exist download the data
    if not os.path.exists(os.path.join(scardec_stf_dir, 'data')):

        # Make parent data directory
        os.makedirs(os.path.join(scardec_stf_dir, 'data'))

        # Move the preprocessed data to data
        networks = ",".join({_sta.split('.')[0] for _sta in cmt3_ds_base.meta.stations.codes})
        stations = ",".join({_sta.split('.')[1] for _sta in cmt3_ds_base.meta.stations.codes})

        # Run download function
        opl.download_data(
            os.path.join(scardec_stf_dir, 'data'),
            starttime=cmt3.origin_time - 300,
            endtime=cmt3.origin_time + 3600 + 300,
            network=networks, station=stations,
            channel_priorities=['BH[ZNE12]', 'HH[ZNE12]', 'LH[ZNE12]'],
            location_priorities=['00', '10', '01']
            )

    # Read the data
    raw = obspy.read(os.path.join(scardec_stf_dir, 'data', 'waveforms/*.mseed'))
    inv = obspy.read_inventory(os.path.join(scardec_stf_dir, 'data', 'stations/*.xml'))

    # Preprocess the data
    onp.preprocess(raw, inv=inv, starttime=starttime-50,
                   sps=sampling_rate, length_in_s=length_in_s+100,
                   water_level=water_level, pre_filt=pre_filt,
                   rr_output=rr_output,
                   interpolate=True,
                   filter=False)

    # Save the preprocessed data
    raw.write(os.path.join(scardec_stf_dir, 'preprocessed', 'waveforms.mseed'), format='MSEED')


# %%

# Read the preprocessed data and the station info
inv = obspy.read_inventory(os.path.join(scardec_stf_dir, 'data', 'stations/*.xml'))
prep = obspy.read(os.path.join(scardec_stf_dir, 'preprocessed', 'waveforms.mseed'))
praw = onp.Dataset.from_stream(prep,
                               components=["N", "E", "Z"],
                               inv=inv,
                               event_latitude=cmt3.latitude,
                               event_longitude=cmt3.longitude)

# %%

obsd_ds = praw.copy()
cmt3_ds = cmt3_ds_base.copy()
gree_ds = cmt3_ds_base.copy()

tshift = 200.0
starttime = cmt3.origin_time - tshift
sps = 2.0
length_in_s = 3600 + tshift

osl.process(obsd_ds, starttime, length_in_s, sps, step=False)
osl.process(cmt3_ds, starttime, length_in_s, sps, step=True)
osl.process(gree_ds, starttime, length_in_s, sps, step=True)


# %%
# Convolve with STF for the respective cmt solution

t = np.arange(0, cmt3_ds.meta.npts * cmt3_ds.meta.delta, cmt3_ds.meta.delta)

cstf = osl.STF.triangle(
    origin=cmt3_stf.origin_time, t=t, tc=cmt3_stf.time_shift,
    tshift=tshift, hdur=cmt3_stf.hdur, M0=cmt3.M0/1e7)

plt.figure(figsize=(8,2.5))
cstf.plot(normalize=False, label='CMT3D+')
plt.xlim(0, 125)
plt.xlabel('Time since origin [s]')
plt.ylabel('N$\cdot$m')
plt.legend(frameon=False, fontsize='small')
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.9)
plt.savefig('peru_data_cmt3d+_stf.pdf', dpi=300)

# %%
# Convolve the CMT3D with the STF but not the Green functions

ds_obsd, ds_gree = obsd_ds.intersection(gree_ds)
_, ds_cmt3 = obsd_ds.intersection(cmt3_ds)

ds_cmt3.convolve(cstf.f/cstf.M0, tshift)

# %%
# Get all stations where the epicentral distance is large than 30 dg

fig = plt.figure()
osl.plot.plot_check_section([ds_obsd, ds_cmt3, ds_gree], component='Z', scale=30.0,
                            labels=['Observed', 'CMT3D+', 'Green Functions'],)
plt.subplots_adjust(left=0.2, right=0.975, top=0.975, bottom=0.1)
fig.set_size_inches(8, 6)
fig.savefig('peru_data_cmt3d_green.pdf')
plt.close(fig)


# %%
# Note that any P is really my definition of any P arrival from taup P
# and that could be P, Pdiff, PKP

phases = ['P', 'S', 'Rayleigh', 'Love', 'anyP', 'anyS']

for phase in phases:

    onp.tt.get_arrivals(cmt3, ds_obsd, phase=phase)
    onp.tt.get_arrivals(cmt3, ds_gree, phase=phase)
    onp.tt.get_arrivals(cmt3, ds_cmt3, phase=phase)

# %%
# Now given the selected traces we want to use the corresponding windows to taper
# the traces such that we can perform the inversion only on the relevant windows
# and not the whole trace.

phase = 'body'
component = 'Z'

# Subselect the seignals based on distance phases etc.
ds_gree_tt = onp.tt.select_traveltime_subset(ds_gree, component=component, phase=phase, maxdist=145.0)

# Get the corresponding observed traces
ds_gree_tt, ds_obsd_tt = ds_gree_tt.intersection(ds_obsd)
_, ds_cmt3_tt = ds_gree_tt.intersection(ds_cmt3)

# Reomve components from observed arrays
ds_obsd_tt = ds_obsd_tt.subset(components=component)
ds_cmt3_tt = ds_cmt3_tt.subset(components=component)

#%%
fig = plt.figure()
osl.plot.plot_check_section([ds_obsd_tt, ds_cmt3_tt, ds_gree_tt], labels=['Observed', 'CMT3D+', 'Green Functions'])
plt.subplots_adjust(left=0.2, right=0.975, top=0.975, bottom=0.1)
fig.set_size_inches(8, 6)
fig.savefig('peru_data_cmt3d_green_tt_select.pdf')
plt.close(fig)


# %%
# Now we want to taper the traces to the selected windows

tds_obsd_tt = onp.tt.taper_dataset(ds_obsd_tt, phase, tshift)
tds_cmt3_tt = onp.tt.taper_dataset(ds_cmt3_tt, phase, tshift)
tds_gree_tt = onp.tt.taper_dataset(ds_gree_tt, phase, tshift)

# %%
fig = plt.figure()
osl.plot.plot_check_section([tds_obsd_tt, tds_cmt3_tt, tds_gree_tt], labels=['Observed', 'CMT3D+', 'Green Functions'],
                   scale=5.0, limits=[5*60,30*60], plot_misfit_reduction=True,)
plt.subplots_adjust(left=0.2, right=0.9, top=0.975, bottom=0.1)
fig.set_size_inches(8, 10)
fig.savefig('peru_data_cmt3d_green_tt_taper.pdf')
plt.close(fig)

# %%

fobsd, fcmt3, idx = onp.utils.remove_misfits(tds_obsd_tt, tds_cmt3_tt)
fgree = tds_gree_tt.subset(stations=idx)


#%%
# Function to plot a beach ball into a specific axis

def plotb(
    x,
    y,
    tensor,
    linewidth=0.25,
    width=100,
    facecolor="k",
    clip_on=True,
    alpha=1.0,
    normalized_axes=True,
    ax=None,
    pdf=False,
    **kwargs,
):

    from matplotlib import transforms
    from obspy.imaging.beachball import beach as obspy_beach

    if normalized_axes or ax is None:
        if ax is None:
            ax = plt.gca()
        pax = opl.axes_from_axes(ax, 948230, extent=[0, 0, 1, 1], zorder=10)
        pax.set_xlim(0, 1)
        pax.set_ylim(0, 1)
        pax.axis("off")
    else:
        if ax is None:
            ax = plt.gca()
        pax = ax
        
    if pdf:
        # This ratio original width * (pdf_dpi / figure_dpi / 2)
        # No idea where the 2 comes from. It's a magic number
        width = width * (72 / 100 / 2)
    else:
        width = width

    # Plot beach ball
    bb = obspy_beach(
        tensor,
        linewidth=linewidth,
        facecolor=facecolor,
        bgcolor="w",
        edgecolor="k",
        alpha=alpha,
        xy=(x, y),
        width=width,
        size=100,  # Defines number of interpolation points
        axes=pax,
        **kwargs,
    )
    bb.set(clip_on=clip_on)

    # # This fixes pdf output issue
    # bb.set_transform(transforms.Affine2D(np.identity(3)))

    pax.add_collection(bb)

# %%

fig = plt.figure()

ax = osl.plot.plot_check_section([fobsd, fcmt3, fgree], labels=['Observed', 'CMT3D+', 'Green Functions'],
                   scale=5.0, start_idx=0, step_idx=1, limits=[0*60,60*60], plot_misfit_reduction=True,
                   legendkwargs=dict(loc='center right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0, 
                                     columnspacing=1.0, fontsize='small'))

subax = opl.axes_from_axes(ax, 12341, [0.0, 1.0, 0.35, 0.075])
cstf.plot(normalize=cstf.M0, lw=0.75, c='tab:red')
# cstf.plot(normalize=gstf.M0, lw=0.75, c='tab:blue')
plt.xlim(0, 125)

# remove left,right and top spines from subax
subax.spines['top'].set_visible(False)
subax.spines['right'].set_visible(False)
subax.spines['left'].set_visible(False)

# offset bottom spine
subax.spines['bottom'].set_position(('outward', 2))

# Make tick labels small
subax.tick_params(which='both', axis='both', labelsize='small')
subax.tick_params(which='both', axis='y', labelleft=False, left=False)

beachax = opl.axes_from_axes(ax, 12342, [-0.2, 1.0, 0.2, 0.075])
beachax.axis('off')
plotb(
        0.5,
        0.5,
        gcmt.tensor,
        linewidth=0.25,
        width=75,
        facecolor='tab:red',
        normalized_axes=True,
        ax=beachax,
        clip_on=False,
        pdf=True
    )
beachax.set_xlim(0,1)
beachax.set_ylim(0,1)

plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
fig.set_size_inches(8, 10)
fig.savefig('peru_data_cmt3d_green_tt_taper_removed_outliers.pdf')
plt.close(fig)


# %%
# Now do the inversion.
















# %%
misfit_gcmt = onp.utils.L2(fobsd, fcmt3)


if True:
    # Misfit reduction due to the SCARDEC STF
    print(f"GCMT: {np.mean(misfit_gcmt):.2f}")
    print(f"SCARDEC: {np.mean(misfit_cmt3):.2f}")
    print(f"Number of stations: {len(fobsd.data)}")
    print(f"Misfit Reduction: {100*(np.mean(misfit_gcmt) - np.mean(misfit_cmt3))/np.mean(misfit_gcmt):.0f}%")
    print(f"Percentage of original misfit: {100*np.mean(misfit_cmt3)/np.mean(misfit_gcmt):.0f}%")

# %%


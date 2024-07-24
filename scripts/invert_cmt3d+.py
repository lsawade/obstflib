# %%
# convolution testing
import sys
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
scardec_stf_dir = './STF/FCTs_20060420_232502_EASTERN_SIBERIA__RUSSIA'
scardec_stf_dir = './STF/FCTs_20181205_041808_SOUTHEAST_OF_LOYALTY_ISLANDS'

# sys.argv = ['./STF/FCTs_20181205_041808_SOUTHEAST_OF_LOYALTY_ISLANDS']

# scardec_stf_dir = sys.argv[1]
scardec_id = scardec_stf_dir.split('/')[-1]



try:
    print(os.path.join(scardec_stf_dir, f'*_CMT3D+'))
    print(glob.glob(os.path.join(scardec_stf_dir, f'*_CMT3D+')))
    cmt3_file = glob.glob(os.path.join(scardec_stf_dir, f'*_CMT3D+'))[0]
except IndexError:
    print(f"No CMT3D+ file found in {scardec_stf_dir}")
    sys.exit()
    

# Which selection: P, Ptrain, S, Strain, Rayleigh, Love, body
phase = 'S'

# For duration of the STF
component = 'T'

outdir = os.path.join('/lustre/orion/geo111/scratch/lsawade', f'{scardec_id}', phase, component)
plotdir = os.path.join(outdir, 'plots')
os.makedirs(plotdir, exist_ok=True)



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
# Create plot dir before plotting

if not os.path.exists(os.path.join(scardec_stf_dir, 'plots')):
    os.makedirs(os.path.join(scardec_stf_dir, 'plots'))
    
# %% 
# Convolve with STF for the respective cmt solution

t = np.arange(0, cmt3_ds.meta.npts * cmt3_ds.meta.delta, cmt3_ds.meta.delta)

cstf = osl.STF.triangle(
    origin=cmt3_stf.origin_time, t=t, tc=cmt3_stf.time_shift,
    tshift=tshift, hdur=cmt3_stf.hdur, M0=cmt3.M0)

plt.figure(figsize=(8,2.5))
cstf.plot(normalize=1e7, label='CMT3D+')
plt.xlim(0, 125)
plt.xlabel('Time since origin [s]')
plt.ylabel('N$\cdot$m')
plt.legend(frameon=False, fontsize='small')
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.9)
plt.savefig(os.path.join(plotdir, 'data_cmt3d+_stf.pdf'), dpi=300)

# %%
# Convolve the CMT3D with the STF but not the Green functions

ds_obsd, ds_gree = obsd_ds.intersection(gree_ds)
_, ds_cmt3 = obsd_ds.intersection(cmt3_ds)

ds_cmt3.convolve(cstf.f/cstf.M0, tshift)

# %%
# Get all stations where the epicentral distance is large than 30 dg

fig = plt.figure()
osl.plot.plot_check_section([ds_obsd, ds_cmt3, ds_gree], component=component, scale=30.0,
                            labels=['Observed', 'CMT3D+', 'Green Functions'],)
plt.subplots_adjust(left=0.2, right=0.975, top=0.975, bottom=0.1)
fig.set_size_inches(8, 6)
fig.savefig(os.path.join(plotdir, 'data_cmt3d_green.pdf'))
plt.close(fig)


# %%
# Note that any P is really my definition of any P arrival from taup P
# and that could be P, Pdiff, PKP

phases = ['P', 'S', 'Rayleigh', 'Love', 'anyP', 'anyS']

for _phase in phases:

    onp.tt.get_arrivals(cmt3, ds_obsd, phase=_phase)
    onp.tt.get_arrivals(cmt3, ds_gree, phase=_phase)
    onp.tt.get_arrivals(cmt3, ds_cmt3, phase=_phase)

# %%
# Now given the selected traces we want to use the corresponding windows to taper
# the traces such that we can perform the inversion only on the relevant windows
# and not the whole trace.

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
osl.plot.plot_check_section([ds_obsd_tt, ds_cmt3_tt, ds_gree_tt], labels=['Observed', 'CMT3D+', 'Green Functions'],
                            component=component)
plt.subplots_adjust(left=0.2, right=0.975, top=0.975, bottom=0.1)
fig.set_size_inches(8, 6)
fig.savefig(os.path.join(plotdir, 'data_cmt3d_green_tt_select.pdf'))
plt.close(fig)


# %%
# Now we want to taper the traces to the selected windows

tds_obsd_tt = onp.tt.taper_dataset(ds_obsd_tt, phase, tshift, gf_shift=-20.0)
tds_cmt3_tt = onp.tt.taper_dataset(ds_cmt3_tt, phase, tshift, gf_shift=-20.0)
tds_gree_tt = onp.tt.taper_dataset(ds_gree_tt, phase, tshift, gf_shift=-125.0)

# %%
fig = plt.figure()
osl.plot.plot_check_section([tds_obsd_tt, tds_cmt3_tt, tds_gree_tt], labels=['Observed', 'CMT3D+', 'Green Functions'],
                   scale=5.0, limits=[5*60,30*60], plot_misfit_reduction=True,component=component)
plt.subplots_adjust(left=0.2, right=0.9, top=0.975, bottom=0.1)
fig.set_size_inches(8, 10)
fig.savefig(os.path.join(plotdir, 'data_cmt3d_green_tt_taper.pdf'))
plt.close(fig)

# %%

# fobsd, fcmt3, idx = onp.utils.remove_misfits(tds_obsd_tt, tds_cmt3_tt,
#                                              misfit_quantile=0.975, ratio_quantile_above=0.95, ratio_quantile_below=0.1)
fobsd, fcmt3, idx = onp.utils.remove_misfits(tds_obsd_tt, tds_cmt3_tt,
                                             misfit_quantile=0.8, ratio_quantile_above=0.9, ratio_quantile_below=0.2)

fgree = tds_gree_tt.subset(stations=idx)

    

# %%
# Plot section with stuff removed!
osl.plot.plot_full_section([fobsd, fcmt3, fgree], ['Observed', 'CMT3D+', 'Green Functions'], [cstf], cmt3, 
                           scale=5.0, limits=[0*60,60*60],component=component, 
                           outfile=os.path.join(plotdir, 'data_cmt3d_green_tt_taper_removed_outliers.pdf'))


osl.plot.plot_full_section([fobsd, fcmt3], ['Observed', 'CMT3D+'], [cstf], cmt3, 
                           scale=5.0, limits=[0*60,60*60], component=component, 
                           outfile=os.path.join(plotdir, 'data_cmt3d_removed_outliers.pdf'))

# %%
# Now do the inversion.

def compute_costs_grads(t, tmaxs, d, G, config, parallel=True, threshold = 0.01,
              integrated=True, integrated_threshold=0.9):
    
    def compute_cost(tmax, t, d, G, config):
        config.update(dict(Tmax=tmax))
        inv1 = osl.Inversion(t, d, G, config=config)
        x = osl.utils.gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
        x = 2 * x / np.sum(x)
        inv1.optimize_smooth_bound0N(x=x)
        return inv1.cost

    if parallel:
        costs = Parallel(n_jobs=10)(delayed(compute_cost)(_tmax, t, d, G, config.copy()) for _tmax in tmaxs)
    else:
        costs = []
        for _tmax in tmaxs:
            costs.append(compute_cost(_tmax, t, d, G, config))
    
    # Compute the gradient of the costs
    grad = np.gradient(costs, tmaxs)
    
    # make all values non-positive
    grad = grad - np.max(grad)

    # normalize, and positivives
    grad = grad / np.min(grad)
    
    # Normalize the costs
    costs = costs - np.min(costs)
    costs = costs / np.max(costs)
    
    return costs, grad

# %%
config = dict(
    Tmin=-10, Tmax=100, knots_per_second=1.0, 
    A=1.0, 
    penalty_weight=10.0,
    smooth_weight=1.0,
    bound_weight=1000.0,
    maxiter=150,
    verbose=False)


tmaxs = np.arange(5, 200, 5)

costs_main, grads_main = compute_costs_grads(fobsd.t, tmaxs, fobsd.data[:, 0,:], fgree.data[:, 0, :], config, parallel=True,
                                             threshold=0.01,
                                             integrated=False, integrated_threshold=0.9975)  

# %%


# Write cost and grad to npz file
tmaxs_main = tmaxs
costs_main = np.array(costs_main)
grads_main = np.array(grads_main)
np.savez(os.path.join(outdir, 'costs_grads_main.npz'), costs=costs_main, grads=grads_main, tmaxs=tmaxs_main)

# %%
tmax_idx = osl.utils.find_Tmax(tmaxs, costs_main, grads_main, Npad=150)

# %%
# Update the dictionary with the new tmax
config = dict(
    Tmin=0, Tmax=tmaxs[tmax_idx], 
    knots_per_second=1.0, 
    A=1.0, 
    penalty_weight=100.0,
    smooth_weight=1.0,
    bound_weight=100.0,
    maxiter=300,
    verbose=True)

idx_shift = int(fgree.meta.delta * 50)
inv = osl.Inversion(fobsd.t, fobsd.data[:,0,:], fgree.data[:,0,:], config=config)
x = osl.utils.gaussn(inv.npknots[: -inv.k - 1], 30, 20)
x = 2*x / np.sum(x)
inv.optimize_smooth_bound0N(x=x)


inv.print_results()


# %%
# Now we want to compute the misfit reduction between the CMT3D+ and B-STF and the SCARDEC STF

scardec = osl.STF.scardecdir(scardec_stf_dir, stftype="optimal")
scardec.interp(t, origin=cmt3.origin_time, tshift=0)
scardec.f = scardec.f * 1e7
scardec.M0 *= 1e7

bstf = osl.STF(t=t, f=inv.construct_f(inv.model)*cmt3.M0, tshift=0, origin=cmt3.origin_time, label='B-STF')
bstf.M0 = cmt3.M0

plt.figure(figsize=(6, 2.5))
bstf.plot(normalize=cstf.M0, lw=1.0, c='tab:blue', label='B-STF')
scardec.plot(normalize=cstf.M0, lw=1.0, c='tab:red', label='SCARDEC')
# plt.xlim(0, 125)
plt.legend()
plt.savefig(os.path.join(plotdir, 'bstf_vs_scardec_stf.pdf'))
plt.close()

# %%
# Plot duration and inversion summary
osl.plot.plot_stf_comparison([cstf, bstf, scardec], ['CMT3D+', 'B-STF','SCARDEC'], tmaxs, tmax_idx, costs_main, grads_main, outfile=os.path.join(plotdir, 'bstf_vs_scardec_stf_duration.pdf'))

#%%

# Create two new datasets
fscar = fgree.copy()
fbstf = fgree.copy()

# Convolve the Green functions with the optimal STF
fscar.convolve(scardec.f/cmt3.M0, 0)
fbstf.convolve(bstf.f/cmt3.M0, 0)

# %%
    
osl.plot.plot_full_section([fobsd, fscar, fbstf], ['Observed', 'SCARDEC', 'B-STF'], [scardec, bstf], cmt3, scale=5.0, limits=[0*60,60*60], component=component, outfile=os.path.join(plotdir, 'data_scardec_bstf.pdf'))

# %%

osl.plot.plot_sub_section([fobsd, fscar, fbstf], ['Observed', 'SCARDEC', 'B-STF'], [scardec, bstf], cmt3, stf_scale=cmt3.M0, scale=5.0, limits=[12.5*60,42.5*60], component=component, outfile=os.path.join(plotdir, 'data_scardec_bstf_sub.pdf'))


# %%
# Station-wise source time function inversion

# New tmaxs range for station wise inversions
# This reduces to overall number of duration length inversions
newtmaxs = np.arange(5, tmaxs[tmax_idx] + 40, 5)


def inversion(i, t, tmaxs, d, G, config):
    
    config['verbose'] = False
    costs, grads = compute_costs_grads(t, tmaxs, d, G, config, parallel=False)
    
    config['Tmax'] = tmaxs[tmax_idx]
    
    print(f"[{i:>03d}]: Tmax = {tmaxs[tmax_idx]:>5.0f}")
    
    inv1 = osl.Inversion(t, d, G, config=config)
    x = osl.utils.gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    inv1.optimize_smooth_bound0N(x=x)
    
    print(f"[{i:>03d}]: Done.")

    return i, inv1, costs, grads


invs = Parallel(n_jobs=25)(delayed(inversion)(
    i, t, newtmaxs, 
    fobsd.data[i:i+1, 0,:],
    fgree.data[i:i+1, 0, :], 
    config.copy()) for i in range(len(fobsd.data)))


# %%
fstfs = fgree.copy()
fcosts = fgree.copy()
fgrads = fgree.copy()
Ntmaxs = len(newtmaxs)
for _inv in invs:
    i, inv1, costs, grads = _inv
    f = inv1.construct_f(inv1.model)
    print(i, np.sum(f))
    fstfs.data[i, 0, :] = f
    fcosts.data[i, 0, :] = 0
    fgrads.data[i, 0, :] = 0
    
    fcosts.data[i, 0, :Ntmaxs] = costs
    fgrads.data[i, 0, :Ntmaxs] = grads
    

fstfs.meta.starttime = fstfs.meta.stations.attributes.origin_time
fcosts.meta.starttime = fcosts.meta.stations.attributes.origin_time + 5.0
fgrads.meta.starttime = fgrads.meta.stations.attributes.origin_time + 5.0
fcosts.meta.delta = 5.0
fgrads.meta.delta = 5.0


# %%

osl.plot.plot_sub_section([fstfs], ['STF'], [bstf], cmt3, scale=5.0, limits=[0,200], component=component, outfile=os.path.join(plotdir, 'stationwise_stfs.pdf'))


#%%
avg_stf = np.mean(fstfs.data[:,0,:], axis=0)
std_stf = np.std(fstfs.data[:,0,:], axis=0)


fig, axes = plt.subplots(3,1, gridspec_kw={'height_ratios': [1, 2, 1]}, figsize=(6, 7), )
ax1 = axes[0]
ax1.plot(bstf.t, bstf.f/bstf.M0, 'k', label='Optimal', zorder=1)
ax1.plot(fstfs.t, avg_stf, label='Mean', zorder=0, c='gray')
ax1.plot(scardec.t, scardec.f/bstf.M0, label='Scardec', zorder=0, c='tab:red')
ax1.fill_between(fstfs.t, np.maximum(0, avg_stf-std_stf), avg_stf+std_stf, alpha=0.5, zorder=-1, label='Std', color='gray')
ax1.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize='small')
ax1.set_xlim(0,np.max(newtmaxs))
opl.plot_label(ax1, '(a)', location=2, box=False)

# Remove top and right spine
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='x', which='both', labelbottom=False)


ax2 = axes[1]
plt.sca(ax2)
osl.plot.plot_check_section([fstfs], ['STF'], scale=50.0, limits=[0,125], fill=True, fillkwargs=dict(alpha=0.75, color='gray', linewidth=0.0), step_idx=1, colors=['w'], azi=True, plot_real_distance=True, lw=0.25,
                            remove_spines=False, component=component)

# Remove top and right spine
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='x', which='both', labelbottom=False)
opl.plot_label(ax2, '(b)', location=2, box=False)

# Set ylabel
ax2.set_ylabel('Azimuth [$^\circ$]')
ax2.set_xlabel('')
plt.xlim(0,np.max(newtmaxs))
plt.ylim(0,399)


ax3 = axes[2]
labelcost = r"$C^{\mathrm{corr}}$"
labelgrad = r"$G^{\mathrm{corr}}$"
avg_cost = np.mean(fcosts.data[:,0,:Ntmaxs], axis=0)
std_cost = np.std(fcosts.data[:,0,:Ntmaxs], axis=0)
avg_grad = np.mean(fgrads.data[:,0,:Ntmaxs], axis=0)
std_grad = np.std(fgrads.data[:,0,:Ntmaxs], axis=0)
ax3.fill_between(newtmaxs, np.maximum(0, avg_cost-std_cost), avg_cost+std_cost, alpha=0.5, zorder=-1, color='gray', linewidth=0.0)
ax3.fill_between(newtmaxs, np.maximum(0, avg_grad-std_grad), avg_grad+std_grad, alpha=0.5, zorder=-1, color='gray', linewidth=0.0)
ax3.plot(newtmaxs, avg_cost, 'k', label=labelcost)
ax3.plot(newtmaxs, avg_grad, 'k--', label=labelgrad)

opl.plot_label(ax3, '(c)', location=2, box=False)
ax3.legend(frameon=False, loc='upper center', ncol=2)
ax3.set_xlim(0,np.max(newtmaxs))
ax3.set_ylim(0,1.0)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_yticks([0, 1], ['0', '1'])
# ax3.tick_params(axis='y', which='both', labelleft=False)
ax3.set_xlabel('Time since origin [s]')

plt.savefig(os.path.join(plotdir, 'stationwise_stfs_all.pdf'))


# %%

# Save the results
np.savez(os.path.join(outdir, 'results.npz'),
    t=fstfs.t,
    optimal_stf=bstf.f,
    stations=fstfs.meta.stations.codes,
    azimuths=fstfs.meta.stations.azimuths,
    distances=fstfs.meta.stations.distances,
    optimal_tmaxs=tmaxs_main,
    optimal_costs=costs_main,
    stationwise_stfs=fstfs.data[:,0,:],
    stationwise_tmaxs=newtmaxs,
    stationwise_costs=fcosts.data[:,0,:]
)
    
    
    
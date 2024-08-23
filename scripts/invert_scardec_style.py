# %%
# convolution testing
import sys
import datetime
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
from gf3d.source import CMTSOLUTION
import obsnumpy as onp
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def log(msg):
    length = 80
    length_msg = len(msg)
    length_right = (length - length_msg) - 1
    if length_right < 0:
        fill = ""
    else:
        fill = "-" * length_right
    print(f'[{datetime.datetime.now()}] {msg} {fill}', flush=True)
    
log("Starting inversion")

# %%
# Files

if ("IPython" in sys.argv[0]) or ("ipython" in sys.argv[0]):
    # scardec_stf_dir = './STF/FCTs_20070815_234057_NEAR_COAST_OF_PERU'
    # scardec_stf_dir = './STF/FCTs_20060420_232502_EASTERN_SIBERIA__RUSSIA'
    # scardec_stf_dir = './STF/FCTs_20181205_041808_SOUTHEAST_OF_LOYALTY_ISLANDS'
    # scardec_stf_dir = './STF/FCTs_19930115_110605_HOKKAIDO__JAPAN_REGION'
    # scardec_stf_dir = './STF/FCTs_19930712_131711_HOKKAIDO__JAPAN_REGION'
    # scardec_stf_dir = './STF/FCTs_19940309_232806_FIJI_ISLANDS_REGION'
    # scardec_stf_dir = './STF/FCTs_20030925_195006_HOKKAIDO__JAPAN_REGION'
    # scardec_stf_dir = './STF/FCTs_20030925_195006_HOKKAIDO__JAPAN_REGION'
    scardec_stf_dir = './STF/FCTs_19951203_180108_KURIL_ISLANDS'
else:
    
    
    # Which directory
    scardec_stf_dir = sys.argv[1]
    
    if len(sys.argv) > 2:
        nocoast = True
    else:
        nocoast = False


phases = ['Ptrain', 'Strain']
components = ['Z', 'T']

# Get the Scardec ID
scardec_id = scardec_stf_dir.split('/')[-1]
region = ' '.join(scardec_stf_dir.split('/')[-1].split('_')[3:])

# Inversion parameters
smooth_weight=[10.0, 10.0]
knots_per_second=0.5

log(f"Scardec ID: {scardec_id}")
log(f"Region: {region}")
log(f"Components: {components}")
log(f"Phases: {phases}")

try:
    gcmt_file = glob.glob(os.path.join(scardec_stf_dir, f'CMTSOLUTION'))[0]
    cmt3_file = glob.glob(os.path.join(scardec_stf_dir, f'*_CMT3D+'))[0]
except IndexError:
    print(f"No CMT3D+ file found in {scardec_stf_dir}")
    sys.exit()

# %%


outdir = os.path.join('/lustre/orion/geo111/scratch/lsawade', 'STF_results_scardec', f'{scardec_id}')
plotdir = os.path.join(outdir, 'plots')
os.makedirs(plotdir, exist_ok=True)



# %%
# Get the event record at II BFO for the Alaska event using the GFManager
gcmt = CMTSOLUTION.read(gcmt_file)
cmt3 = CMTSOLUTION.read(cmt3_file)

cmt3_stf = deepcopy(cmt3)
cmt3.time_shift = 0.0


# %%
log("Loading Green function subset")

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
log("Getting and/or preprocessing the data")

# Base parameters
tshift = 200.0
starttime = cmt3.origin_time - tshift
sps = 2.0
length_in_s = 3600 + tshift

# Preprocess variables
pre_filt = (0.005, 0.006, 1, 2)

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

log("Loading preprocessed data")

# Read the preprocessed data and the station info
inv = obspy.read_inventory(os.path.join(scardec_stf_dir, 'data', 'stations/*.xml'))
prep = obspy.read(os.path.join(scardec_stf_dir, 'preprocessed', 'waveforms.mseed'))
praw = onp.Dataset.from_stream(prep,
                               components=["N", "E", "Z"],
                               inv=inv,
                               event_latitude=cmt3.latitude,
                               event_longitude=cmt3.longitude)

# %%
# Remove zero traces
log("Removing zero traces")
praw = onp.utils.remove_zero_traces(praw)

# %%

log("Processing data")

obsd_ds = praw.copy()
cmt3_ds = cmt3_ds_base.copy()
gree_ds = cmt3_ds_base.copy()

tshift = 200.0
starttime = cmt3.origin_time - tshift
sps = 1.0
length_in_s = 3600 + tshift

log("Processing data")
osl.process(obsd_ds, starttime, length_in_s, sps, step=False)
log("Processing cmt3d+ synthetics")
osl.process(cmt3_ds, starttime, length_in_s, sps, step=True)
log("Processing Green functions")
osl.process(gree_ds, starttime, length_in_s, sps, step=True)


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

for _component, _phase in zip(components, phases):
    fig = plt.figure()
    osl.plot.plot_check_section([ds_obsd, ds_cmt3, ds_gree], component=_component, scale=1.0,
                            labels=['Observed', 'CMT3D+', 'Green Functions'],)
    plt.subplots_adjust(left=0.2, right=0.975, top=0.975, bottom=0.1)
    fig.set_size_inches(8, 6)
    fig.savefig(os.path.join(plotdir, f'{_component}_{_phase}_data_cmt3d_green.pdf'))
    plt.close(fig)


# %%
# Note that any P is really my definition of any P arrival from taup P
# and that could be P, Pdiff, PKP
log("Getting arrivals")

allphases = ['P', 'S', 'Rayleigh', 'Love', 'anyP', 'anyS']

for _phase in allphases:

    onp.tt.get_arrivals(cmt3, ds_obsd, phase=_phase)
    onp.tt.get_arrivals(cmt3, ds_gree, phase=_phase)
    onp.tt.get_arrivals(cmt3, ds_cmt3, phase=_phase)

# %%
# Compute SNR
# log("Computing SNR")
# onp.utils.compute_snr(ds_obsd, tshift, period=17.0)

#%%
# Create SNR figure
def plot_snr(ds, component, phase, plotdir, ds2=None):

    idx = ds.meta.components.index(component)
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(ds.meta.stations.attributes.snr_int[:, idx], ds.meta.stations.attributes.snr_max[:, idx], 'o')
    plt.xlabel('SNR int')
    plt.ylabel('SNR max')
    plt.subplot(2,2,2)
    plt.hist(ds.meta.stations.attributes.snr_max[:, idx])
    plt.xlabel('SNR max')
    plt.subplot(2,2,3)
    plt.hist(ds.meta.stations.attributes.snr_int[:, idx])
    plt.xlabel('SNR int')
    plt.savefig(os.path.join(plotdir, f'{component}_{phase}_snr.pdf'))
    
        
    if ds2 is not None:
        
        fig = plt.figure(figsize=(10,8))
        osl.plot.plot_check_section([ds, ds2], labels=['Observed', 'CMT3D+'],
                        scale=5.0, limits=[0*60,60*60], plot_misfit_reduction=False,component=component,
                        vals=ds.meta.stations.attributes.snr_int[:, idx], valtitle='\mathrm{SNR}_{I}',
                        valformat='{:7.2g}', plot_right_annotations=True )
        plt.subplots_adjust(left=0.2, right=0.9, top=0.975, bottom=0.1)
        fig.set_size_inches(8, 10)
        fig.savefig(os.path.join(plotdir, f'{component}_{phase}_snr_data_cmt3d.pdf'))
        plt.close(fig)

def plot_misfits(ds1, ds2, component, phase, plotdir, label='misfit_data_cmt3d'):
    idx = ds1.meta.components.index(component)
    fig = plt.figure()
    osl.plot.plot_check_section([ds1, ds2], labels=['Observed', 'CMT3D+'],
                    scale=5.0, limits=[0*60,60*60], plot_misfit_reduction=False,component=component,
                    vals=onp.utils.L2(ds1, ds2, normalize=True)[:, idx], valtitle='L^2_N',
                    valformat='{:5.3f}', plot_right_annotations=True )
    plt.subplots_adjust(left=0.2, right=0.9, top=0.975, bottom=0.1)
    fig.set_size_inches(8, 10)
    fig.savefig(os.path.join(plotdir, f'{component}_{phase}_{label}.pdf'))
    plt.close(fig)

# %%
# Now given the selected traces we want to use the corresponding windows to taper
# the traces such that we can perform the inversion only on the relevant windows
# and not the whole trace.
log("Selecting windows")
obsd_tt = []
gree_tt = []
cmt3_tt = []
gree_rat = []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    # Subselect the seignals based on distance phases etc.
    _obsd_tt = onp.tt.select_traveltime_subset(ds_obsd, component=_component, phase=_phase, 
                                               mindist=30.0, maxdist=np.inf, minwindow=300.0)
    log(f"Reduced traces by component/phase selection {_component} {_phase}: {ds_obsd.data.shape[0]} -> {_obsd_tt.data.shape[0]}")
    
    # Remove traces with low SNR
    onp.utils.compute_snr(_obsd_tt, tshift, period=17.0, phase=_phase[0])
    _, _cmt3 =_obsd_tt.intersection(ds_cmt3)
    
    
    plot_snr(_obsd_tt, _component, _phase, plotdir, ds2=_cmt3)
    _obsd_tt_snr, _ = onp.utils.remove_snr(_obsd_tt, snr_int_min_threshold=50.0,snr_int_max_threshold=1000000.0,
                                     snr_max_min_threshold=5.0, snr_max_max_threshold=10000.0,
                                     component=_component)
    log(f"Removing low/high SNR traces {_obsd_tt.data.shape[0]:d} --> {_obsd_tt_snr.data.shape[0]:d}")
    
    # # Subselect the seignals based on distance phases etc.
    # gree_tt.append(onp.tt.select_traveltime_subset(ds_gree_snr, component=_component, phase=_phase, 
    #                                                mindist=30.0, maxdist=np.inf, minwindow=300.0))

    # Get the corresponding observed traces
    # ds_obsd_tt, ds_gree_tt = obsd_tt[_i].intersection(ds_gree)
    _obsd, _gree = _obsd_tt_snr.intersection(ds_gree)
    obsd_tt.append(_obsd)
    gree_tt.append(_gree)
    
    _, _cmt3 = obsd_tt[_i].intersection(ds_cmt3)
    cmt3_tt.append(_cmt3)

    # Remove components from observed arrays
    gree_tt[_i] = gree_tt[_i].subset(components=_component)
    cmt3_tt[_i] = cmt3_tt[_i].subset(components=_component)

#%%
for _i, (_component, _phase) in enumerate(zip(components, phases)):
   
    fig = plt.figure()
    osl.plot.plot_check_section([obsd_tt[_i], cmt3_tt[_i], gree_tt[_i]], labels=['Observed', 'CMT3D+', 'Green Functions'],
                                component=_component)
    plt.subplots_adjust(left=0.2, right=0.975, top=0.975, bottom=0.1)
    fig.set_size_inches(8, 6)
    fig.savefig(os.path.join(plotdir, f'{_component}_{_phase}_data_cmt3d_green_tt_select.pdf'))
    plt.close(fig)


# %%
# Now we want to taper the traces to the selected windows
log("Tapering windows")
# ttt denotes travel, time, taper
obsd_ttt = []
gree_ttt = []
cmt3_ttt = []
taper_ds_cp = [] # cp = component, phase

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    print(_i, type(obsd_tt), type(obsd_tt[_i]))
    
    # Taper datasets, 
    _tds_obsd_tt, _taperds = onp.tt.taper_dataset(obsd_tt[_i], _phase, tshift, gf_shift=-20.0, return_taper=True)
    _tds_cmt3_tt = onp.tt.taper_dataset(cmt3_tt[_i], _phase, tshift, gf_shift=-20.0)
    _tds_gree_tt = gree_tt[_i].copy() 
    
    # Append to lists
    obsd_ttt.append(_tds_obsd_tt)
    gree_ttt.append(_tds_gree_tt)
    cmt3_ttt.append(_tds_cmt3_tt)
    taper_ds_cp.append(_taperds)

# %%
for _i, (_component, _phase) in enumerate(zip(components, phases)):
    print(_i, _component, _phase)
    plot_misfits(obsd_ttt[_i], cmt3_ttt[_i], _component, _phase, plotdir, label='misfit_data_cmt3d_tapered')

# %%
log("Removing outliers")
# fobsd, fcmt3, idx = onp.utils.remove_misfits(tds_obsd_tt, tds_cmt3_tt,
#                                              misfit_quantile=0.975, ratio_quantile_above=0.95, ratio_quantile_below=0.1)
fobsd = []
fcmt3 = []
fgree = []
ftape = []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    # # Get removed indices
    if obsd_ttt[_i].data.shape[0] > 50:
        _fobsd, _fcmt3, _idx = onp.utils.remove_misfits(obsd_ttt[_i], cmt3_ttt[_i], misfit_quantile_threshold=0.85, ratio_quantile_threshold_above=0.85, ratio_quantile_threshold_below=0.15)
    else:
        _fobsd, _fcmt3, _idx = onp.utils.remove_misfits(obsd_ttt[_i], cmt3_ttt[_i], misfit_quantile_threshold=0.9, ratio_quantile_threshold_above=0.9, ratio_quantile_threshold_below=0.1)

    _fgree = gree_ttt[_i].subset(stations=_idx)
    _ftape = taper_ds_cp[_i].subset(stations=_idx)
    
    
    # append
    fobsd.append(_fobsd)
    fcmt3.append(_fcmt3)
    fgree.append(_fgree)
    ftape.append(_ftape)
    
    # Log reduction in traces
    log(f"Reduced traces for {_component} {_phase}: {obsd_ttt[_i].data.shape[0]} -> {fobsd[_i].data.shape[0]}")
    
    

# %%
# Plot section with stuff removed!
for _i, (_component, _phase) in enumerate(zip(components, phases)):
    osl.plot.plot_full_section([fobsd[_i], fcmt3[_i], fgree[_i]], ['Observed', 'CMT3D+', 'Green Functions'], [cstf], cmt3, 
                            scale=2.0, limits=[0*60,60*60],component=_component, 
                            outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_cmt3d_green_tt_taper_removed_outliers.pdf'))


    osl.plot.plot_full_section([fobsd[_i], fcmt3[_i]], ['Observed', 'CMT3D+'], [cstf], cmt3, 
                            scale=2.0, limits=[0*60,60*60], component=_component, 
                            outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_cmt3d_removed_outliers.pdf'))

#%%
# Get the mean and max window length for AIC choice
maxNs = []
meanNs = []
medianNs = []
minNs = []
statNs = []
for _i, (_component, _phase) in enumerate(zip(components, phases)):
    _maxNs = onp.tt.get_max_window(fobsd[_i].meta.stations.attributes.arrivals, phase=_phase) * sps
    _meanNs = onp.tt.get_mean_window(fobsd[_i].meta.stations.attributes.arrivals, phase=_phase) * sps
    _medianNs = onp.tt.get_median_window(fobsd[_i].meta.stations.attributes.arrivals, phase=_phase) * sps
    _minNs = onp.tt.get_min_window(fobsd[_i].meta.stations.attributes.arrivals, phase=_phase) * sps
    
    log(f"{_component} {_phase}")
    log(f"  Max    window length: {_maxNs:>4.0f} samples")
    log(f"  Mean   window length: {_meanNs:>4.0f} samples")
    log(f"  Median window length: {_medianNs:>4.0f} samples")
    log(f"  Min    window length: {_minNs:>4.0f} samples")
    
    # append
    maxNs.append(_maxNs)
    meanNs.append(_meanNs)
    medianNs.append(_medianNs)
    minNs.append(_minNs)
    
    # Get stationwise sampling length
    _start, _end = onp.tt.get_windows(fobsd[_i].meta.stations.attributes.arrivals, phase=_phase)
    statNs.append((_end - _start) * sps)
    
    

    # Ns = np.maximum(minNs, (minNs+meanNs)/2)
Ns = medianNs
log(f"Using median window length")

# %%
# Now do the inversion.
import psutil
def print_mem(p):
    rss = p.memory_info().rss
    print(f"[{p.pid}] memory usage: {rss / 1e6:0.3} MB")

def get_mem(p):
    rss = p.memory_info().rss
    return rss / 1e6

def print_all_mem():
    p = psutil.Process()
    procs = [p] + p.children(recursive=True)
    rss = 0
    for _p in procs:
        rss += _p.memory_info().rss
    print(f"Memory usage: {rss / 1e6:0.3} MB")

def compute_cost(tmax, t, d, G, config, tapers=None, model=None):
    config.update(dict(Tmax=tmax))
    inv1 = osl.Inversion(t, d, G, tapers=tapers, config=config)
    x = osl.utils.gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
    x = 2 * x / np.sum(x)
    if model is not None:
        inv1.optimize_smooth_bound_diff(x=model[:len(x)], x0=model[:len(x)])
    else:
        inv1.optimize_smooth_bound0N(x=x)
    c = deepcopy(inv1.cost)
    del inv1
    return c
    
def compute_costs_grads(t, tmaxs, d, G, config, tapers=None, model=None, parallel=True, verbose=False):
   
    
    if parallel:
        costs = Parallel(n_jobs=30)(delayed(compute_cost)(_tmax, t, d, G, config.copy(), tapers=tapers, model=model) for _tmax in tmaxs)
        costs = np.array(costs)
    else:
        costs = np.zeros(len(tmaxs))
        for _i, _tmax in enumerate(tmaxs):
            # if verbose:
            #     print_all_mem()
                # config['verbose'] = True
            costs[_i] = compute_cost(_tmax, t, d, G, config, model=model)
            if verbose:
                print(f"cost for tmax={_tmax}: {costs[_i]}")
                
    
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


def station_cost_grads(i, t, tmaxs, d, G, config, tapers=None):
    
    if i == 0:
        verbose=True
    else:
        verbose=False
        
    costs, grads = compute_costs_grads(t, tmaxs, d, G, config, tapers=tapers, 
                                       parallel=False, verbose=verbose)
    log(f" -> Station {i:>3d} done.")

    return i, costs, grads

# %%
def find_station_Tmax(i, tmaxs, costs, grads, window_length, knots_per_second, coeff=1.0):
    
    # Find the optimal Tmax
    tmax_idx = osl.utils.find_Tmax_AIC(tmaxs, costs, grads, knots_per_second, window_length, coeff=coeff) 
    
    print(f"[{i:>03d}]: Tmax = {tmaxs[tmax_idx]:>5.0f}")
    
    return i, tmax_idx


def station_inversion(t, d, G, tapers, config):
    
    inv1 = osl.Inversion(t, d, G, tapers=tapers, config=config)
    x = osl.utils.gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    inv1.optimize_smooth_bound0N(x=x)
    
    print(f"[{i:>03d}]: Done.")

    return i, inv1

def station_inversion_diff(t, c, d, G, tapers, config):
    
    inv1 = osl.Inversion(t, d, G, tapers=tapers, config=config)
    x = osl.utils.gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    inv1.optimize_smooth_bound_diff(x=c[:len(x)], x0=c[:len(x)])
    
    print(f"[{i:>03d}]: Done.")

    return i, inv1


# %%

config = dict(
    Tmin=0, Tmax=100, knots_per_second=0.5, 
    A=1.0, 
    penalty_weight=0.0,
    smooth_weight=1.0,
    bound_weight=100.0,
    maxiter=150,
    verbose=False)

min_tmax = 10
dt_tmax = 5
max_tmax = 300 + dt_tmax
tmaxs = np.arange(min_tmax, max_tmax, dt_tmax)

# %%
inv_stat = []
t = fobsd[0].t.copy()

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    log(f"Inverting station-wise {_component} {_phase}")
    _inv = Parallel(n_jobs=20)(delayed(station_cost_grads)(
    # _inv = []
    # for i in range(len(fobsd[_i].data)):
        # _inv.append(station_cost_grads(
            i, t.copy(), tmaxs.copy(), 
            fobsd[_i].data[i:i+1, 0, :].copy(),
            fgree[_i].data[i:i+1, 0, :].copy(), 
            config.copy(), tapers=ftape[_i].data[i:i+1, 0, :].copy()
        # ))
    ) for i in range(len(fobsd[_i].data)))

    inv_stat.append(_inv)

# %%


log("Constructing station-wise STFs")
fstfs = []
fcosts = []
fgrads = []
_tmaxs = []
Ntmaxs = len(tmaxs)
coeffs = [10.0, 10.0]
models = []
for _i, (_component, _phase, _coeff) in enumerate(zip(components, phases, coeffs)):
    
    _fstfs = fgree[_i].copy()
    _fcosts = fgree[_i].copy()
    _fgrads = fgree[_i].copy()
    _fstfs.meta.stations.attributes.tmaxs = np.zeros(len(_fstfs.meta.stations.codes), dtype=np.float32)
    _fstfs.meta.stations.attributes.ints = np.zeros(len(_fstfs.meta.stations.codes), dtype=np.float32)
    
    __tmaxs = []
    
    _c = []
    
    for i, _inv in enumerate(inv_stat[_i]):
        i, costs, grads = _inv
        
        # print(i, costs, grads)
        
        # Getting tmaxs using AIC
        # _, _tmax_idx = find_station_Tmax(i, tmaxs, costs, grads, statNs[_i][i], knots_per_second[_i], coeff=_coeff)
        _, _tmax_idx = find_station_Tmax(i, tmaxs, costs, grads, fobsd[_i].data.shape[2], knots_per_second, coeff=_coeff)
        _fstfs.meta.stations.attributes.tmaxs[i] = tmaxs[_tmax_idx]
        __tmaxs.append(_fstfs.meta.stations.attributes.tmaxs[i])
        
        # Inverting with optimal Tmax
        config['Tmax'] = np.max(tmaxs) #tmaxs[-1] #_fstfs.meta.stations.attributes.tmaxs[i]
        _, inv1 = station_inversion(t, fobsd[_i].data[i:i+1, 0,:], fgree[_i].data[i:i+1, 0, :], ftape[_i].data[i:i+1, 0, :], config)
        f = inv1.construct_f(inv1.model)
        Nm = inv1.model.shape[0]
        _c.append(inv1.model.copy()**2)
        
        # Integrate the value
        _fstfs.meta.stations.attributes.ints[i] = np.trapz(f, dx=inv1.dt)
        
        # Defining the STF, costs, and grads
        _fstfs.data[i, 0, :]  = f
        _fcosts.data[i, 0, :] = 0
        _fgrads.data[i, 0, :] = 0
        
        _fcosts.data[i, 0, :Ntmaxs] = deepcopy(costs)
        _fgrads.data[i, 0, :Ntmaxs] = deepcopy(grads)
        
    
    # Fix timing        
    _fstfs.meta.starttime = _fstfs.meta.stations.attributes.origin_time
    _fcosts.meta.starttime = _fcosts.meta.stations.attributes.origin_time + min_tmax
    _fgrads.meta.starttime = _fgrads.meta.stations.attributes.origin_time + min_tmax
    _fcosts.meta.delta = dt_tmax
    _fgrads.meta.delta = dt_tmax
    
    # Append results
    fstfs.append(_fstfs)
    fcosts.append(_fcosts)
    fgrads.append(_fgrads)
    _tmaxs.append(np.array(__tmaxs))
    models.append(np.sqrt(np.mean(np.vstack(_c), axis=0)))

# %%
for _i, (_component, _phase) in enumerate(zip(components, phases)):
    fig, axes, _ = osl.plot.plot_stationwise(stfss=[fstfs[_i],], limits=(-20,300))
    plt.subplots_adjust(left=0.35/1, right=1-0.1/1, top=0.95, bottom=0.1, wspace=0.1)      
    plt.savefig(os.path.join(plotdir, f'{_component}_{_phase}_stationwise_stfs.pdf'))
    plt.close(fig)
    
# %%
for _i, (_component, _phase) in enumerate(zip(components, phases)):
    _stf_med = np.median(fstfs[_i].meta.stations.attributes.tmaxs)
    _stf_std = np.std(fstfs[_i].meta.stations.attributes.tmaxs)
    _stf_avg = np.mean(fstfs[_i].meta.stations.attributes.tmaxs)
    _stf_min = np.min(fstfs[_i].meta.stations.attributes.tmaxs)
    _stf_max = np.max(fstfs[_i].meta.stations.attributes.tmaxs)
    
    _int_med = np.median(fstfs[_i].meta.stations.attributes.ints)
    _int_std = np.std(fstfs[_i].meta.stations.attributes.ints)
    _int_avg = np.mean(fstfs[_i].meta.stations.attributes.ints)
    _int_min = np.min(fstfs[_i].meta.stations.attributes.ints)
    _int_max = np.max(fstfs[_i].meta.stations.attributes.ints)

    print("Component: ", _component, "Phase: ", _phase)
    print("        Min    | Mean   | Median  | Std    | Max   ")
    print(f"  Tmax: {_stf_min:>6.0f} | {_stf_avg:>6.0f} | {_stf_med:>6.0f} | {_stf_std:>6.0f} | {_stf_max:>6.0f}")
    print(f"  Int:  {_int_min:>6.4f} | {_int_avg:>6.4f} | {_int_med:>6.4f} | {_int_std:>6.4f} | {_int_max:>6.4f}")
    
# Compute average of P and S median and means
A = np.median(fstfs[1].meta.stations.attributes.ints)

ISTF = np.trapz(np.mean(fstfs[1].data[:,0,:], axis=0), dx=fstfs[1].meta.delta)
log(f"Average STF: {ISTF:0.4f}")

#%%
config['penalty_weight'] = 1.0
config['A'] = A
config['smooth_weight'] = 1.0
config['diff_weight'] = 5000.0
fstfs = []
fcosts = []
fgrads = []
_tmaxs = []
coeffs = [1.0, 1.0]
for _i, (_component, _phase, _coeff) in enumerate(zip(components, phases, coeffs)):
    
    _fstfs = fgree[_i].copy()
    _fcosts = fgree[_i].copy()
    _fgrads = fgree[_i].copy()
    _fstfs.meta.stations.attributes.tmaxs = np.zeros(len(_fstfs.meta.stations.codes), dtype=np.float32)
    _fstfs.meta.stations.attributes.ints = np.zeros(len(_fstfs.meta.stations.codes), dtype=np.float32)
    
    __tmaxs = []
    _c = []
    
    for i, _inv in enumerate(inv_stat[_i]):
        i, costs, grads = _inv
        
        # print(i, costs, grads)
        
        # Getting tmaxs using AIC
        # _, _tmax_idx = find_station_Tmax(i, tmaxs, costs, grads, statNs[_i][i], knots_per_second, coeff=_coeff)
        _, _tmax_idx = find_station_Tmax(i, tmaxs, costs, grads, fobsd[_i].data.shape[2], knots_per_second, coeff=_coeff)
        _fstfs.meta.stations.attributes.tmaxs[i] = tmaxs[_tmax_idx]
        __tmaxs.append(_fstfs.meta.stations.attributes.tmaxs[i])
        
        # Inverting with optimal Tmax
        config['Tmax'] = _fstfs.meta.stations.attributes.tmaxs[i]
        # _, inv1 = station_inversion(t, fobsd[_i].data[i:i+1, 0,:], fgree[_i].data[i:i+1, 0, :], ftape[_i].data[i:i+1, 0, :], config)
        _, inv1 = station_inversion_diff(t, models[_i], fobsd[_i].data[i:i+1, 0,:], fgree[_i].data[i:i+1, 0, :], ftape[_i].data[i:i+1, 0, :], config)
        f = inv1.construct_f(inv1.model)
        
        # Pad with zeros to account for different tmaxs
        _c.append(np.pad(inv1.model.copy(), (0, Nm - inv1.model.shape[0]), mode='constant', constant_values=0))
        
        # Integrate the value
        _fstfs.meta.stations.attributes.ints[i] = np.trapz(f, dx=inv1.dt)
        
        # Defining the STF, costs, and grads
        _fstfs.data[i, 0, :] = f
        
        _fcosts.data[i, 0, :] = 0
        _fgrads.data[i, 0, :] = 0
        
        _fcosts.data[i, 0, :Ntmaxs] = deepcopy(costs)
        _fgrads.data[i, 0, :Ntmaxs] = deepcopy(grads)
    
    # Fix timing        
    _fstfs.meta.starttime = _fstfs.meta.stations.attributes.origin_time
    _fcosts.meta.starttime = _fcosts.meta.stations.attributes.origin_time + min_tmax
    _fgrads.meta.starttime = _fgrads.meta.stations.attributes.origin_time + min_tmax
    _fcosts.meta.delta = dt_tmax
    _fgrads.meta.delta = dt_tmax
    
    # Append results
    fstfs.append(_fstfs)
    fcosts.append(_fcosts)
    fgrads.append(_fgrads)
    _tmaxs.append(np.array(__tmaxs))
    models.append(np.sqrt(np.mean(np.vstack(_c)**2, axis=0)))


# %%

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    fig, axes,_ = osl.plot.plot_stationwise(stfss=[fstfs[_i],], limits=(-20,300))
    plt.subplots_adjust(left=0.35/1, right=1-0.1/1, top=0.95, bottom=0.1, wspace=0.1)      
    plt.savefig(os.path.join(plotdir, f'{_component}_{_phase}_stationwise_stfs_fixed.pdf'))
    plt.close(fig)
    
# %%
# Only select STFs that are common to both components
codes = set(fstfs[0].meta.stations.codes).intersection(fstfs[1].meta.stations.codes)
idx0 = np.array([fstfs[0].meta.stations.codes.tolist().index(_code) for _code in codes])
idx1 = np.array([fstfs[1].meta.stations.codes.tolist().index(_code) for _code in codes])

subfstfs = [fstfs[0].subset(stations=idx0), fstfs[1].subset(stations=idx1)]

# %%
# Plot the stationwise STFs
fig, axes,_ = osl.plot.plot_stationwise(stfss=fstfs, limits=(-20,300))
opl.plot_label(axes[0], '(a)', box=False, location=2, fontsize='small')
opl.plot_label(axes[1], '(b)', box=False, location=2, fontsize='small')
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, wspace=0.3)      
plt.savefig(os.path.join(plotdir, f'stationwise_stfs.pdf'))
plt.close(fig)

# %%
# station = 'II.TAU'
# _scost = fcosts[0].subset(stations=station)
# _sgrad = fgrads[0].subset(stations=station)
# _sstf = fstfs[0].subset(stations=station)
# fig, axes = plt.subplots(2,1, sharex=True)
# plt.suptitle(f'Cost check for {station}')
# ax0 = axes[0]
# plt.sca(ax0)
# plt.plot(_scost.t, _scost.data[0,0,:])
# plt.plot(_sgrad.t, _sgrad.data[0,0,:])
# ax1 = axes[1]
# plt.sca(ax1)
# plt.plot(_sstf.t, _sstf.data[0,0,:])
# plt.xlabel('Time [s]')
# plt.xlim(0,200)
# plt.savefig(os.path.join(plotdir, f'cost_check_{station}.pdf'))



# %%
log("Computing costs and gradients for multiple durations")

config = dict(
    Tmin=0, Tmax=100, knots_per_second=0.5,
    A=A,
    penalty_weight=10.0,
    smooth_weight=10.0,
    bound_weight=100.0,
    diff_weight=10000.0,
    maxiter=200,
    verbose=False)

costs_main = []
grads_main = []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    log(f"Computing main costs and gradients for {_component} {_phase}")
    _costs_main, _grads_main = compute_costs_grads(fobsd[_i].t, tmaxs, 
                                                   fobsd[_i].data[:, 0,: ], 
                                                   fgree[_i].data[:, 0, :],
                                                   config, 
                                                   tapers=ftape[_i].data[:,0,:], 
                                                #    model=models[_i],
                                                   parallel=True)
    costs_main.append(_costs_main)
    grads_main.append(_grads_main)
    
    # Write cost and grad to npz file
    _costs_main = np.array(_costs_main)
    _grads_main = np.array(_grads_main)
    np.savez(os.path.join(outdir, f'{_component}_{_phase}_costs_grads_main.npz'), 
            costs=costs_main, grads=grads_main, tmaxs=tmaxs)


# %%
plt.figure()
plt.plot(tmaxs, costs_main[0])
plt.plot(tmaxs, costs_main[1])
plt.xlabel('Tmax [s]')
plt.ylabel('Normalized cost')
plt.savefig(os.path.join(plotdir, 'costs_main.pdf'))


# %%
# Depending on where the x-th percentile is, we can choose the optimal Tmax

#%%
from scipy.integrate import cumtrapz

def plot_cumulative_stf(fstfs, tmaxs, components, phases, plotdir):
    
    plt.figure()
    for _i, (_component, _phase) in enumerate(zip(components, phases)):
        plt.subplot(2,1,_i+1)
        osl.plot.plot_stf_end_pick(fstfs[_i].t, np.mean(fstfs[_i].data[:,0,:], axis=0), tmaxs, label=f'{_component} {_phase}',
                                   exp=True, exp_label_outside=False)        
        plt.legend(frameon=False, ncol=5, bbox_to_anchor=(0.0, 1.0), loc='lower left',
                   borderaxespad=0., fontsize='small')
        
        if _i == 1:
            plt.xlabel('Time [s]')
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(plotdir, 'stf_cumulative.pdf'))
    
plot_cumulative_stf(fstfs, tmaxs, components, phases, plotdir)

# %%
# Setting the AIC coefficient depending on energy in the Average STF

# Take average T component STF and use it for estimating where the majority of the energy is
_stf = np.mean(fstfs[1].data[:,0,:], axis=0)
_total = np.trapz(_stf, dx=fstfs[1].meta.delta)
_STF = cumtrapz(_stf/_total, dx=fstfs[1].meta.delta, initial=0)


# Find the time where 60% of the energy is
_idx40 = np.argmin(np.abs(_STF - 0.40))
_t40 = fstfs[1].t[_idx40]
_idx50 = np.argmin(np.abs(_STF - 0.50))
_t50 = fstfs[1].t[_idx50]
_idx = np.argmin(np.abs(_STF - 0.60))
_t = fstfs[1].t[_idx]

# Set coefficients lower for shorter STFs 
if _t40 <= 30:
    coeffs = [0.025, 0.025]
    
elif _t40 <= 40:
    coeffs = [0.1, 0.1]

elif _t50 <= 50:
    coeffs = [0.2, 0.2]
    
elif _t <= 60:
    coeffs = [0.2, 0.2]
    
elif _t <= 100:
    coeffs = [1.0, 1.0]

elif _t <= 150:
    coeffs = [2.0, 2.0]
    
else:
    coeffs = [4.0, 4.0]

log(f"Setting coefficients to {coeffs[0]:0.2f} and {coeffs[1]:0.2f}")

# %%
# Getting the optimal Tmax
log("Finding Tmax")

# for _i, (_component, _phase) in enumerate(zip(components, phases)):
#     _tmax_idx = osl.utils.find_Tmax_AIC(tmaxs, costs_main[_i], grads_main[_i], knots_per_second, fobsd[_i].data.shape[-1], coeff=coeffs[_i])
#     tmax_idx.append(_tmax_idx)
#     log(f"Inverting with optimal Tmax {_component} {_phase}: {tmaxs[_tmax_idx]:.0f}")
weights = [2/3, 1/3]

# COmpute the weighted average of the stfs
avg_stf = weights[0] * np.mean(fstfs[0].data[:,0,:], axis=0) \
        + weights[1] * np.mean(fstfs[1].data[:,0,:], axis=0)

# Compute the normalize cumulative STF
_total = np.trapz(avg_stf, dx=fstfs[0].meta.delta)
_STF = cumtrapz(avg_stf/_total, dx=fstfs[0].meta.delta, initial=0)

_idx90 = np.argmin(np.abs(_STF - 0.90))
idx = osl.utils.find_elbow_point(fstfs[0].t[:_idx90], 1-_STF[:_idx90]) 
print(idx)
idx += 10/fstfs[0].meta.delta
idx = int(idx)
tmax = fstfs[0].t[idx]

    
log(f"Inverting with optimal Tmax {_component} {_phase}: {tmax:.0f}")



# %%
# Update the dictionary with the new tmax

config = dict(
    Tmin=0, Tmax=100, knots_per_second=0.5,
    A=A,
    penalty_weight=0.1, # 100.0
    smooth_weight=1.0,
    bound_weight=100.0,
    diff_weight=100.0,
    maxiter=200,
    verbose=False)

invs = []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    config ['Tmax'] = tmax # tmaxs[tmax_idx]
    _inv = osl.Inversion(fobsd[_i].t, fobsd[_i].data[:,0,:], fgree[_i].data[:,0,:], config=config, tapers=ftape[_i].data[:,0,:])
    x = osl.utils.gaussn(_inv.npknots[: -_inv.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    # _inv.optimize_smooth_bound_diff(x=x, x0=models[_i][:len(x)])
    _inv.optimize_smooth_bound0N(x=x)
    _inv.print_results()
    
    invs.append(_inv)

# %%
# Combined inversion

config ['Tmax'] = tmax # tmaxs[tmax_idx]

cinvs= []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    _inv = osl.Inversion(fobsd[_i].t, fobsd[_i].data[:,0,:], fgree[_i].data[:,0,:], config=config, tapers=ftape[_i].data[:,0,:])
    cinvs.append(_inv)
    
cinv = osl.CombinedInversion(cinvs, weights)

x = osl.utils.gaussn(_inv.npknots[: -cinv.k - 1], 30, 20)
x = 2*x / np.sum(x)

# cinv.optimize_diff(x=models[1][:len(x)], x0=models[1][:len(x)])# 
cinv.optimize(x=models[1][:len(x)])
cinv.print_results()

# %%
# Now we want to compute the misfit reduction between the CMT3D+ and B-STF and the SCARDEC STF

scardec = osl.STF.scardecdir(scardec_stf_dir, stftype="optimal")
scardec.interp(t, origin=cmt3.origin_time, tshift=0)
scardec.f = scardec.f * 1e7
scardec.M0 *= 1e7

plt.figure(figsize=(6, 2.5))
combstf = osl.STF(t=t, f=cinv.construct_f(cinv.model)*cmt3.M0, tshift=0, origin=cmt3.origin_time, label='CSTF')
combstf.M0 = np.trapz(combstf.f, combstf.t)
combstf.plot(normalize=cstf.M0, lw=1.0, c='tab:blue', label=f'CSTF')
pcstfs = []
colors = ['tab:orange', 'tab:green']
for _i, (_component, _phase) in enumerate(zip(components, phases)):
    pcstf = osl.STF(t=t, f=invs[_i].construct_f(invs[_i].model)*cmt3.M0, tshift=0, origin=cmt3.origin_time, label='B-STF')
    pcstf.M0 = np.trapz(pcstf.f, pcstf.t)
    pcstfs.append(pcstf)
    pcstfs[_i].plot(normalize=cstf.M0, lw=0.5, c=colors[_i], label=f'{_component}/{_phase}')
scardec.plot(normalize=cstf.M0, lw=1.0, c='tab:red', label='SCARDEC')
plt.xlim(0, 300)
plt.legend()
plt.savefig(os.path.join(plotdir, 'bstf_vs_scardec_stf.pdf'))
plt.close()


# %% 
# Plot panel for the misfit functions
fm_scardec = CMTSOLUTION.from_sdr(
        s=scardec.strike1,
        d=scardec.dip1,
        r=scardec.rake1,
        M0=scardec.M0,  # Convert to dyn*cm
        latitude=scardec.latitude,
        longitude=scardec.longitude,
        depth=scardec.depth,
    )

gcmt_stf = osl.STF.triangle(
    origin=cmt3_stf.origin_time, t=t, tc=cmt3_stf.time_shift,
    tshift=tshift, hdur=cmt3_stf.hdur, M0=cmt3.M0)


# %%

if 'sumatra' in scardec_id.lower():
    nocoast = True

osl.plot.plot_summary(combstf, fstfs, pcstfs, gcmt_stf, scardec,
                      tmaxs, fcosts,
                      costs_main,
                      gcmt, cmt3_stf, fm_scardec,
                      knots_per_second,
                      limits=(0, np.max(tmaxs)),
                      coeffs=coeffs,
                      nocoast=nocoast,
                      region=region)


plt.savefig(os.path.join(plotdir, 'stf_summary.pdf'))

plt.close('all')

# %%
# Plotting the data 
for _i, (_component, _phase) in enumerate(zip(components, phases)):
    # Create two new datasets
    fscar = fgree[_i].copy()
    fbstf = fgree[_i].copy()

    # Convolve the Green functions with the optimal STF
    fscar.convolve(scardec.f/cmt3.M0, 0)
    # fbstf.convolve(pcstfs[_i].f/cmt3.M0, 0)
    fbstf.convolve(combstf.f/cmt3.M0, 0)

    # Taper datasets
    fscar  = onp.tt.taper_dataset(fscar, _phase, tshift, gf_shift=-20.0)
    fbstf  = onp.tt.taper_dataset(fbstf, _phase, tshift, gf_shift=-20.0)

    osl.plot.plot_full_section([fobsd[_i], fscar, fbstf], ['Observed', 'SCARDEC', 'B-STF'], [scardec, combstf], cmt3, 
                           scale=5.0, limits=[0*60,60*60], component=_component, 
                           outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_scardec_bstf.pdf'))

# %%

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    # Create two new datasets
    fscar = fgree[_i].copy()
    fbstf = fgree[_i].copy()

    # Convolve the Green functions with the optimal STF
    fscar.convolve(scardec.f/cmt3.M0, 0)
    # fbstf.convolve(pcstfs[_i].f/cmt3.M0, 0)
    
    for _j in range(fstfs[_i].data.shape[0]):
        _stf = osl.STF(t=t, f=fstfs[_i].data[_j,]*cmt3.M0, tshift=0, origin=cmt3.origin_time, label='B-STF')
        _stf.M0 = np.trapz(pcstf.f, pcstf.t)
    
        fbstf.convolve_trace(_j, _stf.f/cmt3.M0, 0)

    # Taper datasets
    fscar  = onp.tt.taper_dataset(fscar, _phase, tshift, gf_shift=-20.0)
    fbstf  = onp.tt.taper_dataset(fbstf, _phase, tshift, gf_shift=-20.0)

    osl.plot.plot_full_section([fobsd[_i], fscar, fbstf], ['Observed', 'SCARDEC', 'B-STF'], [scardec, combstf], cmt3, 
                           scale=5.0, limits=[0*60,60*60], component=_component, 
                           outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_scardec_bstf_tracewise.pdf'))



# %%

# osl.plot.plot_sub_section([fobsd, fscar, fbstf], ['Observed', 'SCARDEC', 'B-STF'], [scardec, bstf], cmt3, 
#                           stf_scale=cmt3.M0, scale=5.0, limits=[12.5*60,52.5*60], component=component, 
#                           outfile=os.path.join(plotdir, 'data_scardec_bstf_sub.pdf'))


# %%
# Station-wise source time function inversion

# New tmaxs range for station wise inversions
# This reduces to overall number of duration length inversions
# newtmaxs = np.arange(5, tmaxs[tmax_idx] + 25, 5)

# %%
# log("Storing all results")

# # Save the results
# np.savez(os.path.join(outdir, 'results.npz'),
#     t=fstfs.t,
#     optimal_stf=bstf.f,
#     stations=fstfs.meta.stations.codes,
#     azimuths=fstfs.meta.stations.azimuths,
#     distances=fstfs.meta.stations.distances,
#     optimal_tmaxs=tmaxs_main,
#     optimal_costs=costs_main,
#     stationwise_stfs=fstfs.data[:,0,:],
#     stationwise_tmaxs=tmaxs,
#     stationwise_costs=fcosts.data[:,0,:]
# )
    
    
    
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

    
osl.utils.log("Starting inversion")

# %%
# Files

if ("IPython" in sys.argv[0]) or ("ipython" in sys.argv[0]):
    # scardec_stf_dir = './STF/FCTs_20070815_234057_NEAR_COAST_OF_PERU'
    # scardec_stf_dir = './STF/FCTs_20060420_232502_EASTERN_SIBERIA__RUSSIA'
    # scardec_stf_dir = './STF/FCTs_20181205_041808_SOUTHEAST_OF_LOYALTY_ISLANDS'
    # scardec_stf_dir = './STF/FCTs_19930115_110605_HOKKAIDO__JAPAN_REGION'
    # scardec_stf_dir = './STF/FCTs_19930712_131711_HOKKAIDO__JAPAN_REGION'
    # scardec_stf_dir = './STF/FCTs_19940309_232806_FIJI_ISLANDS_REGION'
    # scardec_stf_dir = './STF/FCTs_19951203_180108_KURIL_ISLANDS'
    # scardec_stf_dir = './STF/FCTs_19980325_031225_BALLENY_ISLANDS_REGION'
    scardec_stf_dir = './STF/FCTs_20000604_162826_SOUTHERN_SUMATRA__INDONESIA'
    # scardec_stf_dir = './STF/FCTs_20001116_045456_NEW_IRELAND_REGION__P.N.G'
    # scardec_stf_dir = './STF/FCTs_20030925_195006_HOKKAIDO__JAPAN_REGION'
    # scardec_stf_dir = './STF/FCTs_20190526_074115_NORTHERN_PERU'
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

osl.utils.log(f"Scardec ID: {scardec_id}")
osl.utils.log(f"Region: {region}")
osl.utils.log(f"Components: {components}")
osl.utils.log(f"Phases: {phases}")

try:
    gcmt_file = glob.glob(os.path.join(scardec_stf_dir, f'CMTSOLUTION'))[0]
    cmt3_file = glob.glob(os.path.join(scardec_stf_dir, f'*_CMT3D+'))[0]
except IndexError:
    print(f"No CMT3D+ file found in {scardec_stf_dir}")
    sys.exit()

# %%


outdir = os.path.join('/lustre/orion/geo111/scratch/lsawade', 'STF_results_scardec', f'{scardec_id}')
plotdir = os.path.join(outdir, 'plots')
datadir = os.path.join(outdir, 'data')
os.makedirs(plotdir, exist_ok=True)
os.makedirs(datadir, exist_ok=True)


# %%
# Get the event record at II BFO for the Alaska event using the GFManager
gcmt = CMTSOLUTION.read(gcmt_file)
cmt3 = CMTSOLUTION.read(cmt3_file)

cmt3_stf = deepcopy(cmt3)
cmt3.time_shift = 0.0


# %%
osl.utils.log("Loading Green function subset")

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
osl.utils.log("Getting and/or preprocessing the data")

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
    raw = onp.preprocess(raw, inv=inv, starttime=starttime-50,
                   sps=sampling_rate, length_in_s=length_in_s+100,
                   water_level=water_level, pre_filt=pre_filt,
                   rr_output=rr_output,
                   interpolate=True,
                   filter=False)

    # Save the preprocessed data
    
    raw.write(os.path.join(scardec_stf_dir, 'preprocessed', 'waveforms.mseed'), format='MSEED',
              encoding='FLOAT64')


# %%

osl.utils.log("Loading preprocessed data")

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
osl.utils.log("Removing zero traces")
praw = onp.utils.remove_zero_traces(praw)

# %%

osl.utils.log("Processing data")

obsd_ds = praw.copy()
cmt3_ds = cmt3_ds_base.copy()
gree_ds = cmt3_ds_base.copy()

tshift = 200.0
starttime = cmt3.origin_time - tshift
sps = 1.0
length_in_s = 3600 + tshift

osl.utils.log("Processing data")
osl.process(obsd_ds, starttime, length_in_s, sps, step=False)
osl.utils.log("Processing cmt3d+ synthetics")
osl.process(cmt3_ds, starttime, length_in_s, sps, step=True)
osl.utils.log("Processing Green functions")
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
osl.utils.log("Getting arrivals")

allphases = ['P', 'S', 'Rayleigh', 'Love', 'anyP', 'anyS']

for _phase in allphases:

    onp.tt.get_arrivals(cmt3, ds_obsd, phase=_phase)
    onp.tt.get_arrivals(cmt3, ds_gree, phase=_phase)
    onp.tt.get_arrivals(cmt3, ds_cmt3, phase=_phase)

# %%
# Compute SNR
# osl.utils.log("Computing SNR")
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
osl.utils.log("Selecting windows")
obsd_tt = []
gree_tt = []
cmt3_tt = []
gree_rat = []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    # Subselect the seignals based on distance phases etc.
    _obsd_tt = onp.tt.select_traveltime_subset(ds_obsd, component=_component, phase=_phase, 
                                               mindist=30.0, maxdist=np.inf, minwindow=300.0)
    osl.utils.log(f"Reduced traces by component/phase selection {_component} {_phase}: {ds_obsd.data.shape[0]} -> {_obsd_tt.data.shape[0]}")
    
    # Remove traces with low SNR
    onp.utils.compute_snr(_obsd_tt, tshift, period=17.0, phase=_phase[0])
    _, _cmt3 =_obsd_tt.intersection(ds_cmt3)
    
    
    plot_snr(_obsd_tt, _component, _phase, plotdir, ds2=_cmt3)
    _obsd_tt_snr, _ = onp.utils.remove_snr(_obsd_tt, snr_int_min_threshold=50.0,snr_int_max_threshold=1000000.0,
                                     snr_max_min_threshold=5.0, snr_max_max_threshold=10000.0,
                                     component=_component)
    osl.utils.log(f"Removing low/high SNR traces {_obsd_tt.data.shape[0]:d} --> {_obsd_tt_snr.data.shape[0]:d}")
    
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
osl.utils.log("Tapering windows")
# ttt denotes travel, time, taper
obsd_ttt = []
gree_ttt = []
cmt3_ttt = []
taper_ds_cp = [] # cp = component, phase

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
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
    plot_misfits(obsd_ttt[_i], cmt3_ttt[_i], _component, _phase, plotdir, label='misfit_data_cmt3d_tapered')

# %%
osl.utils.log("Removing outliers")
# fobsd, fcmt3, idx = onp.utils.remove_misfits(tds_obsd_tt, tds_cmt3_tt,
#                                              misfit_quantile=0.975, ratio_quantile_above=0.95, ratio_quantile_below=0.1)
fobsd = []
fcmt3 = []
fgree = []
ftape = []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    osl.utils.log("Removing outliers based on ratio and L2 norm misfit")
    osl.utils.log(f"Component: {_component} -- Phase: {_phase}")
    osl.utils.log("===================================================")
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
    osl.utils.log(f"-----|| {obsd_ttt[_i].data.shape[0]} -> {fobsd[_i].data.shape[0]}")
    osl.utils.log("////")
    

# %%
# Plot section with stuff removed!
for _i, (_component, _phase) in enumerate(zip(components, phases)):
    osl.plot.plot_full_section([fobsd[_i], fcmt3[_i], fgree[_i]], ['Observed', 'CMT3D+', 'Green Functions'], [cstf], cmt3, 
                            scale=2.0, limits=[0*60,60*60],component=_component, 
                            outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_cmt3d_green_tt_taper_removed_outliers.pdf'))


    osl.plot.plot_full_section([fobsd[_i], fcmt3[_i]], ['Observed', 'CMT3D+'], [cstf], cmt3, 
                            scale=2.0, limits=[0*60,60*60], component=_component, 
                            outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_cmt3d_removed_outliers.pdf'))

# %%
def station_inversion(i, t, d, G, tapers, config):
    
    inv1 = osl.Inversion(t, d, G, tapers=tapers, config=config)
    x = osl.utils.gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    inv1.optimize_smooth_bound0N(x=x)
    
    print(f"[{i:>03d}]: Done.")

    return i, inv1

def station_inversion_diff_tmax(i, t, c, d, G, tapers, config):
    
    inv1 = osl.Inversion(t, d, G, tapers=tapers, config=config)
    x = osl.utils.gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    inv1.optimize_smooth_bound_diff(x=c[:len(x)], x0=c[:len(x)])
    
    print(f"[{i:>03d}]: Done.")

    return i, inv1


def station_inversion_diff(i, t, c, d, G, tapers, config):
    
    inv1 = osl.Inversion(t, d, G, tapers=tapers, config=config)
    x = osl.utils.gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    inv1.optimize_smooth_bound_diff(x=c[:len(x)], x0=c[:len(x)])
    
    print(f"[{i:>03d}]: Done.")

    return i, inv1

# %%
tmax_main = 300.0

config = dict(
    Tmin=0, Tmax=tmax_main, knots_per_second=0.5, 
    A=1.0, 
    penalty_weight=0.0,
    smooth_weight=10.0,
    bound_weight=100.0,
    maxiter=150,
    verbose=False)

# %%

osl.utils.log("Constructing station-wise STFs -- 1st pass")
fstfs_first = []
models_first_pass = []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    _fstfs = fgree[_i].copy()
    _fstfs.meta.stations.attributes.tmaxs = np.zeros(len(_fstfs.meta.stations.codes), dtype=np.float32)
    _fstfs.meta.stations.attributes.ints = np.zeros(len(_fstfs.meta.stations.codes), dtype=np.float32)
       
    _c = []
    
        
    # Inverting with optimal Tmax
    invs = Parallel(n_jobs=20)(delayed(station_inversion)(i, t, fobsd[_i].data[i:i+1, 0,:], fgree[_i].data[i:i+1, 0, :], ftape[_i].data[i:i+1, 0, :], config) for i in range(fobsd[_i].data.shape[0]))
        
    for i, _inv in enumerate(invs):
        _, inv1 = _inv
        f = inv1.construct_f(inv1.model)
        Nm = inv1.model.shape[0]
        _c.append(inv1.model.copy())
        
        # Integrate the value
        _fstfs.meta.stations.attributes.ints[i] = np.trapz(f, dx=inv1.dt)
        
        # Defining the STF, costs, and grads
        _fstfs.data[i, 0, :]  = f
    
    # Fix timing        
    _fstfs.meta.starttime = _fstfs.meta.stations.attributes.origin_time
    
    # Append results
    fstfs_first.append(_fstfs)
    models_first_pass.append(np.sqrt(np.mean(np.vstack(_c)**2, axis=0)))

# %%
for _i, (_component, _phase) in enumerate(zip(components, phases)):
    fig, axes, _ = osl.plot.plot_stationwise(stfss=[fstfs_first[_i],], limits=(-20,300))
    plt.subplots_adjust(left=0.35/1, right=1-0.1/1, top=0.95, bottom=0.1, wspace=0.1)      
    plt.savefig(os.path.join(plotdir, f'{_component}_{_phase}_stationwise_stfs.pdf'))
    plt.close(fig)
    
# %%
for _i, (_component, _phase) in enumerate(zip(components, phases)):
   
    _int_med = np.median(fstfs_first[_i].meta.stations.attributes.ints)
    _int_std = np.std(fstfs_first[_i].meta.stations.attributes.ints)
    _int_avg = np.mean(fstfs_first[_i].meta.stations.attributes.ints)
    _int_min = np.min(fstfs_first[_i].meta.stations.attributes.ints)
    _int_max = np.max(fstfs_first[_i].meta.stations.attributes.ints)

    osl.utils.log(f"Component: { _component} -- Phase: {_phase}")
    osl.utils.log("        Min    | Mean   | Median  | Std    | Max   ")
    osl.utils.log(f"  Int:  {_int_min:>6.4f} | {_int_avg:>6.4f} | {_int_med:>6.4f} | {_int_std:>6.4f} | {_int_max:>6.4f}")
    
# Compute average of P and S median and means
A = np.median(fstfs_first[1].meta.stations.attributes.ints)

ISTF = np.trapz(np.mean(fstfs_first[1].data[:,0,:], axis=0), dx=fstfs_first[1].meta.delta)
osl.utils.log(f"Average STF: {ISTF:0.4f}")
osl.utils.log(f"Average Int: {A:0.4f}")

#%%

osl.utils.log("Constructing station-wise STFs -- 2nd pass")

config['penalty_weight'] = 1.0
config['A'] = A
config['smooth_weight'] = 1.0
config['diff_weight'] = 5000.0
fstfs = []


models_second_pass = []
diff_weights = [1250.0, 10000.0]
for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    config['diff_weight'] = diff_weights[_i]
    
    _fstfs = fgree[_i].copy()
    _fstfs.meta.stations.attributes.tmaxs = np.zeros(len(_fstfs.meta.stations.codes), dtype=np.float32)
    _fstfs.meta.stations.attributes.ints = np.zeros(len(_fstfs.meta.stations.codes), dtype=np.float32)
    
    _c = []
    # Inverting with optimal Tmax
    invs = Parallel(n_jobs=20)(delayed(station_inversion_diff)(i, t, models_first_pass[_i], fobsd[_i].data[i:i+1, 0,:], fgree[_i].data[i:i+1, 0, :], ftape[_i].data[i:i+1, 0, :], config) for i in range(fobsd[_i].data.shape[0]))
        
    for i, _inv in enumerate(invs):
        _, inv1 = _inv
        f = inv1.construct_f(inv1.model)
        Nm = inv1.model.shape[0]
        _c.append(inv1.model.copy())
        
        # Integrate the value
        _fstfs.meta.stations.attributes.ints[i] = np.trapz(f, dx=inv1.dt)
        
        # Defining the STF, costs, and grads
        _fstfs.data[i, 0, :]  = f
        
    # Fix timing        
    _fstfs.meta.starttime = _fstfs.meta.stations.attributes.origin_time
  
    # Append results
    fstfs.append(_fstfs)
    models_second_pass.append(np.sqrt(np.mean(np.vstack(_c)**2, axis=0)))


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
fig, axes,_ = osl.plot.plot_stationwise(stfss=fstfs, limits=(-20,300), plot_tmaxs=False)
opl.plot_label(axes[0], '(a)', box=False, location=2, fontsize='small')
opl.plot_label(axes[1], '(b)', box=False, location=2, fontsize='small')
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, wspace=0.3)      
plt.savefig(os.path.join(plotdir, f'stationwise_stfs.pdf'))
plt.close(fig)

#%%
from scipy.integrate import cumtrapz

def plot_cumulative_stf(fstfs, components, phases, plotdir):
    
    plt.figure()
    for _i, (_component, _phase) in enumerate(zip(components, phases)):
        plt.subplot(2,1,_i+1)
        osl.plot.plot_stf_end_pick(fstfs[_i].t, np.mean(fstfs[_i].data[:,0,:], axis=0), label=f'{_component} {_phase}',
                                   extension=True, label_outside=False)        
        plt.legend(frameon=False, ncol=6, bbox_to_anchor=(0.0, 1.0), loc='lower left',
                   borderaxespad=0., fontsize='small')
        if _i == 1:
            plt.xlabel('Time [s]')
        
        plt.ylim(-0.05, 1.05)
        plt.axhline(0.0, c=(0.7,0.7,0.7), ls='-', lw=1.0, zorder=-1)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(plotdir, 'stf_cumulative.pdf'))
    
plot_cumulative_stf(fstfs, components, phases, plotdir)
plt.close('all')

# %%
# Getting the optimal Tmax
osl.utils.log("Finding Tmax using the integrated STF and the elbow method")

weights = [2/3, 1/3]

# COmpute the weighted average of the stfs
avg_stf = weights[0] * np.mean(fstfs[0].data[:,0,:], axis=0) \
        + weights[1] * np.mean(fstfs[1].data[:,0,:], axis=0)
      
# Compute the normalize cumulative STF
_total = np.trapz(avg_stf, dx=fstfs[0].meta.delta)
_STF = cumtrapz(avg_stf/_total, dx=fstfs[0].meta.delta, initial=0)


_idx, _, _, long_stf = osl.utils.find_cond_elbow_idx(t, _STF)
idx = osl.utils.find_elbow_point(fstfs[0].t[:_idx], 1-_STF[:_idx]) 
idx += 10/fstfs[0].meta.delta
idx = int(idx)
tmax = fstfs[0].t[idx]

osl.utils.log(f"Inverting with optimal Tmax {_component} {_phase}: {tmax:.0f}")
# %%

# Plot the stf pick plot separately
osl.plot.plot_stf_end_pick(fstfs[0].t, avg_stf, label='', 
                        extension=True, label_outside=False)        
plt.legend(frameon=False, ncol=6, bbox_to_anchor=(0.0, 1.0), loc='lower left',
        borderaxespad=0., fontsize='small')
if _i == 1:
    plt.xlabel('Time [s]')
    
if long_stf:
    opl.plot_label(plt.gca(), f'LSTF', fontsize='small', box=False, location=18, dist=0.0)
    
plt.ylim(-0.05, 1.05)
plt.axhline(0.0, c=(0.7,0.7,0.7), ls='-', lw=1.0, zorder=-1)
plt.savefig(os.path.join(plotdir, 'stf_cumulative_choice.pdf'))
plt.close('all')

# %%

osl.plot.plot_elbow_point_selection(fstfs[0].t, avg_stf, label=f'', 
                        extension=True, label_outside=False)    

plt.savefig(os.path.join(plotdir, 'stf_cumulative_choice_elbow.pdf'))
plt.close('all')
# %%

osl.utils.log("Constructing station-wise STFs -- 3rd pass with tmax")



def station_inversion_diff_tmax_bu(i, t, c, d, G, tapers, config, tmax, stf):
    
    # Get unique tmax
    from scipy.integrate import cumtrapz
    STF = cumtrapz(stf, dx=t[1]-t[0], initial=0)
    
    # Clip min and clip max
    clipmin = np.maximum(10, tmax-25)
    clipmax = np.minimum(300, tmax+50)
    
    # Somtimes finding new tmax fails then simply use the old one
    try:
        # Very relaxed thresholds
        _tmax = osl.utils.find_tmax(t, STF, A_exp_thresh=3.0, A_log_thresh=5.0, B_exp_thresh=-10, B_log_thresh=-10,
                                    extra_long=True)[0]
    except RuntimeError:
        _tmax = False
        
    # Making sure that the tmax is within the clipmin and clipmax and tmax if not found
    if not _tmax:
        __tmax = tmax
    else:
        if _tmax < clipmin or _tmax > clipmax:
            __tmax = tmax
        else:
            __tmax = _tmax
        
    
    # Actual inversion
    config = deepcopy(config)
    config['Tmax'] = __tmax
    inv1 = osl.Inversion(t, d, G, tapers=tapers, config=config)
    x = osl.utils.gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    x0 = c[:len(x)]
    x0[-1] = 0 # Setting the last value to zero to mak sure bound and diff penalties aren't interfering
    inv1.optimize_smooth_bound_diff(x=c[:len(x)], x0=x0)
    
    
    print(f"[{i:>03d}]: Done. Tmax: {tmax:.0f} -- station tmax: {_tmax} --clip--> {__tmax:.0f}")

    return i, inv1, __tmax


def station_inversion_diff_tmax(i, t, c, d, G, tapers, config, tmax, stf):
    
    # Get unique tmax
    from scipy.integrate import cumtrapz
    STF = cumtrapz(stf, dx=t[1]-t[0], initial=0)
    
    # Clip min and clip max
    clipmin = np.maximum(10, tmax-25)
    clipmax = np.minimum(300, tmax+35)
    idxmin = np.maximum(0,np.argmin( np.abs( t - (tmax-25))))
    idxmax = np.argmin( np.abs( t - (tmax+50)))
    
    # Somtimes finding new tmax fails then simply use the old one
    try:
        # Very relaxed thresholds
        idx = osl.utils.find_elbow_point(t[idxmin:idxmax], STF[idxmin:idxmax]) + idxmin
        
        idx = int(idx)
        _tmax = t[idx]
                
        # _tmax = osl.utils.find_tmax(t, STF, A_exp_thresh=3.0, A_log_thresh=5.0, B_exp_thresh=-10, B_log_thresh=-10,
        #                             extra_long=True)[0]
    except RuntimeError:
        _tmax = False
        
    # Making sure that the tmax is within the clipmin and clipmax and tmax if not found
    if not _tmax:
        __tmax = tmax
    else:
        if _tmax < clipmin or _tmax > clipmax:
            __tmax = tmax
        else:
            __tmax = _tmax
        
    
    # Actual inversion
    config = deepcopy(config)
    config['Tmax'] = __tmax
    inv1 = osl.Inversion(t, d, G, tapers=tapers, config=config)
    x = osl.utils.gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    x0 = c[:len(x)]
    x0[-1] = 0 # Setting the last value to zero to mak sure bound and diff penalties aren't interfering
    inv1.optimize_smooth_bound_diff(x=c[:len(x)], x0=x0)
    
    
    print(f"[{i:>03d}]: Done. Tmax: {tmax:.0f} -- station tmax: {_tmax} --clip--> {__tmax:.0f}")

    return i, inv1, __tmax


# Here adding 25 sceonds to the STF to account for station by station STF variability
config = dict(
    Tmin=0, Tmax=tmax, knots_per_second=0.5,
    A=A,
    penalty_weight=1.0, # 100.0
    smooth_weight=.5,
    bound_weight=100.0,
    diff_weight=1250.0, #5000.0
    maxiter=200,
    verbose=False)

fstfs_tmax = []
tmaxs = []
models_third_pass = []
diff_weights = [1250.0, 5000.0]
for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    config['diff_weight'] = diff_weights[_i]
    
    _fstfs_tmax = fgree[_i].copy()
    _fstfs_tmax.meta.stations.attributes.tmaxs = np.zeros(len(_fstfs_tmax.meta.stations.codes), dtype=np.float32)
    _fstfs_tmax.meta.stations.attributes.ints = np.zeros(len(_fstfs_tmax.meta.stations.codes), dtype=np.float32)
    
    # Inverting with optimal Tmax
    invs = Parallel(n_jobs=20)(delayed(station_inversion_diff_tmax)(i, t, models_second_pass[_i], fobsd[_i].data[i:i+1, 0,:], fgree[_i].data[i:i+1, 0, :], ftape[_i].data[i:i+1, 0, :], config, tmax, fstfs[_i].data[i,0,:]) for i in range(fobsd[_i].data.shape[0]))
        
    _c = []
    for i, _inv in enumerate(invs):
        _, inv1, _tmax = _inv
        f = inv1.construct_f(inv1.model)
        Nm = inv1.model.shape[0]
        _c.append(inv1.model.copy())
        
        # Integrate the value
        _fstfs_tmax.meta.stations.attributes.ints[i] = np.trapz(f, dx=inv1.dt)
        
        # Defining the STF, costs, and grads
        _fstfs_tmax.data[i, 0, :]  = f
        _fstfs_tmax.meta.stations.attributes.tmaxs[i] = _tmax
        
    # Fix timing        
    _fstfs_tmax.meta.starttime = _fstfs_tmax.meta.stations.attributes.origin_time
  
    # Append results
    fstfs_tmax.append(_fstfs_tmax)
    
    # Pad models without many parameters
    Nm_max = np.max([_c[_i].shape[0] for _i in range(len(_c))])
    _c = [np.pad(_c[_i], (0, Nm_max - _c[_i].shape[0]), mode='constant') for _i in range(len(_c))]
    models_third_pass.append(np.sqrt(np.mean(np.vstack(_c)**2, axis=0)))

# Compute average of P and S median and means
A_tmax = np.median(fstfs_tmax[1].meta.stations.attributes.ints)
ISTF_tmax = np.trapz(np.mean(fstfs_tmax[1].data[:,0,:], axis=0), dx=fstfs_tmax[1].meta.delta)

osl.utils.log(f"Average STF: {ISTF:0.4f} --tmax--> {ISTF_tmax:0.4f}")
osl.utils.log(f"Average Int: {A:0.4f} --tmax--> {A_tmax:0.4f}")

# %%

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    fig, axes,_ = osl.plot.plot_stationwise(stfss=[fstfs_tmax[_i],], limits=(-20,300))
    plt.subplots_adjust(left=0.35/1, right=1-0.1/1, top=0.95, bottom=0.1, wspace=0.1)      
    plt.savefig(os.path.join(plotdir, f'{_component}_{_phase}_stationwise_stfs_fixed_tmax.pdf'))
    plt.close(fig)
    
# %%
# Only select STFs that are common to both components
codes = set(fstfs_tmax[0].meta.stations.codes).intersection(fstfs_tmax[1].meta.stations.codes)
idx0 = np.array([fstfs[0].meta.stations.codes.tolist().index(_code) for _code in codes])
idx1 = np.array([fstfs[1].meta.stations.codes.tolist().index(_code) for _code in codes])

subfstfs_tmax = [fstfs_tmax[0].subset(stations=idx0), fstfs_tmax[1].subset(stations=idx1)]

# %%
# Plot the stationwise STFs
fig, axes,_ = osl.plot.plot_stationwise(stfss=fstfs_tmax, limits=(-20,300), plot_tmaxs=False)
opl.plot_label(axes[0], '(a)', box=False, location=2, fontsize='small')
opl.plot_label(axes[1], '(b)', box=False, location=2, fontsize='small')
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, wspace=0.3)      
plt.savefig(os.path.join(plotdir, f'stationwise_stfs_tmax.pdf'))
plt.close(fig)

# %%
# Update the dictionary with theA new tmax

config = dict(
    Tmin=0, Tmax=tmax , knots_per_second=0.5,
    A=A_tmax,
    penalty_weight=0.25, # 100.0
    smooth_weight=5.0,
    bound_weight=100.0,
    diff_weight=10.0,
    maxiter=200,
    verbose=False)

invs = []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    # osl.utils.find_tmax()
    
    _inv = osl.Inversion(fobsd[_i].t, fobsd[_i].data[:,0,:], fgree[_i].data[:,0,:], config=config, tapers=ftape[_i].data[:,0,:])
    x = osl.utils.gaussn(_inv.npknots[: -_inv.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    # _inv.optimize_smooth_bound_diff(x=x, x0=models[_i][:len(x)])
    _inv.optimize_smooth_bound0N(x=x)
    _inv.print_results()
    
    invs.append(_inv)

# %%
# Combined inversion

cinvs= []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    _inv = osl.Inversion(fobsd[_i].t, fobsd[_i].data[:,0,:], fgree[_i].data[:,0,:], config=config, tapers=ftape[_i].data[:,0,:])
    cinvs.append(_inv)
    
cinv = osl.CombinedInversion(cinvs, weights)

_model = weights[0] * models_second_pass[0] + weights[1] * models_second_pass[1]
_model = _model[:len(x)]
_model[-1] = 0.0
_model[0] = 0.0

x = osl.utils.gaussn(_inv.npknots[: -cinv.k - 1], 30, 20)
x = 2*x / np.sum(x)

# cinv.optimize_diff(x=models[1][:len(x)], x0=models[1][:len(x)])# 
cinv.optimize(x=_model[:len(x)])
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
# Plotting the data 
misfits_phase = []

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

    # Compute combined misfit
    m_scardec = onp.utils.L2(fobsd[_i], fscar, normalize=True)
    m_bstf = onp.utils.L2(fobsd[_i], fbstf, normalize=True)
        
    misfits_phase.append([m_scardec, m_bstf])    

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

if 'sumatra' in scardec_id.lower() or 'java' in scardec_id.lower():
    nocoast = True
else:
    nocoast = False

osl.plot.plot_summary(combstf, fstfs, fstfs_tmax, pcstfs, gcmt_stf, scardec,
                      gcmt, cmt3_stf, fm_scardec,
                      knots_per_second,
                      limits=(0, tmax_main),
                      nocoast=nocoast,
                      region=region, 
                      misfits=misfits_phase)


plt.savefig(os.path.join(plotdir, 'stf_summary.pdf'))

plt.close('all')


# %%
# Save the results

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    stationwise_file = os.path.join(datadir, f'{_component}_{_phase}_stationwise')
    fstfs[_i].write(stationwise_file + '.npy', stationwise_file + '.json')

    stationwise_file_tmax = os.path.join(datadir, f'{_component}_{_phase}_stationwise_tmax')
    fstfs_tmax[_i].write(stationwise_file_tmax + '.npy', stationwise_file_tmax + '.json')
    
    # Optimal component wise STFs.

# Optimal STFs 
optimal_file = os.path.join(datadir, f'optimal_PDE_{gcmt.origin_time.isoformat()}.txt')
outdata = np.vstack((t, combstf.f/1e7))
np.savetxt(optimal_file, outdata.T, fmt='%.6e', delimiter=' ', header='Time since origin [s]  STF [N.m]')


# %%
# osl.utils.log("Storing all results")

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
    
    
    
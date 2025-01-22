# %%
# convolution testing
import json
import sys
import datetime
import os
import glob
from copy import deepcopy
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
from scipy.integrate import cumtrapz

    
osl.utils.log("Starting inversion")

# %%
# Files

if ("IPython" in sys.argv[0]) or ("ipython" in sys.argv[0]):
    # scardec_stf_dir = './STF/FCTs_19930115_110605_HOKKAIDO__JAPAN_REGION'
    # scardec_stf_dir = './STF/FCTs_19930712_131711_HOKKAIDO__JAPAN_REGION'
    # scardec_stf_dir = './STF/FCTs_19930608_130336_NEAR_EAST_COAST_OF_KAMCHATKA'
    # scardec_stf_dir = './STF/FCTs_19940309_232806_FIJI_ISLANDS_REGION'
    # scardec_stf_dir = './STF/FCTs_19951203_180108_KURIL_ISLANDS'
    # scardec_stf_dir = './STF/FCTs_19980325_031225_BALLENY_ISLANDS_REGION'
    # scardec_stf_dir = './STF/FCTs_20000604_162826_SOUTHERN_SUMATRA__INDONESIA'
    # scardec_stf_dir = './STF/FCTs_20001116_045456_NEW_IRELAND_REGION__P.N.G'
    # scardec_stf_dir = './STF/FCTs_20021010_105020_IRIAN_JAYA_REGION__INDONESIA'
    # scardec_stf_dir = './STF/FCTs_20030925_195006_HOKKAIDO__JAPAN_REGION'
    # scardec_stf_dir = './STF/FCTs_20060420_232502_EASTERN_SIBERIA__RUSSIA'
    # scardec_stf_dir = './STF/FCTs_20050926_015537_NORTHERN_PERU'
    scardec_stf_dir = './STF/FCTs_20070815_234057_NEAR_COAST_OF_PERU'
    # scardec_stf_dir = './STF/FCTs_20070912_111026_SOUTHERN_SUMATRA__INDONESIA'
    # scardec_stf_dir = './STF/FCTs_20170122_043022_SOLOMON_ISLANDS'
    # scardec_stf_dir = './STF/FCTs_20090810_195538_ANDAMAN_ISLANDS__INDIA_REGION'
    # scardec_stf_dir = './STF/FCTs_20090103_194350_IRIAN_JAYA_REGION__INDONESIA'
    # scardec_stf_dir = './STF/FCTs_20090930_101609_SOUTHERN_SUMATRA__INDONESIA'
    # scardec_stf_dir = './STF/FCTs_20100227_063411_NEAR_COAST_OF_CENTRAL_CHILE'
    # scardec_stf_dir = './STF/FCTs_20160302_124948_SOUTHWEST_OF_SUMATRA__INDONESIA'
    # scardec_stf_dir = './STF/FCTs_20181205_041808_SOUTHEAST_OF_LOYALTY_ISLANDS'
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


outdir = os.path.join('/lustre/orion/geo111/scratch/lsawade', 'STF_results_surface', f'{scardec_id}')
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
# This is really doin the heavy lifting!!!!
fobsd = []
fcmt3 = []
fgree = []
ftape = []

for _component, _phase in zip(components, phases):
    
    _fobsd, _fcmt3, _fgree, _ftape = osl.full_preparation(ds_obsd, ds_cmt3, ds_gree, _phase, _component, cmt3, [cstf], 
                                                          green_is_synthetic=False, plotdir=plotdir, 
                                                          labels=['Observed', 'CMT3D+', 'Green Functions'], gf_shift=-20.0)
    fobsd.append(_fobsd)
    fcmt3.append(_fcmt3)
    fgree.append(_fgree)
    ftape.append(_ftape)
    


# %%
tmax_main = 300.0

config = dict(
    Tmin=0, Tmax=tmax_main, 
    knots_per_second=1.0, 
    A=1.0, 
    penalty_weight=0.0,
    smooth_weight=1e5,
    bound_weight=100.0,
    maxiter=150,
    verbose=False)


osl.utils.log("Constructing station-wise STFs -- 1st pass")
fstfs_first = []
models_first_pass = []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    _fstfs, _c = osl.stationwise_first_pass(fobsd[_i], fgree[_i], ftape[_i], config)
    
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

config['penalty_weight'] = 0.1
config['A'] = A
config['smooth_weight'] = 1e3
diff_weights = [1e4, 1e4]


fstfs = []
models_second_pass = []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    config['diff_weight'] = diff_weights[_i]
    
    _fstfs, _c = osl.stationwise_second_pass(fobsd[_i], fgree[_i], ftape[_i], config, models_first_pass[_i])
  
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


# %%
# Get tmax for the stations
weights = [2/3, 1/3]

tmax = osl.compute_tmax(fstfs, weights, plotdir=plotdir, phases=phases, components=components,
                        plot_intermediate_figures=True)

# %%

osl.utils.log("Constructing station-wise STFs -- 3rd pass with tmax")

# Here adding 25 sceonds to the STF to account for station by station STF variability
config = dict(
    Tmin=0, Tmax=tmax, knots_per_second=1.0,
    A=A,
    penalty_weight=100.0, # 1.0, # 100.0
    smooth_weight=1e2, #.5,
    bound_weight=100.0,
    # diff_weight=1e1, #5000.0 Set during Loop
    maxiter=200,
    verbose=False)
diff_weights = [1e3, 1e3]


fstfs_tmax = []
models_third_pass = []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    config['diff_weight'] = diff_weights[_i]
    
    _fstfs_tmax, _c = osl.stationwise_third_pass(fobsd[_i], fgree[_i], ftape[_i], config, models_second_pass[_i], tmax, fstfs[_i])
    # Append results
    fstfs_tmax.append(_fstfs_tmax)
    
    # Pad models without many parameters
    Nm_max = np.max([_c[_i].shape[0] for _i in range(len(_c))])
    _c = [np.pad(_c[_i], (0, Nm_max - _c[_i].shape[0]), mode='constant') for _i in range(len(_c))]
    models_third_pass.append(np.sqrt(np.mean(np.vstack(_c)**2, axis=0)))


# %%

# Compute average of P and S median and means
A_tmaxP = np.median(fstfs_tmax[0].meta.stations.attributes.ints)
A_tmaxS = np.median(fstfs_tmax[1].meta.stations.attributes.ints)

if A_tmaxP > A_tmaxS and tmax < 30.0:
    A_tmax = A_tmaxP
else:
    A_tmax = A_tmaxS


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

if tmax < 30.0:
    config = dict(
        Tmin=0, Tmax=tmax, knots_per_second=1.0,
        A=A_tmax,
        penalty_weight=0.5, #1.0, # 1.0, # 100.0
        smooth_weight=1e1, #1e2, #.5,
        bound_weight=1.0,
        diff_weight=1e1, #5000.0
        maxiter=200,
        verbose=False)

elif tmax < 40.0:
    
    config = dict(
        Tmin=0, Tmax=tmax, knots_per_second=1.0,
        A=A_tmax,
        penalty_weight=0.1, #1.0, # 1.0, # 100.0
        smooth_weight=1e2, #1e2, #.5,
        bound_weight=1.0,
        diff_weight=1e1, #5000.0
        maxiter=200,
        verbose=False)

else:
    config = dict(
        Tmin=0, Tmax=tmax , knots_per_second=1.0,
        A=A_tmax,
        penalty_weight=10.0, #1.0, # 1.0, # 100.0
        smooth_weight=1e2, #1e2, #.5,
        bound_weight=10.0,
        diff_weight=1e3, #5000.0
        maxiter=200,
        verbose=False)


invs = []

if fobsd[1].data.shape[0] < 10: 
    config['penalty_weight'] = 0.5

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    # osl.utils.find_tmax()
    azimuthalweights = osl.utils.compute_azimuthal_weights(fobsd[_i].meta.stations.azimuths)
    
    _inv = osl.Inversion(fobsd[_i].t, fobsd[_i].data[:,0,:], fgree[_i].data[:,0,:], config=config, tapers=ftape[_i].data[:,0,:],
                         azimuthalweights=azimuthalweights)
    x = osl.utils.gaussn(_inv.npknots[: -_inv.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    
    x0 = models_third_pass[_i][:len(x)]
    if len(models_third_pass[_i]) < len(x):
        x0 = np.pad(x0, (0, len(x) - len(x0)), mode='constant', constant_values=0)
    else:
        x0 = x0[:len(x)]
        
    _inv.optimize_smooth_bound_diff(x=x, x0=x0)
    
    # _inv.optimize_smooth_bound0N(x=x)
    _inv.print_results()
    
    invs.append(_inv)

# %%

# Combined inversion
# config.update(dict(smooth_weight=0))

cinvs= []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    azimuthalweights = osl.utils.compute_azimuthal_weights(fobsd[_i].meta.stations.azimuths)
    _inv = osl.Inversion(fobsd[_i].t, fobsd[_i].data[:,0,:], fgree[_i].data[:,0,:], config=config, tapers=ftape[_i].data[:,0,:],
                         azimuthalweights=azimuthalweights)
    cinvs.append(_inv)
    
cinv = osl.CombinedInversion(cinvs, weights)

_model = weights[0] * models_second_pass[0] + weights[1] * models_second_pass[1]
_model = _model[:len(x)]
_model[-1] = 0.0
_model[0] = 0.0

x = osl.utils.gaussn(_inv.npknots[: -cinv.k - 1], 30, 20)
x = 2*x / np.sum(x)

if len(_model) < len(x):
    x0 = np.pad(_model, (0, len(x) - len(_model)), mode='constant', constant_values=0)
else:
    x0 = _model[:len(x)]
    
   

cinv.optimize_diff(x=_model, x0=x0)# 
# cinv.optimize(x=_model[:len(x)])
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

# # plot spectrum optimal STF
# def spectrum(combstf):
#     t = combstf.t
#     dt = t[1] - t[0]
#     Fs = 1 / dt  # sampling frequency
    
#     ps = np.abs(np.fft.fft(combstf.f))**2
#     freqs = np.fft.fftfreq(len(ps), dt)
#     idx = np.argsort(freqs)
#     plt.figure()
#     ax = plt.axes()
#     plt.plot(freqs[idx], ps[idx])
#     plt.xlim(0,0.2)
#     xt = ax.get_xticks()
#     plt.xticks(xt, [f"{1/_x:.2f}" if _x != 0 else '' for _x in xt ])
    
#     plt.savefig('test.pdf')

# spectrum(combstf)


# %%
# Plotting the data 
misfits_phase = []
fscar = []
fbstf = []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    # Create two new datasets
    _fscar = fgree[_i].copy()
    _fbstf = fgree[_i].copy()

    # Convolve the Green functions with the optimal STF
    _fscar.convolve(scardec.f/cmt3.M0, 0)
    # fbstf.convolve(pcstfs[_i].f/cmt3.M0, 0)
    _fbstf.convolve(combstf.f/cmt3.M0, 0)

    # Taper datasets
    _fscar  = onp.tt.taper_dataset(_fscar, _phase, tshift, gf_shift=-20.0)
    _fbstf  = onp.tt.taper_dataset(_fbstf, _phase, tshift, gf_shift=-20.0)
    
    fscar.append(_fscar)
    fbstf.append(_fbstf)
    
    # make some measurements
    # onp.utils.window_measurements(fobsd[_i], fbstf, phase=_phase)

    # Compute combined misfit
    m_scardec = onp.utils.L2(fobsd[_i], _fscar, normalize=True)
    m_bstf = onp.utils.L2(fobsd[_i], _fbstf, normalize=True)
        
    misfits_phase.append([m_scardec, m_bstf])    

    osl.plot.plot_full_section([fobsd[_i], _fscar, _fbstf], ['Observed', 'SCARDEC', 'B-STF'], [scardec, combstf], cmt3, 
                           scale=5.0, limits=[0*60,60*60], component=_component, 
                           outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_scardec_bstf.pdf'))
    
# %%

# for _i, (_component, _phase) in enumerate(zip(components, phases)):
#     # Create two new datasets
#     fscar = fgree[_i].copy()
#     fbstf = fgree[_i].copy()

#     # Convolve the Green functions with the optimal STF
#     fscar.convolve(scardec.f/cmt3.M0, 0)
#     # fbstf.convolve(pcstfs[_i].f/cmt3.M0, 0)
    
#     for _j in range(fstfs[_i].data.shape[0]):
#         _stf = osl.STF(t=t, f=fstfs[_i].data[_j,]*cmt3.M0, tshift=0, origin=cmt3.origin_time, label='B-STF')
#         _stf.M0 = np.trapz(pcstf.f, pcstf.t)
    
#         fbstf.convolve_trace(_j, _stf.f/cmt3.M0, 0)

#     # Taper datasets
#     fscar  = onp.tt.taper_dataset(fscar, _phase, tshift, gf_shift=-20.0)
#     fbstf  = onp.tt.taper_dataset(fbstf, _phase, tshift, gf_shift=-20.0)

#     osl.plot.plot_full_section([fobsd[_i], fscar, fbstf], ['Observed', 'SCARDEC', 'B-STF'], [scardec, combstf], cmt3, 
#                            scale=5.0, limits=[0*60,60*60], component=_component, 
#                            outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_scardec_bstf_tracewise.pdf'))

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
                      limits=(0, 225),
                      nocoast=nocoast,
                      region=region, 
                      misfits=misfits_phase)


plt.savefig(os.path.join(plotdir, 'stf_summary.pdf'))

plt.close('all')


#%%
#

osl.utils.log("Selecting windows for surface wave measurements")
# Band pass filter for the long preiod comparison
bp = [1/250, 1/100]

fobsd_surface = []
fcmt3_surface = []
fscar_surface = []
fbstf_surface = []

surface_components = ['Z', 'T']
surface_phases = ['Rayleigh', 'Love']

# Plot fit between observed and CMT3D+ and Green functiions convolved with the STF
combstf = osl.STF(t=t, f=cinv.construct_f(cinv.model)*cmt3.M0, tshift=0, origin=cmt3.origin_time, label='CSTF')
combstf.M0 = np.trapz(combstf.f, combstf.t)

# Convolve the green function with the 
bstf_synt = ds_gree.copy()
bstf_synt.convolve(combstf.f/cmt3.M0, 0)

# Convolve the green function with the 
scar_synt = ds_gree.copy()
scar_synt.convolve(scardec.f/cmt3.M0, 0)


for _i, (_component, _phase) in enumerate(zip(surface_components, surface_phases)):
    
    _fobsd_surface, _fbstf_surface, _fcmt3_surface, _ =  osl.full_preparation(ds_obsd, bstf_synt, ds_cmt3,  _phase, _component, cmt3, [cstf, combstf],
                                                                                           green_is_synthetic=True, plotdir=plotdir, snr=False, plot_intermediate_figures=True,
                                                                                           labels=['Observed', 'B-STF', 'CMT3D+'], gf_shift=-20.0, bp=bp
                                                                                           )

    fobsd_surface.append(_fobsd_surface)
    fcmt3_surface.append(_fcmt3_surface)
    fbstf_surface.append(_fbstf_surface)
    
    osl.plot.plot_full_section([_fobsd_surface, _fcmt3_surface, _fbstf_surface], ['Observed', 'CMT3D+', 'B-STF'], [scardec, combstf], cmt3, 
                           scale=2.5, limits=[0*60,60*60], component=_component, 
                           outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_cmt3_bstf.pdf'))
    
    
    # Make comparison between observed, SCARDEC and BSTF
    _fobsd_surface, _fbstf_surface, _fscardec_surface, _ =  osl.full_preparation(ds_obsd, bstf_synt, scar_synt, _phase, _component, cmt3, [scardec, combstf],
                                                                                           green_is_synthetic=True, plotdir=plotdir, snr=False, plot_intermediate_figures=False,
                                                                                           labels=['Observed', 'B-STF', 'SCARDEC'], gf_shift=-20.0, bp=bp
                                                                                           )

    
    fscar_surface.append(_fscardec_surface)
    
    osl.plot.plot_full_section([_fobsd_surface, _fscardec_surface, _fbstf_surface], ['Observed', 'SCARDEC', 'B-STF'], [scardec, combstf], cmt3, 
                                scale=2.5, limits=[0*60,60*60], component=_component, 
                                outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_scardec_bstf.pdf'))

# %%
# Create measurements and plot associated histograms.

measurements = dict()
wavetypes = ['body', 'surface']
misfitlabels = ['CMT3D+', 'SCARDEC', 'B-STF']
misfitcolors = ['k', 'tab:red', 'tab:blue']
bodysynths = [fcmt3, fscar, fbstf]
surfsynths = [fcmt3_surface, fscar_surface, fbstf_surface]

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    # Add phase and component labels
    pc = f"{_component}_{_phase}"
    
    measurements[pc] = dict()
    # Compute misfits between observed and CMT3D+ and BSTF
    for _j, (_ml, _synth) in enumerate(zip(misfitlabels, bodysynths)):
        
        measurements[pc][_ml] = onp.utils.window_measurements(fobsd[_i], _synth[_i], phase=_phase, dict_only=True)


for _i, (_component, _phase) in enumerate(zip(surface_components, surface_phases)):
    # Add phase and component labels
    pc = f"{_component}_{_phase}"
    
    measurements[pc] = dict()
    # Compute misfits between observed and CMT3D+ and BSTF
    for _j, (_ml, _synth) in enumerate(zip(misfitlabels, surfsynths)):
        
        measurements[pc][_ml] = onp.utils.window_measurements(fobsd_surface[_i], _synth[_i], phase=_phase, dict_only=True)

phasecomp_order = [
    "Z_Ptrain",
    "T_Strain",
    "Z_Rayleigh",
    "T_Love"
]

# %%
# outfile = os.path.join(plotdir, 'measurements_histograms.pdf')
# osl.plot.plot_measurements_histograms(measurements, phasecomp_order, misfitlabels, misfitcolors, outfile)

# %%
onp.utils.save_json(measurements, os.path.join(datadir, 'measurements.json'))

# %%
# Compute the correlation ratio between the observed and the CMT3D+ and the BSTF
for _i, (_component, _phase) in enumerate(zip(surface_components, surface_phases)):
    onp.utils.window_measurements(fobsd_surface[_i], fbstf_surface[_i], phase=_phase)
    
ZR_amp = np.median(fobsd_surface[0].meta.stations.attributes.measurements.corr_ratio)
TL_amp = np.median(fobsd_surface[1].meta.stations.attributes.measurements.corr_ratio)
S_amp = np.array([ZR_amp, TL_amp])

scalar_moment_fix = np.sum(S_amp)/2

osl.utils.log(f"Average Correlation Ratio: {scalar_moment_fix:0.4f}")
osl.utils.log(f"Average Correlation Ratio ZR: {ZR_amp:0.4f}")
osl.utils.log(f"Average Correlation Ratio TL: {TL_amp:0.4f}")
# Average ratio of maximum amplitudes
pavgratio = np.abs(fobsd_surface[0].data[:,0,:]).max(axis=1).mean() / np.abs(fbstf_surface[0].data[:,0,:]).max(axis=1).mean()
savgratio = np.abs(fobsd_surface[1].data[:,0,:]).max(axis=1).mean() / np.abs(fbstf_surface[1].data[:,0,:]).max(axis=1).mean()

osl.utils.log(f"Average Ratio of Maximum Amplitudes: {pavgratio:0.4f}")
osl.utils.log(f"Average Ratio of Maximum Amplitudes: {savgratio:0.4f}")

# %%
# Redo STF optimal inversion with fixed amplitude
# Integral value of the STF
A_FIX = np.trapz(combstf.f/cmt3.M0,t)*scalar_moment_fix


# %%

osl.utils.log("Constructing station-wise STFs with the surface wave amplitude fix")

# Here adding 25 sceonds to the STF to account for station by station STF variability
config = dict(
    Tmin=0, Tmax=tmax, knots_per_second=1.0,
    A=A_FIX,
    penalty_weight=100.0, # 1.0, # 100.0
    smooth_weight=1e2, #.5,
    bound_weight=100.0,
    # diff_weight=1e1, #5000.0 Set during Loop
    maxiter=200,
    verbose=False)
diff_weights = [1e3, 1e3]


fstfs_tmax = []
models_third_pass = []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    config['diff_weight'] = diff_weights[_i]
    
    _fstfs_tmax, _c = osl.stationwise_third_pass(fobsd[_i], fgree[_i], ftape[_i], config, models_second_pass[_i], tmax, fstfs[_i])
    # Append results
    fstfs_tmax.append(_fstfs_tmax)
    
    # Pad models without many parameters
    Nm_max = np.max([_c[_i].shape[0] for _i in range(len(_c))])
    _c = [np.pad(_c[_i], (0, Nm_max - _c[_i].shape[0]), mode='constant') for _i in range(len(_c))]
    models_third_pass.append(np.sqrt(np.mean(np.vstack(_c)**2, axis=0)))

# %%

if tmax < 30.0:
    config = dict(
        Tmin=0, Tmax=tmax, knots_per_second=1.0,
        A=A_FIX,
        penalty_weight=10.0, #1.0, # 1.0, # 100.0
        smooth_weight=1e1, #1e2, #.5,
        bound_weight=1.0,
        diff_weight=1e1, #5000.0
        maxiter=200,
        verbose=False)

elif tmax < 40.0:
    
    config = dict(
        Tmin=0, Tmax=tmax, knots_per_second=1.0,
        A=A_FIX,
        penalty_weight=10.0, #1.0, # 1.0, # 100.0
        smooth_weight=1e2, #1e2, #.5,
        bound_weight=1.0,
        diff_weight=1e1, #5000.0
        maxiter=200,
        verbose=False)

else:
    config = dict(
        Tmin=0, Tmax=tmax , knots_per_second=1.0,
        A=A_FIX,
        penalty_weight=10.0, #1.0, # 1.0, # 100.0
        smooth_weight=1e2, #1e2, #.5,
        bound_weight=10.0,
        diff_weight=1e3, #5000.0
        maxiter=200,
        verbose=False)


invs = []

if fobsd[1].data.shape[0] < 10: 
    config['penalty_weight'] = 0.5

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    # osl.utils.find_tmax()
    azimuthalweights = osl.utils.compute_azimuthal_weights(fobsd[_i].meta.stations.azimuths)
    
    _inv = osl.Inversion(fobsd[_i].t, fobsd[_i].data[:,0,:], fgree[_i].data[:,0,:], config=config, tapers=ftape[_i].data[:,0,:],
                         azimuthalweights=azimuthalweights)
    x = osl.utils.gaussn(_inv.npknots[: -_inv.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    
    x0 = models_third_pass[_i][:len(x)]
    if len(models_third_pass[_i]) < len(x):
        x0 = np.pad(x0, (0, len(x) - len(x0)), mode='constant', constant_values=0)
    else:
        x0 = x0[:len(x)]
        
    _inv.optimize_smooth_bound_diff(x=x, x0=x0)
    
    # _inv.optimize_smooth_bound0N(x=x)
    _inv.print_results()
    
    invs.append(_inv)

# %%

# Combined inversion
# config.update(dict(smooth_weight=0))

cinvs= []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    azimuthalweights = osl.utils.compute_azimuthal_weights(fobsd[_i].meta.stations.azimuths)
    _inv = osl.Inversion(fobsd[_i].t, fobsd[_i].data[:,0,:], fgree[_i].data[:,0,:], config=config, tapers=ftape[_i].data[:,0,:],
                         azimuthalweights=azimuthalweights)
    cinvs.append(_inv)
    
cinv = osl.CombinedInversion(cinvs, weights)

_model = weights[0] * models_second_pass[0] + weights[1] * models_second_pass[1]
_model = _model[:len(x)]
_model[-1] = 0.0
_model[0] = 0.0

x = osl.utils.gaussn(_inv.npknots[: -cinv.k - 1], 30, 20)
x = 2*x / np.sum(x)

if len(_model) < len(x):
    x0 = np.pad(_model, (0, len(x) - len(_model)), mode='constant', constant_values=0)
else:
    x0 = _model[:len(x)]
    
   

cinv.optimize_diff(x=_model, x0=x0)# 
# cinv.optimize(x=_model[:len(x)])
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
plt.savefig(os.path.join(plotdir, 'bstf_vs_scardec_stf_fix.pdf'))
plt.close()

# # plot spectrum optimal STF
# def spectrum(combstf):
#     t = combstf.t
#     dt = t[1] - t[0]
#     Fs = 1 / dt  # sampling frequency
    
#     ps = np.abs(np.fft.fft(combstf.f))**2
#     freqs = np.fft.fftfreq(len(ps), dt)
#     idx = np.argsort(freqs)
#     plt.figure()
#     ax = plt.axes()
#     plt.plot(freqs[idx], ps[idx])
#     plt.xlim(0,0.2)
#     xt = ax.get_xticks()
#     plt.xticks(xt, [f"{1/_x:.2f}" if _x != 0 else '' for _x in xt ])
    
#     plt.savefig('test.pdf')

# spectrum(combstf)


# %%
# Plotting the data 
misfits_phase = []
fscar = []
fbstf = []

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    # Create two new datasets
    _fscar = fgree[_i].copy()
    _fbstf = fgree[_i].copy()

    # Convolve the Green functions with the optimal STF
    _fscar.convolve(scardec.f/cmt3.M0, 0)
    # fbstf.convolve(pcstfs[_i].f/cmt3.M0, 0)
    _fbstf.convolve(combstf.f/cmt3.M0, 0)

    # Taper datasets
    _fscar  = onp.tt.taper_dataset(_fscar, _phase, tshift, gf_shift=-20.0)
    _fbstf  = onp.tt.taper_dataset(_fbstf, _phase, tshift, gf_shift=-20.0)
    
    fscar.append(_fscar)
    fbstf.append(_fbstf)
    
    # make some measurements
    # onp.utils.window_measurements(fobsd[_i], fbstf, phase=_phase)

    # Compute combined misfit
    m_scardec = onp.utils.L2(fobsd[_i], _fscar, normalize=True)
    m_bstf = onp.utils.L2(fobsd[_i], _fbstf, normalize=True)
        
    misfits_phase.append([m_scardec, m_bstf])    

    osl.plot.plot_full_section([fobsd[_i], _fscar, _fbstf], ['Observed', 'SCARDEC', 'B-STF'], [scardec, combstf], cmt3, 
                           scale=5.0, limits=[0*60,60*60], component=_component, 
                           outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_scardec_bstf_fix.pdf'))


# %%
for _i, (_component, _phase) in enumerate(zip(components, phases)):
    if scardec_id == 'FCTs_20070815_234057_NEAR_COAST_OF_PERU' and _i == 0:
        
        osl.plot.plot_sub_section([fobsd[_i], fscar[_i], fbstf[_i]], ['Observed', 'SCARDEC', 'B-STF'], [scardec, combstf], cmt3, 
                                  stf_scale=cmt3.M0, scale=5.0, limits=[12.5*60,27.5*60], component=_component, 
                                  outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_scardec_bstf_sub_fix.pdf'))
    
# %%

# # %%
# for _i, (_component, _phase) in enumerate(zip(components, phases)):
#     # Create two new datasets
#     fscar = fgree[_i].copy()
#     fbstf = fgree[_i].copy()

#     # Convolve the Green functions with the optimal STF
#     fscar.convolve(scardec.f/cmt3.M0, 0)
#     # fbstf.convolve(pcstfs[_i].f/cmt3.M0, 0)
    
#     for _j in range(fstfs[_i].data.shape[0]):
#         _stf = osl.STF(t=t, f=fstfs[_i].data[_j,]*cmt3.M0, tshift=0, origin=cmt3.origin_time, label='B-STF')
#         _stf.M0 = np.trapz(pcstf.f, pcstf.t)
    
#         fbstf.convolve_trace(_j, _stf.f/cmt3.M0, 0)

#     # Taper datasets
#     fscar  = onp.tt.taper_dataset(fscar, _phase, tshift, gf_shift=-20.0)
#     fbstf  = onp.tt.taper_dataset(fbstf, _phase, tshift, gf_shift=-20.0)

#     osl.plot.plot_full_section([fobsd[_i], fscar, fbstf], ['Observed', 'SCARDEC', 'B-STF'], [scardec, combstf], cmt3, 
#                            scale=5.0, limits=[0*60,60*60], component=_component, 
#                            outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_scardec_bstf_tracewis_fix.pdf'))

# # %%
# osl.plot.plot_circular_STF(fstfs[0])

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
                      limits=(0, 225),
                      nocoast=nocoast,
                      region=region, 
                      misfits=misfits_phase)


plt.savefig(os.path.join(plotdir, 'stf_summary_fix.pdf'))

plt.close('all')

# %%
osl.utils.log("Selecting windows for surface wave measurements")
fobsd_surface = []
fcmt3_surface = []
fscar_surface = []
fbstf_surface = []

surface_components = ['Z', 'T']
surface_phases = ['Rayleigh', 'Love']

# Plot fit between observed and CMT3D+ and Green functiions convolved with the STF
combstf = osl.STF(t=t, f=cinv.construct_f(cinv.model)*cmt3.M0, tshift=0, origin=cmt3.origin_time, label='CSTF')
combstf.M0 = np.trapz(combstf.f, combstf.t)

# Convolve the green function with the 
bstf_synt = ds_gree.copy()
bstf_synt.convolve(combstf.f/cmt3.M0, 0)

# Convolve the green function with the 
scar_synt = ds_gree.copy()
scar_synt.convolve(scardec.f/cmt3.M0, 0)


for _i, (_component, _phase) in enumerate(zip(surface_components, surface_phases)):
    
    _fobsd_surface, _fbstf_surface, _fcmt3_surface, _ =  osl.full_preparation(ds_obsd, bstf_synt, ds_cmt3,  _phase, _component, cmt3, [cstf, combstf],
                                                                                           green_is_synthetic=True, plotdir=plotdir, snr=False, plot_intermediate_figures=False,
                                                                                           labels=['Observed', 'B-STF', 'CMT3D+'], gf_shift=-20.0, bp=bp
                                                                                           )

    fobsd_surface.append(_fobsd_surface)
    fcmt3_surface.append(_fcmt3_surface)
    fbstf_surface.append(_fbstf_surface)
    
    osl.plot.plot_full_section([_fobsd_surface, _fcmt3_surface, _fbstf_surface], ['Observed', 'CMT3D+', 'B-STF'], [scardec, combstf], cmt3, 
                           scale=2.5, limits=[0*60,60*60], component=_component, 
                           outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_cmt3_bstf_fix.pdf'))
    
    
    # Make comparison between observed, SCARDEC and BSTF
    _fobsd_surface, _fbstf_surface, _fscardec_surface, _ =  osl.full_preparation(ds_obsd, bstf_synt, scar_synt, _phase, _component, cmt3, [scardec, combstf],
                                                                                           green_is_synthetic=True, plotdir=plotdir, snr=False, plot_intermediate_figures=False,
                                                                                           labels=['Observed', 'B-STF', 'SCARDEC'], gf_shift=-20.0, bp=bp
                                                                                           )

    
    fscar_surface.append(_fscardec_surface)
    
    osl.plot.plot_full_section([_fobsd_surface, _fscardec_surface, _fbstf_surface], ['Observed', 'SCARDEC', 'B-STF'], [scardec, combstf], cmt3, 
                                scale=2.5, limits=[0*60,60*60], component=_component, 
                                outfile=os.path.join(plotdir, f'{_component}_{_phase}_data_scardec_bstf_fix.pdf'))

# %%
# Compute and plot measurements

measurements_fix = dict()
wavetypes = ['body', 'surface']
misfitlabels = ['CMT3D+', 'SCARDEC', 'B-STF']
misfitcolors = ['k', 'tab:red', 'tab:blue']
bodysynths = [fcmt3, fscar, fbstf]
surfsynths = [fcmt3_surface, fscar_surface, fbstf_surface]

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    # Add phase and component labels
    pc = f"{_component}_{_phase}"
    
    measurements_fix[pc] = dict()
    # Compute misfits between observed and CMT3D+ and BSTF
    for _j, (_ml, _synth) in enumerate(zip(misfitlabels, bodysynths)):
        
        measurements_fix[pc][_ml] = onp.utils.window_measurements(fobsd[_i], _synth[_i], phase=_phase, dict_only=True)


for _i, (_component, _phase) in enumerate(zip(surface_components, surface_phases)):
    # Add phase and component labels
    pc = f"{_component}_{_phase}"
    
    measurements_fix[pc] = dict()
    # Compute misfits between observed and CMT3D+ and BSTF
    for _j, (_ml, _synth) in enumerate(zip(misfitlabels, surfsynths)):
        
        measurements_fix[pc][_ml] = onp.utils.window_measurements(fobsd_surface[_i], _synth[_i], phase=_phase, dict_only=True)


# %%
# Now add the misfits from for _phase_label in measurements_fix.keys():
measurements_comb = deepcopy(measurements_fix)

for _phase_label in measurements_comb.keys():    
    measurements_comb[_phase_label]["B-STF-B"] = measurements[_phase_label]["B-STF"]


# %%
wavetypes = ['body', 'surface']
misfitlabels = ['CMT3D+', 'SCARDEC', 'B-STF', 'B-STF-B']
misfitcolors = ['k', 'tab:red', 'tab:blue', 'tab:blue']
misfitlinestyles = ['-', '-', '-', '--']

phasecomp_order = [
    "Z_Ptrain",
    "T_Strain",
    "Z_Rayleigh",
    "T_Love"
]

outfile = os.path.join(plotdir, 'measurements_histograms_combined.pdf')


osl.plot.plot_measurements_histograms(measurements_comb, phasecomp_order, misfitlabels, misfitcolors, misfitlinestyles, outfile,
                                      remove_outliers=False)
    
    
# %% 
# Store the measurements as a json file
onp.utils.save_json(measurements, os.path.join(datadir, 'measurements_fix.json'))

# %%
gcmt_stf = osl.STF.triangle(
    origin=cmt3_stf.origin_time, t=t, tc=cmt3_stf.time_shift,
    tshift=0.0, hdur=cmt3_stf.hdur, M0=cmt3.M0)

# %%
# Save the results
def store_stf(t, stf, filename):
    outdata = np.vstack((t, stf))
    np.savetxt(filename, outdata.T, fmt='%.6e', delimiter=' ', header='Time since origin [s]  STF [N.m]')


stfdir = os.path.join(datadir, 'stf')
stationwise_stf_dir = os.path.join(stfdir, 'stationwise')
fm_dir = os.path.join(datadir, 'fms')

os.makedirs(stfdir, exist_ok=True)
os.makedirs(stationwise_stf_dir, exist_ok=True)
os.makedirs(fm_dir, exist_ok=True)

for _i, (_component, _phase) in enumerate(zip(components, phases)):
    
    stationwise_file = os.path.join(stationwise_stf_dir, f'{_component}_{_phase}_stationwise')
    fstfs[_i].write(stationwise_file + '.npy', stationwise_file + '.json')

    stationwise_file_tmax = os.path.join(stationwise_stf_dir, f'{_component}_{_phase}_stationwise_tmax')
    fstfs_tmax[_i].write(stationwise_file_tmax + '.npy', stationwise_file_tmax + '.json')
    
    # Optimal component wise STFs.
    optimal_file = os.path.join(stfdir, f'optimal_{_component}_{_phase}_{gcmt.origin_time.isoformat()}.txt')
    store_stf(t, pcstfs[_i].f/1e7, optimal_file)


# Optimal STFs 
optimal_file = os.path.join(stfdir, f'optimal_PDE_{gcmt.origin_time.isoformat()}.txt')
outdata = np.vstack((t, combstf.f/1e7))
store_stf(t, combstf.f/1e7, optimal_file)

# Store average STF
average_file = os.path.join(stfdir, f'average_PDE_{gcmt.origin_time.isoformat()}.txt')
avg_stf = (weights[0] * np.mean(fstfs[0].data[:,0,:], axis=0) 
        + weights[1] * np.mean(fstfs[1].data[:,0,:], axis=0) ) * cmt3.M0 / 1e7 # to Nm
store_stf(t, avg_stf, average_file)

# Store average STF with 
average_file = os.path.join(stfdir, f'average_PDE_{gcmt.origin_time.isoformat()}_tmax.txt')
avg_stf = (weights[0] * np.mean(fstfs_tmax[0].data[:,0,:], axis=0) 
        + weights[1] * np.mean(fstfs_tmax[1].data[:,0,:], axis=0) ) * cmt3.M0 / 1e7 # to Nm
store_stf(t, avg_stf, average_file)

# Store the other STFs
gcmt_stf_file = os.path.join(stfdir, f'gcmt_{gcmt.origin_time.isoformat()}.txt')
store_stf(t, gcmt_stf.f/1e7, gcmt_stf_file)

# Scardec STF
scardec_file = os.path.join(stfdir, f'scardec_{scardec_id}.txt')
store_stf(t, scardec.f/1e7, scardec_file)


# Store Focal mechanism
gcmt_file = os.path.join(fm_dir, f'gcmt_cmtsolution.txt')
fm_scardec_file = os.path.join(fm_dir, f'scardec_cmtsolution.txt')
cmt3_file = os.path.join(fm_dir, f'cmt3_cmtsolution.txt')

gcmt.write(gcmt_file)
fm_scardec.write(fm_scardec_file)
cmt3.write(cmt3_file)

# Store the misfits
configuration = dict(
    knots_per_second=knots_per_second,
    limits=(0, tmax_main),
    region=region,
    nocoast=nocoast,
    misfits=misfits_phase)


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
    
# convert azimiuths to cartesian coordinates
# posP = np.where(fstfs_tmax[0].meta.stations.attributes.clipped==0)[0]
# posS = np.where(fstfs_tmax[1].meta.stations.attributes.clipped==0)[0]


posP = np.where(fstfs_tmax[0].meta.stations.attributes.clipped==0)[0]
posS = np.where(fstfs_tmax[1].meta.stations.attributes.clipped==0)[0]

if len(posP) < 10 and len(posS) < 10:
    osl.utils.log("Not enough stations to compute directivity")
    sys.exit()
    
from matplotlib.patches import Ellipse
plt.figure()
directivity_fit = dict()
if len(posP) > 10:
    # Get tmax and azimuth
    azP = np.radians(fstfs_tmax[0].meta.stations.azimuths[posP])    
    rP = fstfs_tmax[0].meta.stations.attributes.tmaxs[posP]
    
    # Convert to cartesian
    xP = rP*np.sin(azP)
    yP = rP*np.cos(azP)
    
    # Plot
    plt.scatter(xP, yP, c='tab:blue', label='P')

    # Fit ellipse
    X = np.array(list(zip(xP, yP)))
    regP = osl.LsqEllipse().fit(X)
    centerP, widthP, heightP, phiP = regP.as_parameters()
    
    osl.utils.log(f"Center: {centerP} -- Width: {widthP} -- Height: {heightP} -- Phi: {phiP}")

    directivity_fit["P"] = dict(center=centerP, width=widthP, height=heightP, phi=phiP),

    ellipseP = Ellipse(
            xy=centerP, width=2*widthP, height=2*heightP, angle=np.rad2deg(phiP),
            edgecolor='tab:blue', fc='None', lw=2, label='P Fit', zorder=2
        )
    plt.gca().add_patch(ellipseP)
    

if len(posS) > 10:
    
    # Get tmax and azimuth
    azS = np.radians(fstfs_tmax[1].meta.stations.azimuths[posS])    
    rS = fstfs_tmax[1].meta.stations.attributes.tmaxs[posS]
    
    # Convert to cartesian where the conversion is clockwise from north/yaxis
    xS = rS*np.sin(azS)
    yS = rS*np.cos(azS)

    # Plot
    plt.scatter(xS, yS, c='tab:orange', label='S')

    X = np.array(list(zip(xS, yS)))
    regS = osl.LsqEllipse().fit(X)
    centerS, widthS, heightS, phiS = regS.as_parameters()

    osl.utils.log(f"Center: {centerS} -- Width: {widthS} -- Height: {heightS} -- Phi: {phiS}")
    
    directivity_fit["S"] = dict(center=centerS, width=widthS, height=heightS, phi=phiS)

    ellipseS = Ellipse(
            xy=centerS, width=2*widthS, height=2*heightS, angle=np.rad2deg(phiS),
            edgecolor='tab:orange', fc='None', lw=2, label='S Fit', zorder=2
        )
    plt.gca().add_patch(ellipseS)
    
plt.axis('equal')
plt.savefig(os.path.join(plotdir, 'tmax_map.pdf'))
plt.close()

    

# Save to json
with open(os.path.join(datadir, 'directivity_fit.json'), 'w') as f:
    json.dump(directivity_fit, f, indent=4)



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
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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

cmt = CMTSOLUTION.read(cmt_file)
cmt.time_shift = 0.0

cmt3d_file = os.path.join(scardec_stf_dir, f'{cmt.eventname}_CMT3D+')

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


gf_base_ds = get_ds(gfm, cmt, raw=True)

#%%
gf_base_ds.compute_geometry(cmt.latitude, cmt.longitude)

# %%
# Download the corresponding data

# Base parameters
tshift = 200.0
starttime = cmt.origin_time - tshift
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
        networks = ",".join({_sta.split('.')[0] for _sta in gf_base_ds.meta.stations.codes})
        stations = ",".join({_sta.split('.')[1] for _sta in gf_base_ds.meta.stations.codes})

        # Run download function
        opl.download_data(
            os.path.join(scardec_stf_dir, 'data'),
            starttime=cmt.origin_time - 300,
            endtime=cmt.origin_time + 3600 + 300,
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
                               event_latitude=cmt.latitude,
                               event_longitude=cmt.longitude)

# %%

gf_ds = gf_base_ds.copy()
obs_ds = praw.copy()

tshift = 200.0
starttime = cmt.origin_time - tshift
sps = 2.0
length_in_s = 3600 + tshift

osl.process(obs_ds, starttime, length_in_s, sps, step=False)
osl.process(gf_ds, starttime, length_in_s, sps, step=True)


# %%
# Create STF

# Convolve with STF
t = np.arange(0, gf_ds.meta.npts * gf_ds.meta.delta, gf_ds.meta.delta)

stfcmt = CMTSOLUTION.read(cmt_file)
stfcmt = osl.STF.triangle(
    origin=stfcmt.origin_time, t=t, tc=stfcmt.time_shift,
    tshift=tshift, hdur=stfcmt.hdur, M0=stfcmt.M0/1e7)

stf = deepcopy(scardec)
stf.interp(t, origin=cmt.origin_time, tshift=tshift)

plt.figure(figsize=(10,2))
stf.plot()
stfcmt.plot()
plt.savefig('astf.pdf', dpi=300)

# %%
# Make synthetics with the STF
syn_ds = gf_ds.copy()

# syn_ds.convolve(stf.f/stf.M0, tshift)
syn_ds.convolve(stf.f, tshift)

osl.plot.plot_check(obs_ds, syn_ds, outfile='atest_syn.pdf', label1='Observed', label2='Synthetic')

# %%
# Get all stations where the epicentral distance is large than 30 dg

ds_obs, ds_gfs = obs_ds.intersection(gf_ds)
ds_syn = ds_gfs.copy()
ds_syn.convolve(stf.f/stf.M0, tshift)
osl.plot.plot_check_section([ds_obs, ds_syn], scale=30.0, outfile='atestsection.pdf')


# %%
# Note that any P is really my definition of any P arrival from taup P
# and that could be P, Pdiff, PKP

phases = ['P', 'S', 'Rayleigh', 'Love', 'anyP', 'anyS']

for phase in phases:

    onp.tt.get_arrivals(cmt, ds_obs, phase=phase)
    onp.tt.get_arrivals(cmt, ds_syn, phase=phase)
    onp.tt.get_arrivals(cmt, ds_gfs, phase=phase)


osl.utils.check_meta(ds_obs)

#%%

# components = ['Z', 'R', 'T']
# phases = ['P', 'S', 'Rayleigh', 'Love', 'body', 'Ptrain', 'Strain']
# offset_window_dict = {
#     'P': (60, 60),  # Window is from P to S
#     'S': (60, 60),   # Window is from S to max v Love waves
#     'Ptrain': (10, -120),  # Window is from P to S
#     'Strain': (60, 60),   # Window is from S to max v Love waves
#     'Rayleigh': (60, 60), # Window is max to min v Rayleigh waves
#     'Love': (60, 60), # Window is max to min v Love waves
#     'body': (60, 60), # Window is from P to max v Love waves
# }

# dy = 0.05
# phase = 'body'
# component = 'Z'

# specific_phase = None #'body' # Set to None to plot all phases

# # outdir = os.path.join(scardec_stf_dir, 'plots', 'azimuthal')
# outdir = os.path.join(scardec_stf_dir, 'plots', 'azimuthal')

# if not os.path.exists(outdir):
#     os.makedirs(outdir)

# for phase in phases:
#     for component in components:

#         if specific_phase:
#             if phase != specific_phase:
#                 continue

#         # Subselect the seignals based on distance phases etc.
#         ds_tt = onp.tt.select_traveltime_subset(
#             ds_syn, component=component, phase=phase, maxdist=145.0)

#         # Bin the signals into azimuthal bins that are good for plotting
#         ds_tt, theta_bin = osl.plot.azi_plot_bin(ds_tt, dy=dy)

#         # Get the indeces of the stations in the original array, and the component
#         indeces = [ds_obs.index(station) for station in ds_tt.meta.stations.codes]
#         icx = ds_obs.meta.components.index(component)

#         ds_obs_tt = ds_obs.subset(stations=indeces, components=component)
#         osl.plot.azi_plot([ds_obs_tt, ds_tt],
#                            theta_bin,
#                            dy=dy, hspace=0.0,
#                            phase=phase,
#                            tshift=tshift,
#                            window = 200.0,
#                            pre_window=offset_window_dict[phase][0],
#                            post_window=offset_window_dict[phase][1],
#                            bottomspines=False)

#         plt.savefig(os.path.join(outdir, f'{phase}_{component}.pdf'), dpi=300)
#         plt.close()

# %%
# Now given the selected traces we want to use the corresponding windows to taper
# the traces such that we can perform the inversion only on the relevant windows
# and not the whole trace.

phase = 'body'
component = 'Z'

# Subselect the seignals based on distance phases etc.
ds_gfs_tt = onp.tt.select_traveltime_subset(ds_gfs, component=component, phase=phase, maxdist=145.0)

# Get the corresponding observed traces
ds_gfs_tt, ds_obs_tt = ds_gfs_tt.intersection(ds_obs)
_, ds_syn_tt = ds_gfs_tt.intersection(ds_syn)

# Reomve components from observed arrays
ds_obs_tt = ds_obs_tt.subset(components=component)
ds_syn_tt = ds_syn_tt.subset(components=component)

#%%

osl.plot.plot_check_section([ds_obs_tt, ds_syn_tt], labels=['Observed', 'Green'],
                   outfile='atestsection_traveltime_select.pdf')



#%%
# Now we want to taper the traces to the selected windows

tds_obs_tt = onp.tt.taper_dataset(ds_obs_tt, phase, tshift)
tds_syn_tt = onp.tt.taper_dataset(ds_syn_tt, phase, tshift)
tds_gfs_tt = onp.tt.taper_dataset(ds_gfs_tt, phase, tshift)

# %%

osl.plot.plot_check_section([tds_obs_tt, tds_syn_tt], labels=['Original', 'Tapered'],
                   scale=5.0, outfile='atestsection_taper_syn.pdf')

osl.plot.plot_check_section([tds_obs_tt, tds_gfs_tt], labels=['Original', 'Tapered'],
                   scale=5.0, outfile='atestsection_taper_gfs.pdf')


# %%

def L2(obs, syn, normalize=True):

    l2 = np.sum((obs.data - syn.data)**2, axis=-1)

    if normalize:
        l2 /= np.sum(obs.data**2, axis=-1)

    return l2


misfit = L2(tds_obs_tt, tds_syn_tt)

#%%
# remove where the misfit is too large

def remove_large_misfit(obs, syn, misfit, threshold=1.0):

    idx = np.where(misfit < threshold)[0]
    return obs.subset(stations=idx), syn.subset(stations=idx), idx

fobs, fsyn, idx = remove_large_misfit(tds_obs_tt, tds_syn_tt, misfit, threshold=np.quantile(misfit, 0.25))
fgfs = tds_gfs_tt.subset(stations=idx)


#%%
# Plot check
osl.plot.plot_check_section([fobs, fsyn], labels=['Observed', 'Synthetic'],
                   scale=5.0, outfile='atestsection_misfit_removal.pdf')


# %%
# Load scardec STFs and plot the scardec

stfs = np.genfromtxt(os.path.join(scardec_stf_dir, 'station_stfs','fctsource_appP_time_Nm'))
stations = np.genfromtxt(os.path.join(scardec_stf_dir, 'station_stfs','resumP_HF'),
                         converters={
                             0: lambda s: s.decode('UTF-8'),
                             4: lambda s: s.decode('UTF-8')})
stf = stfs[:, 1].reshape((len(stations), 8192))
t_scardec = stfs[:, 0].reshape(( len(stations),8192))


# %%
stationcodes = [f"{_sta[-1]}.{_sta[0]}" for _sta in stations]
epidist = np.array([_sta[1] for _sta in stations])
azimuths = np.array([_sta[2] for _sta in stations])
pp_p_time = np.array([_sta[2] for _sta in stations])

idx = np.argsort(azimuths)

N = np.max(stf)

pos = np.arange(len(azimuths))
plt.figure(figsize=(5,15))
plt.axis('tight')
plt.plot(t_scardec[0,:], stf[idx, :].T/N + pos, 'k', lw=0.25)
plt.vlines(0, -1, len(azimuths), ls='-', color='gray', zorder=-1, lw=0.25)
plt.xlim(-10, +130)
plt.yticks(pos, [f"{stationcodes[i]}\n{azimuths[i]:.2f}" for i in idx], rotation=0)
plt.savefig("atest_stfs_scardec.pdf")


# %%
# Given the station codes of the Scardec stations, we can select the corresponding stations from our dataset
stalist = fobs.meta.stations.codes.tolist()
idstat = []
for _sc in stationcodes:
    if _sc not in fobs.meta.stations.codes:
        continue

    idstat.append(stalist.index(_sc))


# %%
fobs = fobs.subset(stations=idstat)
fgfs = fgfs.subset(stations=idstat)

# %%


t = np.arange(0, length_in_s, 1/sps)
teststf = osl.STF.gaussian(origin=obspy.UTCDateTime(0), t=t, tshift=tshift, hdur=75, tc=100.0)

# Fake data
testdata = fgfs.copy()
testdata.convolve(teststf.f, tshift)

# Read the Green functions
gcmt = CMTSOLUTION.read(cmt_file)
scardec = osl.STF.scardecdir(scardec_stf_dir, 'optimal')

originshift = gcmt.origin_time - scardec.origin
print(originshift)

#%%

gcmt.hdur

misfits = []
Trange = np.arange(2*gcmt.hdur, gcmt.hdur * 8, 2.5)
for i in Trange:
    stfinv = osl.Inversion(
        np.roll(fobs.data[:, 0, :], int((tshift + originshift) * sps), axis=-1),
        fgfs.data[:, 0, :], t, 1/sps, .001,
        maxT=tshift+i, lamb=1e-3, type="1", maxiter=100,
        taper_type="tukey")

    stfinv.landweber()
    misfits.append(stfinv.chi)

#%%

# Plot the misfits
c = misfits/np.max(misfits)
g = np.gradient(misfits, Trange)
a = np.gradient(g, Trange)

# Picking the end
idx1 = np.argmin(a)
idx2 = np.argmax(a)
diff_idx = idx2 - idx1
fidx = idx2 + 3 * diff_idx
maxT = Trange[fidx]


# Picking another Tmax
idxg = np.argmin(g)
idxg0 = np.where(g[idxg:] == 0)[0][0]

fidxg = idxg + idxg0
print(Trange[fidxg])
#%%

from scipy.optimize import curve_fit

inverted_normalized_misfit = 1-misfits/np.max(misfits)
grad_inv_mf = np.gradient(inverted_normalized_misfit, Trange)

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [np.max(grad_inv_mf), Trange[np.argmax(grad_inv_mf)], 1.]

coeff, var_matrix = curve_fit(gauss, Trange, grad_inv_mf, p0=p0)

# Get the fitted curve
gradfit = gauss(Trange, *coeff)
plt.figure()
plt.plot(Trange, grad_inv_mf, label='Test data')
plt.plot(Trange, gradfit, label='Fitted data')

# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
print('Fitted mean = ', coeff[1])
print('Fitted standard deviation = ', coeff[2])
plt.legend()
plt.savefig('atest_gaussian_fit.pdf', dpi=300)

# %%
_, mu, sigma = coeff
Tmax = mu + 2*4*sigma # Set T max to 2x 4 standard deviations
fidxg = np.where(Trange > Tmax)[0][0]

# %%

xlim = [np.min(Trange), np.max(Trange)]
ylim = [-1.5, 1.5]

plt.figure()
plt.subplot(211)
plt.plot(Trange, c, 'k', label='Misfit')
plt.plot(Trange, g, 'tab:red', label='Misfit Gradient')
plt.plot(Trange, a, 'tab:blue', label='Misfit Acceleration')

# Measurement markers
plt.plot(Trange[idx1], a[idx1], 'o', label='T1', color='tab:blue', markeredgecolor='k', markeredgewidth=0.25)
plt.plot(Trange[idx2], a[idx2], 'd', label='T2', color='tab:blue', markeredgecolor='k', markeredgewidth=0.25)
plt.plot(maxT, c[fidx], '*', label='Tmax', color='tab:blue', markersize=10, markeredgecolor='k', markeredgewidth=0.25)
plt.hlines(1.25, Trange[idx1], Trange[idx2], ls='-', color='gray', zorder=-1, clip_on=False)
plt.hlines(1.6, Trange[idx1], Trange[fidx], ls='-', color='gray', zorder=-1, clip_on=False)
plt.annotate(f"T2-T1", ((Trange[idx1] + Trange[idx2])/2, 1.3), ha='center', annotation_clip=False, zorder=10)
plt.annotate(f"T2 + 3*(T2-T1)", ((Trange[idx1] + Trange[fidx])/2, 1.65), ha='center', annotation_clip=False, zorder=10)
plt.vlines([Trange[_i] for _i in [idx1, idx2, fidx]], -1.5, 1.25, ls=':', color='gray', zorder=-1)

# Fixing the plot extent
plt.xlim(xlim)
plt.ylim(ylim)
plt.gca().tick_params(axis='x', which='both', labelbottom=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.ylabel('Norm. Misfit')

plt.legend(loc='upper right',fontsize='x-small', frameon=False, ncol=2)

plt.subplot(212)

# Plot the misfits and gradients
plt.plot(Trange, c, 'k', label='Misfit')
plt.plot(Trange, g, 'tab:red', label='Misfit Gradient')
plt.plot(Trange, a, 'tab:blue', label='Misfit Acceleration')

# Measurement markers
plt.plot(Trange[idxg], g[idxg], 'o', label='T1', color='tab:red', markeredgecolor='k', markeredgewidth=0.25)
plt.plot(Trange[fidxg], c[fidxg], '*', label='Tmax', color='tab:red', markersize=10, markeredgecolor='k', markeredgewidth=0.25)
plt.vlines([Trange[_i] for _i in [idxg, fidxg]], -1.5, 1.25, ls=':', color='gray', zorder=-1)

# Fixing the plot extent
plt.xlim(xlim)
plt.ylim(ylim)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xlabel('Max. STF Length [s]')
plt.ylabel('Norm. Misfit')

plt.legend(loc='upper right',fontsize='x-small', frameon=False, ncol=2)


plt.savefig('atest_misfit_STF_Length.pdf', dpi=300)


# %%
stfinv = osl.Inversion(
    np.roll(fobs.data[:, 0, :], int((tshift + originshift) * sps), axis=-1),
    fgfs.data[:, 0, :], t, 1/sps, .001,
    minT=tshift, maxT=tshift+Trange[fidxg], lamb=1e-3, type="1", maxiter=200,
    taper_type="tukey")

stfinv.landweber()

#%%

# stfinv = osl.LagrangeInversion(
#     np.roll(fobs.data[:, 0, :], int((tshift + originshift) * sps), axis=-1),
#     fgfs.data[:, 0, :], t, 1/sps, .001,
#     minT=tshift, maxT=tshift+Trange[fidxg], lamb=1e-3, type="1", maxiter=300,
#     taper_type="tukey")

# opt = stfinv.optimize()



# %%
t = np.arange(0, len(stfinv.src) * 1 / sps , 1/sps)
gcmt = CMTSOLUTION.read(cmt_file)
scardec_stf = osl.STF.scardecdir(scardec_stf_dir, 'optimal')
cmt_stf = osl.STF.gaussian(origin=gcmt.origin_time, t=t, tshift=tshift, hdur=gcmt.hdur, tc=gcmt.time_shift)


norm = 1 # np.trapz(stfinv.src, t)
print(norm)
plt.figure(figsize=(6,2.5))
plt.plot(cmt_stf.t-tshift, cmt_stf.f, 'tab:red', label='GCMT', lw=0.75)
plt.plot(scardec_stf.t, scardec_stf.f/scardec_stf.M0, 'tab:blue', label='SCARDEC', lw=0.75)
plt.plot(stfinv.t-tshift, stfinv.src, 'tab:orange', label='GLAD-M35', lw=0.75)
plt.xlim(-5, +200)
plt.xlabel('Time [s]')
plt.ylabel('Norm. Moment Rate')
plt.legend(frameon=False, fontsize='small', loc='upper left')
plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.25)
plt.savefig('stfinv.pdf', dpi=300)



# %%
# Do the deconvolution trace by trace

from scipy.optimize import curve_fit
# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

# %%
from scipy.interpolate import BSpline
import numpy as np
import scipy.optimize as so
from scipy.fftpack import fft, ifft

def get_Tmax_from_gauss(misfits):
    """This setup only works because the misfit curve ends up being an
    approximate inverted error function. taking the gradient allows us to approximate
    that a gaussian that represents the gradient after which we can estimate
    the Tmax"""

    # Invert and normalize the misfit
    inverted_normalized_misfit = 1-misfits/np.max(misfits)

    # Get the gradient
    grad_inv_mf = np.gradient(inverted_normalized_misfit, Trange)

    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    p0 = [np.max(grad_inv_mf), Trange[np.argmax(grad_inv_mf)], 1.]

    # Fit the gaussian
    coeff, var_matrix = curve_fit(gauss, Trange, grad_inv_mf, p0=p0)

    # Get coefficients
    _, mu, sigma = coeff

    return mu + 4*4*sigma

# %%


#%%

i = 0
d = np.roll(fobs.data[:, 0,:], int((tshift + originshift) * sps), axis=1) #fobs.data[i, 0,:]
G = fgfs.data[:, 0, :]
# config = dict(Tmin=tshift-5.0, Tmax=tshift+125, knots_per_second=0.15)
# inv = Inversion(t, d, G, config=config)
# inv.optimize()
config = dict(
    Tmin=tshift, Tmax=tshift+120, knots_per_second=1.0, 
    A=1.2, 
    weight=1.0,
    constraint_weight=0.0,
    penalty_weight=2.0,
    smooth_weight=10.0,
    bound_weight=1.0,
    maxiter=500)

inv1 = Inversion(t, d, G, config=config)
x = gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
x = 2*x / np.sum(x)
inv1.optimize_smooth_bound0N(x=x)


fig, axes = inv1.plot(tshift=tshift)

inv1.print_results()

# %%
ax1 = axes[0]
scardec_plot = osl.STF.scardecdir(scardec_stf_dir, 'optimal')
# ax1.plot(scardec_plot.t, scardec_plot.f/scardec_plot.M0, "-g", label="SCARDEC Opt.")
ax1.legend(frameon=False, loc='upper right', ncol=3)
plt.savefig('atest_bspline_vs_scardec.pdf')

# %%
# Now we need to find the maximum time since the function will always be after time 0


config = dict(
    Tmin=tshift, Tmax=tshift+125, 
    knots_per_second=1.0, 
    A=1.2, 
    weight=1.0,
    constraint_weight=0.0,
    penalty_weight=2.0,
    smooth_weight=10.0,
    bound_weight=1.0,
    maxiter=200)


def find_tmax(t, tmaxs, d, G, config, parallel=True):
    
    def find_tmax_sub(tmax, t, d, G, config):
        config.update(dict(Tmax=tmax))
        inv1 = Inversion(t, d, G, config=config)
        x = gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
        x = 2*x / np.sum(x)
        inv1.optimize_smooth_bound0N(x=x)
        return inv1.cost

    if parallel:
        costs = Parallel(n_jobs=10)(delayed(find_tmax_sub)(_tmax, t, d, G, config.copy()) for _tmax in tmaxs)
    else:
        costs = []
        for _tmax in tmaxs:
            costs.append(find_tmax_sub(_tmax, t, d, G, config))
    
    # Compute the gradient of the costs
    grad = np.gradient(costs, tmaxs - tshift)
    
    # Grab the index for the first time after the origin time and choose one to the right for good measure
    try:
        tmax_idx = np.where(np.abs(grad) / np.max(np.abs(grad)) < 0.001)[0][0]
    except:
        tmax_idx = np.argmin(np.abs(grad))
    
    return tmax_idx, costs


tmaxs = np.arange(10, 200, 5) + tshift

tmax_idx, costs = find_tmax(t, tmaxs, np.roll(fobs.data[:, 0,:], int((tshift + originshift) * sps), axis=1), fgfs.data[:, 0, :], config, tshift)
    
# %%

config = dict(
    Tmin=tshift, Tmax=tshift+tmaxs[tmax_idx], 
    knots_per_second=1.0, 
    A=1.2,
    weight=1.0,
    constraint_weight=0.0,
    penalty_weight=2.0,
    smooth_weight=10.0,
    bound_weight=1.0,
    maxiter=500)

# New tmaxs range for station wise inversions
newtmaxs = np.arange(np.maximum(tmaxs[tmax_idx]-tshift-20, 10), tmaxs[tmax_idx] - tshift + 40, 5) + tshift


def inversion(i, t, tmaxs, d, G, config):
    
    tmax_idx, costs = find_tmax(t, tmaxs, d, G, config, parallel=False)
    print(f"[{i:>03d}]: Done with finding Tmax.")
    config.update(dict(Tmax=tmaxs[tmax_idx]))
    
    inv1 = Inversion(t, d, G, config=config)
    x = gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    inv1.optimize_smooth_bound0N(x=x)
    
    print(f"[{i:>03d}]: Done.")

    return i, inv1


invs = Parallel(n_jobs=11)(delayed(inversion)(
    i, t, newtmaxs,
    np.roll(fobs.data[i:i+1, 0,:], int((tshift + originshift) * sps)), 
    fgfs.data[i:i+1, 0, :], 
    config.copy()) for i in range(len(fobs.data)))

#%%
# invs = []
# for i in range(len(fobs.data)):
#     invs.append(inversion(i, np.roll(fobs.data[i, 0,:], int((tshift + originshift) * sps)), fgfs.data[i, 0, :]))
    

azimuths = fobs.meta.stations.azimuths
distances = fobs.meta.stations.distances
codes = fobs.meta.stations.codes
idx = np.argsort(azimuths)

my_stfs = np.array([_inv[1].construct_f(_inv[1].model) for _inv in invs])

#%%
for _i, _inv in enumerate(invs):
    print()
    
    I_f1 = np.trapz( _inv[1].construct_f( _inv[1].model), _inv[1].t)
    # I_f2 = np.trapz( _inv[2].construct_f( _inv[2].model), _inv[2].t)
    print(f"{codes[_i]:<8} {azimuths[_i]:>8.2f} {distances[_i]:>8.2f} {I_f1:>8.2f} {_inv[1].opt.nit:>03d} {_inv[1].opt.success} {_inv[1].opt.message} ")
    knots = _inv[1].knots
    # _inv[1].print_results(constrained=False)
    # _inv[2].print_results(constrained=True)

idx2 = [stationcodes.index(codes[_i]) for _i in idx]

ints = np.array([np.trapz(my_stfs[_i,:], t-tshift) for _i in idx])
ints2 = np.array([np.trapz(stf[_i,:].T/scardec_stf.M0, t_scardec[0, :]) for _i in idx2])

pos = np.arange(len(azimuths))

#%%
plt.figure(figsize=(7.5,12))
plt.plot(t_scardec[0, :], 25*stf[idx2,:].T/scardec_stf.M0 + pos, 'k', lw=0.5)
plt.plot(t-tshift, 25*my_stfs[idx, :].T + pos, 'r', lw=0.5)
plt.xlim(-10, +175)
plt.ylim(-0.25, len(azimuths)+0.25)
plt.vlines(0, -1, len(azimuths)+0.25, ls='-', color='gray', zorder=-1, lw=1.0)
# plt.vlines(knots-tshift, -1, len(azimuths)+0.25, ls='-', color='gray', zorder=-1, lw=0.25)
plt.yticks(pos, [f"{codes[i]}\n{azimuths[i]:.2f}\n{distances[i]:.2f}" for i in idx], rotation=0)
plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.1)
plt.xlabel('Time since origin [s]')
plt.savefig("atest_stfs.pdf")


# %%

scardec_opt = osl.STF.scardecdir(scardec_stf_dir, 'optimal')
scardec_ave = osl.STF.scardecdir(scardec_stf_dir, 'average')

plt.figure(figsize=(5,8))
plt.plot(t_scardec[0, :], np.sum(stf[idx2,:])/scardec_stf.M0, 'k', lw=0.5, label='SCARDEC Average')
plt.plot(t-tshift, 25*my_stfs[idx, :].T + pos, 'r', lw=0.5)
plt.xlim(-10, +130)
plt.ylim(-0.25, len(azimuths)+0.25)
plt.yticks(pos, [f"{codes[i]}\n{azimuths[i]:.2f}" for i in idx], rotation=0)
plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.1)
plt.xlabel('Time since origin [s]')
plt.savefig("atest_stfs.pdf")




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
# %%
import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import obstflib as osl
import obsnumpy as onp
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


# %%
# Load the data


if ("IPython" in sys.argv[0]) or ("ipython" in sys.argv[0]):
    # scardec_id = "FCTs_20070815_234057_NEAR_COAST_OF_PERU"
    # scardec_id = "FCTs_19951009_153553_NEAR_COAST_OF_JALISCO__MEXICO"
    scardec_id = "FCTs_19950730_051123_NEAR_COAST_OF_NORTHERN_CHILE"
    scardec_id = "FCTs_20210729_061549_ALASKA_PENINSULA"
else:
    
    
    # Which directory
    scardec_id = sys.argv[1]
    
    if len(sys.argv) > 2:
        nocoast = True
    else:
        nocoast = False




# scardec_id = "FCTs_20010623_203314_NEAR_COAST_OF_PERU"
result_dir = "STF_results_surface"

datadir = os.path.join(result_dir, scardec_id, "data")
plotdir = os.path.join(result_dir, scardec_id, "plots")


# %%

# Load the data
pstf = onp.Dataset.read(os.path.join(datadir, 'stf', 'stationwise', 'Z_Ptrain_stationwise_tmax.npy'), os.path.join(datadir, 'stf', 'stationwise', 'Z_Ptrain_stationwise_tmax.json'))
sstf = onp.Dataset.read(os.path.join(datadir, 'stf', 'stationwise', 'T_Strain_stationwise_tmax.npy'), os.path.join(datadir, 'stf', 'stationwise', 'T_Strain_stationwise_tmax.json'))

if len(pstf.meta.stations) <= 7 and  len(sstf.meta.stations) <= 7 :
    osl.utils.log("Only few stations will mark results as disputable.")
    few_stations=True
else:
    few_stations=False

# %%
# Smooth the STF with a Gaussian filter
pstf_smooth = pstf.copy()
sstf_smooth = sstf.copy()

# Make Guassian
t = t = np.arange(0, pstf.meta.npts * pstf.meta.delta, pstf.meta.delta)

# Gaussian shift
g = osl.utils.gaussian_stf(t, 0.0, 33.0)

# Smooth STFs
pstf_smooth.convolve(g)
sstf_smooth.convolve(g)

# %%
# For every trace compute the cumulative stf and find values at 10%, 50%, 90%
from scipy.integrate import cumtrapz

pstf_cum = cumtrapz(pstf_smooth.data[:,0,:], t, axis=-1)
pstf_cum = pstf_cum / pstf_cum[:,-1][:,None]
sstf_cum = cumtrapz(sstf_smooth.data[:,0,:], t, axis=-1)
sstf_cum = sstf_cum / sstf_cum[:,-1][:,None]

# Find the times at 10%, 90%
NP = len(pstf_smooth)
NS = len(sstf_smooth)
t10P = np.zeros(NP)
t90P = np.zeros(NP)
t10S = np.zeros(NS)
t90S = np.zeros(NS)

for i in range(NP):
    t10P[i] = t[np.argmin(np.abs(pstf_cum[i,:] - 0.1))]
    t90P[i] = t[np.argmin(np.abs(pstf_cum[i,:] - 0.9))]
    
for i in range(NS):
    t10S[i] = t[np.argmin(np.abs(sstf_cum[i,:] - 0.1))]
    t90S[i] = t[np.argmin(np.abs(sstf_cum[i,:] - 0.9))]
    
# Compute centroid
centroidP = np.trapz(t[None, :] * pstf_smooth.data[:, 0, :], t, axis=-1) / np.trapz(pstf_smooth.data[:, 0, :], t, axis=-1)
centroidS = np.trapz(t[None, :] * sstf_smooth.data[:, 0, :], t, axis=-1) / np.trapz(sstf_smooth.data[:, 0, :], t, axis=-1)

# Compute the standard deveiation
stdP = np.sqrt(np.trapz((t[None, :] - centroidP[:, None])**2 * pstf_smooth.data[:, 0, :], t, axis=-1) / np.trapz(pstf_smooth.data[:, 0, :], t, axis=-1))
stdS = np.sqrt(np.trapz((t[None, :] - centroidS[:, None])**2 * sstf_smooth.data[:, 0, :], t, axis=-1) / np.trapz(sstf_smooth.data[:, 0, :], t, axis=-1))


# %%
# Plot STFs
    
fig, ax, twinaxes = osl.plot.plot_stationwise(stfss=[pstf_smooth,sstf_smooth], limits=(0, 175), plot_tmaxs=False)
pos = np.argsort(pstf_smooth.meta.stations.azimuths)
y = np.arange(len(pos))
yMax = np.max(y)
for (_t10, _t90, _cntr, _std), _y in zip(zip(t10P[pos], t90P[pos], centroidP[pos], stdP[pos]), y):
    zorder = yMax - _y
    ax[0].plot([_t10, _t90], [_y, _y], c='tab:blue', lw=1, zorder=zorder, marker=2, markersize=10)
    ax[0].plot([_cntr - _std, _cntr + _std], [_y, _y], c='tab:orange', lw=1, zorder=zorder, marker=2, markersize=12.5)
    ax[0].plot(_cntr, _y, c='tab:blue', lw=2, zorder=zorder, ls='None', marker=2, markersize=15)

pos = np.argsort(sstf_smooth.meta.stations.azimuths)
y = np.arange(len(pos))
yMax = np.max(y)
for (_t10, _t90, _cntr, _std), _y in zip(zip(t10S[pos], t90S[pos], centroidS[pos], stdS[pos]), y):
    zorder = yMax - _y
    ax[1].plot([_t10, _t90], [_y, _y], c='tab:blue', lw=1, zorder=zorder, marker=2, markersize=10)
    ax[1].plot([_cntr - _std, _cntr + _std], [_y, _y], c='tab:orange', lw=1, zorder=zorder, marker=2, markersize=12.5)
    ax[1].plot(_cntr, _y, c='tab:blue', lw=2, zorder=zorder, ls='None', marker=2, markersize=15)
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
plt.savefig(os.path.join(plotdir, 'directivity_duration_stationwise.pdf'))

# %%

posP = np.argsort(pstf_smooth.meta.stations.azimuths)
azsortP = pstf_smooth.meta.stations.azimuths[posP]
maxsP = np.max(pstf_smooth.data[:, 0, :], axis=-1)
plt.figure(figsize=(10,8))
plt.subplot(221)
plt.plot(azsortP, t10P[posP], 'o', label='10%')
plt.plot(azsortP, centroidP[posP], 'o', label='Centroid')
plt.plot(azsortP, t90P[posP], 'o', label='90%')
plt.legend()
plt.xlim(0, 360)
ax = plt.subplot(222)
plt.plot(azsortP, t90P[posP]-t10P[posP], '-o', label='Cumul. duration')
plt.plot(azsortP, 2*stdP[posP], '-o', label='2 sigma')
plt.xlim(0, 360)
tax = ax.twinx()
tax.plot(azsortP, maxsP[posP], '-o', label='Max', c='tab:red')
tax.set_xlim(0, 360)
plt.legend()

posS = np.argsort(sstf_smooth.meta.stations.azimuths)
azsortS = sstf_smooth.meta.stations.azimuths[posS]
maxsS = np.max(sstf_smooth.data[:, 0, :], axis=-1)
plt.subplot(223)
plt.plot(azsortS, t10S[posS], 'o', label='10%')
plt.plot(azsortS, centroidS[posS], 'o', label='Centroid')
plt.plot(azsortS, t90S[posS], 'o', label='90%')
plt.legend()
plt.xlim(0, 360)
ax = plt.subplot(224)
plt.plot(azsortS, t90S[posS]-t10S[posS], '-o', label='Cumul. duration')
plt.plot(azsortS, 2*stdS[posS], '-o', label='2 sigma')
plt.xlim(0, 360)
tax = ax.twinx()
tax.plot(azsortS, maxsS[posS], '-o', label='Max', c='tab:red')
tax.set_xlim(0, 360)
plt.legend()


plt.savefig(os.path.join(plotdir, 'directivity_duration_azimuthal.pdf'))


# %%
azP = pstf_smooth.meta.stations.azimuths
posP = np.argsort(pstf_smooth.meta.stations.azimuths)
azsortP = pstf_smooth.meta.stations.azimuths[posP]


dursortP = t90P[posP]-t10P[posP]
# dursortP = circular_movmean(dursortP, 5)

ampsortP = maxsP[posP]
# ampsortP = circular_movmean(ampsortP, 5)
opt, mdP, maP = osl.directivity.get_cosparams(azsortP, dursortP, ampsortP)

# Same but for the S-wave
azS = sstf_smooth.meta.stations.azimuths
posS = np.argsort(sstf_smooth.meta.stations.azimuths)
azsortS = sstf_smooth.meta.stations.azimuths[posS]

dursortS = t90S[posS]-t10S[posS]
# dursortS = circular_movmean(dursortS, 5)

ampsortS = maxsS[posS]
# ampsortS = circular_movmean(ampsortS, 5)
optS, mdS, maS = osl.directivity.get_cosparams(azsortS, dursortS, ampsortS)

# %%

def get_anglefrom_duration(md):
    A, K, theta0 = md
    
    if A > 0:
        theta = theta0  + np.pi
    else:
        theta = theta0 
        
    if theta < 0:
        theta += 2*np.pi
        
    if theta > 2*np.pi:
        theta -= 2*np.pi
        
    return np.degrees(theta)

angleP = get_anglefrom_duration(mdP)
angleS = get_anglefrom_duration(mdS)


fig = plt.figure(figsize=(9,4))
gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.2, width_ratios=[1, 2])

pgs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,0], hspace=0.15)
sgs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1,0], hspace=0.15)
stfgs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[:,1], wspace=0.25)


az_int = np.linspace(0, 2*np.pi, 100)
ax1 = fig.add_subplot(pgs[0])
ax1.plot(azsortP, dursortP, 'o', label='Duration', c='tab:blue', clip_on=False, alpha=0.5, markeredgecolor='k', markeredgewidth=0.25)
ax1.plot(np.degrees(az_int), osl.directivity.cosf(az_int, *mdP), '-', label='Fit', c='tab:blue', clip_on=False)
ax1.axvline(angleP, c='tab:blue', ls='--', label='Directivity')
ax1.set_xlim(0, 360)
ax1.tick_params(axis='both', which='major', labelbottom=False)
ax1.set_ylabel('T [s]')
ax1.set_title('P-wave', fontsize='medium')
# remove top and right spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2 = fig.add_subplot(pgs[1], sharex=ax1)
ax2.plot(azsortP, ampsortP, 'o', label='Maximum Ampl.', c='tab:red', clip_on=False, alpha=0.5, markeredgecolor='k', markeredgewidth=0.25)
ax2.plot(np.degrees(az_int), osl.directivity.cosf(az_int, *maP), '-', label='Fit', c='tab:red', clip_on=False)
ax2.axvline(angleP, c='tab:red', ls='--', label='Directivity')
ax2.set_xlim(0, 360)
ax2.tick_params(axis='both', which='major', labelbottom=False)
ax2.set_ylabel('Norm. A')
# remove top and right spines
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)


ax3 = fig.add_subplot(sgs[0], sharex=ax1)
ax3.plot(azsortS, dursortS, 'o', label='Duration', c='tab:blue', clip_on=False, alpha=0.5, markeredgecolor='k', markeredgewidth=0.25)
ax3.plot(np.degrees(az_int), osl.directivity.cosf(az_int, *mdS), '-', label='Fit', c='tab:blue', clip_on=False)
ax3.axvline(angleS, c='tab:blue', ls='--', label='Directivity')
ax3.set_xlim(0, 360)
ax3.tick_params(axis='both', which='major', labelbottom=False)
ax3.set_ylabel('T [s]')
ax3.set_title('S-wave', fontsize='medium')
# remove top and right spines
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax4 = fig.add_subplot(sgs[1], sharex=ax1)
ax4.plot(azsortS, ampsortS, 'o', label='Maximum Ampl.', c='tab:red', clip_on=False, alpha=0.5, markeredgecolor='k', markeredgewidth=0.25)
ax4.plot(np.degrees(az_int), osl.directivity.cosf(az_int, *maS), '-', label='Fit', c='tab:red', clip_on=False)
ax4.axvline(angleS, c='tab:red', ls='--', label='Directivity')
ax4.set_xlim(0, 360)
ax4.set_ylabel('Norm. A')
ax4.set_xlabel('Azimuth [deg]')
# remove top and right spines
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)


t = plt.gca().text(0.0, 0.0, '', fontsize='xx-small')
xxsmall = t.get_fontsize()
stf_axes = [fig.add_subplot(stfgs[0]), fig.add_subplot(stfgs[1])]

_, ax, _ = osl.plot.plot_stationwise(stfss=[pstf_smooth,sstf_smooth], limits=(0, np.max(t90P*2.5)), plot_tmaxs=False, ax=stf_axes,
                                     tickfontsize=xxsmall * 0.7, labelbox=False)
ax[0].set_title('P-wave', fontsize='medium')
ax[1].set_title('S-wave', fontsize='medium')

pos = np.argsort(pstf_smooth.meta.stations.azimuths)
y = np.arange(len(pos))
yMax = np.max(y)
t = np.arange(0, pstf_smooth.meta.npts * pstf_smooth.meta.delta, pstf_smooth.meta.delta)
scale = 2 * 1/np.max(np.abs(pstf_smooth.data))
for (_t10, _t90, _cntr, _std,_stf), _y in zip(zip(t10P[pos], t90P[pos], centroidP[pos], stdP[pos], pstf_smooth.data[pos,0,:]), y):
    zorder = yMax - _y
        
    # Indicies
    _idx10 = np.argmin(np.abs(t - _t10))
    _idx90 = np.argmin(np.abs(t - _t90))
    _idxcntr = np.argmin(np.abs(t - _cntr))

    # Vertical markers
    ax[0].plot([_t10, _t10], [_y, _y + _stf[_idx10] * scale], c='tab:blue', lw=1, zorder=zorder, ls='-', solid_capstyle='round')
    ax[0].plot([_t90, _t90], [_y, _y + _stf[_idx90] * scale], c='tab:blue', lw=1, zorder=zorder, ls='-' , solid_capstyle='round')
    ax[0].plot([_cntr, _cntr], [ _y, _y+_stf[_idxcntr] * scale], c='tab:blue', lw=1, zorder=zorder, ls='-', solid_capstyle='round')

    # Horizontal line
    ax[0].plot([_t10, _t90], [_y, _y], c='tab:blue', lw=1, zorder=zorder, solid_capstyle='round')

pos = np.argsort(sstf_smooth.meta.stations.azimuths)
y = np.arange(len(pos))
yMax = np.max(y)
scale = 2 * 1/np.max(np.abs(sstf_smooth.data))
for (_t10, _t90, _cntr, _std, _stf), _y in zip(zip(t10S[pos], t90S[pos], centroidS[pos], stdS[pos], sstf_smooth.data[pos,0,:]), y):
    zorder = yMax - _y
        
    # Indicies
    _idx10 = np.argmin(np.abs(t - _t10))
    _idx90 = np.argmin(np.abs(t - _t90))
    _idxcntr = np.argmin(np.abs(t - _cntr))

    # Vertical markers
    ax[1].plot([_t10, _t10], [_y, _y + _stf[_idx10] * scale], c='tab:blue', lw=1, zorder=zorder, ls='-' , solid_capstyle='round')
    ax[1].plot([_t90, _t90], [_y, _y + _stf[_idx90] * scale], c='tab:blue', lw=1, zorder=zorder, ls='-' , solid_capstyle='round')
    ax[1].plot([_cntr, _cntr], [ _y, _y+_stf[_idxcntr] * scale], c='tab:blue', lw=1, zorder=zorder, ls='-', solid_capstyle='round')

    # Horizontal line
    ax[1].plot([_t10, _t90], [_y, _y], c='tab:blue', lw=1, zorder=zorder, solid_capstyle='round')

plt.subplots_adjust(left=0.1, right=0.975, top=0.925, bottom=0.125)
plt.savefig(os.path.join(plotdir, 'directivity_fitting.pdf'))


# %%

# azP = pstf_smooth.meta.stations.azimuths
# posP = np.argsort(pstf_smooth.meta.stations.azimuths)
# azsortP = pstf_smooth.meta.stations.azimuths[posP]

# # dursortP = t90P[posP]-t10P[posP]
# # dursortP = circular_movmean(dursortP, 5)

# ampsortP = maxsP[posP]
# # ampsortP = circular_movmean(ampsortP, 5)
# opt, msP, meP, maP  = osl.directivity.get_cosparams_all(azsortP, t10P[posP], t90P[posP], ampsortP)


# plt.figure()
# ax = plt.subplot(111)
# plt.plot(azsortP, t10P[posP], 'o', label='Start', c='tab:blue')
# plt.plot(azsortP, t90P[posP], 'o', label='End', c='tab:orange')
# # plt.plot(azsortP, 2*stdP[pos], '-o', label='2 sigma', c='tab:orange')
# plt.plot(azsortP, osl.directivity.cosf(np.radians(azsortP), *msP), '--', label='Fit', c='tab:blue')
# plt.plot(azsortP, osl.directivity.cosf(np.radians(azsortP), *meP), '--', label='Fit', c='tab:orange')
# plt.xlim(0, 360)
# tax = ax.twinx()
# tax.plot(azsortP, ampsortP, 'o', label='Maximum Ampl.', c='tab:red')
# tax.plot(azsortP, osl.directivity.cosf(np.radians(azsortP), *maP), '--', label='Fit', c='tab:red')
# tax.set_xlim(0, 360)
# plt.savefig('test.pdf')


# %%


# azS = sstf_smooth.meta.stations.azimuths
# posS = np.argsort(sstf_smooth.meta.stations.azimuths)
# azsortS = sstf_smooth.meta.stations.azimuths[posS]

# # dursortP = t90P[posP]-t10P[posP]
# # dursortP = circular_movmean(dursortP, 5)

# ampsortS = maxsS[posS]
# # ampsortP = circular_movmean(ampsortP, 5)
# opt, msS, meS, maS  = osl.directivity.get_cosparams_all(azsortS, t10S[posS], t90S[posS], ampsortS)


# plt.figure()
# ax = plt.subplot(111)
# plt.plot(azsortS, t10S[posS], 'o', label='Start', c='tab:blue')
# plt.plot(azsortS, t90S[posS], 'o', label='End', c='tab:orange')
# # plt.plot(azsortP, 2*stdP[pos], '-o', label='2 sigma', c='tab:orange')
# plt.plot(azsortS, osl.directivity.cosf(np.radians(azsortS), *msS), '--', label='Fit', c='tab:blue')
# plt.plot(azsortS, osl.directivity.cosf(np.radians(azsortS), *meS), '--', label='Fit', c='tab:orange')
# plt.xlim(0, 360)
# tax = ax.twinx()
# tax.plot(azsortS, ampsortS, 'o', label='Maximum Ampl.', c='tab:red')
# tax.plot(azsortS, osl.directivity.cosf(np.radians(azsortS), *maS), '--', label='Fit', c='tab:red')
# tax.set_xlim(0, 360)
# plt.savefig('test.pdf')



# %%
#Now plot the azimuthal distribution of the duration and maximum amplitude of the STF

from matplotlib.gridspec import GridSpec


def plot_directivity_summary(
    pstf_smooth, sstf_smooth, t10P, t90P, centroidP, stdP, t10S, t90S, centroidS, stdS,
    azsortP, dursortP, ampsortP, mdP,  maP, azsortS, dursortS, ampsortS, mdS, maS):
    
    
    
    
    fig = plt.figure(figsize=(5, 5))
    
    gs = GridSpec(2,2, height_ratios=[1, 3], width_ratios=[1,1], hspace=0.0, wspace=0.3, figure=fig)
    
    STF_axes = [fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])]    
    
    t = plt.gca().text(0.0, 0.0, '', fontsize='xx-small')
    xxsmall = t.get_fontsize()
    
    
    _, _stf_axes, _ = osl.plot.plot_stationwise(stfss=[pstf_smooth,sstf_smooth], limits=(0, np.max(t90P)*2), plot_tmaxs=False,
                                                ax=STF_axes, tickfontsize=xxsmall * 0.5)
    
    pos = np.argsort(pstf_smooth.meta.stations.azimuths)
    y = np.arange(len(pos))
    yMax = np.max(y)
    
    
    
    t = np.arange(0, pstf_smooth.meta.npts * pstf_smooth.meta.delta, pstf_smooth.meta.delta)
    
    for (_t10, _t90, _cntr, _std, _stf), _y in zip(zip(t10P[pos], t90P[pos], centroidP[pos], stdP[pos], pstf_smooth.data[pos, 0, :]), y):
        zorder = yMax - _y
        
        # Indicies
        _idx10 = np.argmin(np.abs(t - _t10))
        _idx90 = np.argmin(np.abs(t - _t90))
        _idxcntr = np.argmin(np.abs(t - _cntr))
    
        # Vertical markers
        _stf_axes[0].plot([_t10, _t10], [_y, _y + _stf[_idx10]], c='tab:blue', lw=1, zorder=zorder, ls='-' )
        _stf_axes[0].plot([_t90, _t90], [_y, _y + _stf[_idx90]], c='tab:blue', lw=1, zorder=zorder, ls='-' )
        _stf_axes[0].plot([_cntr, _cntr], [ _y, _y+_stf[_idxcntr]], c='tab:blue', lw=2, zorder=zorder, ls='-')

        # Horizontal line
        _stf_axes[0].plot([_t10, _t90], [_y, _y], c='tab:blue', lw=1, zorder=zorder)
        
        # _stf_axes[0].plot([_cntr - _std, _cntr + _std], [_y, _y], c='tab:orange', lw=1, zorder=zorder, marker=2, markersize=12.5)

    pos = np.argsort(sstf_smooth.meta.stations.azimuths)
    y = np.arange(len(pos))
    yMax = np.max(y)
    for (_t10, _t90, _cntr, _std, _stf), _y in zip(zip(t10S[pos], t90S[pos], centroidS[pos], stdS[pos], sstf_smooth.data[pos, 0, :]), y):
        zorder = yMax - _y
        
        # Indicies
        _idx10 = np.argmin(np.abs(t - _t10))
        _idx90 = np.argmin(np.abs(t - _t90))
        _idxcntr = np.argmin(np.abs(t - _cntr))
    
        # Vertical markers
        _stf_axes[1].plot([_t10, _t10], [_y, _y + _stf[_idx10]], c='tab:blue', lw=1, zorder=zorder, ls='-' )
        _stf_axes[1].plot([_t90, _t90], [_y, _y + _stf[_idx90]], c='tab:blue', lw=1, zorder=zorder, ls='-' )
        _stf_axes[1].plot([_cntr, _cntr],[ _y, _y+_stf[_idxcntr]], c='tab:blue', lw=2, zorder=zorder, ls='-')

        # Horizontal line
        _stf_axes[1].plot([_t10, _t90], [_y, _y], c='tab:blue', lw=1, zorder=zorder)

    axP = fig.add_subplot(gs[0,0], polar=True)
    axS = fig.add_subplot(gs[0,1], polar=True)
    
    # Interpolate the circle
    az_int = np.linspace(0, 2*np.pi, 100)
    
    # Compute the fits
    durfitP = osl.directivity.cosf(az_int, *mdP)
    durfitS = osl.directivity.cosf(az_int, *mdS)
    
    # Compute amplitude fits
    ampfitP = osl.directivity.cosf(az_int, *maP)
    ampfitS = osl.directivity.cosf(az_int, *maS)

    # Plotting P data 
    zorder=10
    lw = 1
    markersize = 5
    axP.scatter(np.radians(azsortP), dursortP/np.mean(dursortP), s=markersize, c='tab:blue', label='Duration', zorder=zorder)
    axP.scatter(np.radians(azsortP), ampsortP/np.mean(ampsortP), s=markersize, c='tab:blue', label='Max. Ampl.', marker='x', zorder=zorder)
    axP.plot(az_int, durfitP/np.mean(dursortP), '-', label='Fit Duration', c='tab:blue', zorder=zorder, markersize=markersize, lw=lw)
    axP.plot(az_int, ampfitP/np.mean(ampsortP), '--', label='Fit Ampl.', c='tab:blue', zorder=zorder, markersize=markersize, lw=lw)

    # Plotting S data
    axS.scatter(np.radians(azsortS), dursortS/np.mean(dursortS), s=markersize, c='tab:red', label='Duration', zorder=zorder)
    axS.scatter(np.radians(azsortS), ampsortS/np.mean(ampsortS), s=markersize, c='tab:red', label='Max. Ampl.', marker='x')
    axS.plot(az_int, durfitS/np.mean(dursortS), '-', label='Fit Duration', c='tab:red', zorder=zorder, lw=lw)
    axS.plot(az_int, ampfitS/np.mean(ampsortS), '--', label='Fit Ampl.', c='tab:red', zorder=zorder, lw=lw)

    # Find max of the fit
    idminP = np.argmin(durfitP)
    idminS = np.argmin(durfitS)
    iAmaxP = np.argmax(ampfitP)
    iAmaxS = np.argmax(ampfitS)

    axP.plot([az_int[idminP], az_int[idminP]], [0.5, 1.25], '-', c='tab:blue', lw=2, clip_on=False, zorder=zorder, markersize=markersize)
    axS.plot([az_int[idminS], az_int[idminS]], [0.5, 1.25], '-', c='tab:red', lw=2, clip_on=False, zorder=zorder, markersize=markersize)
    axP.plot([az_int[idminP]], [1.25], 'o', c='tab:blue', lw=2, clip_on=False, zorder=zorder, markersize=markersize, markeredgecolor='k')
    axS.plot([az_int[idminS]], [1.25], 'o', c='tab:red', lw=2, clip_on=False, zorder=zorder, markersize=markersize, markeredgecolor='k')
    # axP.plot([az_int[iAmaxP], az_int[iAmaxP]], [0.5, 1.25], '-', c='tab:blue', lw=2, clip_on=False)
    # axS.plot([az_int[iAmaxS], az_int[iAmaxS]], [0.5, 1.25], '-', c='tab:red', lw=2, clip_on=False)


    for _ax, _dur, _amp in zip([axP, axS], [dursortP, dursortS], [ampsortP, ampsortS]):
        _ax.set_theta_direction(-1)
        _ax.set_theta_offset(np.pi/2)
        rmin, rmax = np.min((np.min(_dur/np.mean(_dur)), np.min(_amp/np.mean(_amp)))), \
                     np.max((np.max(_dur/np.mean(_dur)), np.max(_amp/np.mean(_amp))))
        r05 = 0.05 * (rmax-rmin)
        r50 = 0.5 * (rmax-rmin)
        _ax.set_rlim(0.5, rmax+r05)
        _ax.set_rlim(0.5,1.25)
        _ax.set_xticks(np.linspace(0,  2*np.pi, 8, endpoint=False), ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], zorder=20,
                       fontsize='x-small')
        _ax.tick_params(axis='both', which='major', pad=-5)
        _ax.tick_params(axis='y', which='major', labelsize='x-small', zorder=20)
        _ax.grid(zorder=-10, lw=0.5)
        

plot_directivity_summary(pstf_smooth, sstf_smooth, t10P, t90P, centroidP, stdP, t10S, t90S, centroidS, stdS,
                        azsortP, dursortP, ampsortP, mdP, maP, azsortS, dursortS, ampsortS, mdS, maS)
    

plt.subplots_adjust(left=0.1, right=0.975, top=0.95, bottom=0.1)
plt.savefig(os.path.join(plotdir, 'directivity_summary.pdf'))


# %%
# function to compute angle between to vectors in polar coordinates
def angle_between_polar_coordinates(theta1, theta2):
    
    # Convert the angles to cartisian coordinates
    a = np.array([np.cos(theta1), np.sin(theta1)])
    b = np.array([np.cos(theta2), np.sin(theta2)])  
    
    # Compute norms
    an = np.sqrt(np.sum(a**2))
    bn = np.sqrt(np.sum(b**2))
    
    # Compute the angle
    return np.arccos(np.dot(a, b)/(an*bn))
    

# Interpolate the circle
az_int = np.linspace(0, 2*np.pi, 100)

# Compute the fits
durfitP = osl.directivity.cosf(az_int, *mdP)
durfitS = osl.directivity.cosf(az_int, *mdS)

# Compute amplitude fits
ampfitP = osl.directivity.cosf(az_int, *maP)
ampfitS = osl.directivity.cosf(az_int, *maS)

# Find max of the fit
idminP = np.argmin(durfitP)
idminS = np.argmin(durfitS)

# Find the azimuth of the minima
Pdir = np.degrees(az_int[idminP])
Sdir = np.degrees(az_int[idminS])

# find the angle between the two directions
angle = np.degrees(angle_between_polar_coordinates(az_int[idminP], az_int[idminS]))

# %%

if angle < 90 and np.mean(dursortP) > 33.0:
    osl.utils.log(f"Directivity analysis: P-wave directivity is at {Pdir:.1f} and S-wave directivity is at {Sdir:.1f}. The angle between the two is {angle:.1f}.")
    dirok = True
else:
    osl.utils.log(f"Directivity analysis: P-wave directivity is at {Pdir:.1f} and S-wave directivity is at {Sdir:.1f}. The angle between the two is {angle:.1f}, mean duration is {np.mean(dursortP):.1f}s.")
    dirok = False

outdict = dict(
    P=Pdir,
    S=Sdir,
    few_stations=few_stations,
    directivity_ok=dirok
)

with open(os.path.join(datadir, 'directivity_summary.json'), 'w') as f:
    json.dump(outdict, f, indent=4)
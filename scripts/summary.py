# %%
import os, sys
import glob 
import json
import obsnumpy as onp
import obstflib as osl
import obsplotlib.plot as opl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import cartopy
import cartopy.crs as ccrs
import shapefile
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature



if ("IPython" in sys.argv[0]) or ("ipython" in sys.argv[0]):
    result_dir = "STF_results_surface"
else:
        
    # Which directory
    result_dir = sys.argv[1]


scardec_dirs = glob.glob(os.path.join(result_dir, "FCT*"))
scardec_dirs.sort()

# only use directory of it contains data/directivity_summary.json
scardec_dirs = [d for d in scardec_dirs if os.path.exists(os.path.join(d, "data", "directivity_summary.json"))]

# %%
eventids = []
fms = []
optstfs = []
directivities = []

for scardec_dir in scardec_dirs:

    # Get the scardec_id
    scardec_id = os.path.basename(scardec_dir)
    
    # Define directories
    datadir = os.path.join(scardec_dir, "data")
    plotdir = os.path.join(scardec_dir, "plots")
    
    # Check if the final gcmt cmtsolution exists if not continue
    if not os.path.exists(os.path.join(datadir, "fms", "gcmt_cmtsolution.txt")):
        print("Skipping {}".format(scardec_id), "because no GCMT solution")
        continue
    
    # Focal mechanism directories
    fm_dir = os.path.join(datadir, "fms")
    
    
    # Loop over the different methods
    __fms = []
    for _fm in ['gcmt', 'scardec', 'cmt3']:
            
        # Load data
        fm = osl.CMTSOLUTION.read(os.path.join(fm_dir, "{}_cmtsolution.txt".format(_fm)))
        
        if _fm == "cmt3":
            eventid = fm.eventname
        
        __fms.append(fm)
        
    eventids.append(eventid)
    fms.append(__fms)
    
    # Focal mechanism directories
    stf_dir = os.path.join(datadir, "stf")
    
    # Load the STFs
    __stfs = []
    for _stf in ['gcmt', 'scardec', 'optimal_PDE']: 
        _stffile = glob.glob(os.path.join(stf_dir, "{}*.txt".format(_stf)))[0]

        t, stf = np.loadtxt(_stffile).T
        
        __stfs.append(stf)
    
    optstfs.append(__stfs)
    
    # Load the directivity    
    direct_file = os.path.join(datadir, "directivity_summary.json") 
    
    # Load json file
    with open(direct_file, "r") as f:
        directivity = json.load(f)
    
    if directivity['few_stations'] == True or directivity['directivity_ok'] == False:
        directivity = np.nan
    else:
        directivity = directivity['P']
    
    directivities.append(directivity)

# %%
from scipy.integrate import cumtrapz, trapz

def Mw(M0):
    """Moment magnitude M_w"""
    return 2 / 3 * np.log10(7 + M0) - 10.73

# compute scalar moments from stfs and centroid
ctrs = []
mws = []
for _i, stfs in enumerate(optstfs):
    mw = []
    ctr = []
    for _j, stf in enumerate(stfs):
        
        # Compute M0
        M0_in_Nm = trapz(stf, t)
        mw.append(M0_in_Nm * 1e7)
        
        # compute the centroid
        if _j == 0:
            _ctr = fms[_i][_j].time_shift
        else:
            _ctr = np.trapz(t * stf, t) / np.trapz(stf, t)
        
        ctr.append(_ctr)
    mws.append(mw)
    ctrs.append(ctr)


# Compute duration of the STFs usin the cumulative STF
durations = []
for stfs in optstfs:
    duration = []
    for stf in stfs:
        cstf = cumtrapz(stf, t, initial=0)
        startidx = np.where(np.isclose(cstf,0.0))[0][-1]
        endidx = np.where(np.isclose(cstf/np.max(cstf), 1.0))[0][0]
        
        duration.append(t[endidx] - t[startidx])        
    durations.append(duration)
    
# latitudes and longitudes
lats = []
lons = []
depths = []
for fms_ in fms:
    lat = []
    lon = []
    depth = []
    for fm in fms_:
        lat.append(fm.latitude)
        lon.append(fm.longitude)
        depth.append(fm.depth)
    lats.append(lat)
    lons.append(lon)
    depths.append(depth)
    
# %%
# Create table

print("# Event ID, latitude[deg], longitude[deg], depth[km], Mw, centroid [s], duration [s], directivity[deg]")
for _lats, _lons, _depths, _mws, _ctrs, _durations, _directivity, _eventid in zip(lats, lons, depths, mws, ctrs, durations, directivities, eventids):
    # Get the scardec_id
    scardec_id = os.path.basename(scardec_dir)
    
    N = 14
    
    string = f"{_eventid:{N}s}, "
    
    for _i, (_lat, _lon, _depth, _mw, _ctr, _duration) in enumerate(zip(_lats, _lons, _depths, _mws, _ctrs, _durations)):
        
        if _i==0:
            space = ""
        else:
            space = " " * (N + 2)
        
        if _i ==2 and np.isnan(_directivity) == False:
            string += f"{space}{_lat:.2f}, {_lon:.2f}, {_depth:.2f}, {Mw(_mw):.2f}, {_ctr:.2f}, {_duration:.2f}, {_directivity:.2f}\n"
        else:
            string += f"{space}{_lat:.2f}, {_lon:.2f}, {_depth:.2f}, {Mw(_mw):.2f}, {_ctr:.2f}, {_duration:.2f}, \n"
            
    print(string, end="")
    
    
# %%
# Generate booktabs table
printlatex=True
N = len(lats) 

if printlatex:
    print(r'\begin{table}')
    print(r'\centering')
    print(r'\scriptsize')
    print(r'\begin{tabular}{lrrrrrr}')
    print(r'\toprule')
    print(r"\bf{GCMT ID} &  \bf{Latitude[$^\circ$]} & \bf{Longitude[$^\circ$]} & \bf{Depth [km]} & $\bf{M_w}$ & $\bf{T_c}$ \bf{[s]} & \bf{T [s]} \\")
    print(r'\cmidrule(lr){1-7}')
    for _j, (_lats, _lons, _depths, _mws, _ctrs, _durations, _directivity, _eventid) in enumerate(zip(lats[:N], lons[:N], depths[:N], mws[:N], ctrs[:N], durations[:N], directivities[:N], eventids[:N])):
    
        scardec_id = os.path.basename(scardec_dir)
        
        Nspace = 14
        
        string = rf"{_eventid:{Nspace}s} "
                
        for _i, (_lat, _lon, _depth, _mw, _ctr, _duration) in enumerate(zip(_lats, _lons, _depths, _mws, _ctrs, _durations)):
            
            if _i==0:
                space = ""
            else:
                space = " " * (Nspace + 2)
            
            # if _i ==2 and np.isnan(_directivity) == False:
            #     string += f"{space} & {_lat:.2f} & {_lon:.2f} & {_depth:.2f} & {Mw(_mw):.2f} & {_ctr:.2f} &{_duration:.2f} & {_directivity:.2f} \\\\\n"
            # elif _i ==2 and np.isnan(_directivity):
            #     string += f"{space} & {_lat:.2f} & {_lon:.2f} & {_depth:.2f} & {Mw(_mw):.2f} & {_ctr:.2f} & {_duration:.2f} & --/-- \\\\\n"
            # else:
            string += f"{space} & {_lat:.2f} & {_lon:.2f} & {_depth:.2f} & {Mw(_mw):.2f} & {_ctr:.2f} & {_duration:.2f}  \\\\\n"
                
        print(string, end="")
        

        if _j == N - 1:
            print(r'\bottomrule')
        else:
            print(r'\cmidrule(lr){1-7}')
            
            
        if _j % 15 == 0 and _j != 0:
            print(r'Table continues on \\ next page')
            print(r'\end{tabular}')
            print(r'\end{table}')
            print(r'\begin{table}')
            print(r'\centering')
            print(r'\scriptsize')
            print(r'\begin{tabular}{lrrrrrr}')
            print(r'\toprule')
            print(r"\bf{GCMT ID} &  \bf{Latitude[$^\circ$]} & \bf{Longitude[$^\circ$]} & \bf{Depth [km]} & $\bf{M_w}$ & $\bf{T_c}$ \bf{[s]} & \bf{T [s]} \\")
            print(r'\cmidrule(lr){1-7}')
        
    
    print(r'\end{tabular}')
    print(r'\end{table}')
    
    
    
# %%

plt.figure()
ax = plt.gca()
n, bins, _ = ax.hist(Mw(np.array([_mws[0] for _mws in mws])), bins=15, fc='none', lw=1.0, edgecolor='k',  histtype="step")
ax.hist(Mw(np.array([_mws[1] for _mws in mws])), bins=bins, fc='none', lw=1.0, edgecolor='tab:red',  histtype="step")
ax.hist(Mw(np.array([_mws[2] for _mws in mws])), bins=bins, fc='None', lw=1.0, edgecolor='tab:blue', histtype="step")
plt.savefig("mws.pdf")


plt.figure()
ax = plt.gca()
dMw1 = (Mw(np.array([_mws[1] for _mws in mws])) - Mw(np.array([_mws[0] for _mws in mws])) )/Mw(np.array([_mws[0] for _mws in mws]))
dMw2 = (Mw(np.array([_mws[2] for _mws in mws])) - Mw(np.array([_mws[0] for _mws in mws])) )/Mw(np.array([_mws[0] for _mws in mws]))
n, bins, _ = ax.hist(dMw1, bins=15, fc='none', lw=1.0, edgecolor='tab:red',  histtype="step")
ax.hist(dMw2, bins=bins, fc='None', lw=1.0, edgecolor='tab:blue', histtype="step")
plt.savefig("dmws.pdf")


# %%

Mws = np.array(mws)
latitudes = np.array(lats)
longitudes = np.array(lons)
dep = np.array(depths)

# %%
# Summary figure
import matplotlib

def fix_axes(ax: matplotlib.axes.Axes):
    ticklabelfontsize = 'xx-small'
    labelfontsize = 'xx-small'
    legendfontsize = 'xx-small'
    
    ax.tick_params(axis='both', which='major', labelsize=ticklabelfontsize)
    ax.xaxis.get_label().set_fontsize(labelfontsize)
    ax.yaxis.get_label().set_fontsize(labelfontsize)
    
    # Get legends
    legends = [c for c in ax.get_children() if isinstance(c, matplotlib.legend.Legend)]
    
    for _lg in legends:
        plt.setp(_lg.get_texts(), fontsize=legendfontsize)

    
lgdict = dict(handletextpad=-0.5,columnspacing=0.25)

fig = plt.figure(figsize=(5.5, 3.5))

gs = GridSpec(3, 3, figure=fig)

# Changes in geo location
ax = fig.add_subplot(gs[0,0])

opl.plot_label(ax, '(a)', location=17, fontsize='xx-small', box=False, fontweight='bold')

ax.scatter(longitudes[:, 1]- longitudes[:, 0], latitudes[:, 1]- latitudes[:, 0], s=10, marker='o', alpha=0.5, label="SCARDEC-GCMT", linewidth=0.0)
ax.scatter(longitudes[:, 2]- longitudes[:, 0], latitudes[:, 2]- latitudes[:, 0], s=10, marker='o', alpha=0.5, label="CMT3D+ -GCMT", linewidth=0.0)
ax.set_xlabel("$\Delta$Lon [deg]")
ax.set_ylabel("$\Delta$Lat [deg]")
# ax.legend(frameon=False, ncol=2, bbox_to_anchor=(0.5, 1), loc='lower center', **lgdict)

fix_axes(ax)


ax = fig.add_subplot(gs[0,1])
xmw = np.sort(Mws[:, 0])
opl.plot_label(ax, '(b)', location=17, fontsize='xx-small', box=False, fontweight='bold')
params1 = np.polyfit(np.log(Mws[:, 0]), np.log(Mws[:, 1]), 1)
params2 = np.polyfit(np.log(Mws[:, 0]), np.log(Mws[:, 2]), 1)


ax.plot(xmw, xmw, 'k-', zorder = -2, lw=0.5)
ax.plot(xmw, params1[0] * xmw + params1[1], 'tab:blue', zorder = -1, lw=0.25)
ax.plot(xmw, params2[0] * xmw + params2[1], 'tab:orange', zorder = -1, lw=0.25)

ax.scatter(Mws[:, 0], Mws[:, 1], s=4, marker='o', alpha=0.5, linewidth=0.0, c='tab:blue', label="SCARDEC")
ax.scatter(Mws[:, 0], Mws[:, 2], s=4, marker='o', alpha=0.5, linewidth=0.0, c='tab:orange', label="CMT3D+")
ax.set_xlabel("GCMT $M_0$")
ax.set_ylabel("New $M_0$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(frameon=False, ncol=2, bbox_to_anchor=(0.5, 1), loc='lower center', **lgdict, borderaxespad=0.0)
fix_axes(ax)

# Changes in Moment magnitude
ax = fig.add_subplot(gs[0,2])
opl.plot_label(ax, '(c)', location=17, fontsize='xx-small', box=False, fontweight='bold')
# dMw1 = (Mws[:, 1]- Mws[:, 0])/Mws[:, 0] * 100
# dMw2 = (Mws[:, 2]- Mws[:, 0] )/Mws[:, 0] * 100
dMw1 = np.log10((Mws[:, 1])/Mws[:, 0])
dMw2 = np.log10((Mws[:, 2])/Mws[:, 0])
smu, ssigma = np.mean(dMw1), np.std(dMw1)
cmu, csigma = np.mean(dMw2), np.std(dMw2)
label = f"S: $\mu$={smu:4.1f}% $\sigma$={ssigma:3.1f}%\nC: $\mu$={cmu:4.1f}% $\sigma$={csigma:3.1f}%"
_t = opl.plot_label(ax, label, location=2, fontsize='xx-small', box=False)
fs = _t.get_fontsize()
_t.set_fontsize(fs * 0.75)

n, bins, _ = ax.hist(dMw1, bins=15, fc='none', lw=1.0, edgecolor='tab:blue',  histtype="stepfilled")
ax.hist(dMw2, bins=bins, fc='None', lw=1.0, edgecolor='tab:orange', histtype="stepfilled")
ax.set_xlabel("$\log M_1/M_0$")
ax.set_ylim(0, ax.get_ylim()[1]*1.3)
fix_axes(ax)


ax = fig.add_subplot(gs[1,0])
opl.plot_label(ax, '(d)', location=17, fontsize='xx-small', box=False, fontweight='bold')
# change in depth
xmw = np.sort(dep[:, 0])

params1 = np.polyfit(dep[:, 0], dep[:, 1], 1)
params2 = np.polyfit(dep[:, 0], dep[:, 2], 1)

ax.plot(xmw, xmw, 'k-', zorder = -2, lw=0.5)
ax.plot(xmw, params1[0] * xmw + params1[1], 'tab:blue', zorder = -1, lw=0.25)
ax.plot(xmw, params2[0] * xmw + params2[1], 'tab:orange', zorder = -1, lw=0.25)
ax.scatter(dep[:, 0], dep[:, 1], s=4, marker='o', alpha=0.5, linewidth=0.0, c='tab:blue', label="SCARDEC")
ax.scatter(dep[:, 0], dep[:, 2], s=4, marker='o', alpha=0.5, linewidth=0.0, c='tab:orange', label="CMT3D+")
ax.set_xlabel("GCMT Depth [km]")
ax.set_ylabel("New Depth [km]")
ax.set_xscale('log')
ax.set_yscale('log')
fix_axes(ax)

# Changes in Moment magnitude
ax = fig.add_subplot(gs[2,0])
opl.plot_label(ax, '(e)', location=17, fontsize='xx-small', box=False, fontweight='bold')
dz1 = dep[:, 1]- dep[:, 0]
dz2 =  dep[:, 2]- dep[:, 0]

# COmpute statitistics and create label
smu, ssigma = np.mean(dz1), np.std(dz1)
cmu, csigma = np.mean(dz2), np.std(dz2)
label = f"S: $\mu$={smu:4.1f}km $\sigma$={ssigma:3.1f}km\nC: $\mu$={cmu:4.1f}km $\sigma$={csigma:3.1f}km"
_t = opl.plot_label(ax, label, location=2, fontsize='xx-small', box=False)
fs = _t.get_fontsize()
_t.set_fontsize(fs * 0.75)
n, bins, _ = ax.hist(dz1, bins=15, fc='none', lw=1.0, edgecolor='tab:blue',  histtype="stepfilled")
ax.hist(dz2, bins=bins, fc='None', lw=1.0, edgecolor='tab:orange', histtype="stepfilled")
ax.set_xlabel("$\Delta z$ [km]")
fix_axes(ax)
ax.set_ylim(0, ax.get_ylim()[1]*1.3)
plt.subplots_adjust(wspace=0.5, hspace=0.5)

# fig = plt.figure()

_ax = fig.add_subplot(gs[1:,1:])

_ax.axis('off')
opl.plot_label(_ax, '(f)', location=17, fontsize='xx-small', box=False, fontweight='bold', dist=0.01)
ax = opl.axes_from_axes(_ax, 323, [-0.15, 0.0, 1.2, 1.0], projection=ccrs.Robinson(central_longitude=180))
cax = opl.axes_from_axes(ax, 11012391, [0.2, -0.1, 0.6, 0.05])
# ax = fig.add_subplot(111, projection=ccrs.Robinson(central_longitude=180))


vmin, vmax= np.min(dep[:,2]), np.max(dep[:,2])

ax.set_global()
ax.add_feature(cartopy.feature.LAND, facecolor=(0.9, 0.9, 0.9))

file = 'plates/PB2002_boundaries.shp'

r = Reader(file)

for record in r.records():
    if record.attributes["Type"]=="subduction":
        col = 'tab:red'
        lw = 0.75
    else:
        lw = 0.5
        col = 'k'
        
    g = ShapelyFeature(record.geometry, ccrs.PlateCarree(), edgecolor=col, facecolor='none', lw=lw)
    ax.add_feature(g)

for _i, _d in enumerate(directivities):
    if np.isnan(_d):
        continue
    
    
    # COmpute translation 
    length = 6.0
    olat, olon  = osl.utils.reckon(latitudes[_i, 0], longitudes[_i, 0], length, _d)
    L = ax.scatter(longitudes[_i, 2], latitudes[_i, 2], s=Mw(Mws[_i,2]), c=dep[_i, 2], transform=ccrs.Geodetic(),  zorder=2+_i, cmap='rainbow',
                   norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    col = L.to_rgba(dep[_i, 2])
    ax.plot([longitudes[_i, 2], olon], [latitudes[_i, 2], olat], '-', lw=1., c=col, transform=ccrs.Geodetic(), zorder=2+_i)

c = plt.colorbar(L, cax=cax, label="Depth [km]", orientation='horizontal')
c.set_label("Depth [km]", labelpad=0)

fix_axes(cax)

plt.subplots_adjust(wspace=0.45, hspace=0.65, left=0.08, right=0.92, top=0.925, bottom=0.15)

plt.savefig("summary.pdf")


#%%
import matplotlib
    
lgdict = dict(handletextpad=-0.5,columnspacing=0.25)

if True:
    
    fig = plt.figure(figsize=(4, 2.5))

    gs = GridSpec(2, 2, figure=fig)




    ax = fig.add_subplot(gs[0,0])
    opl.plot_label(ax, '(a)', location=17, fontsize='xx-small', box=False, fontweight='bold')
    xmw = np.sort(Mws[:, 0])
    # params1 = np.polyfit(np.log(Mws[:, 0]), np.log(Mws[:, 1]), 1)
    # params2 = np.polyfit(np.log(Mws[:, 0]), np.log(Mws[:, 2]), 1)

    ax.plot(xmw, xmw, 'k-', zorder = -2, lw=0.5)
    # ax.plot(xmw, params1[0] * xmw + params1[1], 'tab:blue', zorder = -1, lw=0.25)
    # ax.plot(xmw, params2[0] * xmw + params2[1], 'tab:orange', zorder = -1, lw=0.25)

    ax.scatter(Mws[:, 0], Mws[:, 1], s=4, marker='o', alpha=0.5, linewidth=0.0, c='tab:red', label="SCARDEC")
    ax.scatter(Mws[:, 0], Mws[:, 2], s=4, marker='o', alpha=0.5, linewidth=0.0, c='tab:blue', label="B-STF")
    ax.set_xlabel("GCMT $M_0$")
    ax.set_ylabel("New $M_0$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False, ncol=2, bbox_to_anchor=(0.5, 1), loc='lower center', **lgdict, borderaxespad=0.0)
    fix_axes(ax)

    # Changes in Moment magnitude
    ax = fig.add_subplot(gs[0,1])
    opl.plot_label(ax, '(b)', location=17, fontsize='xx-small', box=False, fontweight='bold')
    # dMw1 = (Mws[:, 1]- Mws[:, 0])/Mws[:, 0] * 100
    # dMw2 = (Mws[:, 2]- Mws[:, 0] )/Mws[:, 0] * 100
    # from scipy.stats.kde import gaussian_kde
    # kde
    dMw1 = np.log10((Mws[:, 1])/Mws[:, 0])
    dMw2 = np.log10((Mws[:, 2])/Mws[:, 0])
    smu, ssigma = np.mean(dMw1), np.std(dMw1)
    cmu, csigma = np.mean(dMw2), np.std(dMw2)
    label = f"S: $\mu$={smu:6.3f} $\sigma$={ssigma:4.2f}\nB: $\mu$={cmu:6.3f} $\sigma$={csigma:4.2f}"
    _t = opl.plot_label(ax, label, location=1, fontsize='xx-small', box=dict(facecolor='white', edgecolor='none', lw=0.5, alpha=0.75,boxstyle='square,pad=0'))
    fs = _t.get_fontsize()
    _t.set_fontsize(fs * 0.75)

    n, bins, _ = ax.hist(dMw1, bins=15, fc='none', lw=1.0, edgecolor='tab:red',  histtype="stepfilled")
    ax.hist(dMw2, bins=bins, fc='None', lw=1.0, edgecolor='tab:blue', histtype="stepfilled")
    # ax.set_xlabel("$\Delta M_0/M_0$ [%]")
    ax.set_xlabel("$\log M_1/M_0$")
    ax.set_ylim(0, ax.get_ylim()[1]*1.3)
    ax.axvline(0, color='k', lw=0.5, ls='--', zorder= -1)
    fix_axes(ax)

    ax = fig.add_subplot(gs[1,0])
    opl.plot_label(ax, '(c)', location=17, fontsize='xx-small', box=False, fontweight='bold')
    # change in depth
    xmw = np.sort(dep[:, 0])

    # params1 = np.polyfit(dep[:, 0], dep[:, 1], 1)
    # params2 = np.polyfit(dep[:, 0], dep[:, 2], 1)

    ax.plot(xmw, xmw, 'k-', zorder = -2, lw=0.5)
    # ax.plot(xmw, params1[0] * xmw + params1[1], 'tab:blue', zorder = -1, lw=0.25)
    # ax.plot(xmw, params2[0] * xmw + params2[1], 'tab:orange', zorder = -1, lw=0.25)
    ax.scatter(dep[:, 0], dep[:, 1], s=4, marker='o', alpha=0.5, linewidth=0.0, c='tab:red', label="SCARDEC")
    ax.scatter(dep[:, 0], dep[:, 2], s=4, marker='o', alpha=0.5, linewidth=0.0, c='tab:blue', label="B-STF")
    ax.set_xlabel("GCMT Depth [km]")
    ax.set_ylabel("New Depth [km]")
    ax.set_xscale('log')
    ax.set_yscale('log')
    fix_axes(ax)

    # Changes in Moment magnitude
    ax = fig.add_subplot(gs[1,1])
    opl.plot_label(ax, '(d)', location=17, fontsize='xx-small', box=False, fontweight='bold')
    dz1 = dep[:, 1]- dep[:, 0]
    dz2 =  dep[:, 2]- dep[:, 0]

    # COmpute statitistics and create label
    smu, ssigma = np.mean(dz1), np.std(dz1)
    cmu, csigma = np.mean(dz2), np.std(dz2)
    label = f"S: $\mu$={smu:4.1f}km $\sigma$={ssigma:3.1f}km\nB: $\mu$={cmu:4.1f}km $\sigma$={csigma:3.1f}km"
    _t = opl.plot_label(ax, label, location=2, fontsize='xx-small', box=False)
    fs = _t.get_fontsize()
    _t.set_fontsize(fs * 0.75)
    n, bins, _ = ax.hist(dz1, bins=15, fc='none', lw=1.0, edgecolor='tab:red',  histtype="stepfilled")
    ax.hist(dz2, bins=bins, fc='None', lw=1.0, edgecolor='tab:blue', histtype="stepfilled")
    ax.set_xlabel("$\Delta z$ [km]")
    ax.axvline(0, color='k', lw=0.5, ls='--', zorder= -1)
    fix_axes(ax)
    ax.set_ylim(0, ax.get_ylim()[1]*1.3)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # fig = plt.figure()


    plt.subplots_adjust(wspace=0.2, hspace=0.65, left=0.11, right=0.95, top=0.925, bottom=0.15)

    plt.savefig("summary_sub.pdf")




# %%


def all_stfs(t, eventids, opstfs, fms):
    fig = plt.figure(figsize=(6.25, 9))
    gs = GridSpec(1, 2, figure=fig, wspace=0.6)
    axes = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])]
    scale = 20.0
    axestickloc = [[],[]]
    axesticklabels = [[],[]]
    axeslat = [[],[]]
    axeslon = [[],[]]
    axesmw = [[],[]]
    
    div = 59
    for _i, (eventid, stfs, fm) in enumerate(zip(eventids, opstfs, fms)):
    
    
        if _i // div == 0:
            ax = axes[0]
            axesticklabels[0].append(eventid + f'-{_i:02d}')
            axeslat[0].append(fm[2].latitude)
            axeslon[0].append(fm[2].longitude)
            axesmw[0].append(Mw(fm[2].M0))
        else: 
            ax = axes[1]
            axesticklabels[1].append(eventid+ f'-{_i:02d}')
            axeslat[1].append(fm[2].latitude)
            axeslon[1].append(fm[2].longitude)
            axesmw[1].append(Mw(fm[2].M0))
        
        # Correct index for the subplot
        y = _i % div
        axestickloc[_i // div].append(y)
        
        gcmt = osl.utils.triangle_stf(t, fm[0].time_shift, fm[0].hdur)*fm[0].M0/1e7
        # print(fm[0].hdur, fm[0].time_shift)
        ax.plot(t, y + scale*gcmt/fm[0].M0*1e7, color='black',    lw=0.5)
        ax.plot(t, y + scale*stfs[1]/fm[0].M0*1e7, color='tab:red',  lw=0.5)
        ax.plot(t, y + scale*stfs[2]/fm[0].M0*1e7, color='tab:blue', lw=0.5)
    
    for _j, ax in enumerate(axes):
        
        ax.set_xlim(0,200)
        
        # Create the ticklabel
        bbox_dict = dict(fc="w", ec="None", lw=0.0, alpha=0.85, pad=0.1)
            
        
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Create a twin axis on the right side with geographical information
        twin_ax = ax.twinx()
        twin_ax.spines['left'].set_visible(False)
        twin_ax.spines['right'].set_visible(False)
        twin_ax.spines['top'].set_visible(False)
        twin_ax.spines['bottom'].set_visible(False)
        
        _t = opl.plot_label(ax, f'', location=17, fontsize='xx-small', box=False, fontweight='bold')
        xxsmall = _t.get_fontsize()
        
        geoformat = "{:>3.0f},{:>3.0f},{:>3.1f}"
        _d = [geoformat.format(lat, lon, mw) for lat, lon, mw in zip(axeslat[_j], axeslon[_j], axesmw[_j])]
        _d.append(f'Lat [$^\circ$], Lon [$^\circ$], $M_w$')
       
        ticklocs = axestickloc[_j].copy()
        ticklocs.append(ticklocs[-1]+1)
        
        fontscale = 0.9
        ax.set_yticks(axestickloc[_j], axesticklabels[_j], fontsize=xxsmall * fontscale, va='baseline')    
        twin_ax.set_yticks(ticklocs, _d, rotation=0, va='baseline', fontsize=xxsmall * fontscale, ha='right', bbox=bbox_dict)

        ax.set_ylim(-0.25, 60.25)
        twin_ax.set_ylim(-0.25, 60.25)
        
        ax.tick_params(axis='y', left=False,)
        ax.tick_params(axis='x', labelsize='small')
        twin_ax.tick_params(axis='both', left=False, right=False, top=False, bottom=False)

        ax.set_xlabel("Time since PDE [s]", fontsize='small')

t = np.arange(0, 3800, 1, dtype=np.float64)
all_stfs(t, eventids, optstfs, fms)
plt.subplots_adjust(top=0.975, bottom=0.05, left=0.15, right=0.97)
plt.savefig("allstfs.pdf")
plt.close()
        
        
print(len(eventids))
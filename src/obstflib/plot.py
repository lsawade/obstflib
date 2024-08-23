import os
import typing as tp
import numpy as np
import obsnumpy as onp
import obsplotlib.plot as opl
import matplotlib.pyplot as plt
from . import utils

def plot_check(ds1: onp.Dataset, ds2: onp.Dataset, label1='Direct', label2='Obspy',
               network = 'II', station = 'BFO', component = 'Z',
               outfile = 'atest.pdf'):


    # Get the station component on the subset
    idx = ds1.meta.stations.codes.tolist().index(f'{network}.{station}')
    idc = ds1.meta.components.index(component)
    idx_check = ds2.meta.stations.codes.tolist().index(f'{network}.{station}')
    idc_check = ds2.meta.components.index(component)

    plt.figure()
    lw = 0.25
    plt.subplot(111)
    plt.plot(ds1.t, ds1.data[idx,idc, :], "-k", label=label1, lw=lw)
    plt.plot(ds2.t, ds2.data[idx_check, idc_check, :], "r-", label=label2,    lw=lw)
    plt.xlim(0, 3600)
    plt.legend(frameon=False)
    plt.legend(frameon=False)
    plt.savefig(outfile, dpi=300)
    
    
# Create function that automatically set time scaling based on time range
# up to 10 minutes should be in seconds, up to 60 minutes should be in minutes,
# and everything beyond should be in hours
def set_time_scaling(t):
    if t < 600:
        return 1, 's'
    elif t < 7200:
        return 60, 'm'
    else:
        return 3600, 'h'
    


def plot_check_section(dss, labels=['Observed','Synthetic'],
                       component = 'Z', mindist=30.0, maxdist=np.inf,scale=1.0,
                       limits = None, start_idx=0, step_idx=1, end_idx=1000000,
                       plot_misfit_reduction=False, 
                       colors = ['k', 'tab:red', 'tab:blue', 'tab:orange'],
                       lw = 0.75,
                       fill=False,
                       fillcolors = ['k', 'tab:red', 'tab:blue', 'tab:orange'],
                       fillkwargs: dict = {},
                       azi: bool = False,
                       plot_real_distance: bool = False,
                       remove_spines: bool = True,
                       plot_right_annotations: bool = True,
                       legendkwargs: dict = {}, 
                       va='center',
                       tickfontsize='small',
                       vals=None, valformat="{:.2f}", valtitle=None,
                       positive_only=False):

    # Check if the first dataset has an origin time attribute
    if hasattr(dss[0].meta.stations.attributes, 'origin_time'):
        tshift = dss[0].meta.starttime - dss[0].meta.stations.attributes.origin_time
        xlabel = 'Time since origin'
        if limits is not None:
            xlim = limits
        else:
            xlim = (0, np.max(dss[0].t + tshift))
    else:
        tshift = 0.0
        xlabel = 'Time'
        xlim = (np.min(dss[0].t), np.max(dss[0].t))
        
    # Update time scaling
    tscale, unit = set_time_scaling(xlim[1]- xlim[0])
    xlabel = f"{xlabel} [{unit}]"
    xlim = (xlim[0] / tscale, xlim[1] / tscale)

    
    # Create figure
    ax = plt.gca()

    _scale = scale * 1/np.max([np.max(np.abs(ds.data)) for ds in dss])

    minY, maxY = np.inf, -np.inf

    for i, ds in enumerate(dss):
        idx = np.where((ds.meta.stations.distances > mindist) &
                       (ds.meta.stations.distances < maxdist))[0]
        if azi:
            idx2 = np.argsort(ds.meta.stations.azimuths[idx])
        else:
            idx2 = np.argsort(ds.meta.stations.distances[idx])
        
        
        # Get the positions in the array
        pos = idx[idx2]
        
        # Get subset
        pos = pos[start_idx:end_idx:step_idx]
        
        if i == 0:
            POS = pos
            
        # Get which component to plot
        ic = ds.meta.components.index(component)

        # Offset seismograms by value of one
        if plot_real_distance:
            if azi:
                x = ds.meta.stations.azimuths[pos]
            else:
                x = ds.meta.stations.distances[pos]            
        else:
            x = np.arange(len(pos))
        
        # Seismogram array to be plotted
        y = _scale * ds.data[pos, ic, :].T + x
        
        # Get data range
        _minY, _maxY = np.nanmin(y), np.nanmax(y)

        # Update min/max
        minY = np.minimum(_minY, minY)
        maxY = np.maximum(_maxY, maxY)
        
        if fill:
            # Note that we make the zorder dependent on x since ax is growing but the 
            # data are not sorted
            xM = np.max(x)
            for _i, (_y,_x) in enumerate(zip(y.T, x)):
                zorder = xM - _x 
                ax.fill_between((ds.t + tshift)/tscale, _x, _y, fc=fillcolors[i], **fillkwargs, zorder=zorder-0.0001, clip_on=True)
                ax.plot((ds.t + tshift)/tscale, _y, c=colors[i], lw = lw, clip_on=True, zorder=zorder)
        else:
            ax.plot((ds.t + tshift)/tscale, y, c=colors[i], lw = lw)
            ax.plot([],[], c=colors[i], label=labels[i], lw = lw)

    # Splitup network and station codes 
    netcodes = [_code.split('.')[0] for _code in dss[0].meta.stations.codes]
    stacodes = [_code.split('.')[1] for _code in dss[0].meta.stations.codes]
    
    
    # Get ytick labels
    if not plot_real_distance:
        
            _d = [f"{netcodes[i]:>2s}.{stacodes[i]:<4s}" for i in POS]
            ax.set_yticks(x, _d, rotation=0, va=va, fontsize=tickfontsize)
            ax.tick_params(axis="y", direction='out', pad=0)
            
    # Compute misfit reduction only if three datasets are provided
    if plot_misfit_reduction and len(dss) == 3:
        
        # Define normalize l2 misfit
        def norm_l2(obs, syn):
            return np.sum((syn.data - obs.data)**2, axis=-1)/np.sum(obs.data**2, axis=-1)
        
        # Compute the two misfits
        m1 = norm_l2(dss[0], dss[1])[:,0]
        m2 = norm_l2(dss[0], dss[2])[:,0]
        
        # Compute overall misfit reduction
        total_misfit_reduction = 100*(np.sum(m1) - np.sum(m2))/np.sum(m1)
        
        # Trace-wise misfit reduction
        misfit_reduction = 100*(m1 - m2)/m1
        
        # Plot text about the total misfit reduction
        plt.text(1.01, 0.0, f'{total_misfit_reduction:>3.0f}%', ha='left', va='bottom', 
                 fontweight='bold', 
                 transform=ax.transAxes)

        # Create a twin axis on the right side with misfit reduction
        twin_ax = ax.twinx()
        
        # Remove spines if desired
        if remove_spines:
            twin_ax.spines['left'].set_visible(False)
            twin_ax.spines['right'].set_visible(False)
            twin_ax.spines['top'].set_visible(False)
            twin_ax.spines['bottom'].set_visible(False)
            twin_ax.tick_params(axis='both', left=False, right=False, top=False, bottom=False)
        
        twin_ax.set_yticks(x,[f"{misfit_reduction[i]:>3.0f}%" for i in POS],
            rotation=0, va=va, fontsize=tickfontsize)
    
    # If desired plot either azimuth or distance on the right side
    if plot_right_annotations:
        
        # Make sure we arent plotting other things
        if not plot_real_distance and not plot_misfit_reduction:
            
            # Create a twin axis on the right side with misfit reduction
            twin_ax = ax.twinx()
            
            # Remove spines if desired
            if remove_spines:
                twin_ax.spines['left'].set_visible(False)
                twin_ax.spines['right'].set_visible(False)
                twin_ax.spines['top'].set_visible(False)
                twin_ax.spines['bottom'].set_visible(False)
                twin_ax.tick_params(axis='both', left=False, right=False, top=False, bottom=False)
                
            # Plot epicentral distance info on the right side
            _x = np.append(x, x[-1] + 1)
            bbox_dict = dict(fc="w", ec="None", lw=0.0, alpha=0.85, pad=0.1)
            
            if vals is not None:
                geoformat = "{:>3.0f},{:>3.0f}," + valformat
                _d = [geoformat.format(dss[0].meta.stations.distances[i], dss[0].meta.stations.azimuths[i], vals[i]) for i in POS]
                _d.append(f'$\Delta[^\circ],\mathrm{{Az.}}[^\circ],{valtitle}$')
            else:
                geoformat = "{:>3.0f},{:>3.0f}"
                _d = [geoformat.format(dss[0].meta.stations.distances[i], dss[0].meta.stations.azimuths[i]) for i in POS]
                _d.append('$\Delta[^\circ],\mathrm{Az.}$')
                
            twin_ax.set_yticks(_x, _d, rotation=0, va=va, fontsize=tickfontsize, ha='right', bbox=bbox_dict)
            twin_ax.tick_params(axis="y",direction="in", pad=0)
        
    # Get extra legend arguments
    _legendkwargs = {'frameon': False, 'loc': 'upper right', 'ncol': 5, 'fontsize': 'small'}
    _legendkwargs.update(legendkwargs)
    ax.legend(**_legendkwargs)
    
    # Remove left, right and top spines
    if remove_spines:
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Remove ticks on the left
        ax.tick_params(axis='y', left=False)
        
    # Set x axis limits and labels
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    
    # Get some percentage of min/max Y range
    if plot_real_distance:
        dy = 0.01 * (maxY - minY)
        ax.set_ylim(minY - dy, maxY + 5*dy)
    else:
        if positive_only:
            ax.set_ylim(-0.25, np.maximum(15, len(POS) + 1 + scale))
        else:
            ax.set_ylim(-scale, np.maximum(15, len(POS) + 1 + scale) )
    
    if plot_misfit_reduction and len(dss) == 3 or (plot_right_annotations and not plot_real_distance and not plot_misfit_reduction):
        twin_ax.set_xlim(ax.get_xlim())    
        twin_ax.set_ylim(ax.get_ylim())    
        return ax, twin_ax
    else:
        return ax
    
# Function to plot a beach ball into a specific axis

def plotb(
    x,
    y,
    tensor,
    linewidth=0.25,
    width=100,
    facecolor="k",
    clip_on=False,
    alpha=1.0,
    normalized_axes=True,
    ax=None,
    pdf=True,
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

    
def plot_full_section(dss, labels, stfs, mt, stf_scale=1e23, scale=5.0, limits=[0*60,60*60], outfile='full_section.pdf',
                      component='Z'):
        
    fig = plt.figure()

    ax = plot_check_section(dss, labels=labels, component=component,
                    scale=scale, start_idx=0, step_idx=1, limits=limits, plot_misfit_reduction=True,
                    legendkwargs=dict(loc='center right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0, 
                                        columnspacing=1.0, fontsize='small'))
    if isinstance(ax, tuple):
        ax, _= ax
        
    opl.plot_label(ax, '(c)', fontsize='small', location=12, box=False, dist=0.01, fontweight='bold', zorder=10)
    
    colors = ['k', 'tab:red', 'tab:blue', 'tab:orange']
    subax = opl.axes_from_axes(ax, 12341, [0.0, 1.0, 0.35, 0.075])
    
    opl.plot_label(subax, '(b)', fontsize='small', location=12, box=False, dist=0.01, fontweight='bold')
    
    
    # This is because we assume the data is the first element and of course does not
    # come with an STF
    idx_shift = len(dss) - len(stfs)
        
    for _i, _stf in enumerate(stfs):
        _stf.plot(normalize=stf_scale, lw=0.75, c=colors[_i + idx_shift])
        
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
            mt.tensor,
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
    opl.plot_label(beachax, '(a)', fontsize='small', location=3, box=False, dist=0.01, fontweight='bold', zorder=10)
    

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    fig.set_size_inches(8, 10)
    fig.savefig(outfile)
    plt.close(fig)
    
    
def plot_sub_section(dss, labels, stfs, mt, scale=5.0, limits=[12.5*60,27.5*60], outfile='sub_section.pdf',
                     start_idx=20, step_idx=6, end_idx=55, component='Z',
                     stf_scale=1e23):
        
    
    fig = plt.figure()

    ax = plot_check_section(dss, labels=labels, component=component,
                    scale=scale, start_idx=start_idx, step_idx=step_idx, end_idx=end_idx, limits=limits, plot_misfit_reduction=True,
                    legendkwargs=dict(loc='center right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0, 
                                        columnspacing=1.0, fontsize='small'))
    

    colors = ['k', 'tab:red', 'tab:blue', 'tab:orange']
    subax = opl.axes_from_axes(ax, 12341, [0.0, 1.0, 0.35, 0.125])
    
    opl.plot_label(subax, '(b)', fontsize='small', location=12, box=False, dist=0.01, fontweight='bold')
    opl.plot_label(ax, '(c)', fontsize='small', location=12, box=False, dist=0.01, fontweight='bold', zorder=10)
    
    # This is because we assume the data is the first element and of course does not
    # come with an STF
    idx_shift = len(dss) - len(stfs)
        
    for _i, _stf in enumerate(stfs):
        _stf.plot(normalize=stf_scale, lw=0.75, c=colors[_i + idx_shift])
        
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

    beachax = opl.axes_from_axes(ax, 12342, [-0.2, 1.0, 0.2, 0.125])
    beachax.axis('off')
    plotb(
            0.5,
            0.5,
            mt.tensor,
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
    opl.plot_label(beachax, '(a)', fontsize='small', location=3, box=False, dist=0.01, fontweight='bold')

    plt.subplots_adjust(left=0.2, right=0.9, top=0.85, bottom=0.175)
    fig.set_size_inches(8, 3.5)
    fig.savefig(outfile)
    plt.close(fig)
    
    

def plot_stf_comparison(stfs, labels, tmaxs, tmax_idx, costs, knots_per_second, Ns, colors=['k', 'tab:red', 'tab:blue'],
                        outfile="bstf_vs_scardec_stf_duration.pdf",
                        limits=(0, 300)):

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.75, 1]}, figsize=(6, 3.5))

    params = {
        "text.usetex": False,
        "mathtext.fontset": "cm",
    }
    plt.rcParams.update(params)
    
    # Limits endtime next multiple of 25
    tmax = tmaxs[tmax_idx]
    # limits = (0, np.ceil(tmax/50)*50)
    
    #STF plot
    for _stf, _label, _color in zip(stfs, labels, colors):
        # Compute the moment
        I = np.trapz(_stf.f, _stf.t)/stfs[0].M0
        _stf.plot(ax=a0, normalize=stfs[0].M0, lw=1.5, c=_color, label=f"I={I:.2f} " + _label)
    a0.set_xlim(limits)
    a0.set_xlabel('Time since origin [s]')
    a0.legend(frameon=False, ncol=1, fontsize='small')
    opl.plot_label(a0, '(a)', fontsize='small', location=6, box=False, dist=0.01, fontweight='bold')

    # Max duration plot 
    labelcost = r"$C^{\mathrm{corr}}$"
    markersize = 3
    k = tmaxs * knots_per_second + 1
    aic = utils.norm_AIC(np.array(costs), Ns, k, coeff=2.0)
    a1.plot(tmaxs, np.array(costs) / np.max(costs), "-ok", label=labelcost, markersize=markersize)
    a1.plot(tmaxs, aic, "-o", label="Norm. AIC", markersize=markersize*0.66, linewidth=0.5, c='tab:red')
    minAIC = np.argmin(aic)
    a1.plot(tmaxs[minAIC], costs[minAIC], "o", c="w", markersize=markersize*2.0, label="Min. AIC", zorder=-10, 
            markeredgecolor='k', markeredgewidth=1.0)
    # a1.plot(tmaxs, grad, "-o", c="tab:blue", label=labelgrad, markersize=markersize)

    tmax = tmaxs[tmax_idx]
    
    a1.axvline(tmax, color="k", linestyle=":", zorder=-1)
    
    # plt.axhline(0.001, color="k", linestyle=":", zorder=-1)
    
    # if threshold_integrated is not None:
    #     a1.axvline(threshold_integrated, color="tab:red", linestyle=":", zorder=-1)
    #      # Integral tmax
    #     from scipy.integrate import cumtrapz
    #     Ic = np.trapz(np.array(costs), tmaxs)
    #     Ig = np.trapz(np.array(grad), tmaxs)
    #     ic = cumtrapz(np.array(costs), tmaxs, initial=0)/Ic
    #     ig = cumtrapz(np.array(grad), tmaxs, initial=0)/Ig
    #     tmax_idx = np.where((ic > threshold_integrated) & (ig > threshold_integrated))[0][0]
    #     a1.axvline(tmaxs[tmax_idx], color="k", linestyle=":", zorder=-1)
    # else:
    #     a1.axhline(threshold, color="k", linestyle=":", zorder=-1)
        
    a1.text(
        tmax + 1,
        1.0,
        f"Tmax={tmax:.0f}s ",
        color="k",
        horizontalalignment="right",
        verticalalignment="top",
        fontsize="small",
    )

    a1.set_xlim(limits)
    a1.legend(frameon=False, ncol=1, fontsize='small')
    a1.set_xlabel("Maximum Source Duration T [s]")
    opl.plot_label(a1, '(b)', fontsize='small', location=6, box=False, dist=0.01, fontweight='bold')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.125, hspace=0.4)

    plt.savefig(outfile)
    plt.close(f)
    
    

def bin_angles(angles, dy: float = 0.1):
    import numpy as np

    y = np.clip(np.arange(1, -1 - dy, -dy), -1, 1)

    theta = np.arccos(y)
    x = np.sin(theta)

    # Make sure that we wave bins for both sides, positive and negative x
    x = np.concatenate([x[:-1], -x[::-1]])
    y = np.concatenate([y[:-1], y[::-1]])
    theta = np.concatenate([theta[:-1], np.pi + theta])

    # Initialize the new angles, and indeces
    new_angles = []
    new_indeces = []

    # Loop over the bins
    for i in range(len(theta) - 1):

        # Get the angles in the bin
        bin_angles = angles[(angles >= theta[i]) & (angles < theta[i + 1])]

        if len(bin_angles) == 0:
            continue

        # for later I also need the indices
        indices = np.where((angles >= theta[i]) & (angles < theta[i + 1]))

        # Choose the angle that is the closest to the center of the bin
        center = (theta[i] + theta[i + 1]) / 2
        index = np.argmin(np.abs(bin_angles - center))

        # append
        new_angles.append(bin_angles[index])
        new_indeces.append(indices[0][index])

    return (theta, x, y), (new_angles, new_indeces)



def azi_plot_bin(ds, dy=0.1):

    rad_azimuths = ds.meta.stations.azimuths * np.pi / 180

    (theta_bin, x_bin, y_bin), (new_angles, new_indeces) = bin_angles(rad_azimuths, dy=dy)

    # Update meta data
    return  ds.subset(new_indeces), theta_bin


def azi_plot(dss: tp.List[onp.Dataset], theta_bin, dy, hspace=0.0, phase='P', tshift=150.0,
             pre_window=60.0, post_window=60.0, window=200.0,
             bottomspines=False):

    if not isinstance(dss, list):
        dss = [dss,]

    # Colors
    colors = ['k', 'r', 'b']
    central_onset = True if phase in ['Love', 'Rayleigh'] else False

    # Get shorthands for the relevant metadata
    meta = dss[0].meta
    stations = meta.stations
    arrivals = stations.attributes.arrivals

    # Just get the data from the phase
    start_arrivals, end_arrivals = onp.tt.get_windows(arrivals, phase=phase, window=window)

    # Min arrivals have minimum velocity meaning they are later
    # We use this later to make each axes the same length. That way we can
    # ensure that there is skewing of the traces due to different arrival times.
    # and window lengths.
    max_window = np.max(end_arrivals - start_arrivals) + pre_window + post_window

    # Get sampling interval
    delta = meta.delta
    npts = meta.npts
    length_in_s = npts * delta
    t = np.arange(-tshift, length_in_s - tshift + delta, delta)

    # Populate things for plotting the traces.
    slices = []
    npts_window = []
    ts = []

    # Loop over arrival windows
    for _i, (_start, _end) in enumerate(zip(start_arrivals, end_arrivals)):

        idx_start = int(np.argmin(np.abs(t - (_start - pre_window))))
        idx_end = int(np.argmin(np.abs(t - (_end + post_window))))

        if central_onset:
            _t = t[idx_start:idx_end] - (_end + _start)/2
        else:
            _t = t[idx_start:idx_end] - t[idx_start] - pre_window

        ts.append(_t)
        npts_window.append(idx_end - idx_start)
        slices.append(slice(idx_start, idx_end))

    # Make text monospapced
    plt.rcParams["font.family"] = "monospace"

    plt.figure(figsize=(11, 10))
    mainax = plt.gca()
    mainax.set_zorder(20)
    mainax.axis("off")

    # Bin center for each angle
    bin_center = (theta_bin[:-1] + theta_bin[1:]) / 2

    # Initiate an axis dictionary
    axes = {}

    # Get the minimum azimuth larger than pi and the maximum azimuth smaller
    # than pi
    az_gt_pi_pos = np.where(stations.azimuths >= 180)[0]
    az_lt_pi_pos = np.where(stations.azimuths < 180.0)[0]
    idx_left_bottom_ax = az_gt_pi_pos[np.argmin(stations.azimuths[az_gt_pi_pos])]
    idx_right_bottom_ax = az_lt_pi_pos[np.argmax(stations.azimuths[az_lt_pi_pos])]

    for i in range(len(stations.azimuths)):

        try:
            np.max(np.abs(dss[0].data[i, :, slices[i]]))
        except:
            print("Window too small for:", stations.codes[i])
            continue

        # Height of the axes as a function of stretch
        stretch_height = 4.0
        r = 0.5 / stretch_height
        height = 0.5 * dy

        # Distance of the axes from the center
        # r = 0.2
        total_width = 0.5 - r
        percentage_offset = 0.1
        width = total_width * (1 - percentage_offset)
        width_offset = total_width * percentage_offset

        # Height of the axes as a function of stretch
        # stretch_height = 2.0
        # height = stretch_height * dy * r

        # Angle of the axes
        az_true = stations.azimuths[i] /180 * np.pi

        # Get bin index
        az_bin = bin_center[np.digitize(az_true, theta_bin) - 1]

        # Use azimuth to get x,y. Here azimuth is with respect to North and
        # clockkwise
        x = r * np.sin(az_bin) + 0.5
        y = stretch_height * r * np.cos(az_bin) + 0.5

        # Plot bin edges
        # for _i in theta_bin[:-1]:
        #     mainax.plot(
        #         [0.5, 0.5 + r * np.sin(_i)],
        #         [0.5, 0.5 + stretch_height * r * np.cos(_i)],
        #         c="lightgray",
        #         lw=0.1,
        #         ls="-",
        #         clip_on=False,
        #     )

        # If the azimuth is larger than pi the axis is on the left side
        axis_left = az_bin >= np.pi
        ylabel_location = "right" if axis_left else "left"
        yrotation = 0 if axis_left else 0

        # ADjust the left axes to match the width
        if axis_left:
            x = x - width - width_offset
        else:
            x = x + width_offset

        # Create extent
        extent = [x, y - .5 * height * (1 - hspace), width, height * (1 - hspace)]

        # Create axes
        ax = opl.axes_from_axes(
            mainax,
            9809834 + i,
            extent=extent,
        )

        # Add axes to output dictionary
        axes[stations.codes[i]] = ax


        # Remove topright and bottomright spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


        # Remove all ticks and labels
        if i == idx_left_bottom_ax or i == idx_right_bottom_ax:
            labelbottom = True
            ax.set_xlabel("Time since onset [s]", fontsize="small")
            bottomticks = True
        else:
            labelbottom = False
            if bottomspines:
                ax.spines["bottom"].set_visible(True)
                bottomticks = True
            else:
                ax.spines["bottom"].set_visible(False)
                bottomticks = False

        ax.tick_params(
            axis="both",
            which="both",
            bottom=bottomticks,
            top=False,
            left=False,
            right=False,
            labelbottom=labelbottom,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )

        # Remove axis background
        ax.patch.set_visible(False)

        # Set the ylabel
        ax.yaxis.set_label_position(ylabel_location)

        # Get the min/max for all signals
        print(phase, slices[i], npts)
        absmax = np.max([np.max(np.abs(dss[j].data[i, 0, slices[i]])) for j in range(len(dss))])

        # Set axis limits
        if central_onset:
            ax.set_xlim(- 0.5* max_window, 0.5 * max_window)
            ax.hlines(0.0, -0.5* max_window, 0.5 * max_window, color='lightgray', lw=0.5, zorder=-1)
        else:
            ax.set_xlim(-pre_window, max_window-pre_window)
            ax.hlines(0.0, -pre_window, max_window-pre_window, color='lightgray', lw=0.5, zorder=-1)

        ax.set_ylim(-absmax, absmax)

        # Get the xticks
        xticks_major = ax.get_xticks(minor=False)
        xticks_minor = ax.get_xticks(minor=True)

        # Plot vertical lines in the background for the
        majorscale = 0.5
        minorscale = 0.25
        ax.vlines(xticks_major, -absmax*majorscale, absmax*majorscale, color='lightgray', lw=0.5, zorder=-1)
        ax.vlines(xticks_minor, -absmax*minorscale, absmax*minorscale, color='lightgray', lw=0.5, zorder=-1)
        ax.hlines(0.0, -pre_window, max_window-pre_window, color='lightgray', lw=0.5, zorder=-1)

        # Plot the signals
        for j in range(len(dss)):
            ax.plot(ts[i], dss[j].data[i, 0, slices[i]], c=colors[j], lw=0.5)


        # label = f"{az_true*180/np.pi:.0f}",
        stationname = str(stations.codes[i])
        distance = stations.distances[i]
        label = f"{stationname}\n{distance:.0f}deg"
        # label = f"{distance:.1f}deg"

        if axis_left:
            ha = 'left'
        else:
            ha = 'right'

        ax.set_ylabel(
            label,
            rotation=yrotation,
            horizontalalignment=ha,
            verticalalignment="center",
            labelpad=10,
            fontsize="x-small",
        )
        mainax.scatter(
            x,
            y,
            s=20,
            c=az_true,
            marker="o",
            cmap="viridis",
            vmin=0,
            vmax=2 * np.pi,
            clip_on=False,
            zorder=20,
            edgecolor="k",
        )
        mainax.set_xlim(0, 1)
        mainax.set_ylim(0, 1)

    # mainax.set_aspect("equal")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)

    return mainax, axes


def plot_stationwise(stfss, limits=(0, 300), ax=None, plot_tmaxs=True):
    
    
    if ax is None:
        fig, axes = plt.subplots(1, len(stfss), figsize=(1 + 3.0 * len(stfss), 7))
    
        if len(stfss) == 1:
            axes = [axes,]
    
    else:
        axes = ax
        fig = ax[0].figure
        
    azi_true = False
    taxes = []
    for _i, (_ax, _stf) in enumerate(zip(axes, stfss)):
        
        argazi = np.argsort(_stf.meta.stations.azimuths)
        
        if azi_true:
            azis = _stf.meta.stations.azimuths[argazi]
            scale = 25.0
        else:
            azis = np.arange(len(_stf.meta.stations.azimuths))
            scale = 2.0
        
        plt.sca(_ax)
        fillcolor = (0.7,0.7,0.7)
        _, _tax = plot_check_section([_stf], ['STF'], scale=scale, limits=limits, fill=True, 
                                    fillkwargs=dict(alpha=1.0, color=fillcolor, linewidth=0.0), 
                                    step_idx=1, colors=['w'], azi=True, plot_real_distance=azi_true, 
                                    lw=0.25, remove_spines=True, component=_stf.meta.components[0],
                                    plot_right_annotations=True, va='baseline', tickfontsize='x-small',
                                    vals=_stf.meta.stations.attributes.ints[argazi], valformat='{:4.2f}', valtitle='\mathit{Int.}',
                                    positive_only=True)
        taxes.append(_tax)
        
        # if _i > 0:
        #     _ax.tick_params(axis='y', which='both', labelleft=False)
        # if len(stfss) > 1 and _i < len(stfss) - 1:
        #     _tax.tick_params(axis='y', which='both', labelright=False)
        _ax.axvline(0, color='gray', linestyle='-', zorder=-1,linewidth=0.25)
        
        if plot_tmaxs:
            _markerline, _stemlines, _baseline = _ax.stem(azis,
                                                    _stf.meta.stations.attributes.tmaxs[argazi], 
                                                    'k-', orientation='horizontal', basefmt=' ', 
                                                    linefmt='k-', markerfmt='k', 
                                                    bottom=np.max(limits[1]))
            _markerline.set_markersize(4)
            _markerline.set_clip_on(True)
            _markerline.set_zorder(400)
            _markerline.set_markerfacecolor('None')
            _markerline.set_markeredgecolor('k')
            _markerline.set_markeredgewidth(0.5)
            _markerline.set_marker(2)
            _stemlines.set_linewidth(0.5)
            _stemlines.set_zorder(400)
            _stemlines.set_clip_on(True)
            _baseline.set_clip_on(True)
            
        
        # Plot integral value labels at the end of the time limit.
        # for _j, _azi in enumerate(azis):
        #     label = f"{_stf.meta.stations.attributes.ints[argazi[_j]]:3.2f}"
        #     _ax.text(limits[1], _azi+0.05, label, 
        #              fontsize='x-small', ha='right', va='bottom', zorder=10000)
        
            
        _ax.set_xlabel('Time since origin [s]')
        plt.subplots_adjust(left=0.05, right=0.975, top=0.95, bottom=0.05, wspace=0.3)

    return fig, axes, taxes
    

def plot_stationwise_single(bstf, scardec, fstfs, fcosts, tmaxs, costs, tmax_idx, eventname, plotdir, limits=(0, 300), component='Z'):

    # Get info from stf dataset.
    avg_stf = np.mean(fstfs.data[:,0,:], axis=0)
    std_stf = np.std(fstfs.data[:,0,:], axis=0)

    argazi = np.argsort(fstfs.meta.stations.azimuths)

    from matplotlib.gridspec import GridSpec
    

    fig = plt.figure(figsize=(6, 7))
    gs = GridSpec(2,2, height_ratios=[1, 1], hspace=0.75, wspace=0.4, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(bstf.t, bstf.f/bstf.M0, 'k', label='Optimal', zorder=2)
    ax1.plot(fstfs.t, avg_stf, label='Mean', zorder=1, c='gray')
    ax1.fill_between(fstfs.t, np.maximum(0, avg_stf-std_stf), avg_stf+std_stf, alpha=0.5, zorder=-1, label='Std', color='gray')
    ax1.plot(scardec.t, scardec.f/bstf.M0, label='Scardec', zorder=0, c='tab:red')
    ax1.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize='small')
    ax1.set_xlim(*limits)
    opl.plot_label(ax1, '(a)', location=2, box=False, dist=0.0)

    # Remove top and right spine
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', which='both', labelbottom=False)


    ax2 = fig.add_subplot(gs[:, 1], sharex=ax1)
    plt.sca(ax2)
    plot_check_section([fstfs], ['STF'], scale=25.0, limits=[0,125], fill=True, 
                                fillkwargs=dict(alpha=0.75, color='gray', linewidth=0.0), 
                                step_idx=1, colors=['w'], azi=True, plot_real_distance=True, 
                                lw=0.25, remove_spines=False, component=component)
    _markerline, _stemlines, _baseline = ax2.stem(fstfs.meta.stations.azimuths[argazi], 
                                              fstfs.meta.stations.tmaxs[argazi], 
                                              'k-', orientation='horizontal', basefmt=' ', 
                                              linefmt='k-', markerfmt='k|', 
                                              bottom=np.max(limits[1]))
    _markerline.set_markersize(4)
    _markerline.set_clip_on(False)
    _markerline.set_zorder(400)
    _markerline.set_markerfacecolor('None')
    _markerline.set_markeredgecolor('k')
    _markerline.set_markeredgewidth(0.5)
    _stemlines.set_linewidth(0.5)
    _stemlines.set_zorder(400)


    # Remove top and right spine
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='x', which='both', labelbottom=False)
    opl.plot_label(ax2, '(b)', location=2, box=False, dist=0.0)

    # Set ylabel
    ax2.set_ylabel('Azimuth [$^\circ$]')
    ax2.set_xlabel('Time since origin [s]')
    plt.xlim(*limits)
    plt.ylim(0, 385)
    yticks = np.arange(0, 400, 60)
    plt.yticks(yticks, [f'{int(_y):d}' for _y in yticks])


    ax3 = fig.add_subplot(gs[1, 0])
    markersize = 3
    labelcost = r"$C^{\mathrm{corr}}_{\mathrm{avg}}$"
    labelcost_opt = r"$C^{\mathrm{corr}}_{\mathrm{opt}}$"
    labelgrad = r"$G^{\mathrm{corr}}$"
    avg_cost = np.mean(fcosts.data[:,0,:], axis=0)
    std_cost = np.std(fcosts.data[:,0,:], axis=0)
    # avg_grad = np.mean(fgrads.data[:,0,:Ntmaxs], axis=0)
    # std_grad = np.std(fgrads.data[:,0,:Ntmaxs], axis=0)
    ax3.fill_between(tmaxs, np.maximum(0, avg_cost-std_cost), avg_cost+std_cost, 
                    alpha=0.5, zorder=-1, color='gray', linewidth=0.0)
    # ax3.fill_between(newtmaxs, np.maximum(0, avg_grad-std_grad), avg_grad+std_grad, alpha=0.5, zorder=-1, color='gray', linewidth=0.0)
    # ax3.plot(newtmaxs, avg_grad, '--', label=labelgrad, c='k', lw=0.5)
    ax3.plot(tmaxs, avg_cost, '-' , label=labelcost, c='k', lw=0.5)
    ax3.plot(tmaxs, costs, 'k-', label=labelcost_opt)
    ax3.axvline(tmaxs[tmax_idx], c='k', ls='-', lw=1.5)
    ax3.axvline(np.mean(tmaxs), c='k', ls='-', lw=0.5)
    ax3.plot(tmaxs[tmax_idx], costs[tmax_idx], 'o', markersize=5, clip_on=False, 
            markerfacecolor='None', markeredgecolor='tab:red')
    # AIC plot
    # k = tmaxs * knots_per_second + 1
    # aic = utils.norm_AIC(np.array(costs), Ns, k)
    # ax3.plot(tmaxs, aic, "-o", label="Norm. AIC", markersize=markersize*0.66, 
    #         linewidth=0.5, c='tab:red')
    # minAIC = np.argmin(aic)
    # ax3.plot(tmaxs_main[minAIC], costs_main[minAIC], "o", c="w", markersize=markersize*2.0, 
    #         label="Min. AIC", zorder=-10, 
    #             markeredgecolor='k', markeredgewidth=1.0)

    # Annotations
    ax3.axhline(0, c='k', ls=':', lw=.75)
    ax3.axhline(1, c='k', ls=':', lw=.75)

    opl.plot_label(ax3, '(c)', location=2, box=dict(edgecolor='None', facecolor='w'), dist=0.0)
    ax3.legend(frameon=False, ncol=2, fontsize='small', loc='lower center', bbox_to_anchor=(0.5, 1.0))
    ax3.set_xlim(*limits)
    ax3.set_ylim(-0.05,1.05)

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_yticks([0, 1], ['0', '1'])
    # ax3.tick_params(axis='y', which='both', labelleft=False)
    ax3.set_xlabel('Time since origin [s]')

    plt.savefig(os.path.join(plotdir, f'stf_summary_{eventname}.pdf'))
    plt.close('all')

def plot_stf_end_pick(t, stf, label, extension=False, label_outside=False):
    from scipy.integrate import cumtrapz
    
    dt = t[1] - t[0]
    ax = plt.gca()
    total = np.trapz(stf, dx=dt)
    STF = cumtrapz(stf/total, dx=dt, initial=0)
    lw = 0.75
    plt.plot(t, stf/np.max(stf), 'k', lw=lw)
    plt.plot(t, STF, lw=lw)
    # plt.plot(t, 1-STF, lw=lw)
    
    # Fit decay rate of the STF
    if extension:
        
        idx, (func_exp, A_exp, B_exp), (func_log, A_log, B_log), _ = utils.find_cond_elbow_idx(t, STF)
        ax.plot(t, func_exp(t, A_exp, B_exp), 'k--', lw=0.25, zorder=-1)
        ax.plot(t, func_log(t, A_log, B_log), 'k--', lw=0.25, zorder=-1)
        
    else:
        # Mark 90% of the cumulative STF    
        idx = np.argmin(np.abs(STF - 0.90))
    
    if label_outside:
        location=7
    else:
        location=2
        
    # Plot coefficient label
    opl.plot_label(plt.gca(), f"E: {A_exp:6.4f},{B_exp:8.4f}\nL: {A_log:6.4f},{B_log:8.4f}", 
                box=dict(fc="w", ec="None", lw=0.0, alpha=0.85, pad=0.1), 
                location=location, fontsize='xx-small')
    
    # Find elbow point idx
    idx = utils.find_elbow_point(t[:idx], 1-STF[:idx]) 
    idx += 10/dt
    idx = int(idx)
    
    
    plt.axvline(t[idx], c='k', ls='-', label=f'T={int(t[idx]):d}', lw=1.0)
    
    # Mark minimum of the derivative
    opl.plot_label(plt.gca(), label, box=False, location=23, fontsize='small')
    
    # Mark 60, 80, 90, and 95 percent of the cumulative STF
    _idx = np.argmin(np.abs(STF - 0.70))
    plt.axvline(t[_idx], c='tab:blue', ls='--', label='70%', lw=lw)
    _idx = np.argmin(np.abs(STF - 0.8))
    plt.axvline(t[_idx], c='tab:red', ls='--', label='80%', lw=lw)
    _idx = np.argmin(np.abs(STF - 0.9))
    plt.axvline(t[_idx], c='tab:orange', ls='--', label='90%', lw=lw)
    _idx = np.argmin(np.abs(STF - 0.95))
    plt.axvline(t[_idx], c='tab:green', ls='--', label='95%', lw=lw)
    idx100 = np.argmin(np.abs(STF - 1.0))
    
    plt.xlim(0, t[idx100])
    
    ttick = np.arange(0, t[idx100], 10)
    ax.set_xticks(ttick, minor=True)
    ax.grid(which='both', alpha=0.5, axis='x')
    ax.grid(which='major', alpha=0.5, axis='y')    
    
    return idx, t[idx]
            
        
 
def plot_station_geometry(fig, gs, elat, elon, slat, slon, nocoast=False, better=None):
    
    from cartopy.feature import OCEAN
    from cartopy.crs import AzimuthalEquidistant, PlateCarree, Geodetic
    
    # Create axes
    ax_coast = fig.add_subplot(gs,projection=AzimuthalEquidistant(central_latitude=elat, central_longitude=elon))
    ax_grid = fig.add_subplot(gs,projection=AzimuthalEquidistant(central_latitude=90, central_longitude=0))
    ax = fig.add_subplot(gs,projection=AzimuthalEquidistant(central_latitude=elat, central_longitude=elon))

    # Set all axes global limits
    ax_coast.set_global()
    ax_grid.set_global()
    ax.set_global()
    
    # Add coast to bottom axes
    if not nocoast:
        ax_coast.add_feature(OCEAN, linewidth=0.0, facecolor='w', edgecolor='k')
        ax_coast.coastlines(resolution='110m', color='k', linewidth=0.01)
        ax_coast.set_facecolor((0.8,0.8,0.8))
        
    # Add grid to middle axes
    ax_grid.set_facecolor('None')
    gl = ax_grid.gridlines(ylocs=[0, 30, 60, 90], xlocs=None, color='black', linestyle='-', linewidth=0.25)
    gl.xlines = False 
      
    # Plot event-station geometry on top axes
    ax.set_facecolor('None')
    pc = Geodetic()
    
    if better is not None:
        bettercolor='tab:green'
        worsecolor='tab:red'
        colors = [bettercolor if _better else worsecolor for _better in better]
        for _slat, _slon, _color in zip(slat, slon, colors):
            ax.plot([elon, _slon], [elat, _slat], '-', transform=pc, lw=1.0, c=_color)
        
        ax.plot(elon, elat, 'o', transform=pc, markersize=5, markeredgecolor='k', markerfacecolor='tab:blue', markeredgewidth=0.25, zorder=10)
        ax.scatter(slon, slat, c=colors, s=5**2, marker='v', transform=pc, edgecolor='k', linewidth=0.25, zorder=10)
    else:
        for _slat, _slon in zip(slat, slon):
            ax.plot([elon, _slon], [elat, _slat], 'k-', transform=pc, lw=0.5)
        
        ax.plot(elon, elat, 'o', transform=pc, markersize=5, markeredgecolor='k', markerfacecolor='tab:blue', markeredgewidth=0.25, zorder=10)
        ax.plot(slon, slat, 'v', transform=pc, markersize=3, markeredgecolor='k', markerfacecolor='tab:red', markeredgewidth=0.25, zorder=10)
    
    return ax_coast, ax_grid, ax


def underline_annotation(text, ax):
    f = plt.gcf()

    # text isn't drawn immediately and must be given a renderer if one isn't cached.
    # tightbbox return units are in 'figure pixels', transformed to 'figure fraction'.
    tb = text.get_tightbbox(f.canvas.get_renderer()).transformed(f.transFigure.inverted())

    # Use arrowprops to draw a straight line anywhere on the axis.
    ax.annotate('', xy=(tb.xmin,tb.y0), xytext=(tb.xmax,tb.y0),
                xycoords="figure fraction",
                arrowprops=dict(arrowstyle="-", color='k'), zorder=1000)
    



def plot_focal_mechanism(ax, gcmt, cmt3, scardec_cmt,
                         gcmt_stf, scardec, combstf,
                         colors: dict, pdf=True):
    
    from copy import deepcopy
    width = 55
    
    if pdf:
        # This ratio original width * (pdf_dpi / figure_dpi / 2)
        # No idea where the 2 comes from. It's a magic number
        width = width * (72 / 100 / 2)
    else:
        width = width

    ax.axis("off")

    x = 0.0
    y_gcmt = 0.55 
    y_scardec = 0.25
    y_cmt3 = -0.05 
    
    factor_gcmt = 1.0
    factor_scardec = np.trapz(scardec.f/scardec.M0, scardec.t)
    factor_cmt3 = np.trapz(combstf.f/cmt3.M0, combstf.t)
    
    cmt3_l = deepcopy(cmt3)
    cmt3_l.M0 = cmt3_l.M0 * factor_cmt3
    scardec_cmt_l = deepcopy(scardec_cmt)
    scardec_cmt_l.M0 = scardec_cmt_l.M0 * factor_scardec
    
    print(factor_gcmt, factor_scardec, factor_cmt3)
    plotb(
        x,
        y_gcmt,
        gcmt.tensor,
        linewidth=0.25,
        width=width,
        facecolor=colors['GCMT'],
        normalized_axes=True,
        ax=ax,
    )
       
    plotb(
        x,
        y_scardec,
        scardec_cmt.tensor,
        linewidth=0.25,
        width=width,
        facecolor=colors['SCARDEC'],
        normalized_axes=True,
        ax=ax,
    )

    plotb(
        x,
        y_cmt3,
        cmt3.tensor,
        linewidth=0.25,
        width=width,
        facecolor=colors['BSTF'],
        normalized_axes=True,
        ax=ax,
    )
    
    xMw = 0.1
    htext   = "Mw    lat[dg] lon[dg] dep[km]"
    tformat = " {:.2f} {:>5.1f}  {:>6.1f}   {:>5.1f}"
    t = plt.text(xMw, 0.7, htext, fontsize='x-small', ha='left', va='bottom', fontweight='bold')
    plt.text(xMw, y_gcmt, tformat.format(gcmt.Mw, gcmt.latitude, gcmt.longitude, gcmt.depth), fontsize='x-small', ha='left', va='center')
    plt.text(xMw, y_scardec,   tformat.format(scardec_cmt_l.Mw, scardec_cmt_l.latitude, scardec_cmt_l.longitude, scardec_cmt_l.depth), fontsize='x-small', ha='left', va='center')
    plt.text(xMw, y_cmt3, tformat.format(cmt3_l.Mw, cmt3_l.latitude, cmt3_l.longitude, cmt3_l.depth), fontsize='x-small', ha='left', va='center')
    
    
def plot_summary(
    combstf,
    fstfs, fstfs_tmax,
    pcstfs,
    gcmt_stf,
    scardec,
    gcmt, cmt3, scardec_cmt,
    knots_per_second,
    weights=[2/3, 1/3],
    components=['Z', 'T'],
    phases=['P', 'S'],
    limits=(0, 300),
    nocoast=False,
    region=None,
    misfits=None,
):
    plt.rcParams['mathtext.fontset']= 'dejavusans'
    
    colors = dict(
        GCMT='k',
        SCARDEC='tab:red',
        BSTF='tab:blue',
    )
    
    fillcolor = (0.7,0.7,0.7)
    fillcolor_dark = (0.6,0.6,0.6)
    
    # Get scalarmoment scaling factor
    M0 = cmt3.M0
    
    # Get the t vector
    t = fstfs[0].t

    # Compute averages phase wise
    avg_stfs = []
    std_stfs = []
    for _i, (_c, _p) in enumerate(zip(components, phases)):
        avg_stfs.append(np.mean(fstfs[_i].data[:,0,:], axis=0))
        std_stfs.append(np.std(fstfs[_i].data[:,0,:], axis=0))
        
    # Compute full average
    avg_stf = weights[0] * np.mean(fstfs[0].data[:,0,:], axis=0) \
        + weights[1] * np.mean(fstfs[1].data[:,0,:], axis=0)
    std_stf = weights[0] * np.std(fstfs[0].data[:,0,:], axis=0) \
        + weights[1] * np.std(fstfs[1].data[:,0,:], axis=0)
        
    from matplotlib.gridspec import GridSpec
    

    fig = plt.figure(figsize=(8, 10))
    
    gs = GridSpec(7,3, height_ratios=[0.75, 1, 1,  1, 1, 1, .75], width_ratios=[1,1.15,1.15], hspace=0.3, wspace=0.3, figure=fig)
    
    # Plot some event information
    ax_event_info = fig.add_subplot(gs[0, 0])
    
    # Add region tag if available
    if region is not None:
        info_text = f"{region}\n"
    else:
        info_text = ""
        
    # Add rest of pde info
    info_text += (f"{gcmt.origin_time.isoformat().rstrip('0')}\n"
                 f"Lat: {gcmt.pde_lat:.2f} Lon: {gcmt.pde_lon:.2f}\n"
                 f"Dep: {gcmt.pde_depth:.1f} km\n"
                 f"Mw = {gcmt.Mw:.2f}")
    opl.plot_label(ax_event_info, f'Event: {gcmt.eventname}', fontsize='small', fontweight='bold', location=6, box=dict(edgecolor='None', facecolor='w'), dist=0.02)
    opl.plot_label(ax_event_info, info_text, fontsize='small', location=1, box=dict(edgecolor='None', facecolor='w'), dist=0.00)
    
    ax_event_info.axis('off')
    
    
    ax_stf_comp = fig.add_subplot(gs[1, 0])    
    Amax = np.max(combstf.f/M0)
    Aargmax = np.argmax(combstf.f/M0)
        
    plt.sca(ax_stf_comp)
    plt.fill_between(t, 0, combstf.f/M0, facecolor=colors['BSTF'], lw=0.0, alpha=0.25)
    plt.plot(t-gcmt_stf.tshift, gcmt_stf.f/M0, c=colors['GCMT'], lw=1.0, label='GCMT')
    plt.plot(t, scardec.f/M0, c=colors['SCARDEC'], lw=1.0, label='SCARDEC')
    plt.plot(t, combstf.f/M0, c=colors['BSTF'], lw=2.0, label='BSTF')
    plt.plot([t[Aargmax], limits[-1]], [Amax, Amax], c='k', ls='-', lw=0.25, zorder=5)
    plt.text(limits[-1], Amax-0.0005,
             f"{Amax:.3f}", fontsize='x-small',
             ha='right', va='top')
    ax_stf_comp.spines['top'].set_visible(False)
    ax_stf_comp.spines['right'].set_visible(False)
    ax_stf_comp.spines['left'].set_visible(False)
    ax_stf_comp.tick_params(axis='both', labelbottom=True, labelleft=False, left=False, labelsize='small')
    ax_stf_comp.set_xlim(*limits)
    ax_stf_comp.legend(frameon=False, fontsize='x-small', loc='lower right',
                       bbox_to_anchor=(1.0, 1.0), ncols=3,
                       handlelength=0.5, handletextpad=0.5,
                       columnspacing=1.0, borderaxespad=0.0,
                       borderpad=0.0)
    opl.plot_label(ax_stf_comp, '(a)', location=6, box=False, dist=0.0, fontsize='small', fontweight='bold')
    
    # Plot
    tmp_ax_cost_AIC = fig.add_subplot(gs[2, 0])
    tmp_ax_cost_AIC.axis('off')
    ax_cost_AIC = opl.axes_from_axes(tmp_ax_cost_AIC, 948230, extent=[0, .2, 1, .7], zorder=10)
    plt.sca(ax_cost_AIC)
    plot_stf_end_pick(t, avg_stf, "", extension=True, label_outside=True)
    
    # for _i, (_c, _p) in enumerate(zip(['tab:blue', 'tab:orange'], phases)):
    #     ax_cost_AIC.plot(tmaxs, costsmain[_i], '-', c=_c, lw=0.5, label=f'Cost {_p}')
    #     aic = utils.norm_AIC(costsmain[_i], fstfs[_i].data.shape[-1], k, coeff=coeffs[_i])
    #     ax_cost_AIC.plot(tmaxs, aic, "--", label=f"AIC {_p}", markersize=3, linewidth=0.5, c=_c)
    #     argaic = np.argmin(aic)
    #     ax_cost_AIC.plot(tmaxs[argaic], costsmain[_i][argaic], 'o', c=_c, markersize=3)
    # ax_cost_AIC.axhline(0, c='k', ls='-', lw=.25, zorder=-1)
    # opl.plot_label(ax_cost_AIC, f'AIC coeff: {coeffs[0]:.2f}', location=7, box=False, dist=0.0, fontsize='x-small', fontweight='bold')
    
    ax_cost_AIC.set_xlim(*limits)
    ax_cost_AIC.set_ylim(-0.05,1.0)
    # ax_cost_AIC.set_yticks([0, 1], ['0', '1'])
    ax_cost_AIC.spines['top'].set_visible(False)
    ax_cost_AIC.spines['right'].set_visible(False)
    ax_cost_AIC.legend(frameon=False, ncol=1, fontsize='x-small', loc='upper right')
    # ax_cost_AIC.tick_params(axis='both', labelbottom=False)
    ax_cost_AIC.set_xlabel('Time since origin [s]', fontsize='small')
    ax_cost_AIC.tick_params(axis='both', labelbottom=True, labelleft=True, left=True, labelsize='small')
    opl.plot_label(ax_cost_AIC, '(b)', location=6, box=False, dist=0.025, fontsize='small', fontweight='bold')
    
    # Plotting the focal mechanism
    ax_event_focal  = fig.add_subplot(gs[3, 0])
    plot_focal_mechanism(ax_event_focal, gcmt, cmt3, scardec_cmt, gcmt_stf, scardec, combstf, colors, pdf=False) 
    opl.plot_label(ax_event_focal, '(c)', location=1, box=False, dist=0.0, fontsize='small', fontweight='bold')
    
    # Plotting the average STFs in the top right
    ax_av_stf_P = fig.add_subplot(gs[0, 1])
    ax_av_stf_S = fig.add_subplot(gs[0, 2])
    phase_av_axes = [ax_av_stf_P, ax_av_stf_S]
    
    for _i, (_c, _p,_ax) in enumerate(zip(components, phases, phase_av_axes)):
        Amax = np.max(avg_stfs[_i])
        Aargmax = np.argmax(avg_stfs[_i])
                      
        plt.sca(_ax)
        plt.plot(t, avg_stfs[_i], c='k', lw=1.0)
        plt.fill_between(t, np.maximum(0, avg_stfs[_i]-std_stfs[_i]), avg_stfs[_i]+std_stfs[_i], alpha=1.0, zorder=-1, color=fillcolor_dark, linewidth=0.25, linestyle='--', edgecolor='k')
        plt.fill_between(t, 0, avg_stfs[_i], alpha=1.0, zorder=-2, color=fillcolor, linewidth=0.0)
        plt.plot([t[Aargmax], limits[-1]], [Amax, Amax], c='k', ls='-', lw=0.25, zorder=5)
        plt.text(limits[-1], Amax-0.0005, 
                 f"{Amax:.3f}", fontsize='x-small',
                 ha='right', va='top')
        plt.plot(t, pcstfs[_i].f/M0, c=colors['BSTF'], lw=2.0)
        plt.xlim(*limits)
        plt.ylim(0,None)
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)
        _ax.spines['left'].set_visible(False)
        _ax.tick_params(axis='both', labelbottom=False, labelleft=False, left=False, labelsize='small')
        
    opl.plot_label(ax_av_stf_P, '(f)', location=6, box=False, dist=0.0, fontsize='small', fontweight='bold')
    opl.plot_label(ax_av_stf_S, '(i)', location=6, box=False, dist=0.0, fontsize='small', fontweight='bold')
    opl.plot_label(ax_av_stf_P, f'P {weights[0]:.2f}', location=7, box=False, dist=0.0, fontsize='small', fontweight='bold')
    opl.plot_label(ax_av_stf_S, f'S {weights[1]:.2f}', location=7, box=False, dist=0.0, fontsize='small', fontweight='bold')

    # Plot the stationwise STFs
    ax_stat_stf_P = fig.add_subplot(gs[1:6, 1])
    ax_stat_stf_S = fig.add_subplot(gs[1:6, 2])
    phase_stat_axes = [ax_stat_stf_P, ax_stat_stf_S]
    
    fig, _, twinaxes = plot_stationwise(fstfs_tmax, limits=limits, ax=phase_stat_axes, plot_tmaxs=True)
    for _ax, _tax in zip(phase_stat_axes, twinaxes):
        ylimits = -0.25, np.maximum(15, len(fstfs[0].meta.stations.azimuths))+1
        _ax.tick_params(axis='x', labelbottom=False)
        _ax.set_xlabel('')
        _ax.set_ylim(ylimits)
        _tax.set_ylim(ylimits)
    
    opl.plot_label(ax_stat_stf_P, '(g)', location=6, box=False, dist=0.0, fontsize='small', fontweight='bold')
    opl.plot_label(ax_stat_stf_S, '(j)', location=6, box=False, dist=0.0, fontsize='small', fontweight='bold')
    
    
    
    # Plot the cost averages
    ax_av_cost_P = fig.add_subplot(gs[6, 1])
    ax_av_cost_S = fig.add_subplot(gs[6, 2])
    phase_cost_axes = [ax_av_cost_P, ax_av_cost_S]
    
    labelcost = r"$C^{\mathrm{corr}}_{\mathrm{avg}}$"
    markersize = 3
    for _i, _ax in enumerate(phase_cost_axes):
        plt.sca(_ax)
        plot_stf_end_pick(t, avg_stfs[_i], "", extension=True, label_outside=True)
        
        _ax.set_xlim(*limits)
        _ax.set_ylim(-0.05,1.0)
        # ax_cost_AIC.set_yticks([0, 1], ['0', '1'])
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)
        _ax.legend(frameon=False, ncol=1, fontsize='x-small', loc='upper right')
        # ax_cost_AIC.tick_params(axis='both', labelbottom=False)
        _ax.set_xlabel('Time since origin [s]', fontsize='small')
        _ax.tick_params(axis='both', labelbottom=True, labelleft=True, left=True, labelsize='small')
    
    opl.plot_label(ax_av_cost_P, '(h)', location=6, box=False, dist=0.025, fontsize='small', fontweight='bold')
    opl.plot_label(ax_av_cost_S, '(k)', location=6, box=False, dist=0.025, fontsize='small', fontweight='bold')
    
        
    # Only plot in the last axis
    _ax.legend(frameon=False, ncol=2, fontsize='x-small', loc='upper right')
    
    better = None
    if misfits is not None:
        better = misfits[0][0] > misfits[0][1]
                
    _, _, ax_event_map_P = plot_station_geometry(fig, gs[4,0], cmt3.latitude, cmt3.longitude, 
                          fstfs[0].meta.stations.latitudes, 
                          fstfs[0].meta.stations.longitudes, nocoast=nocoast, better=better)
    opl.plot_label(ax_event_map_P, '(d)', location=1, box=False, dist=0.0, fontsize='small', fontweight='bold')
    opl.plot_label(ax_event_map_P, 'P', fontsize='small', fontweight='bold', location=2, box=False, dist=0.01)
    
    
    if misfits is not None:
        better = misfits[1][0] > misfits[1][1]
    _, _, ax_event_map_S = plot_station_geometry(fig, gs[5,0], cmt3.latitude, cmt3.longitude, 
                          fstfs[1].meta.stations.latitudes, 
                          fstfs[1].meta.stations.longitudes, nocoast=nocoast, better=better)
    
    opl.plot_label(ax_event_map_S, '(e)', location=1, box=False, dist=0.0, fontsize='small', fontweight='bold')
    opl.plot_label(ax_event_map_S, 'S', fontsize='small', fontweight='bold', location=2, box=False, dist=0.01)
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.975, bottom=0.05)
    
    return fig, [ax_event_info, ax_stf_comp, ax_cost_AIC, ax_event_focal, ax_event_map_P, ax_event_map_S, ax_av_stf_P, ax_av_stf_S, ax_stat_stf_P, ax_stat_stf_S]
    
    

def get_vectors(i, k, curve):
    linev = np.array([k[-1] - k[0], curve[-1] - curve[0]])
    nlinev = linev / np.sqrt(np.sum(linev**2))
    vecfromfirst_to_point = np.array([k[i] - k[0], curve[i] - curve[0]])
    projectedv = np.dot(vecfromfirst_to_point, nlinev) * nlinev
    return linev, vecfromfirst_to_point, projectedv


def plot_vector(v, origin, color, linestyle, head_width=0.05):
    
    line = plt.plot(
        [origin[0], origin[0] + v[0]],
        [origin[1], origin[1] + v[1]],
        c=color,
        ls=linestyle,
    )

    arrowstart = origin + v * 0.95
    arrowlength = v * 0.05
    arrow = plt.arrow(
        arrowstart[0],
        arrowstart[1],
        arrowlength[0],
        arrowlength[1],
        shape="full",
        lw=0,
        length_includes_head=True,
        head_width=head_width,
        color=color,
    )
    return line, arrow


def plot_geometry(idx, k, curve, linev, point1, projectedv):

    head_width = 0.04

    plt.axhline(0, color="gray", linestyle="-", lw=0.75)
    plt.axvline(0, color="gray", linestyle="-", lw=0.75)

    # Line vector
    v1 = plot_vector(linev, [k[0], curve[0]], "k", "-", head_width=head_width)
    
    # # Vector from start of vurve to best point
    v2 = plot_vector(point1, [k[0], curve[0]], "r", "-", head_width=head_width)
    v3 = plot_vector(projectedv, [k[0], curve[0]], "b", "-.", head_width=head_width)
    v4 = plot_vector(
        projectedv - point1,
        [k[idx], curve[idx]],
        "g",
        "-",
        head_width=head_width,
    )

    # plt.plot(k, curve, "k", lw=1.25)
    plt.plot(k[idx], curve[idx], "ro", markeredgecolor="k", markersize=5)

    # plt.axis("equal")
    # plt.ylim(0, 1)
    # plt.xlim(0, k[-1])
    plt.yticks([0.0, 1.0])

    return v1, v2, v3, v4


def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    import matplotlib.patches as mpatches
    print(height, width)
    p = mpatches.FancyArrow(0, 0.5 * height, width, 0,
        length_includes_head=True,
        head_width=0.75 * height,
        width=0.5 / 2 * height)
    return p

def plot_elbow_point_selection(t, stf, label, extension=False, label_outside=False):
    from scipy.integrate import cumtrapz
    from matplotlib.legend_handler import HandlerPatch
    import matplotlib.patches as mpatches
    dt = t[1] - t[0]

    total = np.trapz(stf, dx=dt)
    STF = cumtrapz(stf/total, dx=dt, initial=0)
    lw = 0.75
    
    # Fit decay rate of the STF
    if extension:
        idx, (func_exp, A_exp, B_exp), (func_log, A_log, B_log), _ = utils.find_cond_elbow_idx(t, STF)
        utils.log(f'Exp: {A_exp:6.4f}, {B_exp:6.4f}')
        utils.log(f'Log: {A_log:6.4f}, {B_log:6.4f}')
    else:
        # Mark 90% of the cumulative STF    
        idx = np.argmin(np.abs(STF - 0.90))
    
    if label_outside:
        location=7
    else:
        location=23
        
    # Find elbow point idx
    idxf = utils.find_elbow_point(t[:idx], 1-STF[:idx]) 
    idxf += 10/dt
    idxf = int(idxf)
    
    ratio = np.maximum(t[idx], 300.0)/300.0
    print('ratio', ratio)
    
    axes_size = 2.0
    leftbuffer = 0.5
    rightbuffer = 1.5
    topbuffer = 0.3
    bottombuffer = 0.5

    ax_height = axes_size
    ax_width  = axes_size * ratio
    
    width = ax_width + leftbuffer + rightbuffer
    height = ax_height + topbuffer + bottombuffer
    
    leftbound = leftbuffer/width
    rightbound = 1 - rightbuffer/width
    topbound = (height - topbuffer) / height
    bottombound = bottombuffer / height
      
    # Create figure
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca()
    
    # Plot coefficient label
    opl.plot_label(plt.gca(), f"E: {A_exp:6.4f},{B_exp:8.4f}\nL: {A_log:6.4f},{B_log:8.4f}", 
                   box=dict(fc="w", ec="None", lw=0.0, alpha=0.85, pad=0.0), dist=0.0025,
                   location=location, fontsize='x-small')
    
    tmax = int(t[idxf])
    _idx95 = np.argmin(np.abs(STF - 0.95))
    _t95 = t[_idx95]
    _tx95 = t[np.argmin(np.abs(t - 2.0*_t95))]
    
    t_orig = t.copy()
    t = t.copy()/300.0
    
    vlines = []    
    vlines.append(plt.axvline(t[idxf], c='k', ls='-', label=f'T={tmax:d}', lw=1.0))
    
    print('hellloleoeoleole')
    
    stfhandles = []
    stfhandles.append(plt.plot(t, stf/np.max(stf), c='gray', lw=lw, label='STF')[0])
    # plt.plot(t, STF, lw=lw)
    stfhandles.append(plt.plot(t, STF, lw=lw, c='k', label='Int. STF')[0])
    # variable A is defined
    if extension:
        stfhandles.append(ax.plot(t, func_exp(t_orig, A_exp, B_exp), 'k--', lw=0.25, zorder=-1,
                                  label='Est. $\exp{}$')[0])
        stfhandles.append(ax.plot(t, func_log(t_orig, A_log, B_log), 'k--', lw=0.25, zorder=-1,
                                  label='Est. $\log{}$')[0])
        
        
    legend0 = ax.legend(handles=stfhandles, frameon=False, ncol=1, bbox_to_anchor=(1.0, 0.0), loc='lower left',
                        fontsize='small')
    
    # Mark minimum of the derivative
    opl.plot_label(plt.gca(), label, box=False, location=23, fontsize='small')
    
    # Mark 60, 80, 90, and 95 percent of the cumulative STF
    # _idx = np.argmin(np.abs(STF - 0.70))
    # vlines.append(plt.axvline(t[_idx], c='tab:blue', ls='--', label='70%', lw=lw))
    # _idx = np.argmin(np.abs(STF - 0.8))
    # vlines.append(plt.axvline(t[_idx], c='tab:red', ls='--', label='80%', lw=lw))
    # _idx = np.argmin(np.abs(STF - 0.9))
    # vlines.append(plt.axvline(t[_idx], c='tab:orange', ls='--', label='90%', lw=lw))
    # _idx = np.argmin(np.abs(STF - 0.95))
    vlines.append(plt.axvline(t[_idx95], c='tab:green', ls='--', label='95%', lw=lw))
    idx100 = np.argmin(np.abs(STF - 1.0))
    
    vlines.append(plt.axvline(_tx95/300, c='k', ls='--', label='T=2x $t_{95\%}$', lw=1.0))
    
    legend1 = ax.legend(handles=vlines, frameon=False, ncol=6, bbox_to_anchor=(0.0, 1.0), loc='lower left',
                        fontsize='small')  
    
    # Get the vectors
    linev, point1, projectedv = get_vectors(idxf, t[:idx], STF[:idx])
    
    # Plot vector geometry
    handles = plot_geometry(idxf, t[:idx], STF[:idx], linev, point1, projectedv)
    
    labels = ["$\mathbf{b}$", "$\mathbf{p}$", "$\hat{\mathbf{b}}(\mathbf{p}\cdot\hat{\mathbf{b}})$", "$\mathbf{d}_{\mathrm{\max}}$"]
    arrow_handles = [_handles[1] for _handles in handles]
    
    print(arrow_handles)
    plt.legend(arrow_handles, labels, 
               frameon=False, ncol=1, bbox_to_anchor=(1.0, 1.0), loc='upper left', 
               handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)},
               fontsize='small')
    
    ax.add_artist(legend0)
    ax.add_artist(legend1)
    
    plt.xlim(0, ratio * 1.1)
    plt.ylim(0.0, 1.1)
    
    
    plt.xlabel("Time since origin [s/300.0s]")

    plt.subplots_adjust(left=leftbound, right=rightbound, 
                        bottom=bottombound, top=topbound)
    
    # plt.subplots_adjust(left=leftbuffer, right=1-rightbuffer, bottom=bottombuffer, top=1-topbuffer)
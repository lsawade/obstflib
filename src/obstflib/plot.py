import typing as tp
import numpy as np
import obsnumpy as onp
import obsplotlib.plot as opl
import matplotlib.pyplot as plt

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
                       outfile = 'atestsection.pdf', limits = None, start_idx=0, step_idx=1, end_idx=1000000,
                       plot_misfit_reduction=False,
                       legendkwargs: dict = {}):

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

    # Get trace colors 
    colors = ['k', 'tab:red', 'tab:blue', 'tab:orange']
    
    # Create figure
    ax = plt.gca()

    scale = scale * 1/np.max([np.max(np.abs(ds.data)) for ds in dss])

    minY, maxY = np.inf, -np.inf

    for i, ds in enumerate(dss):
        idx = np.where((ds.meta.stations.distances > mindist) &
                       (ds.meta.stations.distances < maxdist))[0]
        idx2 = np.argsort(ds.meta.stations.distances[idx])
        
        # Get the positions in the array
        pos = idx[idx2]
        
        # Get subset
        pos = pos[start_idx:end_idx:step_idx]

        # Get which component to plot
        ic = ds.meta.components.index(component)

        # Offset seismograms by value of one
        x = np.arange(len(pos))
        
        # Seismogram array to be plotted
        y = scale * ds.data[pos, ic, :].T + x
        
        # Get data range
        _minY, _maxY = np.min(y), np.max(y)
        
        # Update min/max
        minY = np.minimum(_minY, minY)
        maxY = np.maximum(_maxY, maxY)
        
        ax.plot((ds.t + tshift)/tscale, y, c=colors[i], lw=0.75)
        ax.plot([],[], c=colors[i], label=labels[i], lw=0.75)

    # Splitup network and station codes 
    netcodes = [_code.split('.')[0] for _code in dss[0].meta.stations.codes]
    stacodes = [_code.split('.')[1] for _code in dss[0].meta.stations.codes]
    
    # Get ytick labels
    ax.set_yticks(x, [f"{netcodes[i]:>2s}.{stacodes[i]:<4s}: {dss[-1].meta.stations.distances[i]:6.2f}" for i in pos],
            rotation=0, fontsize='small')
    
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
        twin_ax.spines['left'].set_visible(False)
        twin_ax.spines['right'].set_visible(False)
        twin_ax.spines['top'].set_visible(False)
        twin_ax.spines['bottom'].set_visible(False)
        twin_ax.tick_params(axis='both', left=False, right=False, top=False, bottom=False)
        
        twin_ax.set_yticks(x,[f"{misfit_reduction[i]:>3.0f}%" for i in pos],
            rotation=0, fontsize='small')
    
    # Get extra legend arguments
    _legendkwargs = {'frameon': False, 'loc': 'upper right', 'ncol': 5, 'fontsize': 'small'}
    _legendkwargs.update(legendkwargs)
    ax.legend(**_legendkwargs)
    
    # Remove left, right and top spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Remove ticks on the left
    ax.tick_params(axis='y', left=False)
        
    # Set x axis limits and labels
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    
    # Get some percentage of min/max Y range
    dy = 0.01 * (maxY - minY)
    ax.set_ylim(minY - dy, maxY + 5*dy)
    
    if plot_misfit_reduction and len(dss) == 3:
        twin_ax.set_xlim(ax.get_xlim())    
        twin_ax.set_ylim(ax.get_ylim())    
    
    return ax
    
    
    
    
    

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


import os
from copy import deepcopy
import numpy as np    
import matplotlib.pyplot as plt
import obsnumpy as onp
import obsplotlib.plot as opl
from joblib import Parallel, delayed
from scipy.integrate import cumtrapz

from . import utils
from . import plot
from . inversion import Inversion



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
        plot.plot_check_section([ds, ds2], labels=['Observed', 'CMT3D+'],
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
    plot.plot_check_section([ds1, ds2], labels=['Observed', 'CMT3D+'],
                    scale=5.0, limits=[0*60,60*60], plot_misfit_reduction=False,component=component,
                    vals=onp.utils.L2(ds1, ds2, normalize=True)[:, idx], valtitle='L^2_N',
                    valformat='{:5.3f}', plot_right_annotations=True )
    plt.subplots_adjust(left=0.2, right=0.9, top=0.975, bottom=0.1)
    fig.set_size_inches(8, 10)
    fig.savefig(os.path.join(plotdir, f'{component}_{phase}_{label}.pdf'))
    plt.close(fig)



def full_preparation(obsd: onp.Dataset, synt: onp.Dataset, green: onp.Dataset, phase, component, 
                     event, stfs, green_is_synthetic=False,
                     plotdir="./", snr=True, plot_intermediate_figures=True,
                     labels=['Observed', 'CMT3D+', 'GF'],
                     gf_shift=-20.0, bp=None):
    
    # %%
    # Note that any P is really my definition of any P arrival from taup P
    # and that could be P, Pdiff, PKP
    utils.log("Getting arrivals")

    allphases = ['P', 'S', 'Rayleigh', 'Love', 'anyP', 'anyS']

    # Check if the phase is in the list of allphases
    if obsd.meta.stations.attributes is not None and hasattr(obsd.meta.stations.attributes, 'arrivals'):
        
        phases = []
        
        for _phase_ in allphases:
            if not hasattr(obsd.meta.stations.attributes.arrivals, _phase_):
                phases.append(_phase_)
                
    else:
        phases = allphases

    # For the user
    if len(phases) > 0:
        utils.log(f"Phases not yet computed: {phases}")
    else:
        utils.log("All phases already computed. Skipping traveltime computation.")
        
    for _phase_ in phases:

        onp.tt.get_arrivals(event, obsd, phase=_phase_)
        onp.tt.get_arrivals(event, green, phase=_phase_)
        onp.tt.get_arrivals(event, synt, phase=_phase_)

    # %%
    # Now given the selected traces we want to use the corresponding windows to taper
    # the traces such that we can perform the inversion only on the relevant windows
    # and not the whole trace.
    utils.log("Selecting windows")

    
    # Subselect the seignals based on distance phases etc.
    obsd_tt = onp.tt.select_traveltime_subset(obsd, component=component, phase=phase, 
                                               mindist=30.0, maxdist=np.inf, minwindow=300.0)
    
    utils.log(f"Reduced traces by component/phase selection {component} {phase}: {obsd.data.shape[0]} -> {obsd_tt.data.shape[0]}")
    
    # COmpute the timeshift for the taces from arrivaltime and startitem of the traes
    tshift = obsd.meta.stations.attributes.origin_time - obsd.meta.starttime

    # remove traces depending on the SNR things depending on the SNR
    if snr:
        
        onp.utils.compute_snr(obsd_tt, tshift, period=17.0, phase=phase[0])

        # Plot SNR and misfit
        _, _synt = obsd_tt.intersection(synt)        
        plot_snr(obsd_tt, component, phase, plotdir, ds2=_synt)
        
        obsd_tt_snr, _ = onp.utils.remove_snr(obsd_tt, snr_int_min_threshold=50.0,snr_int_max_threshold=1000000.0,
                                        snr_max_min_threshold=5.0, snr_max_max_threshold=10000.0,
                                        component=component)
        
        utils.log(f"Removing low/high SNR traces {obsd_tt.data.shape[0]:d} --> {obsd_tt_snr.data.shape[0]:d}")
        
        obsd_tt = obsd_tt_snr
        
        del obsd_tt_snr
        
    # Get the intersection between data and synthetics
    obsd_is, green_is = obsd_tt.intersection(green)
    _, synt_is = obsd_tt.intersection(synt)    

    # Remove components from the other arrays
    green_is = green_is.subset(components=component)
    synt_is = synt_is.subset(components=component)
   
   # Plot output if desired
    if plot_intermediate_figures:
        # Plot figure of the selected data 
        fig = plt.figure()
        plot.plot_check_section([obsd_is, synt_is, green_is], labels=labels,
                                component=component)
        plt.subplots_adjust(left=0.2, right=0.975, top=0.975, bottom=0.1)
        fig.set_size_inches(8, 6)
        fig.savefig(os.path.join(plotdir, f'{component}_{phase}_data_synt_green_tt_select.pdf'))
        plt.close(fig)

    # Now we want to taper the traces to the selected windows
    utils.log(f"Tapering windows {component} {phase}")
        
    # Taper datasets, 
    obsd_tap, taper_ds = onp.tt.taper_dataset(obsd_is, phase, tshift, gf_shift=gf_shift, return_taper=True)
    synt_tap = onp.tt.taper_dataset(synt_is, phase, tshift, gf_shift=gf_shift)
    if green_is_synthetic:
        green_tap = onp.tt.taper_dataset(green_is, phase, tshift, gf_shift=gf_shift)
    else:
        green_tap = green_is.copy() # on purpose not tapered!
        
    # Now that the traces are tapered we may bandpass them if desired
    if bp is not None:
        obsd_tap.filter(freqmin=bp[0], freqmax=bp[1], zerophase=True)
        synt_tap.filter(freqmin=bp[0], freqmax=bp[1], zerophase=True)
        green_tap.filter(freqmin=bp[0], freqmax=bp[1], zerophase=True)
    
    # Plot misfit figure
    if plot_intermediate_figures:
        plot_misfits(obsd_tap, synt_tap, component, phase, plotdir, label='misfit_data_cmt3d_tapered')


    utils.log("Removing outliers")
    utils.log("Removing outliers based on ratio and L2 norm misfit")
    utils.log(f"Component: {component} -- Phase: {phase}")
    utils.log("===================================================")
    
    # Reomve traces that are outlier in terms of misfit and get removed indices
    if obsd_tap.data.shape[0] > 50:
        fobsd, fsynt, _idx = onp.utils.remove_misfits(obsd_tap, synt_tap, misfit_quantile_threshold=0.85, ratio_quantile_threshold_above=0.85, ratio_quantile_threshold_below=0.15)
    else:
        fobsd, fsynt, _idx = onp.utils.remove_misfits(obsd_tap, synt_tap, misfit_quantile_threshold=0.9, ratio_quantile_threshold_above=0.9, ratio_quantile_threshold_below=0.1)

    fgreen = green_tap.subset(stations=_idx)
    ftaper = taper_ds.subset(stations=_idx)
    
    # Log reduction in traces
    utils.log(f"-----|| {obsd_tap.data.shape[0]} -> {fobsd.data.shape[0]}")
    utils.log("////")
    

    if plot_intermediate_figures:
        # Plot section with stuff removed!
        plot.plot_full_section([fobsd, fsynt, fgreen], labels, stfs, event, 
                                scale=2.0, limits=[0*60,60*60],component=component, 
                                outfile=os.path.join(plotdir, f'{component}_{phase}_data_{labels[1]}_{labels[2]}_removed_outliers.pdf'))


        plot.plot_full_section([fobsd, fsynt], labels[:2], stfs, event, 
                                scale=2.0, limits=[0*60,60*60], component=component, 
                                outfile=os.path.join(plotdir, f'{component}_{phase}_data_{labels[1]}_removed_outliers.pdf'))



    return fobsd, fsynt, fgreen, ftaper
    
    
def station_inversion(i, t, d, G, tapers, config):
    
    inv1 = Inversion(t, d, G, tapers=tapers, config=config)
    x =utils.gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    inv1.optimize_smooth_bound0N(x=x)
    
    print(f"[{i:>03d}]: Done.")

    return i, inv1


def station_inversion_diff(i, t, c, d, G, tapers, config):
    
    inv1 = Inversion(t, d, G, tapers=tapers, config=config)
    x = utils.gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    inv1.optimize_smooth_bound_diff(x=c[:len(x)], x0=c[:len(x)])
    
    print(f"[{i:>03d}]: Done.")

    return i, inv1


def station_inversion_diff_tmax(i, t, c, d, G, tapers, config, tmax, stf):
    
    # Compute the cumulative STF
    STF = cumtrapz(stf, dx=t[1]-t[0], initial=0)
    
    # Clip min and clip max
    clipmin = np.maximum(10, tmax-25)
    clipmax = np.minimum(300, tmax+25)
    idxmin = np.maximum(0,np.argmin( np.abs( t - (tmax-25))))
    idxmax = np.argmin( np.abs( t - (tmax+50)))
    
    clipped = False

    # Somtimes finding new tmax fails then simply use the old one
    try:
        # Very relaxed thresholds
        idx = utils.find_elbow_point(t[idxmin:idxmax], STF[idxmin:idxmax]) + idxmin
        
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
            clipped = True
            __tmax = tmax
        else:
            __tmax = _tmax
        
    
    # Actual inversion
    config = deepcopy(config)
    config['Tmax'] = __tmax
    inv1 = Inversion(t, d, G, tapers=tapers, config=config)
    x = utils.gaussn(inv1.npknots[: -inv1.k - 1], 30, 20)
    x = 2*x / np.sum(x)
    x0 = c[:len(x)]
    x0[-1] = 0 # Setting the last value to zero to mak sure bound and diff penalties aren't interfering
    inv1.optimize_smooth_bound_diff(x=c[:len(x)], x0=x0)
    

    utils.log(f"[{i:>03d}]: Done. Tmax: {tmax:.0f} -- station tmax: {_tmax} --clip--> {__tmax:.0f}")

    return i, inv1, __tmax, clipped

    
def stationwise_first_pass(fobsd: onp.Dataset, fgree: onp.Dataset, ftape: onp.Dataset, config):
    
    
    fstfs = fgree.copy()
    fstfs.meta.stations.attributes.tmaxs = np.zeros(len(fstfs.meta.stations.codes), dtype=np.float32)
    fstfs.meta.stations.attributes.ints = np.zeros(len(fstfs.meta.stations.codes), dtype=np.float32)
       
    c = []
    
        
    # Inverting with optimal Tmax
    invs = Parallel(n_jobs=20)(delayed(station_inversion)(i, fobsd.t, fobsd.data[i:i+1, 0,:], fgree.data[i:i+1, 0, :], ftape.data[i:i+1, 0, :], config) for i in range(fobsd.data.shape[0]))
        
    for i, _inv in enumerate(invs):
        _, inv1 = _inv
        f = inv1.construct_f(inv1.model)
        Nm = inv1.model.shape[0]
        c.append(inv1.model.copy())
        
        # Integrate the value
        fstfs.meta.stations.attributes.ints[i] = np.trapz(f, dx=inv1.dt)
        
        # Defining the STF, costs, and grads
        fstfs.data[i, 0, :]  = f
    
    # Fix timing        
    fstfs.meta.starttime = fstfs.meta.stations.attributes.origin_time
    
    return fstfs, c
    
    
def stationwise_second_pass(fobsd: onp.Dataset, fgree: onp.Dataset, ftape: onp.Dataset, config,
                            diffmodel):
    
    
    fstfs = fgree.copy()
    fstfs.meta.stations.attributes.tmaxs = np.zeros(len(fstfs.meta.stations.codes), dtype=np.float32)
    fstfs.meta.stations.attributes.ints = np.zeros(len(fstfs.meta.stations.codes), dtype=np.float32)
    
    c = []
    # Inverting with optimal Tmax
    invs = Parallel(n_jobs=20)(delayed(station_inversion_diff)(i, fobsd.t, diffmodel, fobsd.data[i:i+1, 0,:], fgree.data[i:i+1, 0, :], ftape.data[i:i+1, 0, :], config) for i in range(fobsd.data.shape[0]))
        
    # Compute actual source time functions
    for i, _inv in enumerate(invs):
        _, inv1 = _inv
        f = inv1.construct_f(inv1.model)
        c.append(inv1.model.copy())
        
        # Integrate the value
        fstfs.meta.stations.attributes.ints[i] = np.trapz(f, dx=inv1.dt)
        
        # Defining the STF, costs, and grads
        fstfs.data[i, 0, :]  = f
        
    # Fix timing        
    fstfs.meta.starttime = fstfs.meta.stations.attributes.origin_time
    
    return fstfs, c


def compute_tmax(fstfs, weights = [2/3, 1/3], plotdir='./', components=['P', 'S'], phases=['Ptrain', 'Strain'],
                 plot_intermediate_figures=True):
    
    # Getting the optimal Tmax
    utils.log("Finding Tmax using the integrated STF and the elbow method")

    
    # Compute the weighted average of the stfs
    avg_stf = np.zeros(fstfs[0].data.shape[2])
    for i, _fstf in enumerate(fstfs):
        avg_stf += weights[i] * np.mean(_fstf.data[:,0,:], axis=0)
    
        
    # Compute the normalize cumulative STF
    _total = np.trapz(avg_stf, dx=fstfs[0].meta.delta)
    _STF = cumtrapz(avg_stf/_total, dx=fstfs[0].meta.delta, initial=0)

    # Find the optimal Tmax from the computed cumulative STF
    tmax, long_stf = utils.find_tmax(fstfs[0].t, _STF)
    
    utils.log(f"Inverting with optimal Tmax: {tmax:.0f}")

    if plot_intermediate_figures:

        # Plotting all the phase specfific STFs for analysis
        plot.plot_cumulative_stf(fstfs[0].t, 
                                [avg_stf, 
                                np.mean(fstfs[0].data[:, 0, :], axis=0), 
                                np.mean(fstfs[1].data[:, 0, :], axis=0)], 
                                ['Combined', *components], 
                                ['P+S' , *phases], plotdir)
        plt.close('all')

        # Plot the stf pick plot separately
        plot.plot_stf_end_pick(fstfs[0].t, avg_stf, label='', 
                                extension=True, label_outside=False)        
        
        plt.legend(frameon=False, ncol=6, bbox_to_anchor=(0.0, 1.0), loc='lower left',
                borderaxespad=0., fontsize='small')
        plt.xlabel('Time [s]')
            
        if long_stf:
            opl.plot_label(plt.gca(), f'LSTF', fontsize='small', box=False, location=18, dist=0.0)
            
        plt.ylim(-0.05, 1.05)
        plt.axhline(0.0, c=(0.7,0.7,0.7), ls='-', lw=1.0, zorder=-1)
        plt.savefig(os.path.join(plotdir, 'stf_cumulative_choice.pdf'))
        plt.close('all')


        plot.plot_elbow_point_selection(fstfs[0].t, avg_stf, label=f'', 
                                extension=True, label_outside=False)    

        plt.savefig(os.path.join(plotdir, 'stf_cumulative_choice_elbow.pdf'))
        plt.close('all')
        
        
    # Finally return the optimal Tmax
    return tmax


def stationwise_third_pass(fobsd: onp.Dataset, fgree: onp.Dataset, ftape: onp.Dataset, config,
                           diffmodel, tmax, fstfs):
    
    fstfs_tmax = fgree.copy()
    fstfs_tmax.meta.stations.attributes.tmaxs = np.zeros(len(fstfs_tmax.meta.stations.codes), dtype=np.float32)
    fstfs_tmax.meta.stations.attributes.ints = np.zeros(len(fstfs_tmax.meta.stations.codes), dtype=np.float32)
    fstfs_tmax.meta.stations.attributes.clipped = np.zeros(len(fstfs_tmax.meta.stations.codes), dtype=int)
    
    # Inverting with optimal Tmax
    invs = Parallel(n_jobs=20)(delayed(station_inversion_diff_tmax)(i, fobsd.t, diffmodel, fobsd.data[i:i+1, 0,:], fgree.data[i:i+1, 0, :], ftape.data[i:i+1, 0, :], config, tmax, fstfs.data[i,0,:]) for i in range(fobsd.data.shape[0]))
        
    c = []
    for i, _inv in enumerate(invs):
        _, inv1, _tmax, _clipped = _inv
        f = inv1.construct_f(inv1.model)
        c.append(inv1.model.copy())
        
        # Integrate the value
        fstfs_tmax.meta.stations.attributes.ints[i] = np.trapz(f, dx=inv1.dt)
        
        # Defining the STF, costs, and grads
        fstfs_tmax.data[i, 0, :]  = f
        fstfs_tmax.meta.stations.attributes.tmaxs[i] = _tmax
        fstfs_tmax.meta.stations.attributes.clipped[i] = int(_clipped)
        
    # Fix timing        
    fstfs_tmax.meta.starttime = fstfs_tmax.meta.stations.attributes.origin_time
    
    return fstfs_tmax, c
    
# %%
import os, sys
import glob 
import json
import obsnumpy as onp
import obstflib as osl
import obsplotlib.plot as opl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

# %%

if ("IPython" in sys.argv[0]) or ("ipython" in sys.argv[0]):
    result_dir = "STF_results_surface"
else:
        
    # Which directory
    result_dir = sys.argv[1]


scardec_dirs = glob.glob(os.path.join(result_dir, "FCT*"))
scardec_dirs.sort()

# only use directory of it contains data/directivity_summary.json
_scardec_dirs = []
for _d in scardec_dirs:
    if os.path.exists(os.path.join(_d, "data", "measurements.json")):
        _scardec_dirs.append(_d)
    else:
        print(f"Skipping {_d}")
# scardec_dirs = [d for d in scardec_dirs if os.path.exists(os.path.join(d, "data", "measurements_fix.json"))]


# %%



for _i, scardec_dir in enumerate(scardec_dirs):
    
    if _i == 0:
        measurements = onp.utils.load_json(os.path.join(scardec_dir, "data", "measurements.json"))
        measurements_fix = onp.utils.load_json(os.path.join(scardec_dir, "data", "measurements_fix.json"))
        continue
    
    _m = onp.utils.load_json(os.path.join(scardec_dir, "data", "measurements.json"))
    _m_fix = onp.utils.load_json(os.path.join(scardec_dir, "data", "measurements_fix.json"))
    
    for _phase_label in measurements.keys():
        
        for _misfit_label in measurements[_phase_label].keys():
            
            for _att_label in measurements[_phase_label][_misfit_label].keys():
                
                measurements[_phase_label][_misfit_label][_att_label].extend(_m[_phase_label][_misfit_label][_att_label])
                measurements_fix[_phase_label][_misfit_label][_att_label].extend(_m_fix[_phase_label][_misfit_label][_att_label])
        
    
# %%
# Add the unfixed meaurements to the fixed ones
for _phase_label in measurements_fix.keys():
    
    measurements_fix[_phase_label]["B-STF-B"] = measurements[_phase_label]["B-STF"]


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

outfile = os.path.join(".", 'summary_histogram.pdf')

# %%

    
osl.plot.plot_measurements_histograms(measurements_fix, phasecomp_order, misfitlabels, misfitcolors, misfitlinestyles, 
                                      outfile, remove_outliers=True)

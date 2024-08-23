# %%
import os
from glob import glob
from gf3d.seismograms import GFManager
from gf3d.source import CMTSOLUTION

# %%
def create_stf_subset(databasename: str, subsetfilename: str, cmt: CMTSOLUTION,
                      radius_in_km=50, ngll=5, fortran=False):

    # Get event info
    latitude = cmt.latitude
    longitude = cmt.longitude
    depth_in_km = cmt.depth

    # Check for files given database path
    db_globstr = os.path.join(databasename, '*', '*', '*.h5')

    # Get all files
    db_files = glob(db_globstr)

    # Get subset
    GFM = GFManager(db_files)
    GFM.load_header_variables()
    GFM.write_subset_directIO(subsetfilename, latitude, longitude,
                                depth_in_km, radius_in_km, NGLL=ngll,
                                fortran=fortran)

# %%

databaselocation = '/lustre/orion/geo111/scratch/lsawade/DB/GLAD-M35/256_TEST'
subsetdir = '/lustre/orion/geo111/scratch/lsawade/STF_SUBSETS'
STFdir = './STF'

# Get all stf directories
stfdirs = glob(os.path.join(STFdir, 'FCT*'))

# Make sure the subset dir exists
if not os.path.exists(subsetdir):
    os.makedirs(subsetdir)

for _i, stfdir in enumerate(stfdirs):
    cmt = CMTSOLUTION.read(os.path.join(stfdir, 'CMTSOLUTION'))

    # Get basename
    stf_base = os.path.basename(stfdir)

    # Create subsetfilename
    subsetfilename = os.path.join(subsetdir, stf_base, 'subset.h5')

    # Make sure the subset dir exists
    if not os.path.exists(os.path.join(subsetdir, stf_base)):
        os.makedirs(os.path.join(subsetdir, stf_base))

    # Create subset
    print('\n' + 72*'-')
    print('Event #', _i)
    print(databaselocation, subsetfilename)
    print(cmt)

    # This function only ever fails if there are no elements within the
    # radius of the source. This only really happens for deep events, where
    # element size is large. Remeber the element detection algorithm uses the
    # midpoint, if the midpoint is to far from the source location it will not
    # select the element. This makes the detection fast but also events are
    # are often not found.
    try:
        if cmt.depth < 200:
            create_stf_subset(databaselocation, subsetfilename, cmt, radius_in_km=40, ngll=5, fortran=False)
        else:
            create_stf_subset(databaselocation, subsetfilename, cmt, radius_in_km=75, ngll=5, fortran=False)

    except Exception as e:
        print('Error:', e)
        create_stf_subset(databaselocation, subsetfilename, cmt, radius_in_km=100, ngll=5, fortran=False)
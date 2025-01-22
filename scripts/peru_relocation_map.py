# %%

import os, sys
import obstflib as osl
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import pygmt
import numpy as np

#%%


if ("IPython" in sys.argv[0]) or ("ipython" in sys.argv[0]):
    scardec_id = "FCTs_20070815_234057_NEAR_COAST_OF_PERU"
    # scardec_id = "FCTs_19951009_153553_NEAR_COAST_OF_JALISCO__MEXICO"
    # scardec_id = "FCTs_19950730_051123_NEAR_COAST_OF_NORTHERN_CHILE"
    # scardec_id = "FCTs_20210729_061549_ALASKA_PENINSULA"
else:
    
    
    # Which directory
    scardec_id = sys.argv[1]
    
    if len(sys.argv) > 2:
        nocoast = True
    else:
        nocoast = False




# scardec_id = "FCTs_20010623_203314_NEAR_COAST_OF_PERU"
result_dir = "STF_results_scardec"

datadir = os.path.join(result_dir, scardec_id, "data")
plotdir = os.path.join(result_dir, scardec_id, "plots")

# %%

# Make pygmt figure
def make_peru_map():

    fig = pygmt.Figure()
    region = [-78, -74.75, -15.5, -12.5]
    fig.basemap(region=region, projection="M10c", frame=True)

    # ADd topography
    topo = True
    if topo:
        # Load global topography data. This is a 1 arc-minute grid.
        grid_region = pygmt.datasets.load_earth_relief(resolution="30s", region=region)

        # Load earth relief using pygmt
        pygmt.makecpt(cmap="geo", series=[-8000, 8000])
        fig.grdimage(grid=grid_region, frame="a")

    # Add coastlines and borders
    fig.coast(
        shorelines=True,
        resolution="f",
        borders=["1/0.75p,black", "2/0.5p,darkgray"],
    )

    return fig

# %%

gcmt = osl.CMTSOLUTION.read(os.path.join(datadir, "fms", "gcmt_cmtsolution.txt"))
scar = osl.CMTSOLUTION.read(os.path.join(datadir, "fms", "scardec_cmtsolution.txt"))
cmt3 = osl.CMTSOLUTION.read(os.path.join(datadir, "fms", "cmt3_cmtsolution.txt"))
# %%
spec = []

xoffsets = np.array([0.5,  -0.75, -0.5])
yoffsets = np.array([-0.5, +0.1, +0.5])

for _i, cmt in enumerate([gcmt, scar, cmt3]):
    
    # Get the longitude, latitude and depth of the event
    cmt_longitude = cmt.longitude
    cmt_latitude = cmt.latitude
    cmt_depth = cmt.depth
    cmt_tensor = cmt.tensor
    
    # Get the exponents of the moment tensors
    exponent = np.floor(np.log10(np.abs(np.max(cmt_tensor))))

    # Create the spec array required by the GMT plotting tool
    spec.append(np.array([cmt_longitude, cmt_latitude, cmt_depth,
                    *(cmt_tensor/10**exponent), exponent, cmt_longitude-xoffsets[_i], cmt_latitude-yoffsets[_i]]))


# %%


fig = make_peru_map()

cmts = [gcmt, scar, cmt3]
labels = ["GCMT", "SCARDEC", "CMT3D+"]
colors = ["black", "red", "blue"]

txoffsets = np.array([-0.35,  0.85, 0.65])
tyoffsets = np.array([0.75, 0.2, -0.8])


# # Plot event with a circle
fig.plot(x=[-76.6, -76.7], y=[-13.39, -14.],
        style="a0.5c", pen="0.25p", fill='white')


for s, cmt, label, color in zip(spec,cmts, labels, colors):
    
    # add event location with a beachball
    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'w') as f:
        np.savetxt(f, s.reshape(1, len(s)))

    # with open(tmp.name, 'r') as f:
    with open(tmp.name, 'r') as f:
        hello = np.loadtxt(f)
        print(hello)

    fig.meca(spec=tmp.name, convention='mt',
            compressionfill=color, extensionfill="white",
            pen="0.01p,gray30,solid", scale="1c+s5.5",
            # offset the beach to ball to the left
            offset="+p0.01+s0.25",
            event_name='GCMT')


    
fig.text(textfiles=None, 
         x=np.array([cmt.longitude for cmt in cmts])+txoffsets, 
         y=np.array([cmt.latitude for cmt in cmts])+tyoffsets, position=None,
            text=labels,
        angle=0, font='8p,Helvetica-Bold,black', justify='LM', fill='white', frame='+p0.5p,black',
        pen='0.5p,black')

# fig.show()
fig.savefig(os.path.join(plotdir, "peru_map.pdf"))

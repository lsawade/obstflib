# %%
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import obspy
import obsplotlib.plot as opl
import obstflib as osl

STFDIR = "STF"

# %%
# Get the list of all the STF files
stfdirs = glob(os.path.join(STFDIR, "*"))

# %%
# Create Scardec STF dataclass


STF_opt = osl.SCARDECSTF.fromdir(stfdirs[0], "optimal")
STF_avg = osl.SCARDECSTF.fromdir(stfdirs[0], "average")

# %%
# Plot the STF

# Make default font Helvetica
plt.rcParams["font.family"] = "Helvetica"


def plot_scardec_stf(
    stf_opt: opl.SCARDECSTF,
    stf_avg: opl.SCARDECSTF | None = None,
    color_opt="k",
    color_avg="r",
):
    plt.figure(figsize=(7, 3.5))
    plt.plot(stf_opt.time, stf_opt.moment_rate, label="Optimal STF", c=color_opt)
    if stf_avg:
        plt.plot(stf_avg.time, stf_avg.moment_rate, label="Average STF", c=color_avg)
    plt.xlabel("Time (s)")
    plt.ylabel("Moment rate (Nm/s)")
    opl.plot_label(
        ax=plt.gca(),
        label=f"{stf_opt.origin} - Mw {stf_opt.Mw} - Depth "
        f"{stf_opt.depth_in_km} km\n{stf_opt.region}",
        location=6,
        fontsize="medium",
        box=False,
    )
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)


# %%


def M02hdur(M0):
    # GCMT conversion equation
    Nm_conv = 1 / 1e7
    return 2.26 * 10 ** (-6) * (M0 * Nm_conv) ** (1 / 3)


print(M02hdur(10**24))
print(M02hdur(10**27))

# %%

plot_scardec_stf(STF_opt, STF_avg)


def plot_triangular_stf(
    hdur: float, *args, centroid_time: float = 0.0, M0: float = 1.0, **kwargs
):

    startpoint = centroid_time - hdur
    midpoint = centroid_time
    endpoint = centroid_time + hdur

    # total area under triangle has to be 1

    ax = plt.gca()
    ax.plot([startpoint, midpoint, endpoint], [0, M0 / hdur, 0], *args, **kwargs)


def plot_gaussian_stf(
    hdur: float,
    *args,
    centroid_time: float = 0.0,
    M0: float = 1.0,
    alpha: float = 1.628,
    **kwargs,
):

    # Time vector +
    t = np.arange(-1.5 * hdur, 1.5 * hdur, 0.01) + centroid_time

    # Exponent for the Gaussian
    exponent = -((alpha * (t - centroid_time) / hdur) ** 2)

    # Are under the Gaussen -> M0
    gaussian = M0 * alpha / (np.sqrt(np.pi) * hdur) * np.exp(exponent)

    ax = plt.gca()
    ax.plot(t, gaussian, *args, **kwargs)


# Info from globalcmt.org
origin = obspy.UTCDateTime(2017, 9, 8, 4, 49, 19.20)
timeshift = 27.4500
M0 = 2.826165954079838e28
halfduration = 32.0000
centroidtime = origin + timeshift

# Get centroid time with respect to SCARDEC Origin
cmt_time_scardec: float = centroidtime - STF_opt.origin

plot_triangular_stf(
    halfduration,
    "k--",
    centroid_time=cmt_time_scardec,
    M0=(M0 / 1e7),
    label="GCMT",
)

plot_gaussian_stf(
    hdur=halfduration,
    centroid_time=cmt_time_scardec,
    M0=(M0 / 1e7),
    alpha=1.628,
    label="SF3DG",
)

plt.legend(loc="upper right", frameon=False)


# %%

stfs_opt = []
stfs_avg = []
gcmts = []

for _stfdir in stfdirs:
    stfs_opt.append(opl.SCARDECSTF.fromdir(_stfdir, "optimal"))
    stfs_avg.append(opl.SCARDECSTF.fromdir(_stfdir, "average"))
    gcmts.append(opl.CMTSOLUTION.read(_stfdir + "/CMTSOLUTION"))

# %%


def plotb(
    x,
    y,
    tensor,
    linewidth=0.25,
    width=100,
    facecolor="k",
    clip_on=True,
    alpha=1.0,
    normalized_axes=True,
    ax=None,
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
    bb.set_transform(transforms.Affine2D(np.identity(3)))

    pax.add_collection(bb)


def plot_single_stf(stf_opt, stf_avg, gcmt, pdf=False):

    color_opt = "tab:blue"
    color_avg = "tab:orange"
    color_tri = "k"
    color_gauss = "tab:green"

    # Create the figure
    plt.figure(figsize=(7, 3.5))

    # Plot the scardec STFs
    plot_scardec_stf(stf_opt, stf_avg, color_opt=color_opt, color_avg=color_avg)

    fm_scardec = opl.CMTSOLUTION.from_sdr(
        s=stf_opt.strike1,
        d=stf_opt.dip1,
        r=stf_opt.rake1,
        M0=stf_opt.M0 * 1e7,  # Convert to dyn*cm
        latitude=stf_opt.latitude,
        longitude=stf_opt.longitude,
        depth=stf_opt.depth_in_km,
    )

    # Plot the triangular STF as used in GCMT analysis
    plot_triangular_stf(
        gcmt.hdur,
        c=color_tri,
        ls="--",
        centroid_time=gcmt.cmt_time - stf_opt.origin,
        M0=(gcmt.M0 / 1e7),
        label="GCMT",
    )

    # Plot Gaussian STF as used in SF3DG
    plot_gaussian_stf(
        hdur=gcmt.hdur,
        centroid_time=gcmt.cmt_time - stf_opt.origin,
        M0=(gcmt.M0 / 1e7),
        alpha=1.628,
        label="SF3DG",
        c=color_gauss,
        ls="--",
    )
    ax = plt.gca()

    if pdf:
        # This ratio original width * (pdf_dpi / figure_dpi / 2)
        # No idea where the 2 comes from. It's a magic number
        width = 100 * (72 / 100 / 2)
    else:
        width = 100

    plotb(
        0.075,
        0.85,
        gcmt.tensor,
        linewidth=0.25,
        width=width,
        facecolor=color_tri,
        normalized_axes=True,
        ax=ax,
    )

    plotb(
        0.075,
        0.6,
        fm_scardec.tensor,
        linewidth=0.25,
        width=width,
        facecolor=color_opt,
        normalized_axes=True,
        ax=ax,
    )

    ax.set_ylim(bottom=0)
    plt.sca(ax)
    plt.legend(loc="upper right", frameon=False)
    plt.show(block=False)


plot_single_stf(stfs_opt[0], stfs_avg[0], gcmts[0])


# %%
def plot_all_stfs(stfs_opt, stfs_avg, gcmts, outdir="./STF_plots"):

    # Make sure the output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Loop over the STFs and GCMTs
    for stf_opt, stf_avg, gcmt in zip(stfs_opt, stfs_avg, gcmts):

        print(gcmt.eventname)

        # Plot single stf
        plot_single_stf(stf_opt, stf_avg, gcmt, pdf=True)
        plt.savefig(os.path.join(outdir, f"{gcmt.eventname}.pdf"))
        plt.close("all")


# %%
plot_all_stfs(stfs_opt, stfs_avg, gcmts)

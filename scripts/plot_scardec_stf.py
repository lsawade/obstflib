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
chile_dir = os.path.join(STFDIR, 'FCTs_20170908_044919_NEAR_COAST_OF_CHIAPAS__MEXICO')

STF_opt = osl.STF.scardecdir(chile_dir, "optimal")
STF_avg = osl.STF.scardecdir(chile_dir, "average")

# %%
# Plot the STF

# Make default font Helvetica
plt.rcParams["font.family"] = "Helvetica"


def plot_scardec_stf(
    stf_opt: osl.STF,
    stf_avg: osl.STF | None = None,
    color_opt="k",
    color_avg="r",
):
    plt.figure(figsize=(7, 3.5))
    stf_opt.plot(label="Optimal STF", c=color_opt)
    if stf_avg:
        stf_avg.plot(label="Average STF", c=color_avg)
    plt.xlabel("Time (s)")
    plt.ylabel("Moment rate (Nm/s)")
    opl.plot_label(
        ax=plt.gca(),
        label=f"{stf_opt.origin} - Mw {stf_opt.Mw} - Depth "
        f"{stf_opt.depth} km\n{stf_opt.region}",
        location=6,
        fontsize="medium",
        box=False,
    )
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)

plot_scardec_stf(STF_opt, STF_avg)

# %%


def M02hdur(M0):
    # GCMT conversion equation
    Nm_conv = 1 / 1e7
    return 2.26 * 10 ** (-6) * (M0 * Nm_conv) ** (1 / 3)


print(M02hdur(10**24))
print(M02hdur(10**27))


# %%
# Info from globalcmt.org
gcmt = osl.CMTSOLUTION.read(os.path.join(chile_dir, "CMTSOLUTION"))
cmt3 = osl.CMTSOLUTION.read(glob(os.path.join(chile_dir, "*_CMT3D+"))[0])

t = STF_opt.t

# Get centroid time with respect to SCARDEC Origin
cmt_time_scardec: float = gcmt.cmt_time - STF_opt.origin

# Time vector for plotting
t = np.arange(-1.5 * gcmt.hdur, 1.5 * gcmt.hdur, 0.01) + cmt_time_scardec

# Plot triangular STF
stf_tri = osl.STF.triangle(origin=STF_opt.origin, t=t,
                           hdur=gcmt.hdur, tshift=0.0, tc=cmt_time_scardec,
                           M0=gcmt.M0 / 1e7)
stf_tri.plot(label="GCMT", c="k", ls="--")

stf_gau = osl.STF.gaussian(origin=STF_opt.origin, t=t,
                           hdur=gcmt.hdur, tshift=0.0, tc=cmt_time_scardec,
                           M0=cmt3.M0 / 1e7)
stf_gau.plot(label="CMT3D+", c="tab:blue", ls="--")

plt.ylim(0, None)
plt.legend(loc="upper right", frameon=False)
plt.savefig('test_stf_plot.pdf')

# %%

stfs_opt = []
stfs_avg = []
gcmts = []
cmt3s = []

for _stfdir in stfdirs:
    if "FCTs" not in _stfdir:
        continue
    try:
        cmt3s.append(opl.CMTSOLUTION.read(glob(os.path.join(_stfdir, "*_CMT3D+"))[0]))
        stfs_opt.append(osl.STF.scardecdir(_stfdir, "optimal"))
        stfs_avg.append(osl.STF.scardecdir(_stfdir, "average"))
        gcmts.append(opl.CMTSOLUTION.read(_stfdir + "/CMTSOLUTION"))
    except IndexError:
        print(f"IndexError: {_stfdir}")
        continue
    
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


def plot_single_stf(stf_opt, stf_avg, gcmt, cmt3, pdf=False):

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
        depth=stf_opt.depth,
    )

    # Get centroid time with respect to SCARDEC Origin
    gcmt_time_scardec: float = gcmt.cmt_time - stf_opt.origin
    cmt3_time_scardec: float = cmt3.cmt_time - stf_opt.origin

    # Time vector for plotting
    t = np.arange(-1.5 * gcmt.hdur, 1.5 * gcmt.hdur, 0.01) + gcmt_time_scardec

    # Plot triangular STF
    stf_tri = osl.STF.triangle(origin=stf_opt.origin, t=t,
                            hdur=gcmt.hdur, tshift=0.0, tc=gcmt_time_scardec,
                            M0=gcmt.M0 / 1e7)
    stf_tri.plot(label="GCMT", c=color_tri, ls="--")

    # Plot Specfem style Gaussian STF.
    stf_gau = osl.STF.gaussian(origin=stf_opt.origin, t=t,
                            hdur=cmt3.hdur, tshift=0.0, tc=cmt3_time_scardec,
                            M0=cmt3.M0 / 1e7)
    stf_gau.plot(label="CMT3D+", c=color_gauss, ls="--")

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
        fm_scardec.tensor,
        linewidth=0.25,
        width=width,
        facecolor=color_opt,
        normalized_axes=True,
        ax=ax,
    )
    
    plotb(
        0.075,
        0.6,
        gcmt.tensor,
        linewidth=0.25,
        width=width,
        facecolor=color_tri,
        normalized_axes=True,
        ax=ax,
    )

    plotb(
        0.075,
        0.35,
        cmt3.tensor,
        linewidth=0.25,
        width=width,
        facecolor=color_gauss,
        normalized_axes=True,
        ax=ax,
    )

    ax.set_ylim(bottom=0)
    plt.sca(ax)
    plt.legend(loc="upper right", frameon=False)
    plt.show(block=False)


plot_single_stf(stfs_opt[0], stfs_avg[0], gcmts[0], cmt3s[0])


# %%
def plot_all_stfs(stfs_opt, stfs_avg, gcmts, outdir="./STF_plots"):

    # Make sure the output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Loop over the STFs and GCMTs
    for stf_opt, stf_avg, gcmt, cmt3 in zip(stfs_opt, stfs_avg, gcmts, cmt3s):

        print(gcmt.eventname)

        # Plot single stf
        plot_single_stf(stf_opt, stf_avg, gcmt, cmt3, pdf=True)
        plt.savefig(os.path.join(outdir, f"{gcmt.eventname}.pdf"))
        plt.close("all")


# %%
plot_all_stfs(stfs_opt, stfs_avg, gcmts)

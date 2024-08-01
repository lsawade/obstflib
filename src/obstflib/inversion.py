
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import scipy.optimize as opt

from scipy.interpolate import BSpline
import numpy as np
import scipy.optimize as so
from scipy.fftpack import fft, ifft

from .utils import next_power_of_2, gaussn


class Inversion(object):

    Tmin = -5
    Tmax = 250
    knots_per_second = 0.2
    k = 3  # degree of the spline
    A = 1.0  # Area of the STF
    maxiter = 100
    weight = 1.0
    penalty_weight = 1.0
    smooth_weight = 1.0
    bound_weight = 1.0
    verbose: bool = False

    def __init__(self, t, data, G, tapers=None, config=None):

        # Data/synthetic inputs
        self.t = t
        self.data = data
        self.G = G
        self.N = data.shape[0]

        # Get base parameters
        self.nt = data.shape[-1]
        self.ntfft = next_power_of_2(self.nt)
        self.dt = t[1] - t[0]
        self.dataN = np.sum(self.data**2, axis=1) * self.dt
        
        # For speedup, we use separate forward functions for either tapered or non-tapered
        # inversion
        if tapers is not None:
            self.tapers = tapers
            self.forward = self.__forward_tapered__
        else:
            self.forward = self.__forward__

        # Fourier transform of the Green's function
        self.FG = fft(G, n=self.ntfft, axis=-1)

        # Config
        if config is not None:
            for key in config:
                setattr(self, key, config[key])

        # Construct the spline
        self.construct_spline()

        # Create the base Bsplines
        self.create_base_bsplines()

    def construct_spline(self):

        # Define interior knots as a function the total time
        self.n_interior_knots = int((self.Tmax - self.Tmin) * self.knots_per_second)

        # Make sure the number of knots is odd
        if self.n_interior_knots % 2 == 0:
            self.n_interior_knots += 1

        # Define the interior knots
        dt = (self.Tmax - self.Tmin) / self.n_interior_knots
        self.knots = np.linspace(
            self.Tmin - dt, self.Tmax + dt, self.n_interior_knots + 2
        )[1:-1]

        # Correct the scipy function to include the "numpy knots"
        self.npknots = np.hstack(
            (self.Tmin * np.ones(self.k), self.knots, self.Tmax * np.ones(self.k))
        )

        # Create base Bspline object to be used in the inversion
        self.bspl = BSpline(
            t=self.npknots, c=np.zeros_like(self.npknots), k=self.k, extrapolate=False
        )

    def construct_f(self, c):
        self.bspl.c = c**2
        f = self.bspl(self.t)
        f = np.where(np.isnan(f), 0, f)
        return f

    def create_base_bsplines(self):
        # Adding 2 to the range because of the numpyknots correction
        self.Bsplines = []
        self.FBsplines = []
        self.I_Bsplines = []
        self.GBsplines = []

        for i in range(len(self.npknots) - self.k - 1):

            # Compute the base spline
            b = self.bspl.basis_element(self.npknots[i : i + 5], extrapolate=False)

            # Set areas of extrappolation to zero
            b = b(self.t)
            b = np.where(np.isnan(b), 0, b)

            # Add Bspline to the list
            self.Bsplines.append(b)

            # Compute the integral of the Bspline
            self.I_Bsplines.append(np.trapz(b, self.t))

            # Compute the Fourier transform of the Bspline
            self.FBsplines.append(fft(b, n=self.ntfft))

            # Construct G convolved with Bspline matrix so that we can skip a
            # convolution in the gradient computation
            self.GBsplines.append(
                np.real(ifft(self.FG * self.FBsplines[i][None, :])[:, : self.nt])
            )

        # Convert to numpy arrays
        self.I_Bsplines = np.array(self.I_Bsplines)
        self.GBsplines = np.array(self.GBsplines)

    def __forward__(self, c):
        # Convolve spike with the Bspline
        f = self.construct_f(c)
        
        return  (
            np.real(ifft(self.FG * fft(f, n=self.ntfft)[None, :])[:, : self.nt])
            * self.dt
        )
        
    def __forward_tapered__(self, c):
        # Convolve spike with the Bspline
        f = self.construct_f(c)
        
        return  (
            np.real(ifft(self.FG * fft(f, n=self.ntfft)[None, :])[:, : self.nt])
            * self.dt
        ) * self.tapers


    def loss(self, c):
        return 0.5 * np.sum((self.forward(c) - self.data) ** 2) * self.dt / self.N
    

    def gradient(self, c):

        # Compute the residual
        residual = self.forward(c) - self.data

        # Compute the gradient
        grad = np.zeros_like(c)

        # Adding 2 to the range because of the numpyknots correction
        for i in range(len(self.npknots) - self.k - 1):

            # Convolve
            g = (
                2
                * c[i]
                * np.real(ifft(self.FG * self.FBsplines[i][None, :])[:, : self.nt])
                * self.dt
            )

            # Integrate over time
            grad[i] = np.sum(residual * g) * self.dt / self.N

        return grad

    def loss_norm(self, c):

        return (
            0.5
            * np.sum(
                np.sum((self.forward(c) - self.data) ** 2, axis=1)
                * self.dt
                / (np.sum(self.data**2, axis=1) * self.dt)
            )
            / self.N
        )

    def gradient_norm(self, c):

        # Compute the residual
        residual = self.forward(c) - self.data

        # Computing the product of the derivative model parameters and
        # convolutions of Green functions and base Bsplines
        g = 2 * c[:, None, None] * self.GBsplines * self.dt

        # This beast is the gradient of the loss function with respect to the
        grad = (
            np.sum(
                np.sum(residual[None, :, :] * g, axis=-1)
                * self.dt
                / self.dataN[None, :],
                axis=-1,
            )
            / self.N
        )

        return grad

    def loss_integral_penalty(self, c):

        # Integrate f
        I_f = np.trapz(self.construct_f(c), self.t)

        # General cost function
        return 0.5 * (I_f - self.A) ** 2

    def gradient_integral_penalty(self, c):

        # Compute the gradient
        grad = np.zeros(len(c))

        # Integrate f
        I_f = np.trapz(self.construct_f(c), self.t)

        # Get the gradient with respect to the Bspline coefficients
        grad[:] = 2 * c * self.I_Bsplines * (I_f - self.A)

        return grad

    def cost_smoothness_first_order(self, c):
        """Forward FD punishes adjacent model coefficients that are different."""
        # Forward FD
        Cfw = np.sum((c[1:] ** 2 - c[0:-1] ** 2) ** 2)

        Cbw = np.sum((c[-1] ** 2 - c[-2] ** 2) ** 2)

        return Cfw + Cbw

    def gradient_smoothness_first_order(self, c):
        """Forward FD punishes adjacent model coefficients that are different."""

        # Gradient of the above loss_smoothness function with respect to c
        grad = np.zeros_like(c)

        # first element has contributition from fw and centered FD
        grad[0] = -4 * c[0] * (c[1] ** 2 - c[0] ** 2)

        grad[1:-1] = 4 * c[1:-1] * (2 * c[1:-1] ** 2 - c[2:] ** 2 - c[0:-2] ** 2)

        grad[-1] = 8 * c[-1] * (c[-1] ** 2 - c[-2] ** 2)

        return grad

    def cost_bound0(self, c):
        return np.sum((c[: self.k]) ** 2)

    def gradient_bound0(self, c):
        grad = np.zeros_like(c)
        grad[: self.k] = 2 * c[: self.k]
        return grad

    def cost_boundN(self, c):
        return np.sum((c[-1]) ** 2)

    def gradient_boundN(self, c):
        grad = np.zeros_like(c)
        grad[-1] = 2 * c[-1]
        return grad

    def loss_int(self, c):
        return self.weight * self.loss_norm(
            c
        ) + self.penalty_weight * self.loss_integral_penalty(c)

    def grad_int(self, c):
        return self.weight * self.gradient_norm(
            c
        ) + self.penalty_weight * self.gradient_integral_penalty(c)

    def loss_int_smooth1(self, c):
        C1 = self.loss_norm(c)
        C2 = self.loss_integral_penalty(c)
        C3 = self.cost_smoothness_first_order(c)
        if self.verbose:
            print(f"C1={C1:g}, C2={C2:g}, C3={C3:g}")
        return self.weight * C1 + self.penalty_weight * C2 + self.smooth_weight * C3

    def grad_int_smooth1(self, c):
        return (
            self.weight * self.gradient_norm(c)
            + self.penalty_weight * self.gradient_integral_penalty(c)
            + self.smooth_weight * self.gradient_smoothness_first_order(c)
        )

    def loss_int_smooth1_bound0(self, c):
        C1 = self.loss_norm(c)
        C2 = self.loss_integral_penalty(c)
        C3 = self.cost_smoothness_first_order(c)
        C4 = self.cost_bound0(c)
        # print(f"C1={C1:g}, C2={C2:g}, C3={C3:g}, C4={C4:g}")
        return (
            self.weight * C1
            + self.penalty_weight * C2
            + self.smooth_weight * C3
            + self.bound_weight * C4
        )

    def grad_int_smooth1_bound0(self, c):
        return (
            self.weight * self.gradient_norm(c)
            + self.penalty_weight * self.gradient_integral_penalty(c)
            + self.smooth_weight * self.gradient_smoothness_first_order(c)
            + self.bound_weight * self.gradient_bound0(c)
        )

    def loss_int_smooth1_bound0N(self, c):
        C1 = self.loss_norm(c)
        C2 = self.loss_integral_penalty(c)
        C3 = self.cost_smoothness_first_order(c)
        C4 = self.cost_bound0(c)
        C5 = self.cost_boundN(c)
        if self.verbose:
            print(f"C1={C1:g}, C2={C2:g}, C3={C3:g}, C4={C4:g}, C5={C5:g}")
        return (
            self.weight * C1
            + self.penalty_weight * C2
            + self.smooth_weight * C3
            + self.bound_weight * (C4 + C5)
        )

    def grad_int_smooth1_bound0N(self, c):
        return (
            self.weight * self.gradient_norm(c)
            + self.penalty_weight * self.gradient_integral_penalty(c)
            + self.smooth_weight * self.gradient_smoothness_first_order(c)
            + self.bound_weight * (self.gradient_bound0(c) + self.gradient_boundN(c))
        )

    def optimize(self, x=None):

        if x is None:
            x = 0.25 * np.ones(len(self.npknots) - self.k - 1)
            x = gaussn(self.npknots[: -self.k - 1], 30, 20)
            x = x / np.sum(x)

        self.init_model = x

        # opt = so.minimize(
        #     self.loss,
        #     0.25 * np.ones_like(self.npknots),
        #     method="Nelder-Mead",
        #     bounds=[(0, 1)] * len(self.npknots),
        # )

        opt = so.minimize(
            self.loss,
            x,
            jac=self.gradient,
            method="CG",
            options={"maxiter": self.maxiter},
            # bounds=[(0, 1)] * len(x),
        )

        # opt = so.minimize(
        #     self.loss,
        #     x,
        #     jac=self.gradient,
        #     method="BFGS",
        #     bounds=[(0, 1)] * len(x),
        # )

        self.model = opt.x
        self.cost = opt.fun
        self.opt = opt

    def optimize_penalty(self, x=None):

        if x is None:
            x = 0.25 * np.ones(len(self.npknots) - self.k - 1)
            x = gaussn(self.npknots[: -self.k - 1], 30, 20)
            x = x / np.sum(x)

        self.init_model = x

        opt = so.minimize(
            self.loss_int,
            x,
            jac=self.grad_int,
            method="CG",
            options={"maxiter": self.maxiter},
            # bounds=[(0, 1)] * len(x),
        )

        # opt = so.minimize(
        #     self.loss,
        #     x,
        #     jac=self.gradient,
        #     method="BFGS",
        #     bounds=[(0, 1)] * len(x),
        # )

        self.model = opt.x
        self.cost = opt.fun
        self.opt = opt

    def optimize_smooth(self, x=None):

        if x is None:
            x = 0.25 * np.ones(len(self.npknots) - self.k - 1)
            x = gaussn(self.npknots[: -self.k - 1], 30, 20)
            x = x / np.sum(x)

        self.init_model = x

        opt = so.minimize(
            self.loss_int_smooth1,
            x,
            jac=self.grad_int_smooth1,
            method="CG",
            options={"maxiter": self.maxiter},
            # bounds=[(0, 1)] * len(x),
        )

        self.model = opt.x
        self.cost = opt.fun
        self.opt = opt

    def optimize_smooth_bound(self, x=None):

        if x is None:
            x = 0.25 * np.ones(len(self.npknots) - self.k - 1)
            x = gaussn(self.npknots[: -self.k - 1], 30, 20)
            x = x / np.sum(x)

        self.init_model = x

        opt = so.minimize(
            self.loss_int_smooth1_bound0,
            x,
            jac=self.grad_int_smooth1_bound0,
            method="BFGS",
            options={"maxiter": self.maxiter},
            # bounds=[(0, 1)] * len(x),
        )

        self.model = opt.x
        self.cost = opt.fun
        self.opt = opt

    def optimize_smooth_bound0N(self, x=None):

        if x is None:
            x = 0.25 * np.ones(len(self.npknots) - self.k - 1)
            x = gaussn(self.npknots[: -self.k - 1], 30, 20)
            x = x / np.sum(x)

        self.init_model = x

        opt = so.minimize(
            self.loss_int_smooth1_bound0N,
            x,
            jac=self.grad_int_smooth1_bound0N,
            method="BFGS",
            options={"maxiter": self.maxiter},
            # bounds=[(0, 1)] * len(x),
        )

        self.model = opt.x
        self.cost = opt.fun
        self.opt = opt

    
    def print_results(self, title="Smooth+bound0"):
        print(f"{title}:")
        print("I_i = ", np.trapz(self.construct_f(self.init_model), self.t))
        print("I_f = ", np.trapz(self.construct_f(self.model), self.t))
        print("COST", self.cost)
        print("N_knots", len(self.knots))
        print("STATUS", self.opt.success, ":", self.opt.message)
        print("Niter", self.opt.nit)
        
    @staticmethod
    def unique_legend():
        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        plt.legend(handles, labels, loc="best", frameon=False, ncol=3)


    def plot(self, outfile='atest_bspline_inversion.pdf', tshift=200.0):
        fig = plt.figure(figsize=(10, 10))
        ax1 = plt.subplot(311)
        # plt.plot(t, f, "-r", label="Original STF")
        plt.plot(self.t-tshift, self.construct_f(self.model), "-m", label="B-Spline")
        plt.xlim(-20, 200)
        self.unique_legend()

        plt.xlabel("Time [s]")
        ax2 = plt.subplot(312)
        plt.plot(self.t, self.data.T/np.abs(self.data).max()*2 + np.arange(self.N)[None,:], "-k", label="Data")
        plt.plot(self.t, self.G.T/np.abs(self.data).max()*2 + np.arange(self.N)[None,:], "-g", label="Green")
        plt.ylim(-1.5, 1.5 + self.N)
        self.unique_legend()
        plt.xlabel("Time [s]")
        plt.xlim(0, 3600)
        ax3 = plt.subplot(313)
        # plt.plot(t, s, "r", label="Original Synthetics", alpha=0.5, markersize=2)
        plt.plot(self.t, self.data.T/np.abs(self.data).max()*2 + np.arange(self.N)[None,:], "k", label="Noisy Data", alpha=0.5, markersize=2)
        plt.plot(self.t, self.forward(self.model).T/np.abs(self.data).max()*2 + np.arange(self.N)[None,:], "-m", label="Inverted Synthetic")
        plt.xlim(0, 3600)
        plt.ylim(-1.5, 1.5 + self.N)
        self.unique_legend()
        plt.savefig(outfile)
        
        return fig, [ax1, ax2, ax3]




def construct_taper(npts, taper_type="tukey", alpha=0.2):
    """
    Construct taper based on npts

    :param npts: the number of points
    :param taper_type:
    :param alpha: taper width
    :return:
    """
    taper_type = taper_type.lower()
    _options = ['hann', 'boxcar', 'tukey', 'hamming']
    if taper_type not in _options:
        raise ValueError("taper type option: %s" % taper_type)
    if taper_type == "hann":
        taper = signal.windows.hann(npts)
    elif taper_type == "boxcar":
        taper = signal.windows.boxcar(npts)
    elif taper_type == "hamming":
        taper = signal.windows.hamming(npts)
    elif taper_type == "tukey":
        taper = signal.windows.tukey(npts, alpha=alpha)
    else:
        raise ValueError("Taper type not supported: %s" % taper_type)
    return taper




class DeconInversion(object):
    """Actual inversion class."""

    def __init__(self, observed, green, t, dt, critical,
                 minT: float = 0, maxT: float = 50,
                 lamb=None, type="2", maxiter: int = 100,
                 taper_type: str = "tukey"):
        """
        :param obs: observed traces
        :param G: Green's functions
        :param maxT: time after which STF is forced to zero
        :param crit: critical value for stopping the iteration
        :param dt: time sampling
        :param lamb: waterlevel for deconvolution if type 2 is chosen. Unused if
                     type is "1".
        :param type: string defining the type of landweber method. Type 1 is the
                     method using the steepest decent; type 2 is using a Newton
                     step.
        :return:
        """

        # Get data
        self.obs = observed
        self.green = green
        # self.windows = windows # Window measurements and weighting data
        self.dt = dt
        self.t = t

        # Get parameters
        self.minT = minT
        self.maxT = maxT
        self.critical = critical
        self.lamb = lamb
        self.perc = 0.05  # Newton
        self.maxiter = maxiter
        self.taper_type = taper_type
        self.type = type

        # Deconvolution method
        if self.type == "1":
            self.compute_gradient = self.compute_gradient_sd
        elif self.type == "2":
            self.compute_gradient = self.compute_gradient_newton
        else:
            raise ValueError('Type must be "1" or "2"')

        # Get informations about size and initialize src
        self.nr, self.nt = self.obs.shape
        self.src = np.zeros(self.nt)


        # Compute objective function and residual
        self.orig_syn = self.forward(self.src)
        self.syn = self.forward(self.src)
        self.res = self.residual()
        self.chi = self.misfit()
        self.chi0 = self.chi
        self.it = 1

        # Lists to be filled:
        self.src_list = []
        self.chi_list = []

    def landweber(self):
        """Perform Landweber iterative source time function inversion."""

        # Compute the first gradient
        grad, alpha = self.compute_gradient()

        # Create source time function windowing taper
        pos = np.where(np.logical_and((self.minT <= self.t),
                                      (self.t <= self.maxT)))[0]
        Nt = len(pos)
        taperright = construct_taper(len(pos), taper_type=self.taper_type,
                                alpha=0.05)
        taperleft = construct_taper(len(pos), taper_type=self.taper_type,
                                alpha=0.01)

        # Make taper one sided
        Ptime = np.zeros(self.nt)
        Ptime[pos[:-Nt//2]] = taperleft[:-Nt//2]
        Ptime[pos[-Nt//2:]] = taperright[-Nt//2:]

        # Perform iterative deconvolution (inverse problem)
        self.chip = self.chi0

        while self.chi > self.critical * self.chi0 and self.it <= self.maxiter:

            # Regularized gradient
            gradreg = grad

            if type == "1":
                srctmp = self.src + gradreg
            else:
                srctmp = self.src + self.perc * gradreg

            # Window source time function --> zero after some time T
            srctmp = srctmp * Ptime

            # Enforce positivity
            srctmp[np.where(srctmp < 0)[0]] = 0

            # Compute misfit function and gradient
            self.syn = self.forward(srctmp)
            self.res = self.residual()
            self.chi = self.misfit()
            grad, _ = self.compute_gradient()
            self.it = self.it + 1

            # Check convergence
            if self.chi > self.chip:
                print("NOT CONVERGING")
                break

            if abs(self.chi - self.chip)/self.chi < 10**(-5):
                print("CONVERGING TOO LITTLE")
                break

            # Update
            # chi / chi0
            self.chip = self.chi
            self.src = srctmp

            self.chi_list.append(self.chi)
            self.src_list.append(self.src)

        # Final misfit function
        print("Iteration: %d -- Misfit: %1.5f" % (self.it,
                                                        self.chi / self.chi0))

    def residual(self):
        """Computes the residual between the observed data
        and the synthetic data."""

        return self.obs - self.syn

    def misfit(self):
        """Computes the misfit between the observed data and
        the forward modeled synthetic data."""

        # Trace by trace normalized misift
        return 0.5 * np.sum(np.sum((self.obs - self.syn) ** 2, axis=-1)/np.sum((self.obs) ** 2, axis=-1))

    def forward(self, src):
        """ Convolution of set of Green's functions

        :param green: Green's function
        :param src:
        :return:
        """
        # Get frequency spectrum of source time function
        SRC = fft(src)

        # Get frequency spectra of Green's functions
        GRE = fft(self.green, axis=1)

        # Convolve the two and return matrix containing the synthetic
        syn = np.real(ifft(GRE * SRC, axis=1)) * self.dt

        return syn

    def compute_gradient_newton(self):
        """ Compute Gradient using the waterlevel deconvolution which computes
        the Newton Step.

        :param resid: residual
        :param green: green's function
        :param lamb: waterlevel scaling
        :return:
        """

        # FFT of residuals and green functions
        RES = fft(self.res, axis=1)
        GRE = fft(self.green, axis=1)

        # Compute gradient (full wavelet estimation)
        num = np.sum(RES * np.conj(GRE), axis=0)
        den = np.sum(GRE * np.conj(GRE), axis=0)

        # Waterlevel
        wl = self.lamb * np.max(np.abs(den))
        pos = np.where(den < wl)
        den[pos] = wl
        grad = np.real(ifft(num / (den)))


        # Step value
        hmax = 1

        return grad, hmax

    def compute_gradient_sd(self):
        """ Compute the Gradient using the steepest decent method
        :param resid:
        :param green:
        :return:
        """

        # FFT of residuals and green functions
        RES = fft(self.res, axis=-1)
        GRE = fft(self.green, axis=-1)

        # Compute gradient (full wavelet estimation)
        norm = np.sum(self.obs**2,axis=-1)
        num = np.sum(RES * np.conj(GRE) / norm[:, None], axis=0)
        den = np.sum(GRE * np.conj(GRE) / norm[:, None], axis=0)

        mod = np.abs(den)
        cond = np.max(mod)/np.min(mod)

        # print(np.max(mod), factor)

        # Relaxation parameter
        # The factor is chosen to be close to the maximum value of the denominator
        # but slightly smaller so that the gradient converges a little bit faster
        factor = np.max(mod) * 10**(-np.log10(cond)/20)
        tau = 1 / factor
        grad = tau * np.real(ifft(num))

        # Step value
        hmax = 1

        return grad, hmax



class LagrangeInversion(object):
    """Actual inversion class."""

    def __init__(self, observed, green, t, dt, critical,
                 minT: float = 0, maxT: float = 50,
                 lamb=None, type="2", maxiter: int = 100,
                 taper_type: str = "tukey"):
        """
        :param obs: observed traces
        :param G: Green's functions
        :param maxT: time after which STF is forced to zero
        :param crit: critical value for stopping the iteration
        :param dt: time sampling
        :param lamb: waterlevel for deconvolution if type 2 is chosen. Unused if
                     type is "1".
        :param type: string defining the type of landweber method. Type 1 is the
                     method using the steepest decent; type 2 is using a Newton
                     step.
        :return:
        """

        # Get data
        self.obs = observed
        self.green = green
        self.dt = dt
        self.t = t
        self.Nt = len(self.t)
        
        # Get parameters
        self.minT = minT
        self.maxT = maxT
        self.critical = critical
        self.lamb = lamb
        self.perc = 0.05  # Newton
        self.maxiter = maxiter
        self.taper_type = taper_type
        self.type = type
        
        # Get taper parameters
        self.mpos = np.where(np.logical_and((self.minT <= self.t),
                                      (self.t <= self.maxT)))[0]
        self.Nm = len(self.mpos)       
        self.make_tapers()
        
        # Model parameters
        self.src = np.zeros(self.Nt)
        # self.lmd = np.zeros(self.Nm + 1)
        self.lmd = np.zeros(self.Nm)
        
        
        # Compute objective function and residual
        self.orig_syn = self.forward()
        self.syn = self.forward()
        
        print(len(self.src), len(self.syn))
        self.res = self.residual()
        self.chi = self.misfit()
        self.chi0 = self.chi
        self.it = 1

        # Lists to be filled:
        self.src_list = []
        self.chi_list = []
        
        
    def make_tapers(self):
        
        # Create left and right tapers
        taperright = construct_taper(len(self.mpos), taper_type=self.taper_type,
                                alpha=0.05)
        taperleft = construct_taper(len(self.mpos), taper_type=self.taper_type,
                                alpha=0.01)

        # Make taper one sided
        self.taper = np.zeros(self.Nt)
        self.taper[self.mpos[:-self.Nt//2]] = taperleft[:-self.Nt//2]
        self.taper[self.mpos[-self.Nt//2:]] = taperright[-self.Nt//2:]        


    def residual(self):
        """Computes the residual between the observed data
        and the synthetic data."""

        return (self.obs - self.syn)/np.sum((self.obs) ** 2, axis=-1)[:, None]
    
    def cg(self, x):
        
        # Get source part of model vector
        self.src[self.mpos] = x[:self.Nm]
        
        # Get constraint lagrange multipliers
        # self.lmd[:] = x[self.Nm:]
        
        self.forward()
        c = self.misfit()
        g = self.gradient()
        
        return c, g

    def optimize(self):
        
        # return opt.minimize(self.cg, np.zeros(self.Nm + self.Nm + 1), method='Nelder-Mead', jac=True)
        # return opt.minimize(self.cg, np.zeros(self.Nm + self.Nm), method='L-BFGS-B', jac=True)
        return opt.minimize(self.cg, np.zeros(self.Nm), method='Nelder-Mead', jac=True, tol=1e-2, options={'disp': True, 'maxiter': 200})

    def misfit(self):
        """Computes the misfit between the observed data and
        the forward modeled synthetic data."""

        # Trace by trace normalized misift
        C1 = 0.5 * np.sum(np.sum((self.obs - self.syn) ** 2, axis=-1)/np.sum((self.obs) ** 2, axis=-1))
        
        # Constraint for positivity
        # C2 = - np.sum(self.src[self.mpos] * self.lmd[:self.Nm])
        
        # Constraint for 
        # C3 = self.lmd[-1]*(1 - np.sum(self.src[self.mpos]) * self.dt)
        
        return C1 #+ C2 #+ C3

    def forward(self):
        """ Convolution of set of Green's functions

        :param green: Green's function
        :param src:
        :return:
        """
        # Get frequency spectrum of source time function
        SRC = fft(self.src)

        # Get frequency spectra of Green's functions
        GRE = fft(self.green, axis=1)

        # Convolve the two and return matrix containing the synthetic
        self.syn = np.real(ifft(GRE * SRC, axis=-1)) * self.dt
        
        return self.syn

    def gradient(self):
        """ Compute the Gradient using the steepest decent method
        :param resid:
        :param green:
        :return:
        """

        # FFT of residuals and green functions
        RES = fft(self.res, axis=-1)
        GRE = fft(self.green, axis=-1)

        # Compute gradient (full wavelet estimation)
        norm = np.sum(self.obs**2,axis=-1)
        num = np.sum(RES * np.conj(GRE) / norm[:, None], axis=0)
        den = np.sum(GRE * np.conj(GRE) / norm[:, None], axis=0)

        mod = np.abs(den)
        cond = np.max(mod)/np.min(mod)

        # print(np.max(mod), factor)

        # Relaxation parameter
        # The factor is chosen to be close to the maximum value of the denominator
        # but slightly smaller so that the gradient converges a little bit faster
        factor = np.max(mod) * 10**(-np.log10(cond)/20)
        tau = 1 / factor
        
        # Compute the gradient for the wavelet
        dC1df = tau * np.real(ifft(num))
        dC1df = dC1df * self.taper
        dC1df = dC1df[self.mpos]
        
        # Compute the gradient for the constraints of positivity
        # dC2df = - self.lmd[:-1]
        # dC2df = - self.lmd
    
        # Compute the gradient for the sum = 1 constraint
        # dC3df = np.ones(self.Nm)
        # dC3df *= self.lmd[-1] * self.dt 

        # Compute the gradient for the positivity constraint with respect to each lambda
        # dC2dlmd = - self.src[self.mpos]
        
        # Compute the gradient for the sum = 1 constraint with respect to each lambda
        # dC3dlmd = 1 - np.sum(self.src) * self.dt

        # print(len(dC1df), len(dC2df), len(dC3df), self.Nm)
        # Concatenate the gradients
        dJdm = np.zeros(self.Nm)        
        dJdm[:self.Nm] = dC1df #+ dC2df #+ dC3df
        # dJdm[self.Nm:] = dC2dlmd
        # dJdm[self.Nm:-1] = dC2dlmd
        # dJdm[-1] = dC3dlmd
        
        return dJdm


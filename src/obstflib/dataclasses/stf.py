from __future__ import annotations
import os
from glob import glob
import obspy
import numpy as np
from dataclasses import dataclass
import obspy.core.event.event
from ..utils import triangle_stf, boxcar_stf, gaussian_stf, error_stf, interp_stf
import matplotlib.pyplot as plt


@dataclass
class STF:
    """Source Time Function (STF) base class
    """
    origin: obspy.UTCDateTime
    t: np.ndarray
    f: np.ndarray
    tshift: float = 0.0  # starttime with respect to origin
    label: str = ""  # Some label

    @classmethod
    def gaussian(cls, origin: obspy.UTCDateTime, t, hdur: float, tc: float = 0.0,
                 tshift: float = 0.0, alpha: float = 1.628, M0: float = 1.0, **kwargs):
        """Crreat a Gaussian STF

        Parameters
        ----------
        origin : obspy.UTCDateTime
            Origin time where t, and f start
        t : np.ndarray
            time vector
        hdur : float
            half duration of the Gaussian. Sigma = hdur/alpha
        tc : float, optional
            centroid time with respect to tshift and origin, by default 0.0
        tshift : float, optional
            timeshift with respect to origin where t=0 neede for inversion,
            by default 0.0
        M0: float, optional
            Scalar moment, by default 1.0
        alpha : float, optional
            alpha parameter for the Gaussian. The default should probably not
            be changed, by default 1.628

        Returns
        -------
        STF
            Reutrn a STF object with a Gaussian STF
        """

        return cls(origin=origin, tshift=tshift, t=t,
                   f=M0*gaussian_stf(t, tshift+tc, hdur), **kwargs)

    @classmethod
    def error(cls, origin: obspy.UTCDateTime, t, hdur: float, tc: float = 0.0,
               tshift: float = 0.0, M0: float = 1.0, **kwargs):
        """Generates a specfem style error function for the modeling/integration
        of a moment tensor.

        Parameters
        ----------
        origin : obspy.UTCDateTime
            origin time where t, and f start
        t : np.ndarray
            time vector
        hdur : float
            half duration
        tc : float, optional
            centroid time, by default 0.0
        tshift : float, optional
            additional shift to define the source time function, by default 0.0
        M0 : float, optional
            scaleing for moment rate, by default 1.0

        Returns
        -------
        STF
            STF object with the error function
        """

        ret = cls(origin=origin, tshift=tshift, t=t,
                   f=M0*error_stf(t, tshift+tc, hdur), **kwargs)

        ret.M0 = M0
        
        return ret

    @classmethod
    def boxcar(cls, origin: obspy.UTCDateTime, t, hdur: float, tc: float = 0.0,
               tshift: float = 0.0, M0: float = 1.0, **kwargs):
        """Create a boxcar STF

        Parameters
        ----------
        origin : obspy.UTCDateTime
            Origin time where t, and f start
        t : np.ndarray
            time vector
        hdur : float
            half duration of the boxcar
        tc : float, optional
            centroid time with respect to tshift and origin, by default 0.0
        tshift : float, optional
            timeshift with respect to origin where t=0 neede for inversion,
            by default 0.0
        M0: float
            Scalar moment

        Returns
        -------
        STF
            Return a STF object with a boxcar STF
        """

        ret = cls(origin=origin, tshift=tshift, t=t,
                   f=M0*boxcar_stf(t, tshift+tc, hdur), **kwargs)
        
        ret.M0 = M0
        
        return ret

    @classmethod
    def triangle(cls, origin: obspy.UTCDateTime, t, hdur: float, tc: float = 0.0,
                 tshift: float = 0.0, M0=1.0, **kwargs):
        """Create a triangle STF

        Parameters
        ----------
        origin : obspy.UTCDateTime
            Origin time where t, and f start
        t : np.ndarray
            time vector
        hdur : float
            half duration of the triangle
        tc : float, optional
            centroid time with respect to tshift and origin, by default 0.0
        tshift : float, optional
            timeshift with respect to origin where t=0 neede for inversion,
            by default 0.0
        M0: float, by default 1.0
            Scalar moment
        Returns
        -------
        STF
            Return a STF object with a triangle STF
        """

        ret = cls(origin=origin, tshift=tshift, t=t,
                   f=M0*triangle_stf(t, tshift+tc, hdur), **kwargs)
        ret.M0 = M0
        
        return ret

    def interp(self, t: np.ndarray, tshift=0.0,
               origin: obspy.UTCDateTime | None = None) -> np.ndarray:
        """Interpolate the STF to a new time vector

        Parameters
        ----------
        t : np.ndarray
            New time vector

        Returns
        -------
        np.ndarray
            Interpolated STF
        """

        if (origin is not None):
            tshift_origin = self.origin - origin
            self.origin = origin
        else:
            tshift_origin = 0.0

        self.f = interp_stf(t, self.t + tshift - tshift_origin, self.f)
        self.t = t
        self.tshift = tshift


    def gradient(self, new: bool=False) -> STF:
        """Creates new STF object with the gradient of the STF"""

        if new:
            return STF(
                origin=self.origin,
                t=self.t,
                f=np.gradient(self.f, self.t),
                tshift=self.tshift
            )
        else:
            self.f = np.gradient(self.f, self.t)
            return self


    @classmethod
    def scardec(cls, filename):
        """This constructor reads a SCARDEC STF file and generates a STF object
        from the file content. In addition to the STF attributes it also adds
        the following attributes:

        - latitude
        - longitude
        - depth
        - M0
        - Mw
        - strike1
        - dip1
        - rake1
        - strike2
        - dip2
        - rake2
        - region

        The file format is as follows:
        1st line: YYYY MM DD HH MM SS'.0' Latitude Longitude [origin time and
        epicentral location from NEIC]

        2nd line: Depth(km) M0(N.m) Mw strike1(°) dip1(°) rake1(°) strike2(°)
        dip2(°) rake2(°) [all from SCARDEC]

        All the other lines are the temporal
        STF, with format: time(s), moment rate(N.m/s)

        """

        attrdict = dict()

        # Get region from filename
        attrdict["region"] = " ".join(os.path.basename(filename).split("_")[3:])

        with open(filename, "r") as filename:

            lines = filename.readlines()

        line1 = lines[0].split()
        line2 = lines[1].split()

        origin = obspy.UTCDateTime(
            int(line1[0]),
            int(line1[1]),
            int(line1[2]),
            int(line1[3]),
            int(line1[4]),
            float(line1[5]),
        )


        attrdict["latitude"] = float(line1[6])
        attrdict["longitude"] = float(line1[7])

        attrdict["depth"] = float(line2[0])
        attrdict["M0"] = float(line2[1])
        attrdict["Mw"] = float(line2[2])
        attrdict["strike1"] = int(line2[3])
        attrdict["dip1"] = int(line2[4])
        attrdict["rake1"] = int(line2[5])
        attrdict["strike2"] = int(line2[6])
        attrdict["dip2"] = int(line2[7])
        attrdict["rake2"] = int(line2[8])

        # Now get STF
        time = []
        moment_rate = []
        for line in lines[2:]:
            t, m = line.split()
            time.append(float(t))
            moment_rate.append(float(m))

        # Convert to numpy arrays
        time = np.array(time)
        moment_rate = np.array(moment_rate)

        # Create the object
        ret = cls(
            origin=origin,
            t=time,
            f=moment_rate,
            tshift=0.0,
            label='SCARDEC'
        )

        # Set the other attributes as well
        for key, value in attrdict.items():
            setattr(ret, key, value)

        return ret

    @classmethod
    def scardecdir(cls, dirname, stftype="optimal"):
        """Inside each earthquake directory, two files are provided, for the
        average STF (file fctmoysource_YYYYMMDD_HHMMSS_Name) and for the
        optimal STF (file fctoptsource_YYYYMMDD_HHMMSS_Name)"""

        if stftype == "optimal":
            return cls.scardec(glob(os.path.join(dirname, "fctoptsource*"))[0])
        elif stftype == "average":
            return cls.scardec(glob(os.path.join(dirname, "fctmoysource*"))[0])
        else:
            raise ValueError("stftype must be 'optimal' or 'average'")


    def plot(self, *args, ax=None, shift: bool = True, normalize=False, **kwargs):
        """Plot the STF"""
        if shift:
            t = self.t - self.tshift
        else:
            t = self.t
            
        if normalize != False:
            if normalize is True:
                f = self.f/np.max(self.f)
            else:
                f = self.f/normalize
        else:
            f = self.f
            
        if ax is not None:
            ax.plot(t, f, *args, **kwargs)
        else:
            plt.plot(t, f, *args, **kwargs)
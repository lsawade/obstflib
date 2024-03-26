from __future__ import annotations
import os
from glob import glob
import obspy
import numpy as np
from dataclasses import dataclass
import obspy.core.event.event
from ..utils import triangle_stf, boxcar_stf, gaussian_stf, interp_stf

@dataclass
class STF:
    """Source Time Function (STF) base class
    """
    origin: obspy.UTCDateTime
    t: np.ndarray
    f: np.ndarray
    tshift: float = 0.0  # starttime with respect to origin

    @classmethod
    def gaussian(cls, origin: obspy.UTCDateTime, t, hdur: float, tc: float = 0.0,
                 tshift: float = 0.0, alpha: float = 1.628):
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
        alpha : float, optional
            alpha parameter for the Gaussian. The default should probably not
            be changed, by default 1.628

        Returns
        -------
        STF
            Reutrn a STF object with a Gaussian STF
        """

        return cls(origin=origin, tshift=tshift, t=t,
                   f=gaussian_stf(t, tshift+tc, hdur))

    @classmethod
    def boxcar(cls, origin: obspy.UTCDateTime, t, hdur: float, tc: float = 0.0,
               tshift: float = 0.0):
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

        Returns
        -------
        STF
            Return a STF object with a boxcar STF
        """

        return cls(origin=origin, tshift=tshift, t=t,
                   f=boxcar_stf(t, tshift+tc, hdur))

    @classmethod
    def triangle(cls, origin: obspy.UTCDateTime, t, hdur: float, tc: float = 0.0,
                 tshift: float = 0.0):
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

        Returns
        -------
        STF
            Return a STF object with a triangle STF
        """

        return cls(origin=origin, tshift=tshift, t=t,
                   f=triangle_stf(t, tshift+tc, hdur))

    def interp(self, t: np.ndarray) -> np.ndarray:
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

        self.f = interp_stf(self.t, self.f, t)

class SCARDECSTF(STF):
    """Inside each earthquake directory, two files are provided, for the average STF (file fctmoysource_YYYYMMDD_HHMMSS_Name) and for the optimal STF (file fctoptsource_YYYYMMDD_HHMMSS_Name)

     These two STF files have the same format:

    1st line: YYYY MM DD HH MM SS'.0' Latitude Longitude [origin time and epicentral location from NEIC]
    2nd line: Depth(km) M0(N.m) Mw strike1(°) dip1(°) rake1(°) strike2(°) dip2(°) rake2(°) [all from SCARDEC]
    All the other lines are the temporal STF, with format: time(s), moment rate(N.m/s)
    """

    origin: obspy.UTCDateTime
    latitude: float
    longitude: float
    depth_in_km: float
    M0: float
    Mw: float
    strike1: float
    dip1: float
    rake1: float
    strike2: float
    dip2: float
    rake2: float
    time: np.ndarray
    moment_rate: np.ndarray
    region: str

    @classmethod
    def fromfile(cls, filename):

        # Get region from filename
        region = " ".join(os.path.basename(filename).split("_")[3:])

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
        latitude = float(line1[6])
        longitude = float(line1[7])

        depth_in_km = float(line2[0])
        M0 = float(line2[1])
        Mw = float(line2[2])
        strike1 = int(line2[3])
        dip1 = int(line2[4])
        rake1 = int(line2[5])
        strike2 = int(line2[6])
        dip2 = int(line2[7])
        rake2 = int(line2[8])

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
        return cls(
            origin,
            latitude,
            longitude,
            depth_in_km,
            M0,
            Mw,
            strike1,
            dip1,
            rake1,
            strike2,
            dip2,
            rake2,
            time,
            moment_rate,
            region,
        )

    @classmethod
    def fromdir(cls, dirname, stftype="optimal"):
        if stftype == "optimal":
            return cls.fromfile(glob(os.path.join(dirname, "fctoptsource*"))[0])
        elif stftype == "average":
            return cls.fromfile(glob(os.path.join(dirname, "fctmoysource*"))[0])
        else:
            raise ValueError("stftype must be 'optimal' or 'average'")

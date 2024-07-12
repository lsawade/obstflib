from __future__ import annotations
import os
from glob import glob
import warnings
import obspy
import numpy as np
from dataclasses import dataclass
import obspy.core.event.event


@dataclass(kw_only=True)
class CMTSOLUTION:
    """
    Class to handle a seismic moment tensor source including a source time
    function.
    """

    origin_time: obspy.UTCDateTime = obspy.UTCDateTime(0)  # Timestamp
    pde_lat: float = 0.0  # deg
    pde_lon: float = 0.0  # deg
    pde_depth: float = 0.0  # km
    mb: float = 0.0  # magnitude scale
    ms: float = 0.0  # magnitude scale
    region_tag: str = ""  # string
    eventname: str = ""  # event id -> GCMT
    time_shift: float = 0.0  # in s
    hdur: float = 0.0  # in s
    latitude: float = 0.0  # in deg
    longitude: float = 0.0  # in deg
    depth: float = 0.0  # in m
    Mrr: float = 0.0  # dyn*cm
    Mtt: float = 0.0  # dyn*cm
    Mpp: float = 0.0  # dyn*cm
    Mrt: float = 0.0  # dyn*cm
    Mrp: float = 0.0  # dyn*cm
    Mtp: float = 0.0  # dyn*cm

    @classmethod
    def read_quakeml(cls, infile: str):
        """Read quakeML and feed to from_event"""
        return cls.from_event(obspy.read_events(infile)[0])

    @classmethod
    def from_event(cls, event: obspy.core.event.event.Event):
        """Generate CMTSOLUTION from an obspy event object"""
        cmtsolution = None
        pdesolution = None

        for origin in event.origins:
            if origin.origin_type == "centroid":
                cmtsolution = origin
            else:
                pdesolution = origin

        if cmtsolution is None:
            raise ValueError("Moment tensor not found in event.")

        if pdesolution is None:
            raise ValueError("PDE not found in event.")

        origin_time = pdesolution.time
        pde_lat = pdesolution.latitude
        pde_lon = pdesolution.longitude
        pde_depth_in_m = pdesolution.depth
        mb = 0.0
        ms = 0.0
        for mag in event.magnitudes:
            if mag.magnitude_type == "Mb":
                mb = mag.mag
            elif mag.magnitude_type == "MS":
                ms = mag.mag

        # Get region tag
        try:
            region_tag = cmtsolution.region
        except Exception:
            try:
                region_tag = pdesolution.region
            except Exception:
                warnings.warn("Region tag not found.")
        region_tag = "N/A"

        eventname = ""
        for descrip in event.event_descriptions:
            if descrip.type == "earthquake name":
                eventname = descrip.text

        cmt_time = cmtsolution.time
        focal_mechanism = event.focal_mechanisms[0]
        half_duration = (
            focal_mechanism.moment_tensor.source_time_function.duration / 2.0
        )
        latitude = cmtsolution.latitude
        longitude = cmtsolution.longitude
        depth_in_m = cmtsolution.depth
        tensor = focal_mechanism.moment_tensor.tensor
        # Convert to dyn cm
        Mrr = tensor.m_rr * 1e7
        Mtt = tensor.m_tt * 1e7
        Mpp = tensor.m_pp * 1e7
        Mrt = tensor.m_rt * 1e7
        Mrp = tensor.m_rp * 1e7
        Mtp = tensor.m_tp * 1e7

        return cls(
            origin_time=origin_time,
            pde_lat=pde_lat,
            pde_lon=pde_lon,
            mb=mb,
            ms=ms,
            pde_depth=pde_depth_in_m,
            region_tag=region_tag,
            eventname=eventname,
            time_shift=cmt_time - origin_time,
            hdur=half_duration,
            latitude=latitude,
            longitude=longitude,
            depth=depth_in_m / 1000.0,
            Mrr=Mrr,
            Mtt=Mtt,
            Mpp=Mpp,
            Mrt=Mrt,
            Mrp=Mrp,
            Mtp=Mtp,
        )

    def to_gf3d(self):
        """Converts the CMTSOLUTION to a gf3d file"""
        from gf3d.source import CMTSOLUTION as gf3d_CMTSOLUTION

        return gf3d_CMTSOLUTION(
            origin_time=self.origin_time,
            pde_lat=self.pde_lat,
            pde_lon=self.pde_lon,
            pde_depth=self.pde_depth,
            mb=self.mb,
            ms=self.ms,
            region_tag=self.region_tag,
            eventname=self.eventname,
            time_shift=self.time_shift,
            hdur=self.hdur,
            latitude=self.latitude,
            longitude=self.longitude,
            depth=self.depth,
            Mrr=self.Mrr,
            Mtt=self.Mtt,
            Mpp=self.Mpp,
            Mrt=self.Mrt,
            Mrp=self.Mrp,
            Mtp=self.Mtp,
        )



    @classmethod
    def read(cls, infile: str):
        """Reads CMT solution file

        Parameters
        ----------
        infile : str
            CMTSOLUTION file

        Returns
        -------
        CMTSOLUTION class
            A class that contains all the tensor info

        """
        try:
            # Read an actual file
            if os.path.exists(infile):
                with open(infile, "rt") as f:
                    lines = f.readlines()

            # Read a multiline string.
            else:
                lines = infile.strip().split("\n")

        except Exception as e:
            print(e)
            raise IOError("Could not read CMTFile.")

        # Convert first line
        line0 = lines[0]

        # Split up origin time values
        origin_time = line0.strip()[4:].strip().split()[:6]

        # Create datetime values
        values = list(map(int, origin_time[:-1])) + [float(origin_time[-1])]

        # Create datetime stamp
        try:
            origin_time = obspy.UTCDateTime(*values)
        except (TypeError, ValueError):
            warnings.warn("Could not determine origin time from line: %s" % line0)
            origin_time = obspy.UTCDateTime(0)

        otherinfo = line0[4:].strip().split()[6:]
        pde_lat = float(otherinfo[0])
        pde_lon = float(otherinfo[1])
        pde_depth = float(otherinfo[2])
        mb = float(otherinfo[3])
        ms = float(otherinfo[4])
        region_tag = " ".join(otherinfo[5:])

        # Reading second line
        eventname = lines[1].strip().split()[-1]

        # Reading third line
        time_shift = float(lines[2].strip().split()[-1])

        # Reading fourth line
        half_duration = float(lines[3].strip().split()[-1])

        # Reading fifth line
        latitude = float(lines[4].strip().split()[-1])

        # Reading sixth line
        longitude = float(lines[5].strip().split()[-1])

        # Reading seventh line
        depth = float(lines[6].strip().split()[-1])

        # Reading lines 8-13
        Mrr = float(lines[7].strip().split()[-1])
        Mtt = float(lines[8].strip().split()[-1])
        Mpp = float(lines[9].strip().split()[-1])
        Mrt = float(lines[10].strip().split()[-1])
        Mrp = float(lines[11].strip().split()[-1])
        Mtp = float(lines[12].strip().split()[-1])

        return cls(
            origin_time=origin_time,
            pde_lat=pde_lat,
            pde_lon=pde_lon,
            pde_depth=pde_depth,
            mb=mb,
            ms=ms,
            region_tag=region_tag,
            eventname=eventname,
            time_shift=time_shift,
            hdur=half_duration,
            latitude=latitude,
            longitude=longitude,
            depth=depth,
            Mrr=Mrr,
            Mtt=Mtt,
            Mpp=Mpp,
            Mrt=Mrt,
            Mrp=Mrp,
            Mtp=Mtp,
        )

    @classmethod
    def from_sdr(cls, s, d, r, M0=1.0, **kwargs):
        """definition from Stein and Wysession
        s is the strike (phi_f), d is the dip (delta), and r is the slip angle
        (lambda). IM ASSIGNING THE COMPONENTS WRONG. SEE EMAIL TO YURI!!!!!

        Note How in Stein and Wysession the z-axis points UP (Figure 4.2-2) and
        in Aki and Richards the z-axis points down (Figure 4.20).
        This results in a sign changes for the fault normal and slip vector.

        Convention for the code here is
        X-North, Y-East, Z-Down
        R-Down, Theta-North, Phi-East


        """

        # To radians
        s = np.radians(s)
        d = np.radians(d)
        r = np.radians(r)

        # Fault normal
        n = np.array([-np.sin(d) * np.sin(s), +np.sin(d) * np.cos(s), -np.cos(d)])

        # Slip vector
        d = np.array(
            [
                np.cos(r) * np.cos(s) + np.cos(d) * np.sin(r) * np.sin(s),
                np.cos(r) * np.sin(s) - np.cos(d) * np.sin(r) * np.cos(s),
                -np.sin(r) * np.sin(d),
            ]
        )

        # Compute moment tensor Stein and Wysession Style
        Mx = M0 * (np.outer(n, d) + np.outer(d, n))

        # # Full terms from AKI & Richards (Box 4.4)
        # Mxx = -M0 * (np.sin(d) * np.cos(r)
        #   * np.sin(2*s) + np.sin(2*d) * np.sin(r) * np.sin(s)**2)
        # Mxy = +M0 * (np.sin(d) * np.cos(r)
        #   * np.cos(2*s) + 0.5 * np.sin(2*d) * np.sin(r) * np.sin(2*s))
        # Mxz = -M0 * (np.cos(d) * np.cos(r)
        #   * np.cos(s) + np.cos(2*d) * np.sin(r) * np.sin(s))
        # Myy = +M0 * (np.sin(d) * np.cos(r)
        #   * np.sin(2*s) - np.sin(2*d) * np.sin(r) * np.cos(s)**2)
        # Myz = -M0 * (np.cos(d) * np.cos(r)
        #   * np.sin(s) - np.cos(2*d) * np.sin(r) * np.cos(s))
        # Mzz = +M0 * np.sin(2*d) * np.sin(r)

        # # Moment tensor creation
        # Mx = np.array( [
        #     [Mxx, Mxy, Mxz],
        #     [Mxy, Myy, Myz],
        #     [Mxz, Myz, Mzz]
        # ])

        Mr = np.array(
            [
                [Mx[2, 2], Mx[2, 0], -Mx[2, 1]],
                [Mx[0, 2], Mx[0, 0], -Mx[0, 1]],
                [-Mx[1, 2], -Mx[1, 0], Mx[1, 1]],
            ]
        )

        return cls(
            Mrr=Mr[0, 0],
            Mtt=Mr[1, 1],
            Mpp=Mr[2, 2],
            Mrt=Mr[0, 1],
            Mrp=Mr[0, 2],
            Mtp=Mr[1, 2],
            **kwargs,
        )

    def write(self, outfile: str, mode="w"):
        """Writes classic CMTSOLUTION in classic format."""

        with open(outfile, mode) as f:
            f.write(self.__str__())

    def __str__(self):
        """Returns a string in classic CMTSOLUTION format."""

        # Reconstruct the first line as well as possible. All
        # hypocentral information is missing.
        if isinstance(self.origin_time, obspy.UTCDateTime):
            return_str = (
                " PDE %4i %2i %2i %2i %2i %5.2f %8.4f %9.4f %5.1f %.1f %.1f"
                " %s\n"
                % (
                    self.origin_time.year,
                    self.origin_time.month,
                    self.origin_time.day,
                    self.origin_time.hour,
                    self.origin_time.minute,
                    self.origin_time.second + self.origin_time.microsecond / 1e6,
                    self.pde_lat,
                    self.pde_lon,
                    self.pde_depth,
                    self.mb,
                    self.ms,
                    self.region_tag,
                )
            )
        else:
            return_str = "----- CMT Delta: ------- \n"

        return_str += "event name:  %10s\n" % (str(self.eventname),)
        return_str += "time shift:%12.4f\n" % (self.time_shift,)
        return_str += "half duration:%9.4f\n" % (self.hdur,)
        return_str += "latitude:%14.4f\n" % (self.latitude,)
        return_str += "longitude:%13.4f\n" % (self.longitude,)
        return_str += "depth:%17.4f\n" % (self.depth,)
        return_str += "Mrr:%19.6e\n" % (self.Mrr,)
        return_str += "Mtt:%19.6e\n" % (self.Mtt,)
        return_str += "Mpp:%19.6e\n" % (self.Mpp,)
        return_str += "Mrt:%19.6e\n" % (self.Mrt,)
        return_str += "Mrp:%19.6e\n" % (self.Mrp,)
        return_str += "Mtp:%19.6e\n" % (self.Mtp,)

        return return_str

    @property
    def Mw(self):
        """Moment magnitude M_w"""
        return 2 / 3 * np.log10(7 + self.M0) - 10.73

    @property
    def M0(self):
        """Scalar Moment M0 in Nm"""
        return (
            self.Mrr**2
            + self.Mtt**2
            + self.Mpp**2
            + 2 * self.Mrt**2
            + 2 * self.Mrp**2
            + 2 * self.Mtp**2
        ) ** 0.5 * 0.5**0.5

    @property
    def cmt_time(self):
        """UTC Origin + Timeshift"""
        return self.origin_time + self.time_shift

    @property
    def tensor(self):
        """6 element moment tensor"""
        return np.array([self.Mrr, self.Mtt, self.Mpp, self.Mrt, self.Mrp, self.Mtp])

    @property
    def fulltensor(self):
        """Full 3x3 moment tensor"""
        return np.array(
            [
                [self.Mrr, self.Mrt, self.Mrp],
                [self.Mrt, self.Mtt, self.Mtp],
                [self.Mrp, self.Mtp, self.Mpp],
            ]
        )

    @staticmethod
    def same_eventids(id1, id2):
        """Check whether eventids are the same."""

        id1 = id1 if not id1[0].isalpha() else id1[1:]
        id2 = id2 if not id2[0].isalpha() else id2[1:]

        return id1 == id2

    def update_hdur(self):
        """Updates the halfduration if M0 is was reset."""
        # Updates the half duration
        Nm_conv = 1 / 1e7
        self.half_duration = np.round(
            2.26 * 10 ** (-6) * (self.M0 * Nm_conv) ** (1 / 3), decimals=1
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __sub__(self, other: CMTSOLUTION):
        if not isinstance(other, CMTSOLUTION):
            return NotImplemented
        """ USE WITH CAUTION!!
        -> Origin time becomes float of delta t
        -> centroid time becomes float of delta t
        -> half duration is weird to compare like this as well.
        -> the other class will be subtracted from this one and the resulting
           instance will keep the eventname and the region tag from this class
        """

        if not self.same_eventids(self.eventname, other.eventname):
            raise ValueError("CMTSource.eventname must be equal to compare the events")

        # The origin time is the most problematic part
        origin_time = self.origin_time - other.origin_time
        pde_lat = self.pde_lat - other.pde_lat
        pde_lon = self.pde_lon - other.pde_lon
        pde_depth = self.pde_depth - other.pde_depth
        region_tag = self.region_tag
        eventame = self.eventname
        mb = self.mb - other.mb
        ms = self.ms - other.ms
        cmt_time = self.cmt_time - other.cmt_time
        print(self.cmt_time, other.cmt_time, cmt_time)
        half_duration = self.hdur - other.hdur
        latitude = self.latitude - other.latitude
        longitude = self.longitude - other.longitude
        depth = self.depth - other.depth
        Mrr = self.Mrr - other.Mrr
        Mtt = self.Mtt - other.Mtt
        Mpp = self.Mpp - other.Mpp
        Mrt = self.Mrt - other.Mrt
        Mrp = self.Mrp - other.Mrp
        Mtp = self.Mtp - other.Mtp

        return CMTSOLUTION(
            origin_time=origin_time,
            pde_lat=pde_lat,
            pde_lon=pde_lon,
            mb=mb,
            ms=ms,
            pde_depth=pde_depth,
            region_tag=region_tag,
            eventname=eventame,
            time_shift=cmt_time,
            hdur=half_duration,
            latitude=latitude,
            longitude=longitude,
            depth=depth,
            Mrr=Mrr,
            Mtt=Mtt,
            Mpp=Mpp,
            Mrt=Mrt,
            Mrp=Mrp,
            Mtp=Mtp,
        )

    def __ge__(self, other: CMTSOLUTION):
        if not isinstance(other, CMTSOLUTION):
            return NotImplemented
        """This comparison are implemented for the sorting in time."""
        if self.origin_time == other.origin_time:
            return self.time_shift >= other.time_shift
        else:
            return self.origin_time >= other.origin_time

    def __gt__(self, other: CMTSOLUTION):
        if not isinstance(other, CMTSOLUTION):
            return NotImplemented
        """This comparison are implemented for the sorting in time."""
        if self.origin_time == other.origin_time:
            return self.time_shift > other.time_shift
        else:
            return self.origin_time > other.origin_time

    def __eq__(self, other: CMTSOLUTION):
        if not isinstance(other, CMTSOLUTION):
            return NotImplemented
        return (
            self.origin_time,
            self.eventname,
            self.cmt_time,
            self.hdur,
            self.latitude,
            self.longitude,
            self.depth,
            self.Mrr,
            self.Mtt,
            self.Mpp,
            self.Mrt,
            self.Mrp,
            self.Mtp,
        ) == (
            other.origin_time,
            other.eventname,
            other.cmt_time,
            other.hdur,
            other.latitude,
            other.longitude,
            other.depth,
            other.Mrr,
            other.Mtt,
            other.Mpp,
            other.Mrt,
            other.Mrp,
            other.Mtp,
        )

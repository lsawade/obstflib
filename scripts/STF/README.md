# SCARDEC source time functions

I downloaded all SCARDEC source time functions for events larger than MW 7.5.
Then I matched it with all the GCMT events.

From the SCARDEC webpage:
> Inside each earthquake directory, two files are provided, for the average
> STF (file fctmoysource_YYYYMMDD_HHMMSS_Name) and for the optimal STF
> (file fctoptsource_YYYYMMDD_HHMMSS_Name)
> These two STF files have the same format:
>
>   1st line: YYYY MM DD HH MM SS'.0' Latitude Longitude [origin time and epicentral location from NEIC]
>   2nd line: Depth(km) M0(N.m) Mw strike1(°) dip1(°) rake1(°) strike2(°) dip2(°) rake2(°) [all from SCARDEC]
>   All the other lines are the temporal STF, with format: time(s), moment rate(N.m/s)
>

I added one file to each of the directories, with the name CMTSOLUTION, which
is the GCMT solution.


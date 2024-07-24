# %%
import os
import glob
import obstflib as osl
import obspy

large_events_dir = "/autofs/nccs-svm1_home1/lsawade/gcmt/cmt3d/scripts/events_large/cmt3d+_fix"
stf_events_dir = "/autofs/nccs-svm1_home1/lsawade/gcmt/obstflib/scripts/STF"


eventfiles = glob.glob(os.path.join(large_events_dir, "*"))
stf_event_dirs = glob.glob(os.path.join(stf_events_dir, "*"))

cmts = [osl.CMTSOLUTION.read(eventfile) for eventfile in eventfiles]

# %%
# Get the STF times

stftimes = []

for file in stf_event_dirs:
    if os.path.basename(file) == "README.md":
        continue
    date, time = os.path.basename(file).split("_")[1:3]
    
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:])
    
    hour = int(time[:2])
    minute = int(time[2:4])
    second = float(time[4:])
    
    stftime = obspy.UTCDateTime(year, month, day, hour, minute, second) 
    
    stftimes.append(stftime)
    
# %%
# Write CMTs to STF directories
    
for _cmt  in cmts:
    
    # Get the index of stftimes that has the closest time to the origin time of the CMT
    idx = min(range(len(stftimes)), key=lambda i: abs(stftimes[i]-_cmt.origin_time))
    
    # Get the STF directory
    stf_dir = stf_event_dirs[idx]
    
    # Write the CMT to the STF directory
    _cmt.write(os.path.join(stf_dir, f"{_cmt.eventname}_CMT3D+"))
    

    
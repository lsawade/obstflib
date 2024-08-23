#%%
# Imports
import os
import glob
import obstflib as osl

#%%
# Get files

# Path to the directory where the CMTSOLUTION files are stored
scardecdir='./STF'
outputdir='./events'

# Glob cmtsolutions
cmtfiles = glob.glob(os.path.join(scardecdir,'*', 'CMTSOLUTION'))

# %%
# Loop over files and read them and write them to new file

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

for file in cmtfiles:
    # Read the CMTSOLUTION file
    cmt = osl.CMTSOLUTION.read(file)
    
    # Getting the output file name
    outfile = os.path.join(outputdir, cmt.eventname)
    
    # Write file
    cmt.write(outfile)
    
    print(f'Wrote {outfile}')
    
f3d database query subset --fortran --ngll 5 -- glad-m25 subset_53.9_-35.3_10_50.h5 53.9 -35.3 10.0 75.0
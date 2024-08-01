#!/bin/bash
#SBATCH -A GEO111
#SBATCH -t 40:00
#SBATCH -N115
#SBATCH -J STF-Inversion
#SBATCH -p batch
#SBATCH --output=STF.%J.o.txt
#SBATCH --error=STF.%J.e.txt
#SBATCH --mem=0


module purge
module load PrgEnv-cray amd-mixed cray-mpich craype-accel-amd-gfx90a
module load core-personal hdf5-personal

export MPLCONFIGDIR=${LUSTRE}/.matplotlib
# export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=0

source ~/miniconda3/bin/activate gf


# Just a function to ensure that the number of running jobs does not exceed the number of nodes
############################################################################################################
job_limit () {
    # Test for single positive integer input
    if (( $# == 1 )) && [[ $1 =~ ^[1-9][0-9]*$ ]]
    then

        # Check number of running jobs
        joblist=($(jobs -rp))

        # Loop to heck whether jobs are still running
        while (( ${#joblist[*]} >= $1 ))
        do

            # Wait for any job to finish
            command='wait '${joblist[0]}
            for job in ${joblist[@]:1}
            do
                command+=' || wait '$job
            done
            eval $command
            joblist=($(jobs -rp))

        done
    fi
}
############################################################################################################

array=("body Z" "Ptrain Z" "Strain T") # "P Z" "P R" "S Z" "S R" "S T" "Ptrain Z" "Ptrain R" "Strain Z" "Strain R" "Strain T")

stfdir="/ccs/home/lsawade/gcmt/obstflib/scripts/STF"

for ARR in "${array[@]}"; do

    for DIR in $(ls $stfdir); do
        if [ $DIR == "README.md" ]; then
            continue
        fi
        cmd="python invert_cmt3d+.py STF/$DIR $ARR"
        echo "$cmd"
        srun -N1 -n1 -c40 -o logs/$DIR -e logs/$DIR $cmd &
        job_limit $(($SLURM_JOB_NUM_NODES+1))
        
    done
done
wait



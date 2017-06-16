#!/bin/bash

#SBATCH --workdir=/home/hpc/beattyg2/American-Sign-Language-Real-Time-Translation
#SBATCH --job-name=test	 # Job name
#SBATCH --output=job.%j.out # Name of stdout output file (%j expands to jobId)
#SBATCH --nodes=1 # Total number of nodes (a.k.a. servers) requested
#SBATCH --ntasks=1 # Total number of mpi tasks requested
#SBATCH --partition=gpu	 # Partition (a.k.a.queue) to use
#SBATCH --time=24:00:00 # Run time (days-hh:mm:ss or just hh:mm:ss)


# Setup environment
module add cudnn/5.1.10
module add python
source  /home/hpc/beattyg2/mypylibs/bin/activate

# Launch job
echo "Staring job @ `date`"
python3 train.py
echo "Job finished @ `date`."

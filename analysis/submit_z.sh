#!/bin/bash
#
# Batch job for `c02_subsample_z_to_0p5.py`, one model/experiment per job.
#
#   mkdir -p logs
#   sbatch -J z-CM4Xp125-historical  submit_z.sh CM4Xp125 historical
#
# The script is restartable: it skips years already present in both the
# depth-resolved and surface stores, so a job that dies can simply be
# resubmitted. It refuses to fill a year missing from the middle of a store
# (appends only go to the end); in that case delete the store and resubmit.
#
# Runs on the analysis nodes for the same reasons as submit_agessc.sh: the pp
# nodes are pre-AVX Westmere where this environment dies with SIGILL, and are
# extremely slow to read the environment off /work. The exclude list pins the
# job to the modern Xeon Gold nodes an[210-213].

#SBATCH -p analysis
#SBATCH --exclude=an[001-002,005-006,009-012,014,101-108,200-207]
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 14-00:00:00
#SBATCH --mem=120G
#SBATCH -o logs/%x_%j.out

source /home/Henri.Drake/miniconda3/etc/profile.d/conda.sh
conda activate CM4XTransientTracers

cd "$SLURM_SUBMIT_DIR" || exit 1

export DASK_NUM_WORKERS=2
export OMP_NUM_THREADS=1

echo "host      : $(hostname)"
echo "started   : $(date)"
echo "workdir   : $(pwd)"
echo "arguments : $*"

python -u c02_subsample_z_to_0p5.py "$@"
status=$?

echo "finished  : $(date)"
echo "exit code : $status"
exit $status

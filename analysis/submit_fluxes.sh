#!/bin/bash
#
# Batch job for `c01_subsample_fluxes_to_0p5.py`, one model/experiment per job.
#
#   mkdir -p logs
#   sbatch -J fluxes-CM4Xp25-historical         submit_fluxes.sh CM4Xp25  historical
#   sbatch -J fluxes-CM4Xp25-ssp585             submit_fluxes.sh CM4Xp25  ssp585
#   sbatch -J fluxes-CM4Xp125-historical        submit_fluxes.sh CM4Xp125 historical
#   sbatch -J fluxes-CM4Xp125-ssp585            submit_fluxes.sh CM4Xp125 ssp585
#   sbatch -J fluxes-CM4Xp125-piControl         submit_fluxes.sh CM4Xp125 piControl
#   sbatch -J fluxes-CM4Xp125-piControl-cont    submit_fluxes.sh CM4Xp125 piControl-continued
#
# Unlike the agessc job, this one is NOT restartable: each experiment is written
# in a single `to_zarr(mode="w")` call, so a job that hits the time limit loses
# that experiment's work and has to be resubmitted from scratch. Hence the long
# wall time below.
#
# Runs on the analysis nodes for the same reasons as submit_agessc.sh: the pp
# nodes are pre-AVX Westmere where this environment dies with SIGILL, and are
# extremely slow to read the environment off /work. The exclude list pins the
# job to the modern Xeon Gold nodes an[210-213].

#SBATCH -p analysis
#SBATCH --exclude=an[001-002,005-006,009-012,014,101-108,200-207]
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 48:00:00
#SBATCH --mem=60G
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

python -u c01_subsample_fluxes_to_0p5.py "$@"
status=$?

echo "finished  : $(date)"
echo "exit code : $status"
exit $status

#!/bin/bash
#
# Batch job for `c02_subsample_agessc_to_0p5.py`, one experiment per job.
#
# Submit all five CM4Xp125 experiments from this directory:
#
#   mkdir -p logs
#   for exp in historical ssp585 piControl piControl-continued piControl-spinup; do
#       sbatch -J agessc-$exp submit_agessc.sh CM4Xp125 $exp
#   done
#
# The jobs are restartable: each year is appended and validated individually, and
# a re-submission skips whole blocks that are already stored without issuing a
# dmget. So a job that hits the time limit can simply be resubmitted.
#
# Runs in $SLURM_SUBMIT_DIR, so submit from the analysis directory of the checkout
# whose ../data/interim you want written to.

#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --mem=20G
#SBATCH -o logs/%x_%j.out

source /home/Henri.Drake/miniconda3/etc/profile.d/conda.sh
conda activate CM4XTransientTracers

cd "$SLURM_SUBMIT_DIR" || exit 1

# Keep dask's threaded scheduler inside the cgroup: it would otherwise size itself
# from the node's 8 cores rather than the 4 requested above.
export DASK_NUM_WORKERS=4
export OMP_NUM_THREADS=1

echo "host      : $(hostname)"
echo "started   : $(date)"
echo "workdir   : $(pwd)"
echo "arguments : $*"

python -u c02_subsample_agessc_to_0p5.py "$@"
status=$?

echo "finished  : $(date)"
echo "exit code : $status"
exit $status

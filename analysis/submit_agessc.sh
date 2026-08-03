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
#
# Runs on the analysis nodes, NOT the pp nodes. The pp nodes are old Westmere
# (Xeon X5677, no AVX at all) and this conda environment dies there with SIGILL,
# `Illegal instruction (core dumped)`, exit 132. They are also pathologically slow
# at reading the environment off /work: `import numpy` alone took 490 s on pp016
# and 176 s on pp300, versus seconds on an210. The `analysis` partition is
# heterogeneous too -- an001 is pre-AVX Nehalem and an101 is Sandy Bridge -- so the
# exclude list below pins the job to the modern Xeon Gold nodes an[210-213].
# /archive and dmget are reachable from these nodes.

#SBATCH -p analysis
#SBATCH --exclude=an[001-002,005-006,009-012,014,101-108,200-207]
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 24:00:00
#SBATCH --mem=20G
#SBATCH -o logs/%x_%j.out

source /home/Henri.Drake/miniconda3/etc/profile.d/conda.sh
conda activate CM4XTransientTracers

cd "$SLURM_SUBMIT_DIR" || exit 1

# Keep dask's threaded scheduler inside the cgroup: it would otherwise size itself
# from the node total rather than the cores requested above.
export DASK_NUM_WORKERS=2
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

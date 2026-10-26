#!/bin/bash
#
# Batch job that regenerates `data/processed/` by executing the compute
# notebooks in place, in dependency order:
#
#   sbatch -J processed submit_processed.sh
#   sbatch -J processed --dependency=afterok:<jobid> submit_processed.sh
#
# c04 goes first because `common.py` opens `moc_metrics_piControl.nc` at import
# time, so no other notebook can run until it exists. The rest read the interim
# stores through `common.load_datasets()`. The job stops at the first notebook
# that fails, leaving that notebook's partial outputs in place for inspection.
#
# Before touching anything it checks that the monthly surface stores read by
# c08/c10 are complete, so a half-rebuilt store fails loudly here instead of
# silently producing short records.

#SBATCH -p analysis
#SBATCH --exclude=an[001-002,005-006,009-012,014,101-108,200-207]
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 3-00:00:00
#SBATCH --mem=120G
#SBATCH -o logs/%x_%j.out

source /home/Henri.Drake/miniconda3/etc/profile.d/conda.sh
conda activate CM4XTransientTracers

cd "$SLURM_SUBMIT_DIR" || exit 1

export OMP_NUM_THREADS=1

echo "host      : $(hostname)"
echo "started   : $(date)"
echo "commit    : $(git rev-parse HEAD)"

python -u - <<'EOF' || exit 1
from preprocessing import validate_surface_store_after_append

for model in ["CM4Xp125", "CM4Xp25"]:
    for exp, last_year in [("historical", 2014), ("ssp585", 2099)]:
        path = f"../data/interim/{model}_{exp}_transient_tracers_surface.zarr"
        validate_surface_store_after_append(path, expected_year=last_year)
        print(f"ok        : {path}")
EOF

notebooks=(
    c04_moc_metrics
    c03_integrate_zonally_and_vertically
    c05_peak_cfc_content_year
    c07_sink_to_source_transition_years
    c08_tracer_content_by_density_layer_global
    c09_tracer_content_by_density_layer_spatial
)

for nb in "${notebooks[@]}"; do
    echo "running   : $nb ($(date))"
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=cm4xtransienttracers \
        "$nb.ipynb"
    status=$?
    if [ $status -ne 0 ]; then
        echo "FAILED    : $nb (exit $status)"
        exit $status
    fi
    echo "done      : $nb ($(date))"
done

echo "finished  : $(date)"
echo "exit code : 0"

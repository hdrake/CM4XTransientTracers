# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Analysis of transient tracer (CFC-11, CFC-12, SF6) uptake, transport, and outgassing in the GFDL CM4X
coupled climate model, at two ocean resolutions (`CM4Xp25` = 0.25°, `CM4Xp125` = 0.125°). Not a
software package — it is a paper-analysis repository of scripts and notebooks that read raw model
output off GFDL's `/archive` filesystem and reduce it to publishable figures.

## Environment

```bash
conda env create -f environment.yml
conda activate CM4XTransientTracers
python -m ipykernel install --user --name CM4XTransientTracers --display-name "CM4XTransientTracers"
```

Notebooks expect the `cm4xtransienttracers` kernel. The environment pins three of the author's own
GitHub packages that supply nearly all the model-specific plumbing: `CM4Xutils` (pinned to a tag —
`exp_dict`, `pp_dict`, `ds_to_grid`, `fix_geo_coords`, `add_grid_coords`, `coarsen.horizontally_coarsen`),
`gfdl_utils` (`open_frompp`, `open_static`, `get_local` — locating and opening GFDL post-processed
time series), and `doralite` (`dora_metadata` — resolving an experiment ID like `odiv-255` to its
`pathPP`). Bumping the `CM4Xutils` tag can silently change grid/coarsening behavior.

Everything runs relative to `analysis/`, so `cd analysis` first. Long jobs are launched as, e.g.:

```bash
conda activate CM4XTransientTracers
cd analysis/
python c02_subsample_z_to_0p5.py CM4Xp125 historical
```

There are no tests, no linter, and no build step.

## Data flow

The pipeline is three stages, encoded in filename prefixes in `analysis/`. **Run them in numeric
order** — each stage assumes the outputs of the previous one exist.

- `p##_` — *prepare*: build atmospheric tracer boundary-condition files (extending the OMIP CFC
  histories out to year 3000 / piControl year 1300 using SSP5-8.5 global means). These write to
  `/archive/hfd/datasets/input_files/`, i.e. they feed the *model runs*, not the analysis.
- `c##_` — *compute*: read raw `/archive` output, coarsen and reduce, write to `data/interim/`
  (large, gitignored) and `data/processed/` (small, committed).
- `v##_` — *visualize*: read `data/processed/` (and some `data/interim/`) and write PNGs to `figures/`
  (gitignored). Figure filenames encode paper position (`Fig1_...`, `SFig2_...`).

`S_` prefixes one-off side deliverables for collaborators, not part of the pipeline.

Directory roles are also described in `data/README.md` and its subdirectory READMEs:
`data/input/` (committed atmospheric histories), `data/interim/` (bulky Zarr/netCDF, gitignored),
`data/processed/` (small committed netCDF), `data/obs/` (GLODAP, gitignored).

### Key interim products

`data/interim/{model}_{exp}_*` Zarr stores, where `exp` ∈ `historical`, `ssp585`, `piControl`,
`piControl-continued`:

| store | producer | contents |
| --- | --- | --- |
| `..._transient_tracer_fluxes.zarr` | `c01_subsample_fluxes_to_0p5.py` | monthly air–sea fluxes `fg{tracer}` |
| `..._transient_tracers_z.zarr` | `c02_subsample_z_to_0p5.py` | annual-mean, depth-resolved tracers + state (`year` dim) |
| `..._transient_tracers_surface.zarr` | `c02_subsample_z_to_0p5.py` | monthly top level only (`time` dim) |
| `..._transports_rho2.zarr` | `c02_subsample_rho2_to_0p5.py` | monthly `umo`/`vmo` on `rho2` layers |

All are coarsened to a common 0.5° grid — `{X:2, Y:2}` for `CM4Xp25`, `{X:4, Y:4}` for `CM4Xp125` —
via `CM4Xutils.coarsen.horizontally_coarsen`, which is finite-volume-conservative. Do not substitute
a plain `.coarsen().mean()`.

## Conventions that will bite you

**`analysis/common.py` is the shared header for notebooks.** Nearly every `c##`/`v##` notebook opens
with `from common import *` then `grids = load_datasets()`. `load_datasets()` concatenates the
historical and ssp585 Zarr stores into one `xgcm.Grid` per model, keyed `"{model}_forced"`
(e.g. `"CM4Xp125_forced"`), renaming surface variables with a `_surface` suffix. It also defines the
four-layer labels/colors, molar masses, and unit conversions used across all figures. Change a
constant there and every downstream figure changes. Note `common.py` opens
`data/processed/moc_metrics_piControl.nc` at *import* time — to derive the `sigma2_moc_l`/`sigma2_moc_i`
density coordinates — so `c04_moc_metrics.ipynb` must have run before any other notebook can even
import it.

**`analysis/preprocessing.py` is the shared header for the `c02` scripts.** They do `from
preprocessing import *`. It holds the `/archive` loaders (`load_tracer`, `load_state`, `load_rho2`,
`load`) plus a set of restart-safety helpers.

**The `c02` scripts are restartable and append year-by-year.** They loop over 5-year `/archive`
blocks, stage each year to a temp Zarr under `data/interim/tmp_<uuid>/`, append it to the real store,
then validate (`validate_coarse_store_after_append`, `validate_surface_store_after_append`) that the
year landed exactly once and the leading dimension of every array grew together. On re-run they skip
years already present in *both* the coarse and surface stores. If you modify these scripts, preserve
that skip/validate structure — the stores are appended to over many wall-clock hours and a
half-written year is the failure mode being defended against. Only years with all 12 months are kept.

**σ2 is reconstructed, not taken from a diagnostic.** `regrid_sigma2_from_rho2_diags` conservatively
regrids the model's online `rho2`-coordinate thicknesses onto depth levels to get a thickness-weighted
σ2 on the z-grid. `diagnose_sigma2_offline` (a plain `gsw.sigma2(so, thetao)`) exists alongside it as
the cruder offline alternative. Prefer the regridded one; both are carried in the dataset.

**piControl time is shifted to historical-equivalent years.** `control_to_historical_year` /
`assign_historical_dates` add 1749 to control years so a control year 101 lines up with 1850; the
original coordinate is kept as `year_ctrl`/`time_ctrl`. Anything comparing forced to control runs
depends on this offset.

**Zarr stores are written and read with `consolidated=False` and cftime decoding.** Projections run
past the `datetime64[ns]` range, so use `open_zarr_cftime` (or an equivalent
`CFDatetimeCoder(use_cftime=True)`) rather than a bare `xr.open_zarr`.

**Experiments are addressed by Dora ID.** `CM4Xutils.exp_dict[model][exp]` gives IDs like `odiv-255`
(CM4Xp125 historical), `odiv-293` (ssp585), `odiv-231`/`odiv-232` (CM4Xp25 historical/ssp585);
`doralite.dora_metadata(id)["pathPP"]` resolves the archive path. Prefer this to hardcoding
`/archive/Raphael.Dussin/...` paths — some older code does hardcode them and should not be copied.

**Post-processing stream names differ between resolutions.** `ocean_month_z` for p125 vs
`ocean_monthly_z` for p25; tracers live in `ocean_inert_z` / `ocean_inert_month`, densities in
`ocean_month_rho2`, MOC in `ocean_month_rho2_refined`, ideal age in `ocean_annual_z_d2`.
`gu.open_frompp(..., dmget=True)` triggers tape retrieval, so first-touch reads can be very slow.

## Repository history caveat

`main` is stale. The active line of work (`add-agessc-tracer`, branched from
`testing-tracer-diags-for-Anthony`) is many commits ahead and contains the reorganized `analysis/`
layout described above. `origin/main` still has the superseded `notebooks/` + `scripts/` split, where
`notebooks/common.py` used a 3-layer σ2 partition and the subsample scripts had no restart logic.
Do not merge or cherry-pick from `main` without checking which layout you are pulling in.

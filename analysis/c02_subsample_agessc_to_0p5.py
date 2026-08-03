#!/usr/bin/env python
# coding: utf-8

## Example submission from command line:
## conda activate CM4XTransientTracers ; cd /work/hfd/projects/CM4XTransientTracers/analysis/ ; python c02_subsample_agessc_to_0p5.py CM4Xp125 historical
##
## Optional third argument processes a single 5-year block, for smoke testing:
## python c02_subsample_agessc_to_0p5.py CM4Xp125 historical 1850
##
## Writes the ideal age tracer to its own companion store, rather than into
## `{model}_{exp}_transient_tracers_z.zarr`, so that the (expensive) transient
## tracer stores never have to be rebuilt. `common.load_datasets` merges the two.

import warnings
from collections import Counter

import doralite
import gfdl_utils.core as gu
import CM4Xutils
import numpy as np
import xarray as xr
import xgcm

import gsw, xwmt
import zarr
import cftime
import gsw
import dask

import os
import sys
import shutil
import uuid

from preprocessing import *

model = sys.argv[1]
exp = sys.argv[2]
block_override = int(sys.argv[3]) if len(sys.argv) > 3 else None
testing = False

if model == "-f":
    model = "CM4Xp125"

if model != "CM4Xp125":
    raise ValueError(
        f"The ideal age companion store is CM4Xp125-only for now (got '{model}'). "
        f"CM4Xp25 `agessc` lives in `ocean_annual_z` on the same grid as the CFCs, "
        f"and CM4Xp25 ssp585 has no annual/5yr chunks, so it needs its own handling."
    )

print(f"Subsample {model} ideal age output for {exp}.")

# NOTE: 2, not 4. `agessc` is diagnosed on `ocean_annual_z_d2`, which is ALREADY
# coarsened by a factor of 2 relative to the CFC stream (`ocean_inert_z`).
# 1440/2 = 720 and 1120/2 = 560, i.e. exactly the same 0.5 degree target grid the
# CFCs reach from full resolution with {X:4, Y:4}.
dim_coarsen_dict = {"CM4Xp125": {"X": 2, "Y": 2}}
dim_coarsen = dim_coarsen_dict[model]
odivs = CM4Xutils.exp_dict[model]

# -----------------------------------------------------------------------------#
# Time ranges
# -----------------------------------------------------------------------------#

time_ranges = {
    "CM4Xp125": {
        "historical": [1850, 2015],
        "ssp585": [2015, 2100],
        "piControl": [101, 451],
        "piControl-continued": [451, 651],
        # ASSUMPTION: only the pre-1850 portion of the v7 spinup. Under the +1749
        # convention of `control_to_historical_year`, spinup years 1-100 map to
        # 1750-1849, immediately preceding piControl's year 101 -> 1850. The v7
        # record extends to year 0350, but those later years overlap the v8
        # piControl run and would collide in historical-equivalent years. Widen to
        # [1, 351] to process the full 0001-0350 record.
        "piControl-spinup": [1, 101],
    },
}
t_range = time_ranges[model][exp]
if block_override is not None:
    t_range = [block_override, block_override + 5]

print(time_ranges[model])

# Root temporary directory for staging per-year temp Zarrs
tmp_root = os.path.join("..", "data", "interim", f"tmp_{uuid.uuid4().hex}")
os.makedirs(tmp_root, exist_ok=True)

with warnings.catch_warnings(action="ignore", category=UserWarning):
    coarse_path = f"../data/interim/{model}_{exp}_ideal_age_z.zarr"

    existing_years = get_existing_years_from_year_dim(coarse_path, year_dim="year")

    print(f"Experiment {exp}:")
    print(f"  existing years: {sorted(existing_years)}")

    coarse_counts = get_year_occurrences_from_year_dim(coarse_path, year_dim="year")
    coarse_duplicates = {y: c for y, c in coarse_counts.items() if c > 1}
    if coarse_duplicates:
        print("  WARNING: duplicate years already present in store:")
        print(f"    {dict(sorted(coarse_duplicates.items()))}")

    coarse_exists = os.path.exists(coarse_path)

    for t in np.arange(t_range[0], t_range[1], 5):
        tstr = f"{str(t).zfill(4)}*"
        expected_years = np.arange(t, min(t + 5, t_range[1])).astype(int)
        expected_hist_years = [
            control_to_historical_year(y, exp) for y in expected_years
        ]

        # Skip the whole block, and its dmget, when every year is already stored.
        # Block years are deterministic from `t`, so this makes a re-run cost
        # seconds instead of many hours of tape reads.
        if all(y in existing_years for y in expected_hist_years):
            print(f"\n=== Block {tstr}: all years already present; skipping load. ===")
            continue

        print(f"\n=== Loading block {model}-{exp} for t pattern {tstr} ===")

        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            grid = load_agessc(odivs[exp], t=tstr)
            ds = grid._ds

            # `agessc` is already an annual mean, so there is no
            # `groupby("time.year").mean("time")` step here. Convert the time
            # coordinate to a year dimension instead, and assert that we got the
            # block we asked for -- this is what catches an off-by-five error in
            # the annual/10yr chunk slicing.
            if not time_values_strictly_increasing(ds.time.values):
                raise RuntimeError(f"Time coordinate not strictly increasing for block {tstr}")

            block_years = extract_unique_years_from_time_coord(ds.time)
            if not np.array_equal(np.sort(block_years), expected_years):
                raise RuntimeError(
                    f"Year mismatch for block {tstr}: loaded {block_years.tolist()}, "
                    f"expected {expected_years.tolist()}"
                )
            if ds.sizes["time"] != len(expected_years):
                raise RuntimeError(
                    f"Expected {len(expected_years)} annual records for block {tstr}, "
                    f"got {ds.sizes['time']}"
                )

            ds = (
                ds
                .assign_coords({"year": ("time", block_years)})
                .swap_dims({"time": "year"})
                .drop_vars("time")
            )

            print(f"Block years in this chunk: {block_years}")

            for year in block_years:
                hist_year = control_to_historical_year(year, exp)

                if coarse_year_present(coarse_path, hist_year):
                    print(f"  Year {year} (historical {hist_year}) already complete; skipping.")
                    continue

                print(f"  Processing year {year} (historical {hist_year})...")

                ds_year = ds.sel(year=year).expand_dims(year=[int(year)])

                # `skip_coords=True`: the coarsened grid coordinates from this
                # half-resolution stream would not match the transient tracer
                # stores' (the wet-fraction bookkeeping differs), so we keep only
                # `agessc` and inherit coordinates from those stores at merge time.
                ds_year_coarse = CM4Xutils.coarsen.horizontally_coarsen(
                    ds_year, grid, dim_coarsen, skip_coords=True
                )

                # One coarse year is ~56 MB, so realize it once here: it makes the
                # zero-repair count free and the staged write a memory copy.
                ds_year_coarse = ds_year_coarse.compute()

                # Must run before stripping, since it needs the coarsened volcello.
                ds_year_coarse = restore_coarsened_exact_zeros(ds_year_coarse, "agessc")
                ds_year_coarse = strip_to_index_coords(ds_year_coarse, ["agessc"])
                ds_year_coarse["agessc"] = ds_year_coarse["agessc"].astype("float32")

                if "piControl" in exp:
                    ds_year_coarse = assign_historical_dates(ds_year_coarse)

                coarse_year_vals = np.array(ds_year_coarse.year.values).astype(int)
                if len(coarse_year_vals) != 1 or int(coarse_year_vals[0]) != hist_year:
                    raise RuntimeError(
                        f"Coarse year mismatch before append: found {coarse_year_vals}, "
                        f"expected [{hist_year}]"
                    )
                if (ds_year_coarse.sizes["xh"], ds_year_coarse.sizes["yh"]) != (720, 560):
                    raise RuntimeError(
                        f"Unexpected coarse shape {dict(ds_year_coarse.sizes)}; "
                        f"expected xh=720, yh=560"
                    )
                if not np.array_equal(ds_year_coarse.xh.values, np.arange(720)):
                    raise RuntimeError(
                        "xh is not the integer index 0..719 -- coarsening left float "
                        "block means, so the store would not align with the tracer stores"
                    )

                coarse_tmp = os.path.join(
                    tmp_root, f"{model}_{exp}_agessc_year{hist_year}.zarr"
                )
                safe_rmtree(coarse_tmp)

                coarse_chunks = {
                    "year": 1,
                    "z_l": ds_year_coarse.z_l.size,
                    "yh": ds_year_coarse.yh.size,
                    "xh": ds_year_coarse.xh.size,
                }
                ds_coarse_write = (
                    ds_year_coarse
                    .sortby("year")
                    .chunk(coarse_chunks)
                )

                print(f"    Writing temp Zarr for historical year {hist_year} -> {coarse_tmp}")
                ds_coarse_write.to_zarr(coarse_tmp, mode="w", consolidated=False)

                ds_coarse_tmp = open_zarr_cftime(coarse_tmp, consolidated=False)
                expected_year_arrays = get_expected_year_arrays(ds_coarse_tmp, append_dim="year")

                if not coarse_exists:
                    print(f"    Creating new Zarr store: {coarse_path}")
                    ds_coarse_tmp.to_zarr(
                        coarse_path,
                        mode="w",
                        consolidated=False,
                        compute=True,
                    )
                    coarse_exists = True
                else:
                    print(f"    Appending historical year {hist_year} to Zarr: {coarse_path}")
                    ds_coarse_tmp.to_zarr(
                        coarse_path,
                        mode="a",
                        append_dim="year",
                        consolidated=False,
                        compute=True,
                    )

                validate_coarse_store_after_append(
                    coarse_path,
                    expected_year=hist_year,
                    expected_names=expected_year_arrays,
                )

                safe_rmtree(coarse_tmp)
                existing_years.add(hist_year)

                if testing:
                    print("Testing mode: stopping after first year.")
                    break

            if testing:
                break

try:
    shutil.rmtree(tmp_root)
except Exception:
    pass

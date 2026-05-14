#!/usr/bin/env python
# coding: utf-8

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
testing = False

if model == "-f":
    model = "CM4Xp125"

print(f"Subsample {model} transient tracer output for {exp}.")
dim_coarsen_dict = {"CM4Xp25": {"X": 2, "Y": 2}, "CM4Xp125": {"X": 4, "Y": 4}}
dim_coarsen = dim_coarsen_dict[model]
odivs = CM4Xutils.exp_dict[model]

# -----------------------------------------------------------------------------#
# Time ranges
# -----------------------------------------------------------------------------#

time_ranges = {
    "CM4Xp25": {
        "historical": [1850, 2015],
        "ssp585": [2015, 2100],
        "piControl": [101, 361],
        "piControl-continued": [361, 651],
    },
    "CM4Xp125": {
        "historical": [1850, 2015],
        "ssp585": [2015, 2100],
        "piControl": [101, 451],
        "piControl-continued": [451, 651]
    },
}
t_range = time_ranges[model][exp]

print(time_ranges[model])

# Root temporary directory for staging per-year temp Zarrs
tmp_root = os.path.join("..", "data", "interim", f"tmp_{uuid.uuid4().hex}")
os.makedirs(tmp_root, exist_ok=True)

with warnings.catch_warnings(action="ignore", category=UserWarning):
    coarse_path = f"../data/interim/{model}_{exp}_transient_tracers_z.zarr"
    surface_path = f"../data/interim/{model}_{exp}_transient_tracers_surface.zarr"

    existing_coarse_years = get_existing_years_from_year_dim(coarse_path, year_dim="year")
    existing_surface_years = get_existing_years_from_time(surface_path)
    done_years = existing_coarse_years & existing_surface_years

    print(f"Experiment {exp}:")
    print(f"  coarse years:  {sorted(existing_coarse_years)}")
    print(f"  surface years: {sorted(existing_surface_years)}")
    print(f"  completed years (both): {sorted(done_years)}")

    coarse_counts = get_year_occurrences_from_year_dim(coarse_path, year_dim="year")
    coarse_duplicates = {y: c for y, c in coarse_counts.items() if c > 1}
    if coarse_duplicates:
        print("  WARNING: duplicate years already present in coarse store:")
        print(f"    {dict(sorted(coarse_duplicates.items()))}")

    coarse_exists = os.path.exists(coarse_path)
    surface_exists = os.path.exists(surface_path)

    for t in np.arange(t_range[0], t_range[1], 5):
        tstr = f"{str(t).zfill(4)}*"
        print(f"\n=== Loading block {model}-{exp} for t pattern {tstr} ===")

        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            grid = load(odivs[exp], ["cfc11", "cfc12", "sf6"], t=tstr)

            month_counts = grid._ds.time.groupby("time.year").count()
            valid_years = month_counts.where(month_counts == 12, drop=True).year.values

            ds_mean = (
                grid._ds
                .groupby("time.year")
                .mean("time")
                .sel(year=valid_years)
            )

            ds_surface = grid._ds.isel(z_l=[0])

            block_years = ds_mean.year.values.astype(int)
            print(f"Block years in this chunk: {block_years}")

            for year in block_years:
                hist_year = control_to_historical_year(year, exp)

                coarse_done = coarse_year_present(coarse_path, hist_year)
                surface_done = surface_year_present(surface_path, hist_year)

                if coarse_done and surface_done:
                    print(f"  Year {year} (historical {hist_year}) already complete; skipping.")
                    continue

                print(
                    f"  Processing year {year} (historical {hist_year})..."
                    f" coarse_done={coarse_done}, surface_done={surface_done}"
                )

                # -----------------------------------------------------------------
                # COARSE (depth-resolved)
                # -----------------------------------------------------------------
                if not coarse_done:
                    ds_year = ds_mean.sel(year=year).expand_dims(year=[year])

                    ds_year_coarse = CM4Xutils.coarsen.horizontally_coarsen(
                        ds_year, grid, dim_coarsen
                    )

                    if "piControl" in exp:
                        ds_year_coarse = assign_historical_dates(ds_year_coarse)

                    coarse_year_vals = np.array(ds_year_coarse.year.values).astype(int)
                    if len(coarse_year_vals) != 1 or int(coarse_year_vals[0]) != hist_year:
                        raise RuntimeError(
                            f"Coarse year mismatch before append: found {coarse_year_vals}, "
                            f"expected [{hist_year}]"
                        )

                    coarse_tmp = os.path.join(
                        tmp_root, f"{model}_{exp}_coarse_year{hist_year}.zarr"
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

                    print(f"    Writing coarse temp Zarr for historical year {hist_year} -> {coarse_tmp}")
                    ds_coarse_write.to_zarr(coarse_tmp, mode="w", consolidated=False)

                    ds_coarse_tmp = open_zarr_cftime(coarse_tmp, consolidated=False)
                    expected_year_arrays = get_expected_year_arrays(ds_coarse_tmp, append_dim="year")

                    if not coarse_exists:
                        print(f"    Creating new coarse Zarr store: {coarse_path}")
                        ds_coarse_tmp.to_zarr(
                            coarse_path,
                            mode="w",
                            consolidated=False,
                            compute=True,
                        )
                        coarse_exists = True
                    else:
                        print(f"    Appending historical year {hist_year} to coarse Zarr: {coarse_path}")
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
                else:
                    print(f"    Coarse year {hist_year} already exists; skipping coarse append.")

                # -----------------------------------------------------------------
                # SURFACE
                # -----------------------------------------------------------------
                if not surface_done:
                    ds_surface_year = ds_surface.sel(time=ds_surface.time.dt.year == year)

                    if ds_surface_year.sizes["time"] == 0:
                        raise RuntimeError(f"No surface data found for control year {year}")

                    ds_surface_coarse = CM4Xutils.coarsen.horizontally_coarsen(
                        ds_surface_year, grid, dim_coarsen
                    )

                    if "piControl" in exp:
                        ds_surface_coarse = assign_historical_dates(ds_surface_coarse)

                    surface_tmp = os.path.join(
                        tmp_root, f"{model}_{exp}_surface_year{hist_year}.zarr"
                    )
                    safe_rmtree(surface_tmp)

                    surface_chunks = {
                        "time": ds_surface_coarse.time.size,
                        "z_l": 1,
                        "yh": ds_surface_coarse.yh.size,
                        "xh": ds_surface_coarse.xh.size,
                    }
                    ds_surface_write = (
                        ds_surface_coarse
                        .sortby("time")
                        .chunk(surface_chunks)
                    )

                    if not time_values_strictly_increasing(ds_surface_write.time.values):
                        raise RuntimeError(
                            f"Surface time coordinate not strictly increasing for control year {year}"
                        )

                    surface_years_in_tmp = extract_unique_years_from_time_coord(ds_surface_write.time)
                    if len(surface_years_in_tmp) != 1 or int(surface_years_in_tmp[0]) != hist_year:
                        raise RuntimeError(
                            f"Surface year mismatch before append: found {surface_years_in_tmp}, "
                            f"expected [{hist_year}]"
                        )

                    print(f"    Writing surface temp Zarr for historical year {hist_year} -> {surface_tmp}")
                    ds_surface_write.to_zarr(surface_tmp, mode="w", consolidated=False)

                    ds_surface_tmp = open_zarr_cftime(surface_tmp, consolidated=False)

                    if not surface_exists:
                        print(f"    Creating new surface Zarr store: {surface_path}")
                        ds_surface_tmp.to_zarr(
                            surface_path,
                            mode="w",
                            consolidated=False,
                            compute=True,
                        )
                        surface_exists = True
                    else:
                        print(f"    Appending historical year {hist_year} to surface Zarr: {surface_path}")
                        ds_surface_tmp.to_zarr(
                            surface_path,
                            mode="a",
                            append_dim="time",
                            consolidated=False,
                            compute=True,
                        )

                    validate_surface_store_after_append(
                        surface_path,
                        expected_year=hist_year,
                    )

                    safe_rmtree(surface_tmp)
                else:
                    print(f"    Surface year {hist_year} already exists; skipping surface append.")

                coarse_done = coarse_year_present(coarse_path, hist_year)
                surface_done = surface_year_present(surface_path, hist_year)
                if coarse_done and surface_done:
                    done_years.add(hist_year)

                if testing:
                    print("Testing mode: stopping after first year.")
                    break

            if testing:
                break

try:
    shutil.rmtree(tmp_root)
except Exception:
    pass

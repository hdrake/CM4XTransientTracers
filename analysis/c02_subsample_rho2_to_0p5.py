#!/usr/bin/env python
# coding: utf-8

## Example submission from command line:
## conda activate CM4XTransientTracers ; cd /work/hfd/projects/CM4XTransientTracers/analysis/ ; python c02_subsample_rho2_to_0p5.py CM4Xp125 historical

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
    transports_path = f"../data/interim/{model}_{exp}_transports_rho2.zarr"

    existing_transports_years = get_existing_years_from_time(transports_path)
    done_years = existing_transports_years

    print(f"Experiment {exp}:")
    print(f"  transports years:  {sorted(existing_transports_years)}")

    transports_exists = os.path.exists(transports_path)

    for t in np.arange(t_range[0], t_range[1], 5):
        tstr = f"{str(t).zfill(4)}*"
        print(f"\n=== Loading block {model}-{exp} for t pattern {tstr} ===")

        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            grid = load_rho2(odivs[exp], t=tstr, include_transports=True)

            month_counts = grid._ds.time.groupby("time.year").count()
            valid_years = month_counts.where(month_counts == 12, drop=True).year.values

            print(f"Block years in this chunk: {valid_years}")

            for year in valid_years:
                hist_year = control_to_historical_year(year, exp)
            
                transports_done = surface_year_present(transports_path, hist_year)
                
                if transports_done:
                    print(f"  Year {year} (historical {hist_year}) already complete; skipping.")
                    continue
                
                print(
                    f"  Processing year {year} (historical {hist_year})..."
                    f" transports_done={transports_done}"
                )
                
                # -----------------------------------------------------------------
                # TRANSPORTS (depth-resolved)
                # -----------------------------------------------------------------
                if not transports_done:
            
                    ds_transports_year = grid._ds.sel(time=grid._ds.time.dt.year == year)
                    
                    ds_transports_coarse = CM4Xutils.coarsen.horizontally_coarsen(
                        ds_transports_year, grid, dim_coarsen
                    )
                
                    if "piControl" in exp:
                        ds_transports_coarse = assign_historical_dates(ds_transports_coarse)
                
                    transports_tmp = os.path.join(
                        tmp_root, f"{model}_{exp}_transports_year{hist_year}.zarr"
                    )
                    safe_rmtree(transports_tmp)
                
                    transports_chunks = {
                        "time": 1,
                        "rho2_l": ds_transports_coarse.rho2_l.size,
                        "rho2_i": ds_transports_coarse.rho2_i.size,
                        "yh": ds_transports_coarse.yh.size,
                        "yq": ds_transports_coarse.yq.size,
                        "xh": ds_transports_coarse.xh.size,
                        "xq": ds_transports_coarse.xq.size,
                    }
                    ds_transports_write = (
                        ds_transports_coarse
                        .sortby("time")
                        .chunk(transports_chunks)
                    )
                    if "volcello" in ds_transports_write.data_vars:
                        ds_transports_write = ds_transports_write.drop_vars(["volcello"])
                
                    if not time_values_strictly_increasing(ds_transports_write.time.values):
                        raise RuntimeError(
                            f"Transports time coordinate not strictly increasing for control year {year}"
                        )
                
                    transport_years_in_tmp = extract_unique_years_from_time_coord(ds_transports_write.time)
                    if len(transport_years_in_tmp) != 1 or int(transport_years_in_tmp[0]) != hist_year:
                        raise RuntimeError(
                            f"Transport year mismatch before append: found {transport_years_in_tmp}, "
                            f"expected [{hist_year}]"
                        )
                    
                    print(f"    Writing transports temp Zarr for historical year {hist_year} -> {transports_tmp}")
                    ds_transports_write.to_zarr(transports_tmp, mode="w", consolidated=False)
                    ds_transports_tmp = open_zarr_cftime(transports_tmp, consolidated=False)

                    if not transports_exists:
                        print(f"    Creating new transports Zarr store: {transports_path}")
                        ds_transports_tmp.to_zarr(
                            transports_path,
                            mode="w",
                            consolidated=False,
                            compute=True,
                        )
                        transports_exists = True
                    else:
                        print(f"    Appending historical year {hist_year} to transports Zarr: {transports_path}")
                        ds_transports_tmp.to_zarr(
                            transports_path,
                            mode="a",
                            append_dim="time",
                            consolidated=False,
                            compute=True,
                        )

                    safe_rmtree(transports_tmp)
                else:
                    print(f"    Transports year {hist_year} already exists; skipping transports append.")

                transports_done = surface_year_present(transports_path, hist_year)
                if transports_done:
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

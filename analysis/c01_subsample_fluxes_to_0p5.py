#!/usr/bin/env python
# coding: utf-8

## Example submission from command line:
## conda activate CM4XTransientTracers ; cd /work/hfd/projects/CM4XTransientTracers/analysis/ ; python c01_subsample_fluxes_to_0p5.py CM4Xp125 historical
##
## With no arguments, every model and experiment is processed in sequence, as
## before. Passing a model and experiment restricts the run to that one
## combination, so the six of them can be submitted as separate batch jobs.

import warnings
import sys

import doralite
import gfdl_utils.core as gu
import CM4Xutils
import numpy as np
import xarray as xr
import xgcm

import gsw, xwmt
import zarr
import cftime

model_arg = sys.argv[1] if (len(sys.argv) > 1 and sys.argv[1] != "-f") else None
exp_arg = sys.argv[2] if len(sys.argv) > 2 else None

def load_fluxes(exp, tracers):
    model, exp_local = [
        (e, k) for e, d in CM4Xutils.exp_dict.items()
        for k, v in d.items() if exp == v
    ][0]

    # Dora is intermittently unreachable; fall back to the hard-coded paths the
    # same way `CM4Xutils.loading.get_wmt_pathDict` does.
    try:
        pp = doralite.dora_metadata(exp)['pathPP']
    except Exception:
        print("Dora seems to be down. Using hard-coded paths instead.")
        pp = CM4Xutils.pp_dict[model][exp_local]

    ppname = "ocean_inert_month"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    flux_vars = [f"fg{tr}" for tr in tracers]

    # Need to rechunk so that coarsening works correctly
    chunks = {'time':12, 'xh':180, 'yh':140}
    ds = gu.open_frompp(pp, ppname, "ts", local, "*", flux_vars, dmget=True, engine='netcdf4', chunks={})
    ds = ds.chunk(chunks)

    og = gu.open_static(pp, ppname)
    sg = xr.open_dataset(CM4Xutils.exp_dict[model]["hgrid"])
    og = CM4Xutils.fix_geo_coords(og, sg)
    ds = CM4Xutils.add_grid_coords(ds, og)

    return ds

def assign_historical_dates(ds_ctrl):
    # Align dates of a control simulation (with nominal dates starting from year 1)
    # to a historically-referenced simulation (e.g. with dates starting from 1850)
    time_ctrl = ds_ctrl.time.copy()
    ds_ctrl = ds_ctrl.rename({"time": "time_ctrl"})
    historical_equivalent_dates = xr.DataArray(
        np.array([
            cftime.DatetimeNoLeap(
                d.dt.year+1749,
                d.dt.month,
                d.dt.day,
                0,0,0,0,
                has_year_zero=True
            )
            for d in time_ctrl
        ]),
        dims=("time_ctrl",)
    )
    ds_ctrl = (
        ds_ctrl
        .assign_coords({"time": historical_equivalent_dates})
        .swap_dims({"time_ctrl": "time"})
    )
    
    return ds_ctrl

dim_coarsen_dict = {"CM4Xp25": {"X":2, "Y":2}, "CM4Xp125": {"X":4, "Y":4}}
for model, dim_coarsen in dim_coarsen_dict.items():
    if (model_arg is not None) and (model != model_arg): continue
    odivs = CM4Xutils.exp_dict[model]
    datasets = {
        "historical": None,
        "ssp585": None,
        "piControl": None,
        "piControl-continued": None
    }
    with warnings.catch_warnings(action='ignore', category=UserWarning):
        for exp in datasets.keys():
            if (exp_arg is not None) and (exp != exp_arg): continue
            if ("piControl" in exp) & (model=="CM4Xp25"): continue
            print(f"Processing {model} {exp}", flush=True)
            ds = load_fluxes(odivs[exp], ["cfc11", "cfc12", "sf6"])
            grid = CM4Xutils.ds_to_grid(ds)
            datasets[exp] = CM4Xutils.coarsen.horizontally_coarsen(
                ds,
                grid,
                dim_coarsen
            )
            if ("piControl" in exp):
                datasets[exp] = assign_historical_dates(datasets[exp])
                if exp=="piControl":
                    datasets[exp] = datasets[exp].sel(time=slice("1850", "2199"))

            out_path = f"../data/interim/{model}_{exp}_transient_tracer_fluxes.zarr"
            datasets[exp].chunk({"time":12, "xh":-1, "yh":-1}).to_zarr(out_path, mode="w")
            print(f"Wrote {out_path}", flush=True)

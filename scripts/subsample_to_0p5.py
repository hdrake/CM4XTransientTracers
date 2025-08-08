#!/usr/bin/env python
# coding: utf-8

import warnings

import doralite
import gfdl_utils.core as gu
import CM4Xutils
import numpy as np
import xarray as xr
import xgcm

import gsw, xwmt
import zarr
import cftime

import sys
model = sys.argv[1]

print(f"Subsample {model} transient tracer output.")
dim_coarsen_dict = {"CM4Xp25": {"X":2, "Y":2}, "CM4Xp125": {"X":4, "Y":4}}
dim_coarsen = dim_coarsen_dict[model]
odivs = CM4Xutils.exp_dict[model]

def load_tracer(exp, tracers):
    meta = doralite.dora_metadata(exp)
    pp = meta['pathPP']
    print(pp)
    ppname = "ocean_inert_month"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    chunks = {'time':12, 'xh':180, 'yh':140, 'zl':25}
    ds = gu.open_frompp(pp, ppname, out, local, "*", tracers, dmget=True, engine='netcdf4', chunks={})
    ds = ds.chunk(chunks)
    
    og = gu.open_static(pp, ppname)
    model = [e for e,d in CM4Xutils.exp_dict.items() for k,v in d.items() if exp==v][0]
    sg = xr.open_dataset(CM4Xutils.exp_dict[model]["hgrid"])
    og = CM4Xutils.fix_geo_coords(og, sg)
    ds = CM4Xutils.add_grid_coords(ds, og)
    
    return ds

def load_state(exp):
    state_vars = ["thkcello", "thetao", "so"]
    meta = doralite.dora_metadata(exp)
    pp = meta['pathPP']
    ppname = "ocean_annual"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    chunks = {'time':1, 'xh':180, 'yh':140, 'zl':25}
    ds = gu.open_frompp(pp, ppname, out, local, "*", state_vars, dmget=True, engine='netcdf4', chunks={})
    ds = ds.chunk(chunks)

    og = gu.open_static(pp, ppname)
    model = [e for e,d in CM4Xutils.exp_dict.items() for k,v in d.items() if exp==v][0]
    sg = xr.open_dataset(CM4Xutils.exp_dict[model]["hgrid"])
    og = CM4Xutils.fix_geo_coords(og, sg)
    ds = CM4Xutils.add_grid_coords(ds, og)
    ds = add_estimated_layer_interfaces(ds)
    
    return ds

def assign_historical_dates(ds_ctrl):
    # Align dates of a control simulation (with nominal dates starting from year 1)
    # to a historically-referenced simulation (e.g. with dates starting from 1850)
    year_ctrl = ds_ctrl.year.copy()
    ds_ctrl = ds_ctrl.rename({"year": "year_ctrl"})
    historical_equivalent_dates = xr.DataArray(
        np.array([y+1749 for y in year_ctrl]),
        dims=("year_ctrl",)
    )
    ds_ctrl = (
        ds_ctrl
        .assign_coords({"year": historical_equivalent_dates})
        .swap_dims({"year_ctrl": "year"})
    )
    
    return ds_ctrl

def add_estimated_layer_interfaces(ds):
    return ds.assign_coords({"zi": xr.DataArray(
        np.concatenate([[0], 0.5*(ds.zl.values[1:]+ds.zl.values[0:-1]), [6000]]),
        dims=('zi',)
    )})

def load(exp, tracers):
    chunks = {'year':1, 'xh':180, 'yh':140, 'zl':25}
    try:
        ds_tracer = (
            load_tracer(exp, tracers)
            .groupby("time.year").mean()
            .drop_vars("average_DT")
        ).chunk(chunks)
    except:
        ds_tracer = xr.Dataset()
    ds_thickness = (
        load_state(exp)
        .groupby("time.year").mean()
        .drop_vars("average_DT")
    ).chunk(chunks)
    
    ds = xr.merge([ds_tracer, ds_thickness], compat="override")
    ds = ds.chunk(chunks)
    
    grid = CM4Xutils.ds_to_grid(ds)
    
    return grid

datasets = {
    "historical": None,
    "ssp585": None,
    "piControl": None,
    "piControl-continued": None
}
with warnings.catch_warnings(action='ignore', category=UserWarning):
    for exp in datasets.keys():
        grid = load(odivs[exp], ["cfc11", "cfc12", "sf6"])
        datasets[exp] = CM4Xutils.coarsen.horizontally_coarsen(
            grid._ds,
            grid,
            dim_coarsen
        )
        if ("piControl" in exp):
            datasets[exp] = assign_historical_dates(datasets[exp])
            if exp=="piControl":
                datasets[exp] = datasets[exp].sel(year=slice(1850, 2199))

        datasets[exp].chunk({"year":1}).to_zarr(
            f"../data/interim/{model}_{exp}_transient_tracers.zarr", mode="w"
        )

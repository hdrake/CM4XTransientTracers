#!/usr/bin/env python
# coding: utf-8

import warnings

import doralite
import gfdl_utils.core as gu
import CM4Xutils
import numpy as np
import xarray as xr
import xgcm
import gsw

import gsw, xwmt
import zarr
import cftime

def load_tracer(exp, tracers):
    meta = doralite.dora_metadata(exp)
    pp = meta['pathPP']
    ppname = "ocean_inert_z"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    chunks = {'time':12, 'z_l':1}
    ds = gu.open_frompp(pp, ppname, "ts", local, "*", tracers, dmget=True, engine='netcdf4', chunks={})
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
    ppname = "ocean_month_z" if "p125" in meta["expName"] else "ocean_monthly_z"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    chunks = {'time':1, 'z_l':1}
    ds = gu.open_frompp(pp, ppname, "ts", local, "*", state_vars, dmget=True, engine='netcdf4', chunks={})
    ds = ds.chunk(chunks)

    og = gu.open_static(pp, ppname)
    model = [e for e,d in CM4Xutils.exp_dict.items() for k,v in d.items() if exp==v][0]
    sg = xr.open_dataset(CM4Xutils.exp_dict[model]["hgrid"])
    og = CM4Xutils.fix_geo_coords(og, sg)
    ds = CM4Xutils.add_grid_coords(ds, og)   
    return ds

def load(exp, tracers):
    try:
        ds_tracer = load_tracer(exp, tracers).drop_vars("average_DT")
    except:
        ds_tracer = xr.Dataset()
    ds_thickness = load_state(exp)
    ds = xr.merge([ds_tracer, ds_thickness], compat="override")
    
    grid = CM4Xutils.ds_to_grid(ds)
    
    return grid

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
    odivs = CM4Xutils.exp_dict[model]
    datasets = {
        "historical": None,
        "ssp585": None,
        "piControl": None,
        "piControl-continued": None
    }
    with warnings.catch_warnings(action='ignore', category=UserWarning):
        for exp in datasets.keys():
            grid = load(odivs[exp], ["cfc11", "cfc12", "sf6"])
            ds = grid._ds.isel(z_l=0)

            # Compute potential density at surface
            grid._ds["sigma2"] = xr.apply_ufunc(
                gsw.sigma2,
                ds.so,
                ds.thetao,
                dask="parallelized"
            ).rename("sigma2")
            grid._ds["sigma2"].attrs = {
                'long_name': 'Sea Water Potential Density referenced to 2000 dbar',
                'units': 'kg m-3',
                'cell_methods': 'area:mean z_l:mean yh:mean xh:mean time: mean',
                'cell_measures': 'volume: volcello area: areacello',
                'time_avg_info': 'average_T1,average_T2,average_DT',
                'standard_name': 'sea_water_potential_density'
            }
            
            datasets[exp] = CM4Xutils.coarsen.horizontally_coarsen(ds, grid, dim_coarsen)
            if ("piControl" in exp):
                datasets[exp] = assign_historical_dates(datasets[exp])
                if exp=="piControl":
                    datasets[exp] = datasets[exp].sel(time=slice("1850", "2199"))

            datasets[exp].chunk({"time":12, "xh":-1, "yh":-1}).to_zarr(
                f"../data/interim_new/{model}_{exp}_transient_tracer_surface.zarr", mode="w"
            )


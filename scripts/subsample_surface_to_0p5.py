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

chunks = {'time':1, 'xh':360, 'yh':360, 'z_l':-1}
def load_tracer(exp, tracer):
    meta = doralite.dora_metadata(exp)
    pp = meta['pathPP']
    ppname = "ocean_inert_z"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    ds = gu.open_frompp(pp, ppname, "ts", local, "*", tracer, dmget=True, chunks=chunks)
    return ds

def load_state(exp):
    state_vars = ["thkcello", "thetao", "so"]
    meta = doralite.dora_metadata(exp)
    pp = meta['pathPP']
    ppname = "ocean_month_z" if "p125" in meta["expName"] else "ocean_monthly_z"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    ds = gu.open_frompp(pp, ppname, "ts", local, "*", state_vars, dmget=True, chunks=chunks)

    og = gu.open_static(pp, ppname)
    model = [e for e,d in CM4Xutils.exp_dict.items() for k,v in d.items() if exp==v][0]
    sg = xr.open_dataset(CM4Xutils.exp_dict[model]["hgrid"])
    og = CM4Xutils.fix_geo_coords(og, sg)
    ds = CM4Xutils.add_grid_coords(ds, og).drop_dims(["xq", "yq"])    
    return ds

def load(exp, tracers):
    try:
        ds_tracer = load_tracer(exp, tracers).drop_vars("average_DT")
    except:
        ds_tracer = xr.Dataset()
    ds_thickness = load_state(exp)
    ds = xr.merge([ds_tracer, ds_thickness], compat="override")
    
    grid = CM4Xutils.ds_to_grid(ds)

    wm = xwmt.WaterMass(grid)
    wm.get_density("sigma2");
    wm.grid._ds = wm.grid._ds.drop_vars(["thkcello_i", "p", "sa", "ct", "alpha", "beta"])
    
    return wm.grid

def align_dates(ds_ctrl, ds_hist):
    # Align dates of a control simulation (with nominal dates starting from year 0)
    # to a historically-referenced simulation (e.g. with dates starting from 1850)
    time_ctrl = ds_ctrl.time.copy()
    ctrl_years = (time_ctrl.dt.year + (ds_hist.time.dt.year[0] - time_ctrl.time.dt.year[0])).values
    hist_years = ds_hist.time.dt.year.values
    ctrl_years_mask = np.array([y in hist_years for y in ctrl_years])
    hist_years_mask = np.array([y in ctrl_years for y in hist_years])
    ds_ctrl = ds_ctrl.isel(time=ctrl_years_mask)
    ds_hist = ds_hist.isel(time=hist_years_mask)

    ds_ctrl = ds_ctrl.assign_coords({"time": ds_hist.time})
    return ds_ctrl

def area_weighted_coarsen(ds, dim):
    dA = ds.areacello
    ds_center = ds[[v for v in ds.data_vars if np.all([d in ds[v].dims for d in ["yh", "xh"]])]]
    ds_coarse = (ds_center*dA).coarsen(dim=dim).sum() / dA.coarsen(dim=dim).sum()
    dA_coarse = dA.coarsen(dim=dim).sum()
    ds_coarse = ds_coarse.assign_coords({"areacello": dA_coarse})
    return ds_coarse

dim_coarsen_dict = {"CM4Xp25": {"xh":2, "yh":2}, "CM4Xp125": {"xh":4, "yh":4}}
for model, dim_coarsen in dim_coarsen_dict.items():
    if model == "CM4Xp25": continue # Delete later
    odivs = CM4Xutils.exp_dict[model]
    datasets = {"historical": None, "ssp585": None, "piControl": None}
    with warnings.catch_warnings(action='ignore', category=UserWarning):
        for exp in datasets.keys():
            grid = load(odivs[exp], ["cfc11", "cfc12", "sf6"])
            ds = grid._ds.isel(z_l=0)
            datasets[exp] = area_weighted_coarsen(ds, dim=dim_coarsen)
            if exp=="piControl":
                ds_forced = xr.concat([datasets["historical"], datasets["ssp585"]], dim="time")
                datasets["piControl"] = align_dates(datasets["piControl"], ds_forced)
                
            datasets[exp].chunk({"time":12, "xh":-1, "yh":-1}).to_zarr(
                f"/work/hfd/ftp/{model}_{exp}_transient_tracer_surface.zarr", mode="w"
            )


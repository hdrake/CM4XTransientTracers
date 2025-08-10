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

import sys
model = sys.argv[1]

print(f"Subsample {model} transient tracer output.")
dim_coarsen_dict = {"CM4Xp25": {"xh":2, "yh":2}, "CM4Xp125": {"xh":4, "yh":4}}
dim_coarsen = dim_coarsen_dict[model]
odivs = CM4Xutils.exp_dict[model]

def load_tracer(exp, tracer):
    meta = doralite.dora_metadata(exp)
    pp = meta['pathPP']
    ppname = "ocean_inert_month"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    ds = gu.open_frompp(pp, ppname, out, local, "*", tracer, dmget=True)
    return ds

def load_state(exp):
    state_vars = ["thkcello", "thetao", "so"]
    meta = doralite.dora_metadata(exp)
    pp = meta['pathPP']
    ppname = "ocean_annual"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    ds = gu.open_frompp(pp, ppname, out, local, "*", state_vars, dmget=True)

    og = gu.open_static(pp, ppname)
    model = [e for e,d in CM4Xutils.exp_dict.items() for k,v in d.items() if exp==v][0]
    sg = xr.open_dataset(CM4Xutils.exp_dict[model]["hgrid"])
    og = CM4Xutils.fix_geo_coords(og, sg)
    ds = CM4Xutils.add_grid_coords(ds, og).drop_dims(["xq", "yq"])
    ds = add_estimated_layer_interfaces(ds)
    
    return ds

def add_estimated_layer_interfaces(ds):
    return ds.assign_coords({"zi": xr.DataArray(
        np.concatenate([[0], 0.5*(ds.zl.values[1:]+ds.zl.values[0:-1]), [6000]]),
        dims=('zi',)
    )})

def load(exp, tracers):
    try:
        ds_tracer = load_tracer(exp, tracers).groupby("time.year").mean().drop_vars("average_DT")
    except:
        ds_tracer = xr.Dataset()
    ds_thickness = load_state(exp).groupby("time.year").mean().drop_vars("average_DT")
    ds = xr.merge([ds_tracer, ds_thickness], compat="override")
    
    grid = CM4Xutils.ds_to_grid(ds)
    
    return grid

exps = {"historical": None, "ssp585": None, "piControl": None, }
with warnings.catch_warnings(action='ignore', category=UserWarning):
    for exp in exps.keys():
        grid = load(odivs[exp], ["cfc11", "cfc12", "sf6"])
        exps[exp] = weighted_coarsen(grid._ds, dim=dim_coarsen)
        if exp=="piControl":
            exps[exp] = exps[exp].assign_coords(
                {"time": exps[exp].year+(exps["historical"].year[0] - exps[exp].year[0])}
            )
        exps[exp].chunk({"year":1, "xh":-1, "yh":-1, "zl":-1}).to_zarr(
            f"../data/interim_new/{model}_{exp}_transient_tracers.zarr", mode="w"
        )

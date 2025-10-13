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
import gsw
import dask

import os
import sys
model = sys.argv[1]
testing = False

if model=="-f":
    model="CM4Xp25"

print(f"Subsample {model} transient tracer output.")
dim_coarsen_dict = {"CM4Xp25": {"X":2, "Y":2}, "CM4Xp125": {"X":4, "Y":4}}
dim_coarsen = dim_coarsen_dict[model]
odivs = CM4Xutils.exp_dict[model]

def load_tracer(exp, tracers, t="*"):
    meta = doralite.dora_metadata(exp)
    pp = meta['pathPP']
    ppname = "ocean_inert_z"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    ds = gu.open_frompp(pp, ppname, out, local, t, tracers, dmget=True, engine='netcdf4', chunks={})
    chunks = {'z_l':ds.z_l.size}
    ds = ds.chunk(chunks)
    
    og = gu.open_static(pp, ppname)
    model = [e for e,d in CM4Xutils.exp_dict.items() for k,v in d.items() if exp==v][0]
    sg = xr.open_dataset(CM4Xutils.exp_dict[model]["hgrid"])
    og = CM4Xutils.fix_geo_coords(og, sg)
    ds = CM4Xutils.add_grid_coords(ds, og)
    ds = add_estimated_layer_interfaces(ds)
    
    return ds

def load_state(exp, t="*"):
    state_vars = ["volcello", "thkcello", "thetao", "so"]
    meta = doralite.dora_metadata(exp)
    pp = meta['pathPP']
    ppname = "ocean_month_z" if "p125" in meta["expName"] else "ocean_monthly_z"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    ds = gu.open_frompp(pp, ppname, out, local, t, state_vars, dmget=True, engine='netcdf4', chunks={})
    chunks = {'z_l':ds.z_l.size}
    ds = ds.chunk(chunks)

    og = gu.open_static(pp, ppname)
    model = [e for e,d in CM4Xutils.exp_dict.items() for k,v in d.items() if exp==v][0]
    sg = xr.open_dataset(CM4Xutils.exp_dict[model]["hgrid"])
    og = CM4Xutils.fix_geo_coords(og, sg)
    ds = CM4Xutils.add_grid_coords(ds, og)
    ds = add_estimated_layer_interfaces(ds)

    grid = CM4Xutils.ds_to_grid(ds)
    
    # Compute potential density
    diagnose_sigma2_offline(grid)
    
    return grid._ds

def load_rho2(exp, t="*"):
    # Load thickness of density layers
    state_vars = ["thkcello"]
    meta = doralite.dora_metadata(exp)
    pp = meta['pathPP']
    ppname = "ocean_month_rho2"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    ds = gu.open_frompp(pp, ppname, out, local, t, state_vars, dmget=True, engine='netcdf4', chunks={})
    chunks = {'rho2_l':ds.rho2_l.size}
    ds = ds.chunk(chunks)

    og = gu.open_static(pp, ppname)
    model = [e for e,d in CM4Xutils.exp_dict.items() for k,v in d.items() if exp==v][0]
    sg = xr.open_dataset(CM4Xutils.exp_dict[model]["hgrid"])
    og = CM4Xutils.fix_geo_coords(og, sg)
    ds = CM4Xutils.add_grid_coords(ds, og)

    grid_rho2 = CM4Xutils.ds_to_grid(ds)

    # Get depths of density interfaces
    infer_z(grid_rho2)

    return grid_rho2

def assign_historical_dates(ds_ctrl):
    # Align dates of a control simulation (with nominal dates starting from year 1)
    # to a historically-referenced simulation (e.g. with dates starting from 1850)
    if "year" in ds_ctrl.dims:
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
    elif "time" in ds_ctrl.dims:
        time_ctrl = ds_ctrl.time.copy()
        ds_ctrl = ds_ctrl.rename({"time": "time_ctrl"})
        historical_equivalent_dates = xr.DataArray(
            np.array([
                cftime.DatetimeNoLeap(
                    int(d.dt.year) + 1749,
                    int(d.dt.month),
                    int(d.dt.day),
                    0, 0, 0, 0,
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

def add_estimated_layer_interfaces(ds):
    return ds.assign_coords({"z_i": xr.DataArray(
        np.concatenate([[0], 0.5*(ds.z_l.values[1:]+ds.z_l.values[0:-1]), [6750]]),
        dims=('z_i',)
    )})

def diagnose_sigma2_offline(grid):
    grid._ds["sigma2_offline"] = xr.apply_ufunc(
        gsw.sigma2,
        grid._ds.so,
        grid._ds.thetao,
        dask="parallelized"
    ).rename("sigma2_offline")
    grid._ds["sigma2_offline"].attrs = {
        'long_name': 'Sea Water Potential Density referenced to 2000 dbar',
        'units': 'kg m-3',
        'cell_methods': 'area:mean z_l:mean yh:mean xh:mean time: mean',
        'cell_measures': 'volume: volcello area: areacello',
        'time_avg_info': 'average_T1,average_T2,average_DT',
        'standard_name': 'sea_water_potential_density',
        'description': "Diagnosed offline using the `gsw.sigma2` function on `thetao` and `so` on depth levels."
    }

def regrid_sigma2_from_rho2_diags(ds, grid_rho2):
    grid_rho2._ds["sigma2"] = grid_rho2._ds["rho2_l"] - 1000.
    # split up calculation so that coordinates of the
    # numerator and denominator in the volume-weighted
    # mean can be asserted to be the same
    sigma2_numerator = grid_rho2.transform(
        grid_rho2._ds.sigma2*grid_rho2._ds.thkcello.fillna(0.),
        "Z",
        target=ds.z_i,
        target_data=grid_rho2._ds.z_i,
        method="conservative"
    ).fillna(0.).rename({"z_i":"z_l"}).transpose("time", "z_l", "yh", "xh")
    sigma2_numerator = sigma2_numerator.assign_coords({
        k:ds[k] for k in ds.coords if k in sigma2_numerator.coords
    })
    sigma2_denominator = grid_rho2.transform(
        grid_rho2._ds.thkcello.fillna(0.),
        "Z",
        target=ds.z_i,
        target_data=grid_rho2._ds.z_i,
        method="conservative"
    ).rename({"z_i":"z_l"}).transpose("time", "z_l", "yh", "xh")
    sigma2_denominator = sigma2_denominator.assign_coords({
        k:ds[k] for k in ds.coords if k in sigma2_denominator.coords
    })
    sigma2 = sigma2_numerator / sigma2_denominator
    ds["sigma2"] = sigma2
    ds["sigma2"].attrs = {
        'long_name': 'Sea Water Potential Density referenced to 2000 dbar',
        'units': 'kg m-3',
        'cell_methods': 'area:mean z_l:mean yh:mean xh:mean time: mean',
        'cell_measures': 'volume: volcello area: areacello',
        'time_avg_info': 'average_T1,average_T2,average_DT',
        'standard_name': 'sea_water_potential_density',
        'description': "Regridded offline based on online thicknesses between fixed rho2 levels."
    }

def infer_z(grid, zero_ssh=True):
    zl = grid.axes['Z'].coords['center']
    zi = grid.axes['Z'].coords['outer']

    if zero_ssh:
        ssh_ref = 0 
    else:
        # CM4Xp25 does not have col_height variable available, so need to compute from thkcello
        if "col_height" not in grid._ds.data_vars:
            grid._ds["col_height"] = grid._ds["thkcello"].sum(zl)
            grid._ds["col_height"].attrs = {
                'long_name': 'The height of the water column',
                 'units': 'm',
                 'cell_methods': 'area:mean yh:mean xh:mean time: mean',
                 'cell_measures': 'area: areacello',
                 'time_avg_info': 'average_T1,average_T2,average_DT'
            }
        ssh_ref = grid._ds.col_height - grid._ds.deptho

    # Interface depths
    thkcello_cumsum = (
        xr.concat([
            xr.zeros_like(grid._ds.thkcello.isel({zl:0})).expand_dims({zl:[0]}),
            grid._ds.thkcello.cumsum(zl),
        ],dim=zl)
        .rename({zl:zi})
        .assign_coords({zi:grid._ds[zi].values})
    )
    grid._ds["z_i"] = (
        (thkcello_cumsum - ssh_ref)
        .transpose("time", zi, "yh", "xh")
    ).chunk({zi: grid._ds[zi].size})
    grid._ds["z_i"].attrs = {
        'long_name': 'depth of layer interfaces',
        'units': 'm',
        'cell_methods': f'area:mean {zi}:point yh:mean xh:mean time: mean',
        'cell_measures': 'area: areacello',
        'time_avg_info': 'average_T1,average_T2,average_DT',
    }

    # Center depths
    zl_extended = np.concatenate((
        grid._ds[zi][np.array([0])].values,
        grid._ds[zl].values,
        grid._ds[zi][np.array([-1])].values
    ))

    grid._ds["thkcello_i"] = grid.transform(
        grid._ds["thkcello"].fillna(0.),
        "Z",
        zl_extended,
        method="conservative",
    ).fillna(0.).assign_coords({zi:grid._ds[zi].values}).transpose("time", zi, "yh", "xh")

    grid._ds["z_l"] = (
        (grid.cumsum(grid._ds.thkcello_i, "Z", to="center") - ssh_ref)
        .transpose("time", zl, "yh", "xh")
    ).chunk({zl: grid._ds[zl].size})
    grid._ds["z_l"].attrs = {
        'long_name': 'depth of layer centers',
        'units': 'm',
        'cell_methods': f'area:mean {zl}:point yh:mean xh:mean time: mean',
        'cell_measures': 'volume: volcello area: areacello',
        'time_avg_info': 'average_T1,average_T2,average_DT',
    }

def load(exp, tracers, t="*"):
    try:
        ds_tracer = load_tracer(exp, tracers, t=t)
    except:
        ds_tracer = xr.Dataset()
    ds_state = load_state(exp, t=t)
    ds = xr.merge([ds_tracer, ds_state], compat="override")

    # Get accurate density field on depth levels from thkcello(rho2_l)
    grid_rho2 = load_rho2(exp, t=t)
    regrid_sigma2_from_rho2_diags(ds, grid_rho2)
    
    grid = CM4Xutils.ds_to_grid(ds)
    return grid

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
        #"piControl-continued": [451, 651] was having difficult with this!
    }
}

print(time_ranges[model])
with warnings.catch_warnings(action='ignore', category=UserWarning):
    for exp, t_range in time_ranges[model].items():
        coarse_path = f"../data/interim/{model}_{exp}_transient_tracers_z.zarr"  # one per exp
        surface_path = f"../data/interim/{model}_{exp}_transient_tracers_surface.zarr"  # one per exp

        mode = "w" # overwrite if .zarr already file exists
        for t in np.arange(t_range[0], t_range[1], 5):
            tstr = f"{str(t).zfill(4)}*"
            print(f"Coarsening tracers for {model}-{exp} for {tstr}")

            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                grid = load(odivs[exp], ["cfc11", "cfc12", "sf6"], t=tstr)
                
                ds_mean = grid._ds.groupby("time.year").mean("time")
                ds_coarse = CM4Xutils.coarsen.horizontally_coarsen(ds_mean, grid, dim_coarsen)
                
                ds_surface = grid._ds.isel(z_l=[0])
                ds_surface_coarse = CM4Xutils.coarsen.horizontally_coarsen(ds_surface, grid, dim_coarsen)
                
                if "piControl" in exp:
                    ds_coarse = assign_historical_dates(ds_coarse)
                    ds_surface_coarse = assign_historical_dates(ds_surface_coarse)
    
                if not testing:
                    # Coarse
                    chunks = {
                        "year": 1,
                        "z_l": 75,
                        "yh": ds_coarse.yh.size,
                        "xh": ds_coarse.xh.size
                    }
                    ds_write = ds_coarse.chunk(chunks)
                    
                    append_dim = "year" if mode=="a" else None
                    ds_write.to_zarr(
                        coarse_path,
                        mode=mode,
                        append_dim=append_dim     # appending along year dimension
                    )

                    # Coarse surface variables
                    chunks = {
                        "time": 12,
                        "z_l": 1,
                        "yh": ds_surface_coarse.yh.size,
                        "xh": ds_surface_coarse.xh.size
                    }
                    ds_write = ds_surface_coarse.chunk(chunks)
                    
                    append_dim = "time" if mode=="a" else None
                    ds_write.to_zarr(
                        surface_path,
                        mode=mode,
                        append_dim=append_dim     # appending along year dimension
                    )
                    mode = "a" # append subsequent time blocks
                else:
                    break
                    

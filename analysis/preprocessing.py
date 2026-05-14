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

# -----------------------------------------------------------------------------#
# Helper functions for restart-safe Zarr writing
# -----------------------------------------------------------------------------#

def control_to_historical_year(year, exp):
    return int(year + 1749) if "piControl" in exp else int(year)


def open_zarr_cftime(path, consolidated=False):
    """
    Open a Zarr store with explicit cftime decoding so far-future dates
    remain robustly decodable.
    """
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    return xr.open_zarr(path, consolidated=consolidated, decode_times=time_coder)


def get_existing_years_from_year_dim(path, year_dim="year"):
    """
    Return a set of integer years already present in a Zarr store
    that has an explicit year coordinate/dimension.
    """
    if not os.path.exists(path):
        return set()
    try:
        ds = open_zarr_cftime(path, consolidated=False)
    except Exception:
        return set()
    if year_dim not in ds.dims and year_dim not in ds.coords:
        return set()
    return {int(y) for y in ds[year_dim].values}


def get_existing_years_from_time(path):
    """
    Return a set of integer years already present in a Zarr store
    where time is a coordinate or dimension.

    Robust to cftime-backed coordinates and far-future dates.
    """
    if not os.path.exists(path):
        return set()
    try:
        ds = open_zarr_cftime(path, consolidated=False)
    except Exception:
        return set()
    if "time" not in ds.coords and "time" not in ds.dims:
        return set()

    years = []
    for t in ds["time"].values:
        if hasattr(t, "year"):
            years.append(int(t.year))
        else:
            years.append(int(str(np.datetime_as_string(t, unit="D"))[:4]))
    return set(years)


def get_year_occurrences_from_year_dim(path, year_dim="year"):
    """
    Return a Counter of year values stored in a raw Zarr coordinate.
    Uses raw zarr access so it can still diagnose a partially corrupted store.
    """
    if not os.path.exists(path):
        return Counter()
    try:
        root = zarr.open_group(path, mode="r")
    except Exception:
        return Counter()
    if year_dim not in root.array_keys():
        return Counter()
    years = np.array(root[year_dim][:]).astype(int).tolist()
    return Counter(years)


def safe_rmtree(path):
    """Remove a directory tree if it exists."""
    if os.path.exists(path):
        shutil.rmtree(path)


def coarse_year_present(path, hist_year):
    return int(hist_year) in get_existing_years_from_year_dim(path, year_dim="year")


def surface_year_present(path, hist_year):
    return int(hist_year) in get_existing_years_from_time(path)

def get_expected_year_arrays(ds, append_dim="year"):
    """
    Return all arrays in ds whose leading append dimension should grow together.
    Includes data vars and coordinates such as year/year_ctrl.
    """
    names = []

    for name, da in ds.data_vars.items():
        if append_dim in da.dims and da.dims[0] == append_dim:
            names.append(name)

    for name, da in ds.coords.items():
        if append_dim in da.dims and da.dims[0] == append_dim:
            names.append(name)

    return sorted(set(names))


def validate_coarse_store_after_append(path, expected_year, expected_names):
    """
    Validate that:
      - all year-dependent arrays have the same leading length
      - the 'year' coordinate exists
      - expected_year appears exactly once
      - the last stored year equals expected_year

    Uses raw zarr so it can still diagnose a partially corrupted store.
    """
    root = zarr.open_group(path, mode="r")

    if "year" not in root.array_keys():
        raise RuntimeError(f"{path} is missing raw 'year' array")

    year_vals = np.array(root["year"][:]).astype(int)
    year_len = len(year_vals)

    if year_len == 0:
        raise RuntimeError(f"{path} has empty 'year' array")

    counts = Counter(year_vals.tolist())
    if counts[int(expected_year)] != 1:
        raise RuntimeError(
            f"{path}: expected historical year {expected_year} exactly once, "
            f"found {counts[int(expected_year)]} times"
        )

    if int(year_vals[-1]) != int(expected_year):
        raise RuntimeError(
            f"{path}: last stored year is {int(year_vals[-1])}, "
            f"expected {expected_year}"
        )

    bad = {}
    for name in expected_names:
        if name not in root.array_keys():
            raise RuntimeError(f"{path}: expected array '{name}' is missing")
        shape = root[name].shape
        if len(shape) == 0:
            raise RuntimeError(f"{path}: expected array '{name}' is scalar")
        if shape[0] != year_len:
            bad[name] = shape[0]

    if bad:
        raise RuntimeError(
            f"{path}: inconsistent leading year dimension after append. "
            f"'year' has length {year_len}, mismatches: {bad}"
        )


def validate_surface_store_after_append(path, expected_year):
    """
    Validate that the surface store contains the expected historical year.
    Uses cftime-aware reopening because years can exceed datetime64 bounds.
    """
    years = get_existing_years_from_time(path)
    if int(expected_year) not in years:
        tail = sorted(years)[-10:] if years else []
        raise RuntimeError(
            f"Historical year {expected_year} missing after surface append! "
            f"Found years near end: {tail}"
        )


def time_values_strictly_increasing(vals):
    """
    Robust monotonicity check for cftime, datetime.datetime, and numpy datetime64.
    """
    vals = list(vals)
    return all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def extract_unique_years_from_time_coord(time_coord):
    """
    Robust year extraction from a time coordinate for both cftime and numpy-backed
    arrays, without relying on .dt.year.
    """
    years = []
    for t in time_coord.values:
        if hasattr(t, "year"):
            years.append(int(t.year))
        else:
            years.append(int(str(np.datetime_as_string(t, unit="D"))[:4]))
    return np.unique(np.asarray(years, dtype=int))


# -----------------------------------------------------------------------------#
# Existing helper functions
# -----------------------------------------------------------------------------#

def load_tracer(exp, tracers, t="*"):
    meta = doralite.dora_metadata(exp)
    pp = meta["pathPP"]
    ppname = "ocean_inert_z"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    ds = gu.open_frompp(
        pp, ppname, out, local, t, tracers,
        dmget=True, engine="netcdf4", chunks={}
    )
    chunks = {"z_l": ds.z_l.size}
    ds = ds.chunk(chunks)

    og = gu.open_static(pp, ppname)
    model_local = [
        e for e, d in CM4Xutils.exp_dict.items()
        for k, v in d.items() if exp == v
    ][0]
    sg = xr.open_dataset(CM4Xutils.exp_dict[model_local]["hgrid"])
    og = CM4Xutils.fix_geo_coords(og, sg)
    ds = CM4Xutils.add_grid_coords(ds, og)
    ds = add_estimated_layer_interfaces(ds)

    return ds


def load_state(exp, t="*"):
    state_vars = ["volcello", "thkcello", "thetao", "so"]
    meta = doralite.dora_metadata(exp)
    pp = meta["pathPP"]
    ppname = "ocean_month_z" if "p125" in meta["expName"] else "ocean_monthly_z"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    ds = gu.open_frompp(
        pp, ppname, out, local, t, state_vars,
        dmget=True, engine="netcdf4", chunks={}
    )
    chunks = {"z_l": ds.z_l.size}
    ds = ds.chunk(chunks)

    og = gu.open_static(pp, ppname)
    model_local = [
        e for e, d in CM4Xutils.exp_dict.items()
        for k, v in d.items() if exp == v
    ][0]
    sg = xr.open_dataset(CM4Xutils.exp_dict[model_local]["hgrid"])
    og = CM4Xutils.fix_geo_coords(og, sg)
    ds = CM4Xutils.add_grid_coords(ds, og)
    ds = add_estimated_layer_interfaces(ds)

    grid = CM4Xutils.ds_to_grid(ds)

    diagnose_sigma2_offline(grid)

    return grid._ds

def load_rho2(exp, t="*", include_transports=False):
    state_vars = ["thkcello"]
    if include_transports:
        state_vars = state_vars + ["umo", "vmo"]
    meta = doralite.dora_metadata(exp)
    pp = meta["pathPP"]
    ppname = "ocean_month_rho2"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    ds = gu.open_frompp(
        pp, ppname, out, local, t, state_vars,
        dmget=True, engine="netcdf4", chunks={}
    )
    chunks = {"rho2_l": ds.rho2_l.size}
    ds = ds.chunk(chunks)

    og = gu.open_static(pp, ppname)
    model_local = [
        e for e, d in CM4Xutils.exp_dict.items()
        for k, v in d.items() if exp == v
    ][0]
    sg = xr.open_dataset(CM4Xutils.exp_dict[model_local]["hgrid"])
    og = CM4Xutils.fix_geo_coords(og, sg)
    ds = CM4Xutils.add_grid_coords(ds, og)

    grid_rho2 = CM4Xutils.ds_to_grid(ds)
    infer_z(grid_rho2)

    return grid_rho2


def assign_historical_dates(ds_ctrl):
    if "year" in ds_ctrl.dims:
        year_ctrl = ds_ctrl.year.copy()
        ds_ctrl = ds_ctrl.rename({"year": "year_ctrl"})
        historical_equivalent_dates = xr.DataArray(
            np.array([int(y) + 1749 for y in year_ctrl]),
            dims=("year_ctrl",)
        )
        ds_ctrl = (
            ds_ctrl
            .assign_coords({"year": historical_equivalent_dates})
            .swap_dims({"year_ctrl": "year"})
            .sortby("year")
        )
    elif "time" in ds_ctrl.dims:
        time_ctrl = ds_ctrl.time.copy()
        ds_ctrl = ds_ctrl.rename({"time": "time_ctrl"})
        historical_equivalent_dates = xr.DataArray(
            np.array([
                cftime.DatetimeNoLeap(
                    int(d.year) + 1749,
                    int(d.month),
                    int(d.day),
                    0, 0, 0, 0,
                    has_year_zero=True
                )
                for d in time_ctrl.values
            ], dtype=object),
            dims=("time_ctrl",)
        )
        ds_ctrl = (
            ds_ctrl
            .assign_coords({"time": historical_equivalent_dates})
            .swap_dims({"time_ctrl": "time"})
            .sortby("time")
        )

    return ds_ctrl


def add_estimated_layer_interfaces(ds):
    return ds.assign_coords({
        "z_i": xr.DataArray(
            np.concatenate([[0], 0.5 * (ds.z_l.values[1:] + ds.z_l.values[:-1]), [6750]]),
            dims=("z_i",)
        )
    })


def diagnose_sigma2_offline(grid):
    grid._ds["sigma2_offline"] = xr.apply_ufunc(
        gsw.sigma2,
        grid._ds.so,
        grid._ds.thetao,
        dask="parallelized"
    ).rename("sigma2_offline")
    grid._ds["sigma2_offline"].attrs = {
        "long_name": "Sea Water Potential Density referenced to 2000 dbar",
        "units": "kg m-3",
        "cell_methods": "area:mean z_l:mean yh:mean xh:mean time: mean",
        "cell_measures": "volume: volcello area: areacello",
        "time_avg_info": "average_T1,average_T2,average_DT",
        "standard_name": "sea_water_potential_density",
        "description": "Diagnosed offline using the `gsw.sigma2` function on `thetao` and `so` on depth levels.",
    }


def regrid_sigma2_from_rho2_diags(ds, grid_rho2):
    grid_rho2._ds["sigma2"] = grid_rho2._ds["rho2_l"] - 1000.0

    sigma2_numerator = grid_rho2.transform(
        grid_rho2._ds.sigma2 * grid_rho2._ds.thkcello.fillna(0.0),
        "Z",
        target=ds.z_i,
        target_data=grid_rho2._ds.z_i,
        method="conservative",
    ).fillna(0.0).rename({"z_i": "z_l"}).transpose("time", "z_l", "yh", "xh")
    sigma2_numerator = sigma2_numerator.assign_coords({
        k: ds[k] for k in ds.coords if k in sigma2_numerator.coords
    })

    sigma2_denominator = grid_rho2.transform(
        grid_rho2._ds.thkcello.fillna(0.0),
        "Z",
        target=ds.z_i,
        target_data=grid_rho2._ds.z_i,
        method="conservative",
    ).rename({"z_i": "z_l"}).transpose("time", "z_l", "yh", "xh")
    sigma2_denominator = sigma2_denominator.assign_coords({
        k: ds[k] for k in ds.coords if k in sigma2_denominator.coords
    })

    sigma2 = xr.where(
        sigma2_denominator > 0,
        sigma2_numerator / sigma2_denominator,
        np.nan
    )
    ds["sigma2"] = sigma2
    ds["sigma2"].attrs = {
        "long_name": "Sea Water Potential Density referenced to 2000 dbar",
        "units": "kg m-3",
        "cell_methods": "area:mean z_l:mean yh:mean xh:mean time: mean",
        "cell_measures": "volume: volcello area: areacello",
        "time_avg_info": "average_T1,average_T2,average_DT",
        "standard_name": "sea_water_potential_density",
        "description": "Regridded offline based on online thicknesses between fixed rho2 levels.",
    }


def infer_z(grid, zero_ssh=True):
    zl = grid.axes["Z"].coords["center"]
    zi = grid.axes["Z"].coords["outer"]

    if zero_ssh:
        ssh_ref = 0
    else:
        if "col_height" not in grid._ds.data_vars:
            grid._ds["col_height"] = grid._ds["thkcello"].sum(zl)
            grid._ds["col_height"].attrs = {
                "long_name": "The height of the water column",
                "units": "m",
                "cell_methods": "area:mean yh:mean xh:mean time: mean",
                "cell_measures": "area: areacello",
                "time_avg_info": "average_T1,average_T2,average_DT",
            }
        ssh_ref = grid._ds.col_height - grid._ds.deptho

    thkcello_cumsum = (
        xr.concat([
            xr.zeros_like(grid._ds.thkcello.isel({zl: 0})).expand_dims({zl: [0]}),
            grid._ds.thkcello.cumsum(zl),
        ], dim=zl)
        .rename({zl: zi})
        .assign_coords({zi: grid._ds[zi].values})
    )
    grid._ds["z_i"] = (
        (thkcello_cumsum - ssh_ref)
        .transpose("time", zi, "yh", "xh")
    ).chunk({zi: grid._ds[zi].size})
    grid._ds["z_i"].attrs = {
        "long_name": "depth of layer interfaces",
        "units": "m",
        "cell_methods": f"area:mean {zi}:point yh:mean xh:mean time: mean",
        "cell_measures": "area: areacello",
        "time_avg_info": "average_T1,average_T2,average_DT",
    }

    zl_extended = np.concatenate((
        grid._ds[zi][np.array([0])].values,
        grid._ds[zl].values,
        grid._ds[zi][np.array([-1])].values,
    ))

    grid._ds["thkcello_i"] = grid.transform(
        grid._ds["thkcello"].fillna(0.0),
        "Z",
        zl_extended,
        method="conservative",
    ).fillna(0.0).assign_coords({zi: grid._ds[zi].values}).transpose("time", zi, "yh", "xh")

    grid._ds["z_l"] = (
        (grid.cumsum(grid._ds.thkcello_i, "Z", to="center") - ssh_ref)
        .transpose("time", zl, "yh", "xh")
    ).chunk({zl: grid._ds[zl].size})
    grid._ds["z_l"].attrs = {
        "long_name": "depth of layer centers",
        "units": "m",
        "cell_methods": f"area:mean {zl}:point yh:mean xh:mean time: mean",
        "cell_measures": "volume: volcello area: areacello",
        "time_avg_info": "average_T1,average_T2,average_DT",
    }


def load(exp, tracers, t="*"):
    tracer_datasets = []

    for tracer in tracers:
        try:
            ds_t = load_tracer(exp, [tracer], t=t)
            tracer_datasets.append(ds_t)
        except Exception as e:
            print(f"WARNING: tracer {tracer} missing for pattern {t}: {e}")

    if tracer_datasets:
        ds_tracer = xr.merge(tracer_datasets, compat="override")
    else:
        ds_tracer = xr.Dataset()

    ds_state = load_state(exp, t=t)
    ds = xr.merge([ds_tracer, ds_state], compat="override")

    grid_rho2 = load_rho2(exp, t=t)
    regrid_sigma2_from_rho2_diags(ds, grid_rho2)

    grid = CM4Xutils.ds_to_grid(ds)
    return grid

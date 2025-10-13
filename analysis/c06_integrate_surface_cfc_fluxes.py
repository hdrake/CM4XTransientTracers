#!/usr/bin/env python
# coding: utf-8

import warnings
import numpy as np
import xarray as xr

import doralite
import gfdl_utils.core as gu
import CM4Xutils
import xgcm
import zarr
import cftime

SEC_PER_NSEC = 1e-9  # average_DT is in nanoseconds

# ---------- IO helpers ----------

def load_fluxes_minimal(exp, tracers):
    """
    Load just the air–sea tracer fluxes fg{tr} and average_DT (for Δt),
    plus attach fixed grid info (including key variable areacello).
    """
    meta = doralite.dora_metadata(exp)
    pp = meta["pathPP"]
    ppname = "ocean_inert_month"
    out = "ts"
    local = gu.get_local(pp, ppname, out)

    need_vars = [f"fg{tr}" for tr in tracers] + ["average_DT"]

    ds = gu.open_frompp(
        pp, ppname, out, local, "*", need_vars,
        dmget=True, engine="netcdf4", chunks={}
    )
    # Chunks: small over time for reductions
    ds = ds.chunk({"time": 60, "xh": 180, "yh": 140})

    # Static/grid info + consistent coords
    og = gu.open_static(pp, ppname)
    model = [e for e, d in CM4Xutils.exp_dict.items() for k, v in d.items() if exp == v][0]
    sg = xr.open_dataset(CM4Xutils.exp_dict[model]["hgrid"])
    og = CM4Xutils.fix_geo_coords(og, sg)
    ds = CM4Xutils.add_grid_coords(ds, og)

    return ds

def assign_historical_dates(ds_ctrl):
    """Map control years (1,2,...) to historical-like years (1850,1851,...)"""
    time_ctrl = ds_ctrl.time.copy()
    ds_ctrl = ds_ctrl.rename({"time": "time_ctrl"})
    hist_like = xr.DataArray(
        np.array([
            cftime.DatetimeNoLeap(
                d.dt.year + 1749, d.dt.month, d.dt.day, 0, 0, 0, 0,
                has_year_zero=True
            )
            for d in time_ctrl
        ]),
        dims=("time_ctrl",),
    )
    ds_ctrl = ds_ctrl.assign_coords({"time": hist_like}).swap_dims({"time_ctrl": "time"})
    return ds_ctrl

# ---------- Core calculation ----------

def time_integrals_split_by_year(ds_fluxes, ds_inventory, tracers):
    """
    For each tracer `tr`, compute Δt-weighted time-integrals of fg{tr}
    before and after the specified peak calendar year.

        before: time.dt.year < peak_year
        after:  time.dt.year >= peak_year

    fg{tr} is per-area, so multiply by areacello, sum over xh,yh,
    then weight by Δt (seconds) and sum over time.
    """

    ds_fluxes_integrated = xr.Dataset()
    dt_s = ds_fluxes["average_DT"].astype("float64") * SEC_PER_NSEC  # (time,)

    for tr in tracers:
        peak_year = ds_inventory[f"{tr}_peak_year"].values
        
        # Boolean masks on time
        t_before = ds_fluxes.time.dt.year < peak_year
        t_after  = ds_fluxes.time.dt.year >= peak_year

        # areacello-weighted sums as a function of time
        flux = ds_fluxes[f"fg{tr}"]
        flux_global = (ds_fluxes[f"fg{tr}"] * ds_fluxes.areacello).sum(["xh", "yh"])

        # Δt-weighted sums over the two periods as a function of space
        flux_before = (flux.where(t_before) * dt_s.where(t_before)).sum("time")
        flux_after  = (flux.where(t_after ) * dt_s.where(t_after )).sum("time")

        ds_fluxes_integrated[f"fg{tr}_global"] = flux_global
        ds_fluxes_integrated[f"fg{tr}_timeint_before"] = flux_before
        ds_fluxes_integrated[f"fg{tr}_timeint_after"]  = flux_after
        ds_fluxes_integrated[f"{tr}_peak_year"] = ds_inventory[f"{tr}_peak_year"]

    ds_fluxes_integrated = ds_fluxes_integrated.assign_attrs({
        "note": "Flux time-integrals split at provided {tr}_peak_year. "
                "Integrals are Δt-weighted using average_DT (seconds). "
                "fg{tr} are per-area, so global integrals are weighted by areacello."
    })
    ds_fluxes_integrated = ds_fluxes_integrated.assign_coords({
        "geolon_c":ds_fluxes.geolon_c, "geolat_c":ds_fluxes.geolat_c
    })
    return ds_fluxes_integrated

# ---------- Driver ----------
models = ["CM4Xp25", "CM4Xp125"]
tracers = ["cfc11", "cfc12", "sf6"]

for model in models:
    odivs = CM4Xutils.exp_dict[model]
    experiments = ["historical", "ssp585", "piControl", "piControl-continued"]

    # --- Load previous computed peak years ---
    ds_inventory = xr.open_dataset(
        f"../data/interim/transient_tracer_inventory_{model}_forced.nc"
    )
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        for exp in experiments:
            if ("piControl" in exp) and (model == "CM4Xp25"):
                continue

            print(f"Processing {model}-{exp}")
            
            ds_fluxes = load_fluxes_minimal(odivs[exp], tracers)

            # Align control dates and optionally clip, to keep years comparable
            if "piControl" in exp:
                ds_fluxes = assign_historical_dates(ds_fluxes)
                if exp == "piControl":
                    ds_fluxes = ds_fluxes.sel(time=slice("1850", "2199"))
                    
            ds_fluxes_integrated = time_integrals_split_by_year(
                ds_fluxes,
                ds_inventory,
                tracers
            )
            ds_fluxes_integrated = ds_fluxes_integrated.assign_attrs({
                **ds_fluxes_integrated.attrs, "model": model, "experiment": exp
            })

            # Small dataset -> write unchunked
            chunks = dict(ds_fluxes_integrated.dims)
            ds_fluxes_integrated = ds_fluxes_integrated.chunk(chunks)
            ds_fluxes_integrated.to_netcdf(
                f"../data/interim/{model}_{exp}_tracer_flux_integrals.nc",
                mode="w"
            )
            
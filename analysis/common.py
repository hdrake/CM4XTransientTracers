import os

import xarray as xr
import numpy as np
import CM4Xutils

import cmocean
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

moc_metrics = xr.open_dataset("../data/processed/moc_metrics_piControl.nc").drop_vars("region")
moc_metrics = moc_metrics.assign_coords({
    "rho2_moc_l": xr.DataArray((moc_metrics.rho2_moc_i.values[1:] + moc_metrics.rho2_moc_i.values[:-1])/2., dims=("rho2_moc_l",))
})
moc_metrics = moc_metrics.assign_coords({
    "sigma2_moc_l": moc_metrics.rho2_moc_l - 1000.,
    "sigma2_moc_i": moc_metrics.rho2_moc_i - 1000.,
})

layer_labels = ["Surface", "Upper", "Lower", "Bottom"]
layer_labels_short = ["Surface", "Upper", "Lower", "Bottom"]
layer_colors = ["crimson", "seagreen", "darkgoldenrod", "darkslateblue"]
flux_colors = {"upper-to-deep":"olive", "deep-to-bottom":"steelblue"}
facecolor=cmocean.cm.gray(1/1.3)

models = {
    "CM4Xp25"  : {"historical":"odiv-231", "ssp5":"odiv-232"},
    "CM4Xp125" : {"historical":"odiv-255", "ssp5":"odiv-293"}
}

tracers = ["cfc11", "cfc12", "sf6"]
g_per_mol = {
    "cfc11": 137.37,
    "cfc12": 120.91,
    "sf6": 146.06
}
Gg_per_g = 1.e-9
sec_per_day = 24 * 60 * 60
sec_per_year = 365.25 * sec_per_day
sec_per_year_wang21 = 360 * sec_per_day
sec_per_nsec = 1.e-9
m2_per_km2 = 1.e6

def load_ideal_age(model, exps=("historical", "ssp585")):
    """Open and concatenate the per-experiment ideal age companion stores.

    `agessc` is written to its own store by `c02_subsample_agessc_to_0p5.py`, since
    it is diagnosed on a different stream (annual mean, half-resolution grid) than
    the transient tracers and the tracer stores are too expensive to rebuild.

    Returns None, with a message, if any piece is missing: CM4Xp25 has no companion
    store, and a CM4Xp125 run may still be in progress, so absence has to degrade
    gracefully rather than raise.
    """
    paths = [f"../data/interim/{model}_{exp}_ideal_age_z.zarr" for exp in exps]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(
            f"No ideal age companion store for {model} "
            f"(missing {[os.path.basename(p) for p in missing]}); "
            f"`agessc` will be absent from the {model} dataset."
        )
        return None
    return xr.concat([xr.open_zarr(p) for p in paths], dim="year")


def load_datasets():
    grids = {}
    for model in models.keys():
        surface_fluxes = xr.concat([
            xr.open_zarr(f"../data/interim/{model}_historical_transient_tracer_fluxes.zarr"),
            xr.open_zarr(f"../data/interim/{model}_ssp585_transient_tracer_fluxes.zarr")
        ], dim="time")
        surface_tracers = xr.concat([
            xr.open_zarr(f"../data/interim/{model}_historical_transient_tracers_surface.zarr"),
            xr.open_zarr(f"../data/interim/{model}_ssp585_transient_tracers_surface.zarr")
        ], dim="time").squeeze("z_l")
        surface_tracers = surface_tracers.rename({v:f"{v}_surface" for v in surface_tracers.data_vars})
        tracers = xr.concat([
            xr.open_zarr(f"../data/interim/{model}_historical_transient_tracers_z.zarr"),
            xr.open_zarr(f"../data/interim/{model}_ssp585_transient_tracers_z.zarr")
        ], dim="year")

        merge_list = [surface_fluxes, surface_tracers, tracers]

        ideal_age = load_ideal_age(model)
        if ideal_age is not None:
            missing_years = (
                set(int(y) for y in tracers.year.values)
                - set(int(y) for y in ideal_age.year.values)
            )
            if missing_years:
                print(
                    f"WARNING: {model} ideal age store is missing {len(missing_years)} "
                    f"year(s), e.g. {sorted(missing_years)[:5]}; `agessc` will be NaN there."
                )
            # Reindex rather than outer-join, so the merged `year` dimension is set
            # by the tracer stores and cannot be silently extended by stray years.
            merge_list.append(ideal_age.reindex(year=tracers.year))

        ds = xr.merge(merge_list)
        ds = add_estimated_layer_interfaces(ds)
        grids[f"{model}_forced"] = CM4Xutils.ds_to_grid(ds)
    
    return grids

def pad_array(y):
    return np.concatenate(([y[0]], y, [y[-1]]))

def pad_years(x):
    return np.concatenate(([x[0] - 0.5], x+0.5, [x[-1] + 0.5]))

def add_estimated_layer_interfaces(ds):
    if "zl" in ds.coords:
        return ds.assign_coords({"zi": xr.DataArray(
            np.concatenate([[0], 0.5*(ds.zl.values[1:]+ds.zl.values[0:-1]), [6500]]),
            dims=('zi',)
        )})
    elif "z_l" in ds.coords:
        return ds.assign_coords({"z_i": xr.DataArray(
            np.concatenate([[0], 0.5*(ds.z_l.values[1:]+ds.z_l.values[0:-1]), [6500]]),
            dims=('z_i',)
        )})

def cmap_offwhite_fade(layer_color):
    colors = [(1, 1, 1), layer_color] # first color is black, last is red
    cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=200)
    cm = LinearSegmentedColormap.from_list(
            "Custom", [cm(0.05), layer_color], N=200)
    cm.set_bad((0.9, 0.9, 0.9))
    return cm
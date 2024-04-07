import xarray as xr
import numpy as np
import CM4Xutils

import cmocean
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sigma2_i = np.array([0, 36.7, 36.96, 60])
layer_labels = ["Upper layer", "Deep layer", "Bottom layer"]
layer_labels_short = ["Upper", "Deep", "Bottom"]
layer_colors = ["darkgoldenrod", "seagreen", "darkslateblue"]
flux_colors = {"upper-to-deep":"olive", "deep-to-bottom":"steelblue"}
facecolor=cmocean.cm.gray(1/1.3)

models = {
    "CM4Xp25"  : {"historical":"odiv-231", "ssp5":"odiv-232"},
    "CM4Xp125" : {"historical":"odiv-255", "ssp5":"odiv-293"}
}

g_per_mol = {
    "cfc11": 137.37,
    "cfc12": 120.91,
}
Gg_per_g = 1.e-9
sec_per_year = 365.25 * 24 * 60 * 60
sec_per_year_wang21 = 360 * 24 * 60 * 60
sec_per_nsec = 1.e-9
m2_per_km2 = 1.e6

def load_datasets():
    grids = {}
    for model in models.keys():
        surface_fluxes = xr.concat([
            xr.open_zarr(f"/work/hfd/ftp/{model}_historical_transient_tracer_fluxes.zarr"),
            xr.open_zarr(f"/work/hfd/ftp/{model}_ssp585_transient_tracer_fluxes.zarr")
        ], dim="time")
        surface_tracers = xr.concat([
            xr.open_zarr(f"/work/hfd/ftp/{model}_historical_transient_tracer_surface.zarr"),
            xr.open_zarr(f"/work/hfd/ftp/{model}_ssp585_transient_tracer_surface.zarr")
        ], dim="time").drop("z_l")
        surface_tracers = surface_tracers.rename({v:f"{v}_surface" for v in surface_tracers.data_vars})
        tracers = xr.concat([
            xr.open_zarr(f"/work/hfd/ftp/{model}_historical_transient_tracers.zarr"),
            xr.open_zarr(f"/work/hfd/ftp/{model}_ssp585_transient_tracers.zarr")
        ], dim="year")

        ds = xr.merge([surface_fluxes, surface_tracers, tracers])
        ds = add_estimated_layer_interfaces(ds)
        grids[f"{model}_forced"] = CM4Xutils.ds_to_grid(ds)
        
    return grids
    

def pad_array(y):
    return np.concatenate(([y[0]], y, [y[-1]]))

def pad_years(x):
    return np.concatenate(([x[0] - 0.5], x+0.5, [x[-1] + 0.5]))

def add_estimated_layer_interfaces(ds):
        return ds.assign_coords({"zi": xr.DataArray(
            np.concatenate([[0], 0.5*(ds.zl.values[1:]+ds.zl.values[0:-1]), [6000]]),
            dims=('zi',)
        )})

def cmap_offwhite_fade(layer_color):
    colors = [(1, 1, 1), layer_color] # first color is black, last is red
    cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=200)
    cm = LinearSegmentedColormap.from_list(
            "Custom", [cm(0.05), layer_color], N=200)
    cm.set_bad((0.9, 0.9, 0.9))
    return cm
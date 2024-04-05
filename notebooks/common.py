import xarray as xr
import numpy as np
import CM4Xutils

sigma2_i = np.array([0, 36.7, 36.96, 60])
layer_labels = ["Upper layer", "Deep layer", "Bottom layer"]
layer_labels_short = ["Upper", "Deep", "Bottom"]
layer_colors = ["goldenrod", "seagreen", "darkslateblue"]
models = ["CM4Xp25", "CM4Xp125"]

def load_datasets():
    grids = {}
    for model in models:
        surface_fluxes = xr.concat([
            xr.open_zarr(f"/work/hfd/ftp/{model}_historical_transient_tracer_fluxes.zarr"),
            xr.open_zarr(f"/work/hfd/ftp/{model}_ssp585_transient_tracer_fluxes.zarr")
        ], dim="time")
        surface_tracers = xr.concat([
            xr.open_zarr(f"/work/hfd/ftp/{model}_historical_transient_tracer_surface.zarr"),
            xr.open_zarr(f"/work/hfd/ftp/{model}_ssp585_transient_tracer_surface.zarr")
        ], dim="time")
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
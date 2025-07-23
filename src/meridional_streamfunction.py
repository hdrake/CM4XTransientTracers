import xarray as xr 
import numpy as np 
import pandas as pd
import xgcm 
import warnings
from tqdm import tqdm

def zonal_average_by_lat(scalar: xr.DataArray, grid, lat: float) -> xr.DataArray:
    """
    Compute the zonal (and meridional) average of a scalar field *at* a given latitude band.
    
    Parameters
    ----------
    scalar : xr.DataArray
        Any scalar field on the same grid as grid._ds (e.g. ds["z"], ds["thk"], ds["zos"]).
    grid : your grid object
        Must expose grid._ds["geolat_v"] and support grid.diff(..., axis="Y").
    lat : float
        Latitude (°N) at which to extract the zonal average.
    
    Returns
    -------
    avg : xr.DataArray
        The mean profile of `scalar` versus vertical level, tagged with a coordinate `lat=lat`.
    """
    geolat_v = grid._ds["geolat_v"]
    # mask of cells crossing the specified latitude
    diff_lat_mask = np.abs(
        grid.diff((geolat_v >= lat).astype(float), axis="Y")
    )
    diff_lat_mask = diff_lat_mask.where(diff_lat_mask > 0)

    # weighted mean over the horizontal dims
    avg = (scalar * diff_lat_mask).mean(["xh", "yh"], skipna=True)
    return avg.assign_coords(lat=lat)


def meridional_streamfunction_at_lat(grid, lat: float, div_v = None) -> xr.DataArray:
    """
    Compute the meridional volume‐transport streamfunction (Psi) at a given latitude.
    
    Parameters
    ----------
    grid : your grid object
        Must expose grid._ds["vmo"] and support grid.diff(..., axis="Y", boundary="fill").
    geolat : xr.DataArray
        The cell‐center latitude field (e.g. ds.geolat), same horizontal dims as vmo.
    lat : float
        Latitude (°N) at which to compute the cumulative transport.
    
    Returns
    -------
    psi : xr.DataArray
        Psi vs. vertical level in Sverdrups (Sv), tagged with a coordinate `lat=lat`.
    """
    # 1) meridional divergence of vmo
    geolat = grid._ds["geolat"]
    if div_v is None:
        print("no divV provied")
        div_v = (
            grid
            .diff(grid._ds["vmo"].fillna(0.0), axis="Y", boundary="fill")
            .fillna(0.0))

    # 2) sum divergence northward of 'lat'
    psi_lat = (div_v * (geolat >= lat))
    psi_lat = psi_lat.where(psi_lat != 0).sum(["xh", "yh"], skipna=True)

    # 3) integrate from bottom up and convert to Sv
    psi_lat = (psi_lat.cumsum("sigma2_l") - psi_lat.sum("sigma2_l")) / 1e9

    return psi_lat.compute().assign_coords(lat=lat)

def meridional_streamfunction(grid, lats: np.ndarray, div_v = None) -> xr.DataArray:
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)

        grid._ds["geolat_v"] = grid._ds["geolat_v"].compute()
        grid._ds["geolat"] = grid._ds["geolat"].compute()

        psi = xr.concat([meridional_streamfunction_at_lat(grid, lat, div_v = div_v) for lat in tqdm(lats, desc="Computing ψ by latitude")], dim = "lat").sortby("lat") 
        depth = xr.concat([zonal_average_by_lat(grid._ds["z"], grid, lat) for lat in tqdm(lats, desc="Computing z by latitude")], dim = "lat").sortby("lat") 

        psi_ds = -psi.rename("psi").to_dataset()
        psi_ds = psi_ds.assign_coords({'depth': depth})
        psi_geolat = xr.ones_like(psi_ds['depth'].isel(exp = 0).drop_vars("exp")) * psi_ds["lat"]
        psi_ds = psi_ds.assign_coords({'geolat': psi_geolat})

    return psi_ds
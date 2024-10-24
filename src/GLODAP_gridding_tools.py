import pyinterp
import pyinterp.backends.xarray
from scipy import interpolate
import xarray as xr
import dask
import numpy as np
import matplotlib.pyplot as plt
import cartopy  # Map projections libary
import cartopy.crs as ccrs  # Projections list
from scipy.interpolate import CubicSpline
import copy
import pandas as pd

def extract_GLODAP_vars(ds, varn): 
    lat, lon = ds["G2latitude"].to_xarray(), ds["G2longitude"].to_xarray()
    years = ds["G2year"].to_xarray()
    depths = ds["G2depth"].to_xarray()
    
    cast_num = ds["G2cast"].to_xarray().astype("int").astype("str")
    cruise_num = ds["G2cruise"].to_xarray().astype("int").astype("str")
    station_num = ds["G2station"].to_xarray().astype("int").astype("str")

    cast_codes = ds["G2expocode"].to_xarray().astype("str")
    
    tmp = np.char.add(cast_codes.values, "_")
    tmp = np.char.add(tmp, cruise_num)
    tmp = np.char.add(tmp, "_")
    tmp = np.char.add(tmp, station_num)
    tmp = np.char.add(tmp, "_")
    tmp = np.char.add(tmp, cast_num)
    
    all_casts = copy.deepcopy(cast_codes)
    all_casts.values = tmp

    return lat, lon, years, depths, all_casts

def GLODAP_2_regular_grid(ds, regular_z, regular_lat, regular_lon, varns, year = 1990.):

    nvars = len(varns)
    nreglon = len(regular_lon)
    nreglat = len(regular_lat)
    nregz = len(regular_z)
    
    vars_dicts = dict()
    casts_dict = dict()
    
    for (v, varn) in enumerate(varns): 
        vars_gridded_nearest = np.nan * np.zeros((nreglon, nreglat, nregz))
        lat, lon, years, depths, all_casts = extract_GLODAP_vars(ds, varn)
        var_df = ds[varn].to_xarray()
        try: 
            var_df_flag = ds[varn + "f"].to_xarray()
            which_year_and_notnan = (years == year) * (~np.isnan(var_df)) * ( var_df_flag == 2)
        except: 
            which_year_and_notnan = (years == year) * (~np.isnan(var_df))

        depths_ = depths[which_year_and_notnan]
        lat_  = lat[which_year_and_notnan]
        lon_ = lon[which_year_and_notnan]
        var_ = var_df[which_year_and_notnan]
        all_casts_ = all_casts[which_year_and_notnan]
        all_casts_ = np.array(all_casts_.values.tolist())
        unique_casts_ = np.unique(all_casts_)
        
        interp_casts = np.zeros((len(regular_z), len(unique_casts_)))
        LATS = np.zeros(len(unique_casts_))
        LONS = np.zeros(len(unique_casts_))
        cast_list = []
        for j in range(len(unique_casts_)):
            which_cast = (all_casts_ == unique_casts_[j])
            cast_list += [unique_casts_[j]]
            cast_var = var_[which_cast]; cast_depths = depths_[which_cast]
            
            unique_depths = np.unique(cast_depths)
            LATS[j] = lat_[which_cast][0]
            LONS[j] = lon_[which_cast][0]
            time_averaged_cast = np.zeros(len(unique_depths))
            if len(unique_depths) < len(cast_depths):
                for (i, d) in enumerate(unique_depths):
                    time_averaged_cast[i] = np.nanmean(cast_var[cast_depths == d]) 
            else:
                time_averaged_cast[:] = 1 * cast_var[:]  

            f = interpolate.interp1d(unique_depths, time_averaged_cast, fill_value = np.nan, 
                                     bounds_error = False, kind = "linear")
            interp_casts[:, j] = f(regular_z)
            
        interp_casts_ds = xr.Dataset(
                        coords={
                        "points":(["points"], np.arange(len(unique_casts_))),
                        "z":(["z"], regular_z)
                        })
        interp_casts_ds[varn] = (["z", "points"], interp_casts)
        interp_casts_ds["lat"] = (["points"], LATS)
        interp_casts_ds["lon"] = (["points"], LONS)
        interp_casts_ds["cast_id"] = (["points"], cast_list)
        interp_casts_ds["time"] = year

        for k in range(len(regular_z)):
            # binning = pyinterp.Binning2D(
            #     pyinterp.Axis(regular_lon, is_circle=True),
            #     pyinterp.Axis(regular_lat))
            # binning.clear()
            binning = pyinterp.Histogram2D(
                pyinterp.Axis(regular_lon, is_circle=True),
                pyinterp.Axis(regular_lat))
            binning.clear()
            
            binning.push(LONS, LATS, 1 * interp_casts[k, :][:])        
            vars_gridded_nearest[:, :, k] = 1 * binning.variable('mean')            
            
        regular_ds = xr.Dataset(
                        coords={
                        "lon":(["lon"], regular_lon),
                        "lat":(["lat"], regular_lat),
                        "z":(["z"], regular_z), 
                        })
        regular_ds[varn] = (["lon", "lat", "z"], vars_gridded_nearest)
        regular_ds["time"] = year

        vars_dicts[varn] = regular_ds
        casts_dict[varn] = interp_casts_ds

    
    
    return (vars_dicts, casts_dict)
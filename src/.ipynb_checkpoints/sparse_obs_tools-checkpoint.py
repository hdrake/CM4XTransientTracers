import warnings
import matplotlib.pyplot as plt
import numpy as np
import glob
import xarray as xr
import xbudget
import regionate
import datetime
import cftime
import xwmt
import xwmb
import xgcm
import cartopy.crs as ccrs
import CM4Xutils #needed to run pip install nc-time-axis
from regionate import MaskRegions, GriddedRegion
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import xesmf as xe
from scipy import linalg
from scipy import stats
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import BoundaryNorm

def approximate_z(ds, dim = "zl"):
    tmp = ds.thkcello.cumsum(dim = dim)
    #average between 0 and cell bottom
    tmp1 = tmp.isel({dim: 0}) / 2 
    #get top of cell
    tmp2 = tmp.isel({dim : slice(0, -1)}) 
    #get bottom of cell
    tmp3 = tmp.isel({dim : slice(1, None)}) 
    #make sure cell interfaces are on same coordinate
    tmp2.coords[dim] = tmp3.coords[dim]
    #take average
    tmp4 = (tmp2 + tmp3) / 2

    ds["z"] = xr.concat([1. * tmp1, 1. * tmp4], dim = dim)    
    ds["z_bottom"] = 1. * tmp

    ds["z"] = ds["z"].where(ds["thkcello"] > 0) 
    ds["z_bottom"] = ds["z_bottom"].where(ds["thkcello"] > 0) 

    return ds

#float_conversion default is years to seconds

def integrate(ds, varname = "surface_boundary_fluxes", dim = "time", float_conversion = 3.154e+7): 
    
    # Check if time is datetime; if so, convert dt to seconds
    
    if np.issubdtype(ds[dim].dtype, np.datetime64):
        dt = da[dim].diff(dim) / np.timedelta64(1, "s")  # Convert to seconds
    elif np.issubdtype(ds[dim].dtype, "O"): 
        
        dt = ds[dim].diff(dim) / np.timedelta64(1, "s")  # Convert to seconds
    
    elif np.issubdtype((1. * ds[dim]).dtype, np.float64):
        dt = float_conversion * ds[dim].diff(dim)  # Already in numeric format (assume seconds)
    
    dt_arr = xr.zeros_like(ds[dim]) * np.nan

    dt_arr.isel({dim : slice(1, None)}).values[:] = dt.values[:]
    dt_arr = dt_arr.astype("float")

    
    trapz = (ds.shift({dim: 1})[varname] + ds[varname]) * dt_arr / 2.0
    trapz = trapz.where(~np.isnan(trapz), 0) #replace nans with zeros 

    integral_ds = xr.zeros_like(trapz).rename(f"integrated_{varname}")
    integral_ds.values = trapz.cumsum(dim).values

    return integral_ds


def linear_regression(x, y):
    """
    Compute linear regression parameters including R-squared.
    
    R-squared is calculated using: 1 - SS_residual/SS_total
    where SS_residual is sum of squared residuals from the regression line
    and SS_total is sum of squared deviations from the mean
    """
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    
    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return np.array([slope, intercept, r_value, p_value, r_squared])

def regress_spatial(spatial_data, reference_data, common_dim="time"):
    """
    Perform regression between spatial data and reference time series
    """
    # Ensure time is the first dimension for both datasets
    spatial_data = spatial_data.transpose(common_dim, ...)
    
    # Use apply_ufunc to perform regression
    results = xr.apply_ufunc(
        linear_regression,
        reference_data,
        spatial_data,
        input_core_dims=[[common_dim], [common_dim]],
        output_core_dims=[['parameter']],
        vectorize=True,
        output_dtypes=[float],
        output_sizes={'parameter': 5}  # Now 5 parameters instead of 4
    )
    
    # Create a dataset with labeled parameters
    ds = xr.Dataset({
        'slope': results.isel(parameter=0),
        'intercept': results.isel(parameter=1),
        'r_value': results.isel(parameter=2),
        'p_value': results.isel(parameter=3),
        'r_squared': results.isel(parameter=4)
    })
    
    return ds

def cftime_to_decimal_year(cftime_array):
    """Convert a cftime xarray array to a decimal year array."""
    # Extract the year, day of the year, and total days in the year
    year = cftime_array.dt.year
    day_of_year = cftime_array.dt.dayofyear
    total_days = xr.where(
        (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0)), 366, 365
    )  # Account for leap years if necessary

    # Calculate the decimal year
    decimal_years = year + (day_of_year - 1) / total_days
    
    return decimal_years



def plot_median_quantiles(ax, x, median, lower, upper, color='blue', alpha=0.3, label=""):
    ax.plot(x, median, color=color, label=label)
    ax.fill_between(x, lower, upper, color=color, alpha=alpha)

def plot_median_with_errorbars(ax, x, median, lower, upper, color='blue', label=""):
    ax.errorbar(x, median, yerr=[median - lower, upper - median], marker='o', color=color, ecolor=color, capsize=3, label=label)

def plot_bootmedian_with_errorbars(ax, x, median, lower, upper, color='blue', label=""):
    ax.errorbar(x, median, yerr=[median - lower, upper - median], marker='o', color=color, ecolor=color, capsize=3, label=label)

def bootstrap_spatial_stats(da, n_bootstrap=5000, flat_coords = ["xh", "yh"], seed=None):
    """
    Calculate bootstrapped mean and standard error for spatial data.
    
    Parameters:
    -----------
    da : xarray.DataArray
        Input spatial data array
    n_bootstrap : int, optional
        Number of bootstrap samples to generate
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    xarray.Dataset
        Dataset containing the bootstrapped statistics:
        - original_mean: spatial mean of original data
        - original_std: spatial standard deviation of original data
        - boot_mean: mean of bootstrapped means
        - boot_std: standard deviation of bootstrapped means (standard error)
        - confidence_intervals: 95% confidence intervals for the mean
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Flatten spatial dimensions into a single dimension
    
    stacked_da = da.stack(space=flat_coords)
    stacked_da = stacked_da.where(~np.isnan(stacked_da), drop=True)

    # .dropna(dim = "space", how='all')
    
    # Original statistics
    orig_mean = stacked_da.mean('space')
    orig_std = stacked_da.std('space')
    
    # Preallocate array for bootstrap samples
    boot_means = np.zeros((n_bootstrap,) + orig_mean.shape)
    
    # Perform bootstrap sampling
    for i in range(n_bootstrap):
        # Sample with replacement
        n_points = len(stacked_da.space)
        indices = np.random.randint(0, n_points, size=n_points)
        boot_sample = stacked_da.isel(space=indices)
        
        # Calculate mean for this bootstrap sample
        boot_means[i] = boot_sample.mean('space').values
    
    # Convert bootstrap results to xarray
    boot_means_da = xr.DataArray(
        boot_means,
        dims=('bootstrap',) + orig_mean.dims,
        coords={'bootstrap': range(n_bootstrap), **orig_mean.coords}
    )
    
    # Calculate bootstrap statistics
    bootstrap_mean = boot_means_da.mean('bootstrap')
    bootstrap_std = boot_means_da.std('bootstrap')
    
    # # Calculate 95% confidence intervals
    # ci_lower = np.percentile(boot_means, 2.5, axis=0)
    # ci_upper = np.percentile(boot_means, 97.5, axis=0)
    try: 
        ci_lower = boot_means_da.quantile(0.025, dim='bootstrap').values
    
        ci_upper = boot_means_da.quantile(0.975, dim='bootstrap').values
    except: 
        print(boot_means_da)
    
    if np.isnan(np.sum(ci_upper)):
        print("something is wrong^^^^^^")
        
        print("printing sum of bootmean:", np.sum(boot_means))
        print("printing sum of data:", np.sum(stacked_da))

    # Combine results into a dataset
    results = xr.Dataset({
        'original_mean': orig_mean,
        'original_std': orig_std,
        'boot_mean': bootstrap_mean,
        'boot_std': bootstrap_std,
        'ci_lower': (orig_mean.dims, ci_lower),
        'ci_upper': (orig_mean.dims, ci_upper)
    })
    
    return results

def _bootstrap_single_snapshot(data, n_bootstrap=5000, seed=None):
    """
    Helper function to bootstrap a single spatial snapshot.
    Expects numpy array of clean data (no NaNs).
    
    Parameters:
    -----------
    data : numpy.ndarray
        Clean 1D array of data points
    n_bootstrap : int
        Number of bootstrap samples
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray
        Array of statistics [orig_mean, orig_std, boot_mean, boot_std, ci_lower, ci_upper]
    """
    if seed is not None:
        np.random.seed(seed)
    data = data[~np.isnan(data)]
    if len(data) == 0:  # Handle case where all data is NaN
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    
    n_points = len(data)
    boot_means = np.zeros(n_bootstrap)
    
    # Perform bootstrap sampling
    for i in range(n_bootstrap):
        indices = np.random.randint(0, n_points, size=n_points)
        boot_sample = data[indices]
        boot_means[i] = np.mean(boot_sample)
    
    # Calculate statistics
    orig_mean = np.mean(data)
    orig_std = np.std(data)
    boot_mean = np.mean(boot_means)
    boot_std = np.std(boot_means)
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    
    return np.array([orig_mean, orig_std, boot_mean, boot_std, ci_lower, ci_upper])

def bootstrap_spatial_stats_broadcast(da, n_bootstrap=5000, flat_coords=["xh", "yh"], seed=None):
    """
    Calculate bootstrapped mean and standard error for spatial data.
    Uses xarray's apply_ufunc for automatic broadcasting across non-spatial dimensions.
    
    Parameters:
    -----------
    da : xarray.DataArray
        Input spatial data array
    n_bootstrap : int, optional
        Number of bootstrap samples to generate
    flat_coords : list of str, optional
        List of coordinate names to flatten for spatial sampling
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    xarray.Dataset
        Dataset containing the bootstrapped statistics:
        - original_mean: spatial mean of original data
        - original_std: spatial standard deviation of original data
        - boot_mean: mean of bootstrapped means
        - boot_std: standard deviation of bootstrapped means (standard error)
        - ci_lower: lower bound of 95% confidence interval
        - ci_upper: upper bound of 95% confidence interval
    """
    # Stack spatial dimensions
    stacked_da = da.stack(space=flat_coords)
    
    # Define output names for the statistics
    stat_names = ['original_mean', 'original_std', 'boot_mean', 
                  'boot_std', 'ci_lower', 'ci_upper']
    
    # Apply the bootstrap function across non-spatial dimensions
    result = xr.apply_ufunc(
        _bootstrap_single_snapshot,
        stacked_da,
        input_core_dims=[['space']],
        output_core_dims=[['stat']],
        vectorize=True,
        kwargs={'n_bootstrap': n_bootstrap, 
               'seed': seed},
        dask='parallelized',
        output_dtypes=[float],
        output_sizes={'stat': 6}
    )
    
    # Convert to dataset with named variables
    result = result.assign_coords(stat=stat_names)
    ds = result.to_dataset(dim='stat')
    
    return ds



def plot_cfc11_analysis(bottom_cfc11, masked_glodap_model, plot_median_with_errorbars, plot_median_quantiles, 
                       region={"lat": slice(-55, -45), "lon": slice(29, 31)},
                       obs_region={"lat": slice(-80, -30), "lon": slice(20, 40)},
                       figsize=(10, 5)):
    """
    Create a comprehensive visualization of CFC-11 data analysis with observation counts and time series.
    Uses a discrete colorbar with colors centered on integer values.
    """
    # Initialize GridSpec layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 2])
    
    # Observation count plot
    obs_ax = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    
    # Calculate observation counts
    obs_count = ((~np.isnan(bottom_cfc11)).sum("time")).sel(
        lat=slice(obs_region["lat"].start, obs_region["lat"].stop),
        lon=slice(obs_region["lon"].start, obs_region["lon"].stop)
    ).T
    obs_count = obs_count.where(obs_count > 0)
    
    # Create boundaries halfway between integers
    max_count = int(np.ceil(obs_count.max().values))
    min_count = int(np.floor(obs_count.min().values)) if not np.isnan(obs_count.min()) else 0
    
    # Create boundaries array (halfway between integers)
    boundaries = np.arange(min_count - 0.5, max_count + 1.5, 1)
    # Create the levels for the tick marks (integers)
    levels = np.arange(min_count, max_count + 1, 1)
    
    # Create the discrete colormap
    cmap = plt.cm.viridis
    norm = BoundaryNorm(boundaries, cmap.N)
    
    # Plot with discrete colorbar
    mesh = obs_ax.pcolormesh(
        obs_count.lon, 
        obs_count.lat, 
        obs_count.values,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree()
    )
    
    # Add colorbar
    cb = plt.colorbar(
        mesh,
        ax=obs_ax,
        orientation="vertical",
        label="# In-Situ CFC-11 Observations\nat Ocean Bottom",
        ticks=levels,  # use integer levels for tick marks
        boundaries=boundaries,  # use boundaries for color transitions
        pad=0.2,
        fraction=0.035,
        location="left"
    )
    
    # Configure map
    obs_ax.coastlines()
    gl = obs_ax.gridlines(draw_labels=True, alpha=0.3)
    gl.right_labels = False
    gl.top_labels = False
    
    # Add analysis region box
    lat_min, lat_max = region["lat"].start, region["lat"].stop
    lon_min, lon_max = region["lon"].start, region["lon"].stop
    obs_ax.add_patch(Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        edgecolor='red',
        facecolor='none',
        lw=1,
        alpha=0.3,
        transform=ccrs.PlateCarree()
    ))
    
    # Plot time series
    data_labels = ["CM4Xp125 forced", "CM4Xp125 control", "GLODAP"]
    data_colors = ["red", "blue", "green"]
    datasets = [masked_glodap_model.sel(region).isel(exp=i) for i in range(2)] + [bottom_cfc11]
    
    for label, color, dataset in zip(data_labels, data_colors, datasets):
        data_selected = dataset.sel(region)
        median = data_selected.quantile(0.5, dim=["lat", "lon"])
        lower = data_selected.quantile(0.025, dim=["lat", "lon"])
        upper = data_selected.quantile(0.975, dim=["lat", "lon"])
        
        if label == "GLODAP":
            plot_median_with_errorbars(ax1, data_selected.time, median, lower, upper,
                                     color=color, label=label)
            ax2.scatter(data_selected.time,
                       data_selected.count(dim=["lat", "lon"]),
                       color=color, label=label, marker="o", zorder=10)
        else:
            plot_median_quantiles(ax1, data_selected.time, median, lower, upper,
                                color=color, label=label)
    
    # Add model observation counts
    ax2.scatter(
        datasets[0].time,
        datasets[0].sel(region).count(dim=["lat", "lon"]).where(lambda x: x > 0),
        color="k",
        label="model"
    )
    
    # Finalize plot details
    ax1.set_ylabel('CFC-11 Concentration')
    ax1.legend()
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Number of Observations')
    ax2.legend()
    fig.tight_layout()
    
    return fig
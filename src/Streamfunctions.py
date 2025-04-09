import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def compute_streamfunction(ds, z_coord = "sigma2_l", reverse_cumsum = True, 
                                      rho0 = None):
    
    reverse_sigma2 = lambda ds: ds.isel({z_coord: slice(None, None, -1)})
    
    v = -(ds["vmo"])
    v = v.where(np.abs(v) > 1e-8)
    v = v.sum("xh")

    if rho0 is None: 
        rho0 = (v["sigma2_l"] + 1000)
        
    v = v / rho0 #convert to volume transport

    if reverse_cumsum:
        psi = reverse_sigma2(reverse_sigma2(v).cumsum("sigma2_l"))
    else:
        psi = v.cumsum("sigma2_l")

    psi = psi.rename("psi")
    psi["geolat"] = ds["geolat_v"].mean("xh")

    return psi


def density_transform(density_values, power = 50):
    """
    Transform density values to create telescoping effect with higher resolution for denser waters.
    
    Parameters:
    -----------
    density_values : array
        Array of density values
        
    Returns:
    --------
    transformed_values : array
        Transformed density values with emphasis on denser waters
    """
    # Normalized to [0,1] range
    dens_min = np.min(density_values)
    dens_max = np.max(density_values)
    dens_norm = (density_values - dens_min) / (dens_max - dens_min)
    
    # Apply power function (smaller value = more emphasis on denser waters)
    return dens_norm ** power

def create_density_formatter(density_values, transformed_values):
    """
    Create a formatter function to display actual density values on transformed axis.
    
    Parameters:
    -----------
    density_values : array
        Original density values
    transformed_values : array
        Transformed density values
        
    Returns:
    --------
    formatter : function
        Formatter function for matplotlib
    """
    def formatter(y, pos):
        idx = np.abs(transformed_values - y).argmin()
        return f"{density_values[idx]:.2f}"
    
    return formatter

def plot_isopycnal_overturning(fig, ax, ds, streamfunction_var='psi', 
                                      density_dim='density', lat_dim='latitude', 
                                       power = 0.5, levels = None, 
                                       clabel = 'Isopycnal Overturning [Sv]'):
    """
    Minimal code to plot isopycnal overturning with telescoping density axis.
    
    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset containing the streamfunction and coordinates
    streamfunction_var : str
        Name of the streamfunction variable in the dataset
    density_dim : str
        Name of the density dimension in the dataset
    lat_dim : str
        Name of the latitude dimension in the dataset
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Extract the data from xarray
    if isinstance(ds, xr.Dataset):
        psi = ds[streamfunction_var].values
        density = ds[density_dim].values
        latitudes = ds[lat_dim].values
    else:  # Assume it's a DataArray
        psi = ds.values
        density = ds[density_dim].values
        latitudes = ds[lat_dim].values
    
    # Transform the density values to create telescoping effect
    transformed_density = density_transform(density, power)
    
    # Create meshgrid for plotting - handle different dimension orders
    if psi.shape[0] == len(density) and psi.shape[1] == len(latitudes):
        # psi shape is [density, latitude]
        LAT, DENS = np.meshgrid(latitudes, transformed_density)
    else:
        # psi shape is [latitude, density]
        DENS, LAT = np.meshgrid(transformed_density, latitudes)
        psi = psi.T  # Transpose for consistent plotting
    
    if levels is None: 
        max_abs = np.nanmax(np.abs(psi))
        levels = np.linspace(-max_abs, max_abs, 21)
        

    # Plot filled contours
    cf = ax.contourf(LAT, DENS, psi, levels=levels, cmap='RdBu_r', extend='both')
    
    # # Add contour lines
    # ax.contour(LAT, DENS, psi, levels=levels[::2], colors='k', linewidths=0.5, alpha=0.5)
    
    # # Add zero contour
    # ax.contour(LAT, DENS, psi, levels=[0], colors='k', linewidths=1.5)
    
    # Add colorbar
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(clabel)
    
    # Apply formatter to show actual density values on transformed axis
    formatter = create_density_formatter(density, transformed_density)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(formatter))
    
    # Labels
    ax.set_xlabel('Latitude [°N]')
    ax.set_ylabel('Potential Density σ [kg/m³]')

def find_min_locations(data, fixed_dim=None):
    """
    Find locations of minimum values across dimensions in an xarray DataArray.
    When fixed_dim is provided, returns minimum locations for each value along that dimension.
    
    Parameters:
    -----------
    data : xarray.DataArray
        The input data array
    fixed_dim : str, optional
        Name of dimension to hold fixed. If provided, returns minimum locations
        for each value along this dimension.
        
    Returns:
    --------
    xarray.Dataset or dict : 
        If fixed_dim is None, returns a dictionary with dimension names as keys 
        and coordinates of global minimum as values.
        If fixed_dim is provided, returns an xarray Dataset with the fixed dimension preserved
        and new variables for each remaining dimension's minimum coordinates.
    """
    
    if fixed_dim is None:
        # For global minimum
        # This approach ensures we get the actual coordinates
        min_locations = {}
        
        # Find the global minimum value
        min_value = data.min().item()
        
        # Find the first occurrence of the minimum value
        min_idx = np.argwhere(data.values == min_value)[0]
        
        # Map indices to dimension names and coordinates
        for i, dim in enumerate(data.dims):
            min_locations[dim] = data[dim].values[min_idx[i]]
        
        return min_locations
    
    else:
        # Verify dimension exists
        if fixed_dim not in data.dims:
            raise ValueError(f"Dimension '{fixed_dim}' not found in data array")
        
        # Create output dataset
        result = xr.Dataset(coords={fixed_dim: data[fixed_dim]})
        
        # Get other dimensions
        other_dims = [dim for dim in data.dims if dim != fixed_dim]
        
        if not other_dims:
            # If only one dimension, return values directly
            return {x: y for x, y in zip(data[fixed_dim].values, data.values)}
        
        # For each value along the fixed dimension
        for fixed_idx, fixed_val in enumerate(data[fixed_dim].values):
            # Get the slice at this fixed value
            slice_data = data.isel({fixed_dim: fixed_idx})
            
            # Find minimum in this slice
            slice_min = slice_data.min().item()
            
            # Find the indices of this minimum value
            slice_min_idx = np.argwhere(slice_data.values == slice_min)[0]
            
            # Create variables for each dimension's min location
            for i, dim in enumerate(other_dims):
                # Initialize array if first iteration
                if fixed_idx == 0:
                    result[dim] = xr.DataArray(
                        np.full(len(data[fixed_dim]), np.nan),
                        dims=[fixed_dim],
                        coords={fixed_dim: data[fixed_dim]}
                    )
                
                # Set the coordinate value at this position
                result[dim].values[fixed_idx] = slice_data[dim].values[slice_min_idx[i]]
                
        
        return result
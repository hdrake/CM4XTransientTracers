import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xarray as xr
import numpy as np
from tqdm import tqdm
import sectionate
import concurrent.futures
from os import cpu_count

# def generate_moc_grid_indices(grid, dlat=2.):
#     """Generate grid indices for MOC calculation"""
#     lats = np.arange(-89, 89., dlat)
#     section_grid_dicts = {}
#     section_lons = np.arange(0., 360.+5., 5.)
    
#     for lat in tqdm(lats, desc="Processing latitudes", unit="lat"):
#         section_lats = np.full_like(section_lons, lat, dtype=float)
#         i, j, lons, lats_out = sectionate.grid_section(
#             grid, section_lons, section_lats, topology="MOM-tripolar"
#         )
#         section_grid_dicts[lat] = {"i": i, "j": j, "lons": lons, "lats": lats_out}
    
#     return section_grid_dicts
def generate_moc_grid_indices(grid, lats = None, dlat=2., parallel=False, num_workers=None):
    """
    Generate grid indices for MOC calculation
    
    Parameters
    ----------
    grid : xarray.Dataset or similar
        The grid dataset containing coordinates
    dlat : float, optional
        Latitude step size in degrees, default is 2.0
    parallel : bool, optional
        Whether to run the calculation in parallel, default is False
    num_workers : int, optional
        Number of worker threads to use if parallel=True. If None, uses a suitable default.
    
    Returns
    -------
    dict
        Dictionary of grid indices for each latitude
    """

    if lats is None:
        lats = np.arange(-89, 89., dlat)
    else:
        print("using predefined latitudes")
    
    section_grid_dicts = {}
    section_lons = np.arange(0., 360.+5., 5.)
    
    if parallel:
        
        if num_workers is None:
            num_workers = max(1, (cpu_count() - 2) or 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Create a list to store all futures
            futures = []
            
            # Submit all tasks
            for lat in lats:
                futures.append(executor.submit(
                    _process_single_lat, 
                    grid=grid, 
                    lat=lat, 
                    section_lons=section_lons
                ))
            
            # Process results as they complete with tqdm for progress tracking
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Processing latitudes in parallel ({num_workers} workers)",
                unit="lat"
            ):
                lat, result = future.result()
                section_grid_dicts[lat] = result
    else:
        # Original sequential implementation
        for lat in tqdm(lats, desc="Processing latitudes", unit="lat"):
            section_lats = np.full_like(section_lons, lat, dtype=float)
            i, j, lons, lats_out = sectionate.grid_section(
                grid, section_lons, section_lats, topology="MOM-tripolar"
            )
            section_grid_dicts[lat] = {"i": i, "j": j, "lons": lons, "lats": lats_out}
    
    return section_grid_dicts


def _process_single_lat(grid, lat, section_lons):
    """Helper function to process a single latitude"""
    section_lats = np.full_like(section_lons, lat, dtype=float)
    i, j, lons, lats_out = sectionate.grid_section(
        grid, section_lons, section_lats, topology="MOM-tripolar"
    )
    return lat, {"i": i, "j": j, "lons": lons, "lats": lats_out}
    
def transport_across_latitude(grid, grid_indices, lat, 
                              Z_prefix = "sigma2", reverse_cumsum = True):
    """Calculate transport across a latitude line"""
    i, j = grid_indices["i"], grid_indices["j"]
    
    # Since sectionate.convergent_transport doesn't accept a data parameter,
    # we need to use the grid as is
    # The time-specific data should already be set in the grid's dataset
    conv_transport = sectionate.convergent_transport(
        grid, i, j, layer=f"{Z_prefix}_l", interface=f"{Z_prefix}_i")
    
    # Calculate integrated transport
    northward_transport = -(conv_transport['conv_mass_transport'].sum("sect"))
    reverse_sigma2 = lambda ds: ds.isel({f"{Z_prefix}_l": slice(None, None, -1)})
    
    if reverse_cumsum:
        integrated_transport = -reverse_sigma2(reverse_sigma2(northward_transport).cumsum(f"{Z_prefix}_l"))
        return integrated_transport.expand_dims({'lat': [lat]})
    else: 
        integrated_transport = -northward_transport.cumsum(f"{Z_prefix}_l")
        
        return integrated_transport.expand_dims({'lat': [lat]})

def thickness_across_latitude(grid, grid_indices, lat, 
                              Z_prefix = "sigma2"):
    """Calculate transport across a latitude line"""
    i, j = grid_indices["i"], grid_indices["j"]
    
    # Since sectionate.convergent_transport doesn't accept a data parameter,
    # we need to use the grid as is
    # The time-specific data should already be set in the grid's dataset

    thickness = sectionate.extract_tracer("thkcello",grid, i, j, sect_coord="sect")
    
    mean_thickness = thickness.mean("sect")
    
    return mean_thickness.expand_dims({'lat': [lat]})
    
def moc_across_time(grid, grid_indices, time_dim='time', Z_prefix = "sigma2", reverse_cumsum = True):
    """
    Calculate MOC across time using xarray's apply_ufunc
    
    Parameters:
    -----------
    grid : xgcm.Grid
        The xgcm grid object with time dimension
    grid_indices : dict
        Dictionary of grid indices by latitude
    time_dim : str, optional
        Name of the time dimension in the grid's dataset (default: 'time')
    
    Returns:
    --------
    xarray.DataArray
        MOC values with dimensions (lat, sigma2_l, time)
    """
    latitudes = sorted(grid_indices.keys())
    # grid._ds["thkcello"] = grid._ds["thkcello"].where(grid._ds["thkcello"] > 0)

    # Define a function to process one time step
    def process_timestep_thk(grid):
        # Update grid with time-specific data
        
        # Process each latitude
        lat_results = []

        for lat in latitudes:
            result = thickness_across_latitude(grid, grid_indices[lat], lat, Z_prefix = Z_prefix)
            lat_results.append(result)
        
        # Combine latitude results
        combined = xr.concat(lat_results, dim="lat").sortby("lat")

        return combined
    # Define a function to process one time step
    def process_timestep_conv(grid):
        # Update grid with time-specific data
        
        # Process each latitude
        lat_results = []
        for lat in latitudes:
            result = transport_across_latitude(grid, grid_indices[lat], lat, 
                                               Z_prefix = Z_prefix, reverse_cumsum = reverse_cumsum)
            lat_results.append(result)
        
        # Combine latitude results
        combined = xr.concat(lat_results, dim="lat").sortby("lat")
        return combined

        # Define a function to process one time step
    def process_timestep_depth(grid):
        # Update grid with time-specific data
        
        # Process each latitude
        lat_results = []
        for lat in latitudes:
            result = max_depth_across_latitude(grid, grid_indices[lat], lat, Z_prefix = Z_prefix)
            
            lat_results.append(result)
        
        # Combine latitude results
        combined = xr.concat(lat_results, dim="lat").sortby("lat")
        return combined
        
    def set_coords(ds):
        return ds.assign_coords({
                "lat":latitudes,
                f"{Z_prefix}_l":grid._ds[f"{Z_prefix}_l"]}
            )
        
    # Apply the function to each time step
    conv_result = xr.apply_ufunc(
        process_timestep_conv,
        grid,
        input_core_dims=[[]],
        output_core_dims=[['lat', f"{Z_prefix}_l"]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        output_sizes={'lat': len(latitudes), f"{Z_prefix}_l": len(grid._ds[f"{Z_prefix}_l"])}
    )
    
    # Set coordinates
    conv_result = set_coords(conv_result)

        # Apply the function to each time step
    thk_result = xr.apply_ufunc(
        process_timestep_thk,
        grid,
        input_core_dims=[[]],
        output_core_dims=[['lat', f"{Z_prefix}_l"]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        output_sizes={'lat': len(latitudes), f"{Z_prefix}_l": len(grid._ds[f"{Z_prefix}_l"])}
    )

    thk_result = set_coords(thk_result)

    # depth_result = xr.apply_ufunc(
    #     process_timestep_depth,
    #     grid,
    #     input_core_dims=[[]],
    #     output_core_dims=[['lat', f"{Z_prefix}_l"]],
    #     vectorize=True,
    #     dask='parallelized',
    #     output_dtypes=[float],
    #     output_sizes={'lat': len(latitudes), f"{Z_prefix}_l": len(grid._ds[f"{Z_prefix}_l"])}
    # )

    # depth_result = set_coords(depth_result)


    # Transpose to get (lat, sigma2_l, time) order
    return xr.merge([conv_result, thk_result])
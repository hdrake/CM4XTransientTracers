import xarray as xr
def select_from_location(ds, ds_loc, dim = "sigma2_l_target"):
    """Select data from maximum locations in control and forced experiments."""
    results = {}
    
    for exp in ["control", "forced"]:
        exp_data = []
        for y in ds.year:
            ds_sub = ds.sel(exp=exp, year=y)
            loc_sub = ds_loc.sel(exp=exp, year=y).values
            exp_data += [ds_sub.sel({dim:loc_sub})]
        results[exp] = xr.concat(exp_data, dim="year")
    
    return xr.concat(list(results.values()), dim="exp")

def select_from_yearly_location_monthly(ds, ds_loc, dim = "sigma2_l_target"):
    """Select monthly data from maximum locations in control and forced experiments."""
    results = {}
    
    # First, add year as a coordinate if it doesn't exist
    if 'year' not in ds.coords:
        ds = ds.assign_coords(year=ds.time.dt.year)
    
    for exp in ["control", "forced"]:
        exp_monthly_data = []
        
        # Group by year
        for year, year_data in ds.sel(exp=exp).groupby('year'):
            # Get the max location for this year
            max_loc = ds_loc.sel(exp=exp, year=year).values
            
            # Select data at this location for all months in this year
            selected_data = year_data.sel({dim:max_loc})
            
            # Add to our results
            exp_monthly_data.append(selected_data)
        
        # Combine all years for this experiment
        results[exp] = xr.concat(exp_monthly_data, dim="time")
    
    # Combine both experiments
    return xr.concat(list(results.values()), dim="exp")

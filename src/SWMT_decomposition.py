import xarray as xr 

def get_SWMT_heat(ds): 
    ds_heat = ds["surface_ocean_flux_advective_negative_rhs_heat"] + \
              ds["surface_exchange_flux_heat"] + \
              ds["frazil_ice_heat"] + \
              ds["bottom_flux_heat"]

    ds_heat = ds_heat.rename("boundary_fluxes_heat").to_dataset()
    
    ds_heat["SWMT_heat"] = ds["boundary_fluxes_heat"] - ds["bottom_flux_heat"]
    
    ds_heat["SWMT_heat_latent"] = ds["surface_exchange_flux_nonadvective_latent_heat"]
    ds_heat["SWMT_heat_longwave"] = ds["surface_exchange_flux_nonadvective_longwave_heat"]
    ds_heat["SWMT_heat_shortwave"] = ds["surface_exchange_flux_nonadvective_shortwave_heat"]
    ds_heat["SWMT_heat_sensible"] = ds["surface_exchange_flux_nonadvective_sensible_heat"]
    ds_heat["SWMT_heat_frazil"] = ds["frazil_ice_heat"]

    ds_heat["SWMT_heat_mass_transfer"] = -ds["surface_exchange_flux_advective_mass_transfer_heat"]
    
    # Fixed line continuation syntax
    ds_heat["SWMT_heat_approx"] = (ds_heat["SWMT_heat_latent"] + ds_heat["SWMT_heat_longwave"] + \
                                   ds_heat["SWMT_heat_shortwave"] + ds_heat["SWMT_heat_sensible"] + \
                                    ds_heat["SWMT_heat_mass_transfer"] + ds_heat["SWMT_heat_frazil"])
    
    ds_heat["SWMT_heat_residual"] = ds_heat["SWMT_heat"] - ds_heat["SWMT_heat_approx"]
    
    return ds_heat

def get_SWMT_salt(ds): 
    
    ds_salt = ds["surface_ocean_flux_advective_negative_rhs_salt"] +\
                                 ds["surface_exchange_flux_salt"]

    ds_salt = ds_salt.rename("SWMT_salt").to_dataset()
    
    ds_salt["SWMT_salt_evaporation"] = -ds["surface_exchange_flux_advective_evaporation_salt"]
    ds_salt["SWMT_salt_rain_and_ice"] = -ds["surface_exchange_flux_advective_rain_and_ice_salt"]
    ds_salt["SWMT_salt_snow"] = -ds["surface_exchange_flux_advective_snow_salt"]
    ds_salt["SWMT_salt_rivers"] = -ds["surface_exchange_flux_advective_rivers_salt"]
    ds_salt["SWMT_salt_icebergs"] = -ds["surface_exchange_flux_advective_icebergs_salt"]
    ds_salt["SWMT_salt_sea_ice"] = -ds["surface_exchange_flux_advective_sea_ice_salt"]
    
    ds_salt["SWMT_salt_basal_salt"] = ds["surface_exchange_flux_nonadvective_basal_salt"]
    # Fixed line continuation syntax
    ds_salt["SWMT_salt_approx"] = (ds_salt["SWMT_salt_evaporation"] + ds_salt["SWMT_salt_rain_and_ice"] + \
                                   ds_salt["SWMT_salt_snow"] +  ds_salt["SWMT_salt_rivers"] + \
                                   ds_salt["SWMT_salt_sea_ice"] + ds_salt["SWMT_salt_icebergs"] + \
                                ds_salt["SWMT_salt_basal_salt"]   )
    ds_salt["SWMT_salt_residual"] = ds_salt["SWMT_salt"] - ds_salt["SWMT_salt_approx"]
    return ds_salt

def get_SWMT(ds):
    exact_SWMT = ds["boundary_fluxes"] - ds["bottom_flux_heat"]
    
    return xr.merge([get_SWMT_salt(ds), get_SWMT_heat(ds), exact_SWMT.rename("SWMT")])
    
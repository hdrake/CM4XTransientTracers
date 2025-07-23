import xarray as xr 
import numpy as np 
import pandas as pd
def approximate_z_on_boundaries_top_down(ds, dim = "sigma2"):
    thicknesses = ds.thkcello.fillna(0.0)

    eta = (0.0 * thicknesses.isel({f"{dim}_l":0})) + ds.zos
    eta.coords[f"{dim}_l"] = -100

    #calculate the vertical position of cell boundaries using 
    #cumsum
    cell_boundaries = xr.concat([eta, (-thicknesses)], dim = f"{dim}_l")
    cell_boundaries = cell_boundaries.cumsum(dim = f"{dim}_l")
    cell_boundaries = cell_boundaries.rename({f"{dim}_l":f"{dim}_i"})

    cell_boundaries.coords[f"{dim}_i"] = ds.coords[f"{dim}_i"]
    return cell_boundaries.where(ds.wet > 0) #needed for some reason, can't apply to 
        
def approximate_z_top_down(ds, dim = "sigma2"):
    thicknesses = ds.thkcello.fillna(0.0)
    thicknesses_cumsum = thicknesses.cumsum(dim=f"{dim}_l")
    
    cell_boundary_height = approximate_z_on_boundaries_top_down(ds, dim = dim)

    h_n = cell_boundary_height.isel({f"{dim}_i": slice(0, -1)})
    h_np1 = cell_boundary_height.isel({f"{dim}_i": slice(1, None)})
    h_np1.coords[f"{dim}_i"] = h_n.coords[f"{dim}_i"]

    cell_center_height = (h_np1 + h_n) / 2

    cell_center_height = cell_center_height.rename({f"{dim}_i":f"{dim}_l"})
    cell_center_height.coords[f"{dim}_l"] = ds.coords[f"{dim}_l"]
    return cell_center_height.where(ds.wet > 0)
    
def approximate_z_on_boundaries_bottom_up(ds, dim = "sigma2"):
    H = ds.deptho
    
    thicknesses = ds.thkcello.fillna(0.0)
    
    # Flip the thickness array to go from densest to least dense waters
    flipped_thicknesses = thicknesses.isel({f"{dim}_l": slice(None, None, -1)})

    #set H to be maximum depth within a given column
    h_bottom = (0.0 * flipped_thicknesses.isel({f"{dim}_l":0})) + H
    h_bottom.coords[f"{dim}_l"] = 100

    #calculate the vertical position of cell boundaries using 
    #cumsum
    cell_boundaries = xr.concat([h_bottom, (-flipped_thicknesses)], dim = f"{dim}_l")
    cell_boundaries = cell_boundaries.cumsum(dim = f"{dim}_l")
    cell_boundaries = cell_boundaries.rename({f"{dim}_l":f"{dim}_i"})

    #force grid to be ordered from least to most dense 
    cell_boundaries = cell_boundaries.isel({f"{dim}_i": slice(None, None, -1)})

    #make negative values mean "below" surface
    cell_boundaries *= -1 #make     
    cell_boundaries.coords[f"{dim}_i"] = ds.coords[f"{dim}_i"]
    return cell_boundaries
    
def approximate_z_bottom_up(ds, dim="sigma2_l"):
    H = ds.deptho

    thicknesses = ds.thkcello.fillna(0.0)
    # Flip the thickness array to go from densest to least dense
    flipped_thicknesses = thicknesses.isel({dim: slice(None, None, -1)})
    
    h_bottom = (0.0 * flipped_thicknesses.isel({dim:0})) + H
    h_bottom.coords[dim] = 100
    
    cell_boundaries = xr.concat([h_bottom, (-flipped_thicknesses)], dim = dim).cumsum(dim = dim)
    
    h_np1 = cell_boundaries.isel({dim : slice(0, -1)}) 
    h_n = cell_boundaries.isel({dim : slice(1, None)}) 
    h_np1.coords[dim] = h_n.coords[dim]
    
    z_flipped = ((h_np1 + h_n) / 2).where(thicknesses > 0) #midpoint between cell interfaces
    z = z_flipped.isel({dim: slice(None, None, -1)})
    z *= -1 #make 
    return z
    
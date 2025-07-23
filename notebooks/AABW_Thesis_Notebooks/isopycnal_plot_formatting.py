import matplotlib.patches as  mpath
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.pyplot as plt


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

import numpy as np
import matplotlib.patches as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_antarctic_plot(ds, ax, cmap = None, norm = None, 
                        exp = "forced", variable = "mass_tendency", 
                        mass_flux_units_conversion = 1, 
                       draw_gridlines = True, mask_small_values = True):
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    
    ds_convert =  mass_flux_units_conversion * ds.sel(exp = exp)[variable]
    # wmt_mean_budget = wmt_mean_budget.where(wmt_mean_budget != 0.0)
    
    if mask_small_values: 
        ds_convert = ds_convert.where(np.abs(ds_convert) > 1e-8)
        
    cm = ax.pcolormesh(ds_convert.geolon, ds_convert.geolat, 
                         ds_convert, cmap = cmap, transform=ccrs.PlateCarree(), 
                        norm = norm)
    ax.coastlines();
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
   
    if draw_gridlines: 
        # Draw meridian lines with labels around circular boundary
        gls = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, \
                        xlocs=range(-180,171,45), ylocs=range(-90,0,10), \
                        color='gray', alpha=0.25, linestyle='-', zorder=10, 
                        xlabel_style = {"fontsize":9.5})
    
        
    ax.set_boundary(circle, transform=ax.transAxes)

    return cm
    
def plot_antarctic(
    ds, 
    ax, 
    *,
    cmap=None, 
    norm=None, 
    mask_threshold=1e-8, 
    draw_gridlines=True,
    circle_radius=0.5,
    circle_center=(0.5, 0.5),
    gridline_lons=range(-180, 181, 30),
    gridline_lats=range(-90, 0, 10),
):
    """
    Plot a south‐polar (Antarctic) field from an xarray Dataset.

    Parameters
    ----------
    ds : xarray.Dataset or DataArray
        Must contain coords "geolon" and "geolat" and a data variable
        indexed by ("exp", ...) from which `variable` is selected.
    ax : GeoAxesSubplot
        Should already be created with a south‐polar projection, e.g.:
            fig, ax = plt.subplots(subplot_kw={
                "projection": ccrs.SouthPolarStereo()})
    exp : str
        Name of the experiment/coordinate value to select along the "exp" dimension.
    variable : str
        Name of the data variable in ds to plot.
    units_factor : float
        Scalar to multiply the raw field by (e.g. unit conversion).
    cmap : matplotlib Colormap, optional
        Colormap to use for `pcolormesh`. If None, uses Matplotlib default.
    norm : matplotlib.colors.Normalize, optional
        Normalization object for color scaling. If None, auto‐scales.
    mask_threshold : float or None
        If not None, masks out |field| <= threshold. Set to None to skip masking.
    draw_gridlines : bool
        If True, draw meridian/parallel gridlines with labels.
    circle_radius : float
        Radius (in axes‐fraction units) of the circular boundary around Antarctica.
    circle_center : tuple of float
        Center (x, y) in axes‐fraction coordinates of the circular boundary.
    gridline_lons : iterable of int or float
        Longitudes at which to draw meridians.
    gridline_lats : iterable of int or float
        Latitudes at which to draw parallels (should be negative or zero for south).
    
    Returns
    -------
    QuadMesh
        The `QuadMesh` returned by `ax.pcolormesh`, so you can add a colorbar later.
    """
    # Select and convert units
    field = ds

    # Optionally mask very small values
    if mask_threshold is not None:
        field = field.where(np.abs(field) > mask_threshold)

    # Plot with pcolormesh, assuming ds.geolon/geolat are in degrees
    mesh = ax.pcolormesh(
        field.geolon, 
        field.geolat, 
        field, 
        cmap=cmap, 
        norm=norm, 
        transform=ccrs.PlateCarree()
    )

    # Coastlines & land shading
    ax.coastlines(resolution="110m", linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgrey", zorder=1)

    # Optionally draw gridlines
    if draw_gridlines: 
        # Draw meridian lines with labels around circular boundary
        gls = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, \
                        xlocs=range(-180,171,45), ylocs=range(-90,0,10), \
                        color='gray', alpha=0.25, linestyle='-', zorder=10,) 
                        # xlabel_style = {"fontsize":9.5})

    # Draw a circular axes boundary around the south pole
    theta = np.linspace(0, 2 * np.pi, 200)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle_path = mpath.Path(verts * circle_radius + circle_center)
    ax.set_boundary(circle_path, transform=ax.transAxes)

    return mesh

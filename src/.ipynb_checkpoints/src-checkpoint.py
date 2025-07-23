import xarray as xr 
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import numpy as np
import geopy
from geopy import distance
import xesmf as xe
# import sectionate
import re
import gsw


rootdir = "/vortexfs1/home/anthony.meza/scratch/CM4XTransientTracers"
slurmscratchdir = rootdir + "/slurmscratch"
plotsdir = lambda x="": rootdir + "/figures/" + x
datadir = lambda x="": rootdir + "/data/" + x

from meridional_streamfunction import *

from CM4XUtilsFunctions import *
from approximate_z import * 
from remapping import * 
from load_native_files import * 
# def approximate_z(ds, dim = "zl"):
#     tmp = ds.thkcello.cumsum(dim = dim)
#     #average between 0 and cell bottom
#     tmp1 = tmp.isel({dim: 0}) / 2 
#     #get top of cell
#     tmp2 = tmp.isel({dim : slice(0, -1)}) 
#     #get bottom of cell
#     tmp3 = tmp.isel({dim : slice(1, None)}) 
#     #make sure cell interfaces are on same coordinate
#     tmp2.coords[dim] = tmp3.coords[dim]
#     #take average between interfaces
#     tmp4 = (tmp2 + tmp3) / 2

#     ds["z"] = xr.concat([1. * tmp1, 1. * tmp4], dim = dim)    
#     # ds["z_bottom"] = 1. * tmp

#     return ds


def update_thermodynamic_variables(ds, zname = None):
    if zname is None:
        z = approximate_z_top_down(ds)
    else:
        z = ds["z_l"]
    
    p_ref = xr.apply_ufunc(
        gsw.p_from_z, z, ds.geolat, 0, 0, dask="parallelized"
    )
    
    sa = xr.apply_ufunc(
        gsw.SA_from_SP,
        ds["so"],
        p_ref,
        ds["geolon"],
        ds["geolat"],
        dask="parallelized",
    )
    sa.attrs.update({
        "long_name": "Absolute Salinity",
        "standard_name": "sea_water_absolute_salinity",
        "units": "g kg-1",
        "description": "Absolute Salinity (SA) from Practical Salinity (SP) via TEOS‑10."
    })
    
    ct = xr.apply_ufunc(
    gsw.CT_from_pt,
    sa,
    ds["thetao"],
    dask="parallelized")
    ct.attrs.update({
        "long_name": "Conservative Temperature",
        "standard_name": "sea_water_conservative_temperature",
        "units": "degC",
        "description": "Conservative Temperature (CT) from potential temperature (pt) via TEOS‑10."
    })

    ds["ct"] = ct
    ds["sa"] = sa 
    ds = ds.drop_vars(["thetao", "so"])
    return ds
    
def get_thetao(g_ds): #get potential temperature for a GLODAPP section 
    g_ds["z"] = -np.abs(g_ds.depth)
    g_ds = g_ds.rename({"salinity":"so"})

    g_ds['p'] = xr.apply_ufunc(
        gsw.p_from_z, g_ds.z, g_ds.lat, 0, 0, dask="parallelized"
    )
    g_ds['sa'] = xr.apply_ufunc(
        gsw.SA_from_SP,
        g_ds.so,
        g_ds.p,
        g_ds.lon,
        g_ds.lat,
        dask="parallelized",
    )

    g_ds['thetao'] = xr.apply_ufunc(
        gsw.pt0_from_t,
        g_ds.sa,
        g_ds.theta,
        g_ds.p,
        dask="parallelized",
    )
    return g_ds

def calc_sigma2(ds, z_name = "z", lats = None, lons = None, mask_inactive_layers = True): 
    
    if lats is None:
        lats = ds.geolat
    if lons is None:
        lons = ds.geolon
    
    print("using geolatlon")
    
    z = -np.abs(ds[z_name])
    z = z.where(z <= 0).fillna(0.)
    if mask_inactive_layers: 
        thicknesses = ds.thkcello.fillna(0.0)
        ds = ds.where(thicknesses > 0)
    
    p = xr.apply_ufunc(
        gsw.p_from_z, z, lats, 0., 0., dask="parallelized"
    )

    sa = xr.apply_ufunc(
        gsw.SA_from_SP,
        ds.so,
        p,
        lons,
        lats,
        dask="parallelized",
    )
    
    ct = xr.apply_ufunc(
        gsw.CT_from_t,
        sa,
        ds.thetao,
        p,
        dask="parallelized"
    )

    sigma2 = xr.apply_ufunc(
        gsw.sigma2,
        sa,
        ct,
        dask="parallelized"
    )

    return sigma2

def get_sigma2(ds, keep_vars = False): 
    ds['p'] = xr.apply_ufunc(
        gsw.p_from_z, ds.z, ds.geolat, 0, 0, dask="parallelized"
    )


    ds['sa'] = xr.apply_ufunc(
        gsw.SA_from_SP,
        ds.so,
        ds.p,
        ds.geolon,
        ds.geolat,
        dask="parallelized",
    )
    ds['ct'] = xr.apply_ufunc(
        gsw.CT_from_pt,
        ds.sa,
        ds.thetao,
        ds.p,
        dask="parallelized"
    )

    ds['sigma2'] = xr.apply_ufunc(
        gsw.sigma2,
        ds.sa,
        ds.ct,
        dask="parallelized"
    )
    if keep_vars: 
        return ds
    else: 
        return ds.drop_vars(["p", "sa", "ct"])
    
    

def get_sigma2_at_surface(ds, keep_vars = False): 
    p_ref = xr.apply_ufunc(
        gsw.p_from_z, ds.zos, ds.geolat, 0, 0, dask="parallelized"
    )
    
    sa = xr.apply_ufunc(
        gsw.SA_from_SP,
        ds.sos,
        p_ref,
        ds.geolon,
        ds.geolat,
        dask="parallelized",
    )
    # ct = xr.apply_ufunc(
    #     gsw.CT_from_pt,
    #     sa,
    #     ds.tos,
    #     dask="parallelized"
    # )
    ct = ds.tos
    sigma2_surf = xr.apply_ufunc(
        gsw.sigma2,
        sa,
        ct,
        dask="parallelized"
    )

    ds["sa"] = sa
    ds["ct"] = ct
    ds["rho2"] = sigma2_surf

    
    return ds
    

def interpolate_values(ds, original_grid, objective_grid, kind = "linear"):
    unique_points, unique_inds = np.unique(original_grid.values, return_index = True)
    
    if len(unique_inds) > 1:
        ds_subset = ds.values[unique_inds]
        where_nonan = ~np.isnan(unique_points + ds_subset)

        if len(ds_subset[where_nonan]) > 1:
            interp = interp1d(unique_points[where_nonan], ds_subset[where_nonan], 
                              bounds_error = False, kind = kind, 
                              fill_value=np.nan, assume_sorted = False)
            return interp(objective_grid)
        else:
            print("hi")
            return np.nan * objective_grid
    else: 
        return np.nan * objective_grid
    
    
def interpolate_section(ds, objective_grid, 
                        interp_coord = "z", 
                        iterate_coord = "distance",
                        kind = "linear"):
    
    interp_ds = xr.Dataset(coords={interp_coord:objective_grid, 
                                   iterate_coord:ds[iterate_coord].values})
        
    for key,value in ds.drop(interp_coord).items():
        tmp_array = np.zeros((len(objective_grid), len(ds[iterate_coord])))
        for i in range(len(ds[iterate_coord])):
            tmp = interpolate_values(ds[key].isel({iterate_coord:i}), 
                                     ds[interp_coord].isel({iterate_coord:i}), 
                                     objective_grid, kind = kind)
            tmp_array[:, i] = 1 * tmp

        interp_ds = interp_ds.assign({key:([interp_coord,iterate_coord], 1 * tmp_array)})

    return interp_ds

def match_GLODAPP_WOCE(gdset, wdset, time = '2005-2014'):
    years = re.findall(r'\d+', wdset.attrs['all_years_used'])
    ecs = wdset.attrs['expocode'].split(',')
                                         
    ecs_i = list(set(ecs) & set(list(gdset['G2expocode'].unique())))
    print('Available cruise data : '+' '.join(ecs_i))
                                         
    # Subset GLODAP data for available cruise
    cond = gdset['G2expocode']==''
    for ec in ecs:
        cond += (gdset['G2expocode']==ec)
    dfo_all = gdset[cond]
                                         
    # Find cruises that fall within model averaging period
    year_bnds = [int(a) for a in (time.split("-"))]
    leeway = 1
    cond = dfo_all['G2year']==''
    for year in dfo_all['G2year'].unique():
        if year_bnds[0]-leeway <= year <= year_bnds[1]+leeway:
            cond += (dfo_all['G2year']==year)
    dfo = dfo_all[cond]
    if len(dfo)==0:
        yearstr = [str(e) for e in dfo_all['G2year'].unique()]
        raise Exception('No suitable GLODAP data for selected WOCE line and year\n'+
                       'Avaliable data: '+' '.join(ecs_i)+'\n'+
                       'For years: '+' '.join(yearstr))
    return dfo
                                         
def calc_dz(depth):
    d1 = depth.copy()
    zeros = xr.zeros_like(d1.isel(n=0)).expand_dims({'ni':[0]})
    ids = 0.5*(d1+d1.shift(n=-1)).rename({'n':'ni'}).assign_coords({'ni':d1['n'].values+1})
    idepth = xr.concat([zeros,ids],dim='ni')
    dz = idepth.diff('ni')
    dz.loc[{'ni':len(dz['ni'])}]=dz.isel(ni=-2)
    dz = dz.rename({'ni':'n'}).assign_coords({'n':d1['n']})
    return dz

def calc_dz_1d(depth):
    ids = 0.5*(depth[1:]+depth[:-1])
    idepth = np.append(np.array([0]),ids)
    dz = np.diff(idepth)
    dz = np.append(dz,dz[-1])
    return dz
    
def calc_dx(distance):
    d1 = distance.values
    mid = 0.5*(d1[1:]+d1[:-1])
    mid = np.append(-mid[0],mid)
    dx = np.diff(mid)
    dx = np.append(dx,dx[-1])
    return xr.DataArray(dx,dims=distance.dims,coords=distance.coords)    

def extract_GLODAPP_cruises(dfo):                         
    glodap_variables = ['cfc11','cfc12','sf6','theta','salinity']
    sections = {}
    cruises = dfo['G2cruise'].unique()
    for cruise in cruises:
        dfonow = dfo[dfo['G2cruise']==cruise]
        stations = dfonow['G2station'].unique()

        ns = len(stations)

        # Find max no. measurements at each station
        # And distance along section
        maxn = 0
        distance = np.zeros(shape=(ns,))
        avg_time = np.nanmean(dfonow["G2year"].values)

        for i,station in enumerate(stations):
            dfostation = dfonow[dfonow['G2station']==station]
            if i==0:
                lonlast = dfostation['G2longitude'].mean()
                latlast = dfostation['G2latitude'].mean()
            else:
                lon = dfostation['G2longitude'].mean()
                lat = dfostation['G2latitude'].mean()
                distance[i]=geopy.distance.distance((lat,lon),(latlast,lonlast)).km
            if len(dfostation)>maxn:
                maxn = len(dfostation)

        nb = maxn

        # Populate numpy arrays
        lons = np.full(shape=(ns,),fill_value=np.nan)
        lats = np.full(shape=(ns,),fill_value=np.nan)
        depths = np.full(shape=(ns,nb),fill_value=np.nan)
        dzs = np.full(shape=(ns,nb),fill_value=np.nan)
        variables = {}
        for gv in glodap_variables:
            variables[gv]=np.full(shape=(ns,nb),fill_value=np.nan)
        for i,station in enumerate(stations):
            dfostation = dfonow[dfonow['G2station']==station]
            n = len(dfostation)

            lon = dfostation['G2longitude'].mean()
            lat = dfostation['G2latitude'].mean()
            depth = dfostation['G2depth']
            lons[i,]=lon
            lats[i,]=lat
            depths[i,:n]=depth
            dzs[i,:n] = calc_dz_1d(depth.values)

            for variable in variables.keys():
                tmp = dfostation['G2'+variable]
                # Get the variable at unique depths by creating a dictionary
                dt = dict(zip(depth,tmp))
                # Recreate variable at all depths
                tmpnow = [dt[d] for d in depth]
                variables[variable][i,:n] = tmpnow

        # Assign to dataset
        section = xr.Dataset({
            'lon':xr.DataArray(lons,dims=('distance'),coords={'distance':np.cumsum(distance)}),
            'lat':xr.DataArray(lats,dims=('distance'),coords={'distance':np.cumsum(distance)}),
            'depth':xr.DataArray(depths,dims=('distance','n'),coords={'distance':np.cumsum(distance)}),
            'dz':xr.DataArray(dzs,dims=('distance','n'),coords={'distance':np.cumsum(distance)})}
        )
        section["time"] = avg_time
        section["yearmonth"] = avg_time + np.nanmean(dfonow["G2month"].values / 12)

        for variable,array in variables.items():
            section[variable] = xr.DataArray(array,dims=('distance','n'),coords={'distance':np.cumsum(distance)})

        # Calculate appropriate "grid variables"
        section['dx'] = calc_dx(section['distance'])

        sections[cruise]=section
                                         
    
    return sections
                                         
                                         
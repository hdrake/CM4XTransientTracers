#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import xarray as xr
import warnings
import matplotlib.pyplot as plt
import os


# In[3]:


import gfdl_utils.core as gu


# In[4]:


models = {
    "CM4Xp25"  : {"historical":"odiv-231", "ssp5":"odiv-232"},
    "CM4Xp125" : {"historical":"odiv-255", "ssp5":"odiv-293"},
}

pre = "/archive/Raphael.Dussin/"
pp_dict = {
    "odiv-230": f"{pre}FMS2019.01.03_devgfdl_20221223/CM4_piControl_c192_OM4p25_v8/gfdl.ncrc4-intel18-prod-openmp/pp",
    "odiv-231": f"{pre}FMS2019.01.03_devgfdl_20221223/CM4_historical_c192_OM4p25/gfdl.ncrc4-intel18-prod-openmp/pp",
    "odiv-232": f"{pre}FMS2019.01.03_devgfdl_20221223/CM4_ssp585_c192_OM4p25/gfdl.ncrc4-intel18-prod-openmp/pp",
    "odiv-209": f"{pre}FMS2019.01.03_devgfdl_20210706/CM4_piControl_c192_OM4p125_v7/gfdl.ncrc4-intel18-prod-openmp/pp",
    "odiv-313": f"{pre}FMS2019.01.03_devgfdl_20230608/CM4_piControl_c192_OM4p125_v8/gfdl.ncrc5-intel22-prod-openmp/pp/",
    "odiv-255": f"{pre}FMS2019.01.03_devgfdl_20230608/CM4_historical_c192_OM4p125/gfdl.ncrc5-intel22-prod-openmp/pp",
    "odiv-293": f"{pre}FMS2019.01.03_devgfdl_20230608/CM4_ssp585_c192_OM4p125/gfdl.ncrc5-intel22-prod-openmp/pp",
}

pre = "/archive/Raphael.Dussin/datasets/"
sg_dict = {
    "OM4p25": f"{pre}OM4p25/c192_OM4_025_grid_No_mg_drag_v20160808_unpacked/ocean_hgrid.nc",
    "OM4p125": f"{pre}OM4p125/mosaic_c192_om4p125_bedmachine_v20210310_hydrographyKDunne20210614_unpacked/ocean_hgrid.nc"
}

def get_pathDict(run, time="*", snap=False):
    pp = pp_dict[run]
    return {
        "pp": pp,
        "ppname": f"ocean_inert_month",
        "out": "ts",
        "local": "monthly/5yr",
        "time": time,
        "add": "*",
    }


# In[5]:


tracers = ['cfc11', 'cfc12', 'sf6']
g_per_mol = {
    "cfc11": 137.37,
    "cfc12": 120.91,
    "sf6": 146.06
}
Gg_per_g = 1.e-9
sec_per_year = 365.25 * 24 * 60 * 60
sec_per_nsec = 1.e-9


# In[ ]:

data_dict = {}

for model, exps in models.items():
    print(f"Loading inert tracer diagnostics for {model}")
    hist = gu.open_frompp(**get_pathDict(exps["historical"]), dmget=True, chunks={'time':1})
    ssp5 = gu.open_frompp(**get_pathDict(exps["ssp5"]), dmget=True, chunks={'time':1})
    ds = xr.concat([hist, ssp5], dim="time", combine_attrs="override")

    path_dict = get_pathDict(exps["historical"])
    og = xr.open_dataset(gu.get_pathstatic(path_dict["pp"], path_dict["ppname"]))
    
    inv_path = f"../data/transient_tracer_inventory_{model}-SSP585.nc"
    if not(os.path.exists(inv_path)):
        print(f"Computing globally-integrated inventory for {model}: ")
        inv = xr.Dataset()
        for tr in tracers:
            print(tr, end=", ")
            Δt = ds['average_DT'].astype("float64")*sec_per_nsec

            inv[f'{tr}_volumeint'] = (ds[tr]*ds['volcello']).sum(['xh', 'yh', 'zl']).compute()

            timax = inv[f'{tr}_volumeint'].argmax().values
            inv[f'fg{tr}_sink'] = (ds[f'fg{tr}']*Δt).isel(time=slice(0, timax)).sum('time')
            inv[f'fg{tr}_source'] = (ds[f'fg{tr}']*Δt).isel(time=slice(timax, None)).sum('time')

            inv[f'fg{tr}_areaint_NH'] = (
                (ds[f'fg{tr}']*og['areacello'].where(og['geolat']>=0)).sum(['xh', 'yh']).compute()
            )
            inv[f'fg{tr}_areaint_SH'] = (
                (ds[f'fg{tr}']*og['areacello'].where(og['geolat']<0)).sum(['xh', 'yh']).compute()
            )
            inv[f'fg{tr}_areaint'] = inv[f'fg{tr}_areaint_SH'] + inv[f'fg{tr}_areaint_NH']
            inv[f'fg{tr}_areatimeint'] = (inv[f'fg{tr}_areaint']*Δt).cumsum("time").compute()
        print("")
        inv.to_netcdf(inv_path)
    else:
        print(f"Loading globally-integrated inventory for {model}.")
        inv = xr.open_dataset(inv_path)
        
    data_dict[model] = {"ds":ds, "og":og, "inv":inv}
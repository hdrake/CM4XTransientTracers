#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import warnings
import matplotlib.pyplot as plt
import os


import gfdl_utils.core as gu

pre = "/archive/Raphael.Dussin/"
pp = f"{pre}FMS2019.01.03_devgfdl_20230608/CM4_piControl_c192_OM4p125_v8/gfdl.ncrc5-intel22-prod-openmp/pp/"

pp_continued = f"{pre}FMS2019.01.03_devgfdl_20241030/CM4_piControl_c192_OM4p125_v8followup/gfdl.ncrc5-intel23-prod-openmp/pp/"

pre = "/archive/Raphael.Dussin/datasets/"
sg = f"{pre}OM4p125/mosaic_c192_om4p125_bedmachine_v20210310_hydrographyKDunne20210614_unpacked/ocean_hgrid.nc"

def get_pathDict(pp, time="*"):
    return {
        "pp": pp,
        "ppname": f"ocean_inert_month",
        "out": "ts",
        "local": "monthly/5yr",
        "time": time,
        "add": "*",
    }

tracers = ['cfc11', 'cfc12', 'sf6']
g_per_mol = {
    "cfc11": 137.37,
    "cfc12": 120.91,
    "sf6": 146.06
}
Gg_per_g = 1.e-9
sec_per_year = 365.25 * 24 * 60 * 60
sec_per_nsec = 1.e-9

model = "CM4Xp125"
exp = "piControl-continued"

print(f"Loading inert tracer diagnostics for {model}")
ds = gu.open_frompp(**get_pathDict(pp_continued), chunks={'time':1})

path_dict = get_pathDict(pp)
og = xr.open_dataset(gu.get_pathstatic(path_dict["pp"], path_dict["ppname"]))

inv_path = f"../data/interim/transient_tracer_inventory_{model}-{exp}.nc"
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

        inv[f'fg{tr}_areaint_NH'] = (ds[f'fg{tr}']*og['areacello'].where(og['geolat']>=0)).sum(['xh', 'yh']).compute()
        inv[f'fg{tr}_areaint_SH'] = (ds[f'fg{tr}']*og['areacello'].where(og['geolat']<0)).sum(['xh', 'yh']).compute()
        inv[f'fg{tr}_areaint'] = inv[f'fg{tr}_areaint_SH'] + inv[f'fg{tr}_areaint_NH']
        inv[f'fg{tr}_areatimeint'] = (inv[f'fg{tr}_areaint']*Δt).cumsum("time").compute()
    print("")
    inv.to_netcdf(inv_path)
else:
    print(f"Loading globally-integrated inventory for {model}.")
    inv = xr.open_dataset(inv_path)

#!/usr/bin/env python
# coding: utf-8

import doralite
import gfdl_utils.core as gu
import CM4Xutils

import dask
import xarray as xr
import warnings
import matplotlib.pyplot as plt
import os
import cftime

def get_pathDict(pp, time="*", tracers="*"):
    return {
        "pp": pp,
        "ppname": f"ocean_inert_month",
        "out": "ts",
        "local": "monthly/5yr",
        "time": time,
        "add": tracers,
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
odivs = CM4Xutils.exp_dict[model]

for exp in ["piControl", "piControl-continued"]:
    
    print(f"Loading inert tracer diagnostics for {model}-{exp}")
    
    # Define safe chunks
    #chunks = {'time': 1, 'xh': 180, 'yh': 140, 'zl': 25}
    
    open_mfdataset_kwargs = {
        "dmget":True,
        "engine":'netcdf4',
        "chunks":{}, 
        "combine":"nested",
        "concat_dim":"time"
    }

    meta = doralite.dora_metadata(odivs[exp])
    pp = meta['pathPP']

    # Load and truncate the first dataset manually using a boolean mask
    ppdict = get_pathDict(pp, time="*", tracers=tracers + ["volcello"])
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds = gu.open_frompp(**ppdict, **open_mfdataset_kwargs)
    ds["volcello"] = ds["volcello"]#.chunk(chunks)

    # Load grid
    og = xr.open_dataset(gu.get_pathstatic(ppdict["pp"], ppdict["ppname"]))

    # Output path
    inv_path = f"../data/interim/transient_tracer_inventory_{model}-{exp}.nc"
    
    if not os.path.exists(inv_path):
        print(f"Computing globally-integrated tracer inventory for {model}-{exp}: ")
        inv = xr.Dataset()

        print("volcello", end=", ")
        inv[f'volumeint'] = ds['volcello'].sum(['xh', 'yh', 'zl']).compute()
        for tr in tracers:
            print(tr, end=", ")
    
            # compute globally-integrated tracer content
            #ds[tr] = ds[tr].chunk(chunks)
            inv[f'{tr}_volumeint'] = (ds[tr] * ds['volcello']).sum(['xh', 'yh', 'zl']).compute()
            
        inv.to_netcdf(inv_path)
    else:
        print(f"Loading globally-integrated inventory for {model}.")
        inv = xr.open_dataset(inv_path)

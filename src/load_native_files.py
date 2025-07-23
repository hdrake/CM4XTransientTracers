import glob
import xarray as xr
def collect_budget_files(collect_spinup=False, model="CM4Xp125"): 
    """
    Gather a sorted list of budget files for a given model, optionally including spinup files.

    Parameters
    ----------
    collect_spinup : bool, optional
        If True, return all files (including spinup). If False, skip the first 20 files,
        which are assumed to be spinup for this particular dataset layout.
    model : str, optional
        Model name, either "CM4Xp125" (1/8°) or "CM4Xp25" (1/4°). Files for both models
        have been remapped onto a 1.5° grid, and are named accordingly in their directory.

    Returns
    -------
    list of str
        A sorted list of filesystem paths to the budget files. If collect_spinup is False,
        returns only files from index 20 onward (i.e., excluding the first 20 spinup files).
    """

    datadir = lambda x="": (
        "/vortexfs1/home/anthony.meza/scratch/"
        "CM4XTransientTracers/data/model/budgets_sigma2_1p5/" + x
    )
    # Use glob to find all filenames in that directory matching the model prefix
    datafiles = glob.glob(datadir(f"{model}*"))
    # Sort them alphabetically (usually corresponds to chronological order if filenames encode date)
    datafiles = sorted(datafiles)

    if collect_spinup:
        # If the caller specifically wants the spinup files, just return everything
        return datafiles
    else:
        # Otherwise, drop the first 20 entries (assumed spinup) and return the remainder
        return datafiles[20:]


def collect_tracer_files(model="CM4Xp125"):
    """
    Gather tracer files for different experiment categories (spinup, historical+SSP585, piControl)
    for a given model. Returns a dict grouping file lists by experiment type.

    Parameters
    ----------
    model : str, optional
        Model name, either "CM4Xp125" (1/8°) or "CM4Xp25" (1/4°). Files are remapped onto 1.5°,
        and filenames include keywords like "spinup", "historical", "ssp585", or "piControl".

    Returns
    -------
    dict
        Keys are:
          - "spinup":    Sorted list of spinup files
          - "forced":    Sorted list combining historical + SSP585 files
          - "control":   Sorted list of piControl files (excluding those that overlap with spinup)
    """

    datadir = lambda x="": (
        "/vortexfs1/home/anthony.meza/scratch/"
        "CM4XTransientTracers/data/model/tracers_sigma2_1p5/" + x
    )

    spinup_datafiles = sorted(glob.glob(datadir(f"{model}*spinup*")))
    historical_datafiles = sorted(glob.glob(datadir(f"{model}*historical*")))
    ssp585_datafiles = sorted(glob.glob(datadir(f"{model}_ssp585*")))
    picontrol_datafiles = sorted(glob.glob(datadir(f"{model}*piControl*")))

    # Remove any overlap between piControl and spinup sets:
    # Spinup files appear in piControl, so subtract to avoid duplication.
    picontrol_datafiles = sorted(list(set(picontrol_datafiles) - set(spinup_datafiles)))

    # Build a dictionary grouping the file lists by experiment type
    expt_datafiles = {
        "spinup":  spinup_datafiles,                                # pure spinup run
        "forced":  historical_datafiles + ssp585_datafiles,         # transient forced runs
        "control": picontrol_datafiles                              # piControl run (no spinup overlap)
    }

    return expt_datafiles

def infer_budget_file_from_tracer_file(tracer_path):
    """
    Given a tracer Zarr path, compute the corresponding budget Zarr path.
    - For tracer experiments containing "piControl" (or "piControl-spinup"), 
      the year range is in model years (0001→1750, 0005→1754, etc.). 
    - For "historical" or "ssp585", the year range already matches calendar years.
    Returns the first matching budget path or None if not found.
    """
    basename = tracer_path.split("/")[-1]
    if not basename.endswith(".zarr"):
        return None

    # Remove the ".zarr" suffix
    name_no_ext = basename[:-5]  # e.g. "CM4Xp125_piControl-spinup_tracers_sigma2_0001-0005"

    # Split off "<prefix>_tracers_sigma2_<year1>-<year2>"
    try:
        prefix, year_range = name_no_ext.rsplit("_tracers_sigma2_", 1)
    except ValueError:
        return None

    # prefix = e.g. "CM4Xp125_piControl-spinup" or "CM4Xp125_historical" or "CM4Xp125_ssp585"
    parts = prefix.split("_", 1)
    if len(parts) != 2:
        return None
    model, exp = parts  # model="CM4Xp125", exp="piControl-spinup" or "historical" or "ssp585"

    # Get the budget files for this model on the fly
    budget_files = collect_budget_files(collect_spinup=True, model=model)
    # Parse the two integers in the year range
    try:
        start_str, end_str = year_range.split("-")
        start_yr = int(start_str)
        end_yr = int(end_str)
    except ValueError:
        return None

    # If experiment is piControl or piControl-spinup => convert model years to real years
    if "piControl" in exp:
        real_start = start_yr + 1749  # 0001 → 1750, etc.
        real_end   = end_yr   + 1749
    else:
        # historical & ssp585 already use calendar years
        real_start = start_yr
        real_end   = end_yr

    # Build the expected budget filename, e.g. "CM4Xp125_budgets_sigma2_1750-1754.zarr"
    bu_name = f"{model}_budgets_sigma2_{real_start:04d}-{real_end:04d}.zarr"

    # Find it in budget_files
    matches = [b for b in budget_files if bu_name in b]
    return matches[0] if matches else None

def read_tracer_and_zos_from_budget(tracer_path, tracer_preprocess = None, print_budget_path = False):
    """
    Open a tracer Zarr, find its matching budget Zarr, then add the correct 'zos'
    slice from the budget (exp="forced" for historical/SSP, exp="control" for spinup/piControl).
    Returns the updated tracer Dataset, or None if no budget match is found.
    """
    # 1. figure out which budget Zarr goes with this tracer
    bu_path = infer_budget_file_from_tracer_file(tracer_path)
    if bu_path is None:
        print(f"⚠️  No budget Zarr found for {tracer_path}")
        return None
        
    if print_budget_path:
        print(bu_path)

    # 2. open tracer and budget Datasets
    ds_tr = xr.open_mfdataset(
        tracer_path,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        parallel=True,
        preprocess = tracer_preprocess,
        engine="zarr",
    )
    
    ds_bu = xr.open_mfdataset(
        bu_path,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        parallel=True,
        preprocess = lambda ds: ds[["zos"]],
        engine="zarr",
    )

    # 3. decide which 'exp' to grab from the budget
    fname = tracer_path.split("/")[-1]
    if "historical" in fname or "ssp585" in fname:
        exp_key = "forced"
    else:
        # covers both "spinup" and "piControl"
        exp_key = "control"
        ds_tr.coords["time"] = ds_bu.time.values

    # 4. inject 'zos' from the budget into the tracer Dataset
    ds_tr["zos"] = ds_bu.sel(exp=exp_key)["zos"]

    return ds_tr

import numpy as np
import CM4Xutils #needed to run: pip install nc-time-axis
import xarray as xr
from approximate_z import * 
from CM4XUtilsFunctions import * 
import warnings

def make_tanh_grid(N=29, H=7000, alpha=2, eta=2.0):
    """
    Build N+1 interface depths z_k spanning from +eta_max down to -H_max,
    using a tanh‐stretch for finer resolution near the top.

    Parameters
    ----------
    N        : int
        Number of vertical levels (so there are N+1 interfaces).
    H_max    : float
        Maximum (constant) ocean depth in meters. Bottom = -H_max.
    eta_max  : float
        Maximum (constant) surface elevation in meters. Top = +eta_max.
    alpha    : float
        Stretch parameter (>0). Larger alpha => stronger clustering near eta_max.

    Returns
    -------
    z : numpy.ndarray of length N+1
        Interface depths (positive upward), from z[0]=eta_max down to z[N]=-H_max.
    """
    # 1) total span from +eta_max down to -H_max
    L = H + eta

    # 2) uniform parameter s_k in [0,1]
    s = np.linspace(0.0, 1.0, N + 1)   # s[0]=0 (top), s[N]=1 (bottom)

    # 3) build a tanh‐fraction that stays near zero for small s, then grows toward 1 as s→1:
    C = 1.0 - np.tanh(alpha * (1.0 - s)) / np.tanh(alpha)

    # 4) map C ∈ [0,1] onto z ∈ [ eta_max  →  -H_max ]
    z = eta - L * C

    return np.sort(z)
    
def remap_sigma_to_depth(ds, z_i = None, z_l = None, print_warning = False): 
    if z_i is None: 

        N = 50; H = 7000; eta = 25 #N must be odd to get an even number of faces
        if print_warning:
            print("no z_i provided")
            print("using a default setup")
            print(f"N = {N}; H = {H} meters; eta = {eta} meters")

        z_i = make_tanh_grid(N = N, H = H, eta = eta)
                                             #np.arange(-6500, 251, 250)
        z_l = (z_i[1:] + z_i[0:-1]) / 2
    
    ds = ds.assign_coords({"z_l": z_l, "z_i":z_i})

    ds["z"] = approximate_z_on_boundaries_top_down(ds, dim = "sigma2")
    ds = ds.chunk({"sigma2_l":-1, "sigma2_i":-1, "time":1})
    grid = CM4Xutils.ds_to_grid(ds, Zprefix = "sigma2")
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        ds_remap = remap_vertical_coord_custom("z", ds, grid, ds["z"])
        return ds_remap
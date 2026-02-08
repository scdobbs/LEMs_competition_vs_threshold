import TopoAnalysis.dem as d 
import numpy as np
from scipy.stats import linregress
import matplotlib.pylab as plt

def plot_maps_area_slope(dem, area, flow_direction, Z_model, channel_mask = None):
    # Rebuild Elevation object from array
    Z_filled = Z_model.copy()
    elev_model = d.Elevation()
    elev_model._griddata = Z_filled
    elev_model._georef_info = dem._georef_info
    elev_model._nodata_value = np.nan

    slope_orig_obj = d.ChannelSlopeWithSmoothing(
        elevation=dem,
        area=area,
        flow_direction=flow_direction,
        vertical_interval=10,
        min_area=2000
    )

    slope_model_obj = d.ChannelSlopeWithSmoothing(
        elevation=elev_model,
        area=area,
        flow_direction=flow_direction,
        vertical_interval=10,
        min_area=2000
    )

    S_obs = slope_orig_obj._griddata
    S_mod = slope_model_obj._griddata
    A_orig = area._griddata
    Z_orig = dem._griddata

    if channel_mask is not None:
        mask = (channel_mask) & np.isfinite(S_obs) & np.isfinite(S_mod) & np.isfinite(A_orig)
    else:
        mask = np.isfinite(S_obs) & np.isfinite(S_mod) & np.isfinite(A_orig)
    A = A_orig[mask]
    S1 = S_obs[mask]
    S2 = S_mod[mask]

    logA = np.log10(A)
    logS1 = np.log10(S1)
    logS2 = np.log10(S2)

    min_elevation = np.nanmin(dem._griddata)
    max_elevation = np.nanmax(dem._griddata)

    slope1, intercept1, *_ = linregress(logA, logS1)
    slope2, intercept2, *_ = linregress(logA, logS2)
    ks1 = 10**intercept1
    ks2 = 10**intercept2

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), gridspec_kw={'width_ratios': [3, 3, 0.25, 4]})
    axes[0].imshow(dem._griddata, cmap='terrain', origin='lower', vmin = min_elevation, vmax = max_elevation)
    axes[0].set_title("Original Elevation")
    axes[0].axis("off")

    im = axes[1].imshow(Z_model, cmap='terrain', origin='lower', vmin = min_elevation, vmax = max_elevation)
    axes[1].set_title("Modeled Elevation")
    axes[1].axis("off")

    A_fit = np.logspace(np.log10(np.min(A)), np.log10(np.max(A)), 200)
    S_fit1 = ks1 * A_fit ** slope1
    S_fit2 = ks2 * A_fit ** slope2

    fig.colorbar(im, cax=axes[2], orientation='vertical')

    axes[3].loglog(A, S1, '.', alpha=0.3, label='Original')
    axes[3].loglog(A, S2, '.', alpha=0.3, label='Modeled')
    axes[3].loglog(A_fit, S_fit1, 'k--', lw=2, label=f"Original fit: k_s={ks1:.2e}, theta={slope1:.2f}")
    axes[3].loglog(A_fit, S_fit2, 'r--', lw=2, label=f"Modeled fit: k_s={ks2:.2e}, theta={slope2:.2f}")
    axes[3].set_xlabel("Drainage Area A (m²)")
    axes[3].set_ylabel("Slope S")
    axes[3].set_title("Slope–Area Comparison")
    axes[3].legend()
    axes[3].grid(True, which='both', ls='--')

    plt.tight_layout()
    plt.show()
    
def plot_maps_and_channel_mask(dem, Z_model, mask):
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), gridspec_kw={'width_ratios': [3, 3, 0.25, 3]})
    
    min_elevation = np.nanmin(dem._griddata)
    max_elevation = np.nanmax(dem._griddata)
    
    axes[0].imshow(dem._griddata, cmap='terrain', origin='lower', vmin=min_elevation, vmax = max_elevation)
    axes[0].set_title("Original Elevation")
    axes[0].axis("off")

    im = axes[1].imshow(Z_model, cmap='terrain', origin='lower', vmin=min_elevation, vmax = max_elevation)
    axes[1].set_title("Modeled Elevation")
    axes[1].axis("off")

    fig.colorbar(im, cax=axes[2], orientation='vertical')
    
    axes[3].imshow(mask, origin='lower', vmin=0, vmax = 1, cmap='seismic')
    axes[3].set_title("Channel Mask")
    axes[3].axis("off")
    plt.tight_layout()
    plt.show()
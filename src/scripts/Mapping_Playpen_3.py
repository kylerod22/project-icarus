
#!pip install cartopy
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap


def predicted_kp_ovalest_streamlit(forecast_kp=2, solar_lon=0.0, mag_pole_lat=80.7, mag_pole_lon=-72.7, mode="dark_contrast", target_lat=None, target_lon=None, location_name=None):
    
    #base_equatorward_edge = 66.0 - (2.0 * forecast_kp)
    #base_peak_lat = base_equatorward_edge + 4.0 
    #make more accurate
    
    base_equatorward_edge = 70.5 - (2.3 * forecast_kp) #67.5
    base_peak_lat = base_equatorward_edge + 3.5
    
    # Create a grid of geographic latitudes and longitudes
    lons = np.linspace(-180, 180, 360)
    lats = np.linspace(30, 90, 180) 
    lon2d, lat2d = np.meshgrid(lons, lats)
    
    # Convert Geographic to Magnetic Latitude
    phi = np.radians(lat2d)
    lam = np.radians(lon2d)
    phi0 = np.radians(mag_pole_lat)
    lam0 = np.radians(mag_pole_lon)
    
    sin_mlat = np.sin(phi) * np.sin(phi0) + np.cos(phi) * np.cos(phi0) * np.cos(lam - lam0)
    sin_mlat = np.clip(sin_mlat, -1.0, 1.0)
    mlat = np.degrees(np.arcsin(sin_mlat))
    
    # Oval "Squish" Mathematics
    day_night_lat_shift = 4.0 * np.cos(np.radians(lon2d - solar_lon)) #was 5.0 befroe
    dynamic_peak_lat = base_peak_lat + day_night_lat_shift
    
    
    # thickness Scaling width grows proportionally with higher Kp values

    #kp_thickness_factor = 1.0 + (0.15 * forecast_kp)  # Kp=0 -> 1.0x, Kp=5 -> 1.75x, Kp=9 -> 2.35x
    #base_std_dev = 3.5 - 1.5 * np.cos(np.radians(lon2d - solar_lon))
    #dynamic_std_dev = base_std_dev * kp_thickness_factor
    #make a bit tighter
    base_std_dev = 1.8 - 0.5 * np.cos(np.radians(lon2d - solar_lon))
    kp_thickness_factor = 1.0 + (0.12 * forecast_kp)
    dynamic_std_dev = base_std_dev * kp_thickness_factor
    
    probability_grid = np.exp(-0.5 * ((mlat - dynamic_peak_lat) / dynamic_std_dev)**2)
    
    cos_diff = np.cos(np.radians(lon2d - solar_lon))

    # Dynamic Peak Center (Teardrop shape: +5° dayside, -5° nightside)
    base_peak_lat = 70.0 - (1.2 * forecast_kp) #68.0 larger is smaller diam
    dynamic_peak_lat = base_peak_lat + (5.0 * cos_diff)

    # Thickness scaling factor (thin dayside arc, broader nightside)
    thickness_scale = 1.0 - 0.40 * cos_diff

    # Slim, controlled widths for inner/outer fades
    equatorward_std = (1.2 + 0.30 * forecast_kp) * thickness_scale
    poleward_std = (1.5 + 0.35 * forecast_kp) * thickness_scale
    plateau_width = (1.5 * forecast_kp) * thickness_scale #originally 0.5, change to 1 or 1.5 to make inner oval closer to pole as kp inc

    # Distance from peak (+ is Poleward/Inside, - is Equatorward/Outside)
    delta_lat = mlat - dynamic_peak_lat

    # 1. Equatorward (Outer Edge): Slim gradient fading south into transparency
    outer_prob = np.exp(-0.5 * (delta_lat / equatorward_std) ** 2)

    # 2. Poleward (Inner Edge): Solid core plateau that transitions into a slim fade north
    shifted_inner_delta = np.maximum(0.0, delta_lat - plateau_width)
    inner_prob = np.exp(-0.5 * (shifted_inner_delta / poleward_std) ** 2)

    # Combine: Outer edge fades south, Inner edge holds solid peak before fading north
    probability_grid = np.where(delta_lat <= 0, outer_prob, inner_prob)
#new code to get probability from oval
    location_prob_pct = None
    if target_lat is not None and target_lon is not None:
        t_phi = np.radians(target_lat)
        t_lam = np.radians(target_lon)
        t_sin_mlat = np.sin(t_phi) * np.sin(phi0) + np.cos(t_phi) * np.cos(phi0) * np.cos(t_lam - lam0)
        t_mlat = np.degrees(np.arcsin(np.clip(t_sin_mlat, -1.0, 1.0)))

        t_cos = np.cos(np.radians(target_lon - solar_lon))
        t_peak_lat = (70.0 - (1.2 * forecast_kp)) + (5.0 * t_cos)
        t_delta = t_mlat - t_peak_lat
        t_scale = 1.0 - 0.40 * t_cos

        if t_delta <= 0:
            t_eq_std = (1.2 + 0.30 * forecast_kp) * t_scale
            point_prob = np.exp(-0.5 * (t_delta / t_eq_std) ** 2)
        else:
            # updated 1.5 multiplier here too
            t_plat = (1.5 * forecast_kp) * t_scale
            t_pol_std = (1.5 + 0.35 * forecast_kp) * t_scale
            t_shift = max(0.0, t_delta - t_plat)
            point_prob = np.exp(-0.5 * (t_shift / t_pol_std) ** 2)

        raw_pct = point_prob * 100.0
        location_prob_pct = raw_pct if raw_pct >= 0.1 else 0.0
    # Map Theme Styling
    if mode == "light":
        fig_bg = "#ffffff"
        ocean_bg = "#e2f1f8"
        land_bg = "#e5e7eb"
        coast_color = "#374151"
        border_color = "#9ca3af"
        grid_color = "#cbd5e1"
        label_color = "#374151"
        text_color = "#111827"
    else:  # 'dark_contrast'
        fig_bg = "#121722"
        ocean_bg = "#1c2638"
        land_bg = "#343e52"
        coast_color = "#9aa5b8"
        border_color = "#525e75"
        grid_color = "#414d63"
        label_color = "#e2e8f0"
        text_color = "#ffffff"

    # Color Map Semi-transparent green gradient
    colors = [
        (0.0, 1.0, 0.3, 0.0),     
        (0.0, 1.0, 0.3, 0.20),    
        (0.0, 1.0, 0.3, 0.50),    
        (0.0, 1.0, 0.3, 0.80)     
    ]
    aurora_cmap = LinearSegmentedColormap.from_list("aurora_green", colors)
    
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor(fig_bg) 
    
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-90))
    ax.set_facecolor(ocean_bg)
    
    ax.add_feature(cfeature.OCEAN, facecolor=ocean_bg)
    ax.add_feature(cfeature.LAND, facecolor=land_bg, edgecolor=border_color)
    ax.add_feature(cfeature.COASTLINE, edgecolor=coast_color, linewidth=1.1)
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor=border_color)
    
    ax.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
    
    contour = ax.contourf(
        lon2d, lat2d, probability_grid, 
        levels=np.linspace(0.1, 1.0, 20),
        cmap=aurora_cmap, 
        transform=ccrs.PlateCarree(), 
        zorder=2
    )
#New code, plotting location on map
    if target_lat is not None and target_lon is not None:
        ax.plot(
            target_lon, target_lat, 
            marker='o', color='#ff3366', markersize=8, 
            transform=ccrs.PlateCarree(), zorder=5
        )
    gl = ax.gridlines(draw_labels=True, dms=True, color=grid_color, linestyle="--")

    #Styling
    gl.xlabel_style = {"color": label_color, "size": 10, "weight": "bold"}
    gl.ylabel_style = {"color": label_color, "size": 10, "weight": "bold"}
#NEW CODE, commenting out other title
    title_text = f"Auroral Oval Forecast (Kp = {forecast_kp})\n{location_name}: {location_prob_pct:.1f}% Probability"
#    if location_name and location_prob_pct is not None:
 #       title_text = f"Auroral Oval Forecast (Kp = {forecast_kp})\n{location_name}: {location_prob_pct:.1f}% Probability"
  #  else:
   #     title_text = f"Auroral Oval Forecast (Kp = {forecast_kp})"
    #plt.title(
    #    f"Auroral Oval Forecast (Kp = {forecast_kp})",
    #    fontsize=16,
    #    pad=20,
    #    fontweight="bold",
    #    color=text_color,
    #)
    plt.title(title_text, fontsize=15, pad=15, fontweight="bold", color=text_color)

    # Change plt.show() to return fig
    return fig
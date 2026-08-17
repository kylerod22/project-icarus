import streamlit as st
import sys
import subprocess
import importlib.util
import datetime
import requests
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from PIL import Image

import pandas as pd
import os
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
image_path = ROOT_DIR / "images" / "GeomagneticStormsScale.png"



st.set_page_config(page_title="Auroral Oval Predictor", layout="wide")
st.title("Auroral Oval Forecast")

@st.cache_resource
def init_app():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cartopy", "plotly"])
    from scripts import pipeline as Pipeline, Mapping_Playpen_3 as mapping_module
    predict_kp = Pipeline.RT_Pipeline
    predicted_kp_ovalest_streamlit = mapping_module.predicted_kp_ovalest_streamlit
    create_kp_plot = mapping_module.create_kp_plot
    return {
        "Pipeline": predict_kp, 
        "Mapping": predicted_kp_ovalest_streamlit,  
        "Kp_Plot": create_kp_plot
    }



with st.spinner("Initializing app..."):
    app = init_app()

# Text
st.write("Welcome to the Auroral Forecast Visualizer, an interactive tool designed to map and predict the real-time visibility of the Northern Lights (Aurora Borealis). Auroras are breathtaking natural light displays created when energetic particles from the solar wind collide with oxygen and nitrogen molecules in Earth's upper atmosphere, causing them to glow in vibrant shades of green, purple, and red. These atmospheric collisions naturally form a ring-shaped band called the auroral oval, centered around Earth's magnetic poles. Under typical conditions, this ring sits high in the Arctic Circle, making northern regions like Scandinavia, Canada, Alaska, and Iceland the most common places to spot the lights.")

st.write("To quantify solar activity and track geomagnetic disturbances, space scientists use the Kp index which is a global geomagnetic activity scale ranging from 0 to 9. Low Kp values (0 to 2) indicate quiet space weather, keeping the aurora faint and locked close to the polar caps. Higher values (Kp 5 and above) represent geomagnetic storms, which cause the auroral oval to expand southward toward lower latitudes where millions more people can see it. Using the controls to the left, you can select specific target locations, adjust space weather parameters, and explore how local time and storm intensity shape auroral probability across the Northern Hemisphere.")
st.write("*Note: This visualizer is most accurate during the winter months when polar nights provide the darkness required for optical observation; during summer, the midnight sun at high latitudes renders auroras invisible to the naked eye regardless of geomagnetic activity.*")


LOCATIONS = {
    # --- Prime Aurora Hotspots (High Latitude) ---
    "Tromsø, Norway": (69.65, 18.96),
    "Abisko, Sweden": (68.35, 18.83),
    "Rovaniemi, Finland": (66.50, 25.73),
    "Reykjavík, Iceland": (64.15, -21.94),
    "Kangerlussuaq, Greenland": (67.01, -50.72),
    "Longyearbyen, Svalbard": (78.22, 15.63),
    "Fairbanks, Alaska": (64.84, -147.72),
    "Yellowknife, Canada": (62.45, -114.37),
    "Whitehorse, Canada": (60.72, -135.05),
    "Churchill, Canada": (58.77, -94.17),
    "Murmansk, Russia": (68.97, 33.08),

    # --- Mid-Latitude Benchmarks (Great for testing high Kp storms) ---
    "Anchorage, Alaska": (61.21, -149.90),
    "Helsinki, Finland": (60.17, 24.94),
    "Oslo, Norway": (59.91, 10.75),
    "Stockholm, Sweden": (59.33, 18.07),
    "Edinburgh, Scotland": (55.95, -3.19),
    "Dublin, Ireland": (53.35, -6.26),
    "Berlin, Germany": (52.52, 13.40),
    "London, UK": (51.51, -0.13),
    "Calgary, Canada": (51.05, -114.07),
    "Seattle, USA": (47.61, -122.33),
    "Minneapolis, USA": (44.98, -93.27),
    "Boston, USA": (42.36, -71.06),
    "Chicago, USA": (41.88, -87.63),
    "New York City, USA": (40.71, -74.01),
}

TIMEZONES = [
    "UTC",
    "US/Eastern",
    "US/Central",
    "US/Mountain",
    "US/Pacific",
    "Europe/London",
    "Europe/Berlin",
    "Asia/Tokyo",
    "Australia/Sydney",
]

HORIZONS = {
    "Nowcast (0 hours)": "0hr",
    "3 hour forecast": "3hr",
    "6 hour forecast": "6hr"
}

THEMES = {
    "Dark": "dark_contrast",
    "Light": "light"
}

COLOR_THEMES = {
    "Green": "green",
    "Pink": "pink",
    "Purple": "purple",
    "Red": "red"
}

# Default location
if "selected_location_name" not in st.session_state:
    st.session_state.selected_location_name = ("Ann Arbor, Michigan, USA")

if "target_lat" not in st.session_state:
    st.session_state.target_lat = 42.2776

if "target_lon" not in st.session_state:
    st.session_state.target_lon = -83.7409




# ---------------- PREDICTION INPUTS ----------------
st.sidebar.header("Prediction Inputs")

# Location + Lat & Lon
selected_location_name = st.sidebar.selectbox("Select Location", list(LOCATIONS.keys()))
target_lat, target_lon = LOCATIONS[selected_location_name]

# Horizon/Model Selection
forecast_horizon = st.sidebar.selectbox("Select Forecast Window", list(HORIZONS.keys()))
target_horizon = HORIZONS[forecast_horizon]



# ---------------- ADJUSTABLE SETTINGS ----------------

st.sidebar.header(
    "Custom Sandbox Settings", 
    help="Adjust dates, times, and Kp values to simulate custom auroral conditions on the second map."
)

# Kp Index Input
forecast_kp = st.sidebar.slider("Select Kp Index", min_value=0.0, max_value=9.0, value=2.0, step=0.33)

# Date & Time Inputs
selected_time = st.sidebar.time_input("Select Time", value=datetime.time(12, 0))

#Color choices
selected_oval_color = st.sidebar.selectbox("Select Color", list(COLOR_THEMES.keys()))


# ---------------- USER SETTINGS ----------------
st.sidebar.header("User Settings")

# Timezone Selector
selected_tz_str = st.sidebar.selectbox("Select Timezone", TIMEZONES, index=0)

# Theme
theme = st.sidebar.radio("Map Theme", list(THEMES.keys()))
mode_theme = THEMES[theme]





# Time Calculations
def calc_solar_longitude(in_dt):
    # Convert to UTC
    utc_dt = in_dt.astimezone(ZoneInfo("UTC"))

    # Calculate UTC hour in decimal form (e.g., 18:30 -> 18.5)
    utc_hour_decimal = (
        utc_dt.hour + (utc_dt.minute / 60.0) + (utc_dt.second / 3600.0)
    )

    # Solar longitude formula:
    # At 12:00 UTC (Noon), solar_lon = 0.0 (Prime Meridian)
    # Earth rotates 15 degrees per hour
    solar_lon = (12.0 - utc_hour_decimal) * 15.0
    return solar_lon


naive_dt = datetime.datetime.combine(datetime.date.today(), selected_time)
user_local_dt = naive_dt.replace(tzinfo=ZoneInfo(selected_tz_str))
user_solar_lon = calc_solar_longitude(user_local_dt)



@st.cache_data(ttl="10m")
def get_rt_prediction(horizon):
    pred_kp, pred_time = app["Pipeline"](horizon)
    return round(pred_kp, 2), pred_time

with st.spinner("Getting real-time space weather forecast..."):
    pred_kp, pred_time = get_rt_prediction(target_horizon)
    pred_dt = datetime.datetime.fromisoformat(pred_time)
    pred_solar_lon = calc_solar_longitude(pred_dt)


# ---------------- PREDICTION MAP ----------------
st.write(f"### Aurora Visibility for {selected_location_name}")

date_format = "%Y-%m-%d %H:%M:%S"
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Predicted Auroral Oval")

    with st.spinner("Generating predicted auroral oval..."):
        fig_predicted = app["Mapping"](
            forecast_kp=pred_kp,
            solar_lon=pred_solar_lon,
            mode="dark_contrast",
            target_lat=target_lat,
            target_lon=target_lon,
            location_name=selected_location_name
        )

    st.pyplot(
        fig_predicted,
        use_container_width=False
    )

    plt.close(fig_predicted)

with col2:
    st.subheader("Forecast")
    st.metric("Forecast Kp", pred_kp)
    st.metric("Local Time", pred_dt.astimezone(ZoneInfo(selected_tz_str)).strftime("%Y-%m-%d %H:%M %Z"))
    st.metric("Subsolar Longitude", f"{pred_solar_lon:.1f}°")

# ---------------- USER-ADJUSTED MAP ----------------
st.divider()
st.subheader("Sandbox Mode Auroral Oval")
st.write("Note that the map visualization below is an interactive exploratory tool designed for simulation and is not connected to live prediction models. Feel free to use the settings panel on the left to experiment with different dates, times, and Kp values to see how changing geomagnetic conditions and solar orientation dynamically alter auroral coverage across the Northern Hemisphere.")
col1, col2 = st.columns([3, 1])
@st.fragment
def adjustable_oval():

    with col1:
        with st.spinner("Updating auroral oval..."):

            fig_adjusted = app["Mapping"](
                forecast_kp=forecast_kp,
                solar_lon=user_solar_lon,
                mode=mode_theme,
                target_lat=target_lat,
                target_lon=target_lon,
                location_name=selected_location_name,
                color_scheme = COLOR_THEMES[selected_oval_color]
            )

        st.pyplot(
            fig_adjusted,
            use_container_width=False
        )
        plt.close(fig_adjusted)

    with col2:
        st.subheader("Custom ")
        st.metric("Custom Kp", forecast_kp)
        st.metric("Custom Local Time", user_local_dt.astimezone(ZoneInfo(selected_tz_str)).strftime("%Y-%m-%d %H:%M %Z"))
        st.metric("Subsolar Longitude", f"{user_solar_lon:.1f}°")

adjustable_oval()


st.subheader("NOAA Geomagnetic Storm Scale")

st.image(
    image_path,
    caption="NOAA Space Weather Scale for Geomagnetic Storms (G1 - G5), from https://www.spaceweather.gov/noaa-scales-explanation",
    use_container_width=True
)

st.divider()

st.title("Kp Index Predictions")
st.write("This interactive step plot compares predicted Kp values (red) against observed Kp metrics (blue) across 3-hour reporting intervals. The model closely mirrors actual data, accurately capturing quiet background levels along with rapid storm onset and decay spikes. Using the Datetime Slider at the bottom, users can drag the range handles to zoom in on specific multi-day storm events or expand the view to analyze long-term performance across the entire dataset.")

# Load and cache data for performance
@st.cache_data
def load_data():
    return pd.read_csv(ROOT_DIR / "data" / "Processed" / "preds_2025.csv")


df = load_data()

# Render interactive plot for full dataset
fig = app["Kp_Plot"](df)
st.plotly_chart(fig, use_container_width=True)


import streamlit as st
import pandas as pd
import os
import sys
import numpy as np
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "cartopy"])
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap


#import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#from matplotlib.colors import LinearSegmentedColormap
#st.write("Files in current directory:", os.listdir("."))

import datetime
from zoneinfo import ZoneInfo
import streamlit as st



# import importlib.util

# # Load the file directly from the exact file path
# spec = importlib.util.spec_from_file_location("Mapping_Playpen_3", "/work/Mapping_Playpen_3.py")
# mapping_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(mapping_module)


from scripts import Mapping_Playpen_3 as mapping_module
predicted_kp_ovalest_streamlit = mapping_module.predicted_kp_ovalest_streamlit


#if "/work" not in sys.path:
##    sys.path.append("/work")
#from Mapping_Playpen_3 import predicted_kp_ovalest_streamlit
#/work/Mapping_Playpen_3.py

# Page Setup
st.set_page_config(page_title="Auroral Oval Predictor", layout="wide")
st.title("Auroral Oval Forecast")

# Page title
st.title("Mock User Interface")

# Text
st.write("Here is where I will make a mock up user interface. Inputs to include:")
st.write("""
            1. Location (Need to add dropdown, probably limit to N hemis)
            2. Time
            3. 
        """)
st.write("""Outputs: \n
        1. Map
        2. Weather data (none here yet)""")

st.write("Right now, the location does not do anything. Working on polishing fn so maybe the location could be noted on the map and a probability could be given. Time input is working (and affects the visual)! ")
st.write("The predicted kp will end up coming from the predictive model (probably based on the date input), but right now it is a slider so we can see what the visualization is and how the other inputs affect it. ")
st.write("Error that I'm running into rn: The streamlit does not work unless the Mapping Playpen notebook is run, I think because it installs and imports the cartopy library. ")
# Sidebar Controls
st.sidebar.header("Forecast Settings")

# adding Location Dropdown Dictionary
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

selected_location_name = st.sidebar.selectbox(
    "Select Location", list(LOCATIONS.keys())
)
target_lat, target_lon = LOCATIONS[selected_location_name]
st.write(f"{selected_location_name}")
# 1. Kp Index Input
forecast_kp = st.sidebar.slider(
    "Select Kp Index", min_value=0.0, max_value=9.0, value=2.0, step=0.33
)

# 2. Date & Time Inputs
selected_date = st.sidebar.date_input("Select Date", value=datetime.date.today())
selected_time = st.sidebar.time_input(
    "Select Time", value=datetime.time(12, 0)
)

# 3. Timezone Selector
timezones = [
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
selected_tz_str = st.sidebar.selectbox("Select Timezone", timezones, index=0)

# theme selector (aybe should just keep it on dark mode always??)
mode_theme = st.sidebar.radio(
    "Map Theme", ["dark_contrast", "light"], index=0
)

#math section
# combine inputs into a datetime object
naive_dt = datetime.datetime.combine(selected_date, selected_time)
local_dt = naive_dt.replace(tzinfo=ZoneInfo(selected_tz_str))

# Convert to UTC
utc_dt = local_dt.astimezone(ZoneInfo("UTC"))

# Calculate UTC hour in decimal form (e.g., 18:30 -> 18.5)
utc_hour_decimal = (
    utc_dt.hour + (utc_dt.minute / 60.0) + (utc_dt.second / 3600.0)
)

# Solar longitude formula:
# At 12:00 UTC (Noon), solar_lon = 0.0 (Prime Meridian)
# Earth rotates 15 degrees per hour
solar_lon = (12.0 - utc_hour_decimal) * 15.0

# Display Map and info
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Predicted Auroral Oval")
    # Generate the figure using user inputs
    #fig = predicted_kp_ovalest_streamlit(
    #    forecast_kp=forecast_kp, solar_lon=solar_lon, mode=mode_theme
    #)
    fig = predicted_kp_ovalest_streamlit(
        forecast_kp=forecast_kp,
        solar_lon=solar_lon,
        mode=mode_theme,
        target_lat=target_lat,
        target_lon=target_lon,
        location_name=selected_location_name
    )
    # Render Matplotlib figure in Streamlit
    st.pyplot(fig, use_container_width=True)

with col2:
    st.subheader("Calculated Parameters")
    st.metric("Forecast Kp", f"{forecast_kp:.2f}")
    st.metric("Local Time", local_dt.strftime("%Y-%m-%d %H:%M %Z"))
    st.metric("UTC Time", utc_dt.strftime("%Y-%m-%d %H:%M UTC"))
    st.metric("Subsolar Longitude", f"{solar_lon:.1f}°")

# Additional Data Tables
st.divider()
#st.subheader("Training Data Preview")
#try:
#    train_df = pd.read_parquet("Processed/omni_train.parquet")
#    st.dataframe(train_df.head(10), use_container_width=True)
#except Exception as e:
#    st.info("Training data parquet file not found at 'Processed/omni_train.parquet'")













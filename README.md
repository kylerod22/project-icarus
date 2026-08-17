# Project Icarus
## Overview
Project Icarus is an end-to-end machine learning and data visualization project for predicting geomagnetic activity and visualizing the potential visibility of auroras. We combine historical and real-time solar-wind and magnetic-field measurements from NASA OMNIWeb and NOAA with machine learning models to forecast the Kp index at multiple time horizons. These predictions are then used to estimate and visualize the auroral oval across the Northern Hemisphere.

The project also includes an interactive Streamlit web application that allows users to explore predicted auroral conditions, customize simulated space-weather conditions, and examine historical model performance. **You can view the application here: https://project-icarus.streamlit.app/**

## Running Locally
If you wish to run our Streamlit app locally, follow these steps:
### Prerequisites
- Python 3.12 is required, as libraries such as `cartopy` aren't supported by the current 3.14 version.
- Install the required dependencies: `pip install -r requirements.txt`

Now, you should be able to run the app: `streamlit run src\aurora_visualizer_app.py`

Jupyter notebooks are also present in this repository if you wish to explore our data exploration and machine learning development process.

## Data Access
NASA OMNIWeb was used to obtain historical solar-wind and magnetic-field measurements. The OMNI dataset combines measurements from multiple spacecraft and provides time-shifted solar-wind observations near Earth. Variables used in this project include magnetic-field measurements and solar-wind plasma parameters at hourly and higher temporal resolutions. NASA provides a user-interface capable of selectively extracting whichever variables a user chooses: 
- Low resolution (1 hour intervals): https://omniweb.gsfc.nasa.gov/form/dx1.html
- High resolution (1 minute intervals): https://omniweb.gsfc.nasa.gov/form/omni_min.html

NOAA's Space Weather Prediction Center (SWPC) was used for real-time space-weather measurements. In particular, the project's real-time prediction pipeline retrieves solar-wind and magnetic-field measurements through NOAA's publicly available JSON data services. SWPC provides these datasets through its public data service, including real-time JSON products and archived space-weather data. 

These JSON endpoints were used:
- https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json
- https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json

Aurorasaurus data was used to provide human observations of auroral activity. These observations contain information such as observation times, geographic locations, whether an aurora was reported, and associated observational metadata. The data were used to supplement the space-weather measurements and provide ground-based observations of auroral activity. The dataset we used was pulled from https://zenodo.org/records/16783265?preview_file=web_observations_2014-08-01_to_2025-08-02_cleaned.csv

All data were obtained from publicly accessible sources and processed within the Project Icarus data pipeline. The datasets were cleaned, transformed, and combined as necessary for machine-learning model development and auroral visualization. The original data sources remain the authoritative sources for the underlying measurements and observations.


## Contributors
- Sophia Menchaca
- Kyle Rodriguez
- Tony Lett

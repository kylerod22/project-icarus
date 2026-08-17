## QUICK NOTE: Time is in UTC, so don't worry if it doesn't match EST time. I checked and
## the latest timestamp is very close to the current time

import pandas as pd
import numpy as np
import requests
from io import StringIO
import re
import json

FEATURES = ["bt","bx_gsm","by_gsm","bz_gsm","proton_speed","proton_density"]

def get_mag_data():
    url = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
    df = pd.DataFrame(requests.get(url).json())
    df["active"] = df["active"].astype(bool)
    df = (
        df[df["active"]]
        [["time_tag", "bt", "bx_gsm", "by_gsm", "bz_gsm"]]
    )
    return df

def get_proton_data():
    url = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
    df = pd.DataFrame(requests.get(url).json())
    df["active"] = df["active"].astype(bool)
    df = (
        df[df["active"]]
        [["time_tag", "proton_speed", "proton_density"]]
    )
    return df


def get_real_time_data(interval=480):

    # Get separate datasets
    mag_df = get_mag_data()
    proton_df = get_proton_data()

    # Merge
    df = pd.merge(
        mag_df,
        proton_df,
        on="time_tag",
        how="outer"
    )

    # Convert datatypes
    df["time_tag"] = pd.to_datetime(df["time_tag"])
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FEATURES] = df[FEATURES].replace(-9999, np.nan)

    # Get latest timestamp with no NaNs, go back 8 hours/interval from here
    df = df.sort_values("time_tag")
    df = df.drop_duplicates(subset="time_tag", keep="last")
    complete = df.dropna(subset=FEATURES)
    latest = complete["time_tag"].iloc[-1]
    
    full_index = pd.date_range(
        end=latest,
        periods=interval,
        freq="1min"
    )
    df = (
        df.set_index("time_tag")
          .reindex(full_index)
          .rename_axis("time_tag")
          .reset_index()
    )

    df = df.rename(columns={
        "time_tag": "datetime",
        "bt": "mag_avg_nt",
        "bx_gsm": "bx_gsm_nt",
        "by_gsm": "by_gsm_nt",
        "bz_gsm": "bz_gsm_nt",
        "proton_speed": "flow_speed_km_s",
        "proton_density": "proton_density_n_cc"
    })

    return df
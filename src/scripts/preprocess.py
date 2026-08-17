FEATURE_VARS = ["mag_avg_nt", "bx_gsm_nt", "by_gsm_nt", "bz_gsm_nt", "flow_speed_km_s", "proton_density_n_cc"]
def clean_nans(df, limit=60, verbose=False):
    rows_before = len(df)
    total_interpolated = 0

    for feature in FEATURE_VARS:
        nans_before = df[feature].isna().sum()

        df[feature] = df[feature].interpolate(
            method="linear",
            limit=limit,
            limit_direction="both"
        )

        nans_after = df[feature].isna().sum()
        filled = nans_before - nans_after
        total_interpolated += filled

        if verbose:
            print(f"{feature}: {filled} values interpolated")

    df = df.dropna(subset=FEATURE_VARS)

    rows_after = len(df)

    if verbose:
        print("--------------------------------------------------")
        print("Final NaN cleaning results:")
        print(f"Rows before: {rows_before:,}")
        print(f"Rows after:  {rows_after:,}")
        print(f"Rows removed: {rows_before - rows_after:,}")
        print(f"Percent removed: {(rows_before - rows_after) / rows_before:.2%}")
        print(f"Total values interpolated: {total_interpolated}")
        print("--------------------------------------------------")
    return df

def add_time_cols(df):
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.dayofyear
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df = df[["datetime", "year", "month", "day", "hour", "minute", "mag_avg_nt", "bx_gsm_nt", "by_gsm_nt", "bz_gsm_nt", "flow_speed_km_s", "proton_density_n_cc"]]
    return df
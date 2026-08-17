import numpy as np
import pandas as pd
LOG_COLS = ['mag_avg_nt', 'flow_speed_km_s', 'proton_density_n_cc']

KP_FEATURE_VARS = [
    'mag_avg_nt_log',
    'bx_gsm_nt',
    'by_gsm_nt',
    'bz_gsm_nt',
    'flow_speed_km_s_log',
    'proton_density_n_cc_log',
]
KP_LOOKBACK_WINDOWS = [('0_1h', 1), ('1_2h', 2), ('2_3h', 3), ('3_4h', 4)]

def add_log_transformed_columns(df, verbose=False):
    """Adds log1p-transformed copies of the given column(s) as new '<col>_log' columns."""
    df = df.copy()

    for col in LOG_COLS:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            if verbose:
                print(f'Warning: {col} has {negative_count:,} negative values; log1p will produce NaN for these.')
        df[f'{col}_log'] = np.log1p(df[col])
        if verbose:
            print(f'Added {col}_log — min={df[f"{col}_log"].min():.4f}, max={df[f"{col}_log"].max():.4f}, '
                    f'NaNs={df[f"{col}_log"].isna().sum():,}')

    return df

def duplicate_check(df, verbose=False):
    # Drop any duplicate minutes within the same year/day/hour
    duplicate_minutes = df[
        df.duplicated(subset=['year', 'day', 'hour', 'minute'], keep=False)
    ]
    if verbose:
        print(f'Found {len(duplicate_minutes):,} rows with duplicate (year, day, hour, minute) combinations')
    duplicate_minutes.sort_values(['year', 'day', 'hour', 'minute']).head(20)

    if len(duplicate_minutes) > 0:
        before_rows = len(df)
        df = df.drop_duplicates(subset=['year', 'day', 'hour', 'minute'], keep='first')
        if verbose:
            print(f'Dropped {before_rows - len(df):,} duplicate rows, {len(df):,} rows remaining')
    
    return df


def build_kp_interval_features(df, value_cols=KP_FEATURE_VARS, max_missing_minutes=5, has_target=True, verbose=False):
    if verbose:
        print(f'--- Record count audit ---')
        print(f'Input rows (per-minute): {len(df):,}')

    d = df.set_index('datetime').sort_index()

    minute_counts = d.resample('1h').size()
    hourly_stats = d[value_cols].resample('1h').agg(['mean', 'min', 'max'])
    hourly_stats.columns = [f'{col}_{stat}' for col, stat in hourly_stats.columns]
    if has_target:
        kp_hourly_mean = d['kp_10'].resample('1h').mean()

    idx = hourly_stats.index

    if verbose:
        print(f'Resampled hourly buckets ({idx.min()} to {idx.max()}): {len(idx):,}')

    missing = 60 - minute_counts

    # shifts past the edge of the resampled range have no underlying data, so treat them as fully missing
    window_missing = {label: missing.shift(k).fillna(60) for label, k in KP_LOOKBACK_WINDOWS}
    if has_target:
        block_missing = missing + missing.shift(-1).fillna(60) + missing.shift(-2).fillna(60)

    interval_index = idx[idx.hour % 3 == 0]
    if verbose:
        print(f'Candidate 3-hour intervals (hour % 3 == 0): {len(interval_index):,}')

    features = pd.DataFrame(index=interval_index)
    features.index.name = 'datetime'
    if has_target:
        # Kp is only reported once per 3-hour block (the same value repeats across all 3 hours),
        # so the interval's own hourly value is the block value - no averaging across hours needed.
        features['kp_10'] = kp_hourly_mean.loc[interval_index]
        features['kp_index'] = (features['kp_10'] / 10.0).round(2)
    features['year'] = interval_index.year
    features['day'] = interval_index.dayofyear
    features['hour'] = interval_index.hour
    features['minute'] = 0
    features['day_cos'] = np.cos(2 * np.pi * features['day'] / 365)
    features['day_sin'] = np.sin(2 * np.pi * features['day'] / 365)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['minute_cos'] = 1.0
    features['minute_sin'] = 0.0
    if has_target:
        features['data_split'] = df['data_split'].iloc[0]

    for label, shift_by in KP_LOOKBACK_WINDOWS:
        for var in value_cols:
            features[f'{var}_avg_{label}'] = hourly_stats[f'{var}_mean'].shift(shift_by).loc[interval_index]
            features[f'{var}_min_{label}'] = hourly_stats[f'{var}_min'].shift(shift_by).loc[interval_index]
            features[f'{var}_max_{label}'] = hourly_stats[f'{var}_max'].shift(shift_by).loc[interval_index]

    if verbose:
        print(f'Assembled feature rows (before drop filtering): {len(features):,}, columns: {features.shape[1]:,}')

    fail_windows = {
        label: window_missing[label].loc[interval_index] > max_missing_minutes
        for label, _ in KP_LOOKBACK_WINDOWS
    }
    if has_target:
        fail_block = block_missing.loc[interval_index] > max_missing_minutes
        drop_mask = fail_block.copy()
    else:
        drop_mask = pd.Series(False, index=interval_index)
    for label in fail_windows:
        drop_mask |= fail_windows[label]

    reasons = pd.DataFrame({'block_3h': fail_block}, index=interval_index) if has_target else pd.DataFrame(index=interval_index)
    for label in fail_windows:
        reasons[label] = fail_windows[label]
    dropped_reasons = reasons.loc[drop_mask]

    if verbose:
        print(f'Evaluated {len(interval_index):,} candidate 3-hour Kp intervals '
            f'({interval_index.min()} to {interval_index.max()})')
        print(f'Dropping {drop_mask.sum():,} intervals ({drop_mask.mean() * 100:.2f}%) '
            f'for more than {max_missing_minutes} missing minutes in'
            + (' the 3-hour interval or a lookback window:' if has_target else ' a lookback window:'))
        if has_target:
            print(f'  - 3-hour interval itself: {fail_block.sum():,}')
        for label, _ in KP_LOOKBACK_WINDOWS:
            print(f'  - {label} lookback window: {fail_windows[label].sum():,}')
        print(f'Keeping {(~drop_mask).sum():,} intervals')

        if drop_mask.any():
            print('\nSample of dropped intervals and failing checks:')
            print(dropped_reasons.head(10))
            if len(dropped_reasons) > 10:
                print('...')
                print(dropped_reasons.tail(5))

    kept = features.loc[~drop_mask.values].copy()

    if verbose:
        print(f'\n--- Record count audit summary ---')
        print(f'{"Input per-minute rows:":<35}{len(df):>12,}')
        print(f'{"Resampled hourly buckets:":<35}{len(idx):>12,}')
        print(f'{"Candidate 3-hour intervals:":<35}{len(interval_index):>12,}')
        print(f'{"Dropped intervals:":<35}{drop_mask.sum():>12,}')
        print(f'{"Final kept intervals:":<35}{len(kept):>12,}')
        print(f'{"Final output shape:":<35}{str(kept.shape):>12}')
    assert len(kept) == len(interval_index) - drop_mask.sum(), 'kept count does not reconcile with candidates - drops'

    return kept, dropped_reasons


def select_servable_row(df, kp_features, verbose=False):
    # Select the most recently kept interval as the row to serve for this request
    if kp_features.empty:
        raise ValueError('No candidate 3-hour interval survived the missingness check - cannot serve a prediction.')

    most_recent_h = kp_features.index.max()
    servable_row = kp_features.loc[[most_recent_h]]

    # Use the latest available per-minute timestamp as the "now" proxy, since the real-time buffer
    # may not exactly match wall-clock time when this notebook is run against static/demo data.
    now_proxy = df['datetime'].max()
    staleness = now_proxy - most_recent_h

    if verbose:
        print(f'Latest available per-minute data: {now_proxy}')
        print(f'Most recent servable interval anchor (H): {most_recent_h}')
        print(f'Staleness (latest data - H): {staleness}')

        if staleness > pd.Timedelta(hours=3):
            print(f'WARNING: servable row is {staleness} behind the latest data - more than one 3-hour '
                f'cycle stale. This looks like a data gap/outage, not the routine in-progress-hour lag.')
        else:
            print('Staleness is within the expected range for the routine in-progress current hour.')
    
    return servable_row

def create_time_lagged_features(df, verbose=False):
    df = duplicate_check(df, verbose=verbose)
    kp_features, kp_dropped = build_kp_interval_features(df, has_target=False, verbose=verbose)
    servable_row = select_servable_row(df, kp_features, verbose=verbose)
    return servable_row
    

    
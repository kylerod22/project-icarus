import warnings
warnings.filterwarnings('ignore')
from scripts import real_time_data as RT_Data_Module
from scripts import preprocess as Preprocess_Module
from scripts import feature_engineering as Feature_Engineering_Module
from scripts import model_prediction as Prediction_Module
import pandas as pd
import time
from datetime import datetime, timezone, timedelta

def RT_Pipeline(horizon="0hr", verbose=False, benchmark=False):
    overall_start = time.perf_counter()
    start_time = time.perf_counter()

    RT_df = RT_Data_Module.get_real_time_data()
    if benchmark:
        duration = time.perf_counter() - start_time
        print(f"Fetching real-time data took {duration:.6f} seconds")
        start_time = time.perf_counter()

    RT_df = Preprocess_Module.clean_nans(RT_df, verbose=verbose)
    if benchmark:
        duration = time.perf_counter() - start_time
        print(f"Cleaning NaNs took {duration:.6f} seconds")
        start_time = time.perf_counter()

    RT_df = Preprocess_Module.add_time_cols(RT_df)
    if benchmark:
        duration = time.perf_counter() - start_time
        print(f"Adding time columns took {duration:.6f} seconds")
        start_time = time.perf_counter()

    RT_df = Feature_Engineering_Module.add_log_transformed_columns(RT_df, verbose=verbose)
    if benchmark:
        duration = time.perf_counter() - start_time
        print(f"Log transform took {duration:.6f} seconds")
        start_time = time.perf_counter()

    model_input = Feature_Engineering_Module.create_time_lagged_features(RT_df, verbose=verbose)
    if benchmark:
        duration = time.perf_counter() - start_time
        print(f"Time-Lagged feature engineering took {duration:.6f} seconds")
        start_time = time.perf_counter()

    predicted = Prediction_Module.predict(model_input, horizon, verbose=verbose)
    if benchmark:
        duration = time.perf_counter() - start_time
        print(f"Model prediction took {duration:.6f} seconds")
        start_time = time.perf_counter()

    pred_time = datetime.now(timezone.utc)
    if horizon == "3hr":
        pred_time = pred_time + timedelta(hours=3)
    elif horizon == "6hr":
        pred_time = pred_time + timedelta(hours=6)
    

    if benchmark:
        duration = time.perf_counter() - overall_start
        print(f"Entire pipeline took {duration:.6f} seconds to complete!")

    return predicted.iloc[0], str(pred_time)

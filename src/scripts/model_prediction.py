from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
HORIZON_CONFIG = {
    '0hr': {
        'model_path': MODELS_DIR / 'best_lasso_model_0hr.pkl',
        'target': 'kp_index',
    },
    '3hr': {
        'model_path': MODELS_DIR / 'best_lasso_model_3hr.pkl',
        'target': 'kp_index_3_hr_forecast',
    },
    '6hr': {
        'model_path': MODELS_DIR / 'best_lasso_model_6hr.pkl',
        'target': 'kp_index_6_hr_forecast',
    },
}


def load_model(model_path, verbose=False):
    """Load a saved sklearn Pipeline (StandardScaler + Lasso) from a .pkl file."""

    model_path = Path(model_path)
    pipeline = joblib.load(model_path)

    step_names = [name for name, _ in pipeline.steps]
    lasso_alpha = pipeline.named_steps['lasso'].alpha

    if verbose:
        print(f'Model loaded from {model_path}.')
        print(f'Pipeline steps: {step_names}')
        print(f'Lasso alpha: {lasso_alpha}')
        print(f'Expected feature count: {len(pipeline.feature_names_in_)}')

    return pipeline


def prepare_inference_features(df_new, expected_features, verbose=False):
    """Align new data to the exact feature columns/order the pipeline was trained on."""

    expected_features = list(expected_features)
    missing = [col for col in expected_features if col not in df_new.columns]
    if missing:
        raise ValueError(f'New data is missing expected feature columns: {missing}')

    ignored = [col for col in df_new.columns if col not in expected_features]

    X_new = df_new[expected_features]

    rows_before = len(X_new)
    X_new = X_new.dropna()
    rows_after = len(X_new)

    if verbose:
        print(f'Aligned to {len(expected_features)} expected feature columns.')
        print(f'Columns in new data not used as features in the model (ignored): {ignored}')
        print(f'Rows before dropping nulls: {rows_before}.')
        print(f'Rows after dropping nulls: {rows_after}.')

    return X_new

def predict_kp(pipeline, X_new, verbose=False):
    """Generate Kp predictions for new input data, aligned to X_new's index."""

    y_pred = pd.Series(pipeline.predict(X_new), index=X_new.index, name='kp_pred')
    y_pred = y_pred.clip(lower=0)
    
    if verbose:
        print(f'Generated {len(y_pred)} predictions.')
        print(f'Prediction min/mean/max: {y_pred.min():.4f} / {y_pred.mean():.4f} / {y_pred.max():.4f}')

    return y_pred


def predict(model_input, horizon, verbose=False):
    model = load_model(HORIZON_CONFIG[horizon]['model_path'], verbose=verbose)
    X_new = prepare_inference_features(model_input, model.feature_names_in_, verbose=verbose)
    y_pred = predict_kp(model, X_new, verbose=verbose)

    return y_pred


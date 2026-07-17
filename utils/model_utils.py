import os
import json
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from utils.data_utils import GAME_DISPLAY_TO_FOLDER, ROOT_DIR, prepare_game_history, _normalize_game_name


MODEL_ORDER = ["AR", "MA", "ARMA", "ARIMA", "ARIMAX", "SARIMA", "SARIMAX", "XGBoost", "LightGBM", "LSTM", "GRU"]
MODEL_DESCRIPTIONS = {
    "AR": "Classical autoregressive model for short-horizon forecasting.",
    "MA": "Moving average model well-suited for smoothing recent traffic patterns.",
    "ARMA": "Combines autoregressive and moving average behavior for balanced time-series forecasting.",
    "ARIMA": "Statistical model that captures trends and seasonality for stable forecasting.",
    "ARIMAX": "ARIMA model enhanced with exogenous signals for richer traffic forecasting.",
    "SARIMA": "Seasonal ARIMA model aimed at recurring hourly and daily traffic cycles.",
    "SARIMAX": "Seasonal ARIMA with exogenous variables for robust forecasts.",
    "XGBoost": "Gradient-boosted tree model designed for nonlinear traffic relationships.",
    "LightGBM": "Fast gradient-boosting model ideal for quick and accurate player traffic prediction.",
    "LSTM": "Deep learning model capable of learning long-term temporal dependencies.",
    "GRU": "Efficient recurrent model for capturing sequential traffic dynamics.",
}

MODEL_FOLDERS = {
    "AR": "models_ar",
    "MA": "models_ma",
    "ARMA": "models_arma",
    "ARIMA": "models_arima",
    "ARIMAX": "models_arimax",
    "SARIMA": "models_sarima",
    "SARIMAX": "models_sarimax",
    "XGBoost": "models_xgboost",
    "LightGBM": "models_lightgbm",
    "LSTM": "models_lstm",
    "GRU": "models_gru",
}

MODEL_FILE_NAMES = {
    "AR": "best_ar_model.pkl",
    "MA": "best_ma_model.pkl",
    "ARMA": "best_arma_model.pkl",
    "ARIMA": "best_arima_model.pkl",
    "ARIMAX": "best_arimax_model.pkl",
    "SARIMA": "best_sarima_model.pkl",
    "SARIMAX": "best_sarimax_model.pkl",
    "XGBoost": "best_xgboost_model.pkl",
    "LightGBM": "best_lightgbm_model.pkl",
    "LSTM": "best_lstm_model.keras",
    "GRU": "best_gru_model.keras",
}

TREE_MODELS = {"XGBoost", "LightGBM"}
DEEP_MODELS = {"LSTM", "GRU"}
STATISTICAL_MODELS = {"AR", "MA", "ARMA", "ARIMA", "ARIMAX", "SARIMA", "SARIMAX"}

# Full 15-feature set used in XGBoost/LightGBM training (in the same order as the model)
TREE_FEATURE_COLUMNS = [
    "lag_1", "lag_24", "rolling_mean_24", "rolling_std_24",
    "peak_players", "hours_played", "rank",
    "hour", "day_of_week", "is_weekend",
    "month", "quarter", "week_of_year",
    "hour_sin", "hour_cos",
]

# Exog feature candidates used in ARIMAX/SARIMAX (superset; actual ones selected per-game)
EXOG_CANDIDATES = [
    "peak_players", "hours_played", "rank",
    "hour", "day_of_week", "is_weekend", "quarter",
    "hour_sin", "hour_cos", "lag_1", "lag_24",
    "rolling_mean_24", "rolling_std_24",
]


def build_model_path(model_name: str, game_name: str) -> Tuple[str, Optional[str]]:
    if model_name not in MODEL_FOLDERS:
        raise KeyError(f"Unknown model '{model_name}'. Did you forget to resolve '⭐ Best Model' before calling build_model_path?")
    root_dir = ROOT_DIR
    folder = MODEL_FOLDERS[model_name]
    model_file = MODEL_FILE_NAMES[model_name]

    # Primary folder name from the mapping
    primary_folder = GAME_DISPLAY_TO_FOLDER.get(game_name, game_name)

    # Build a list of candidate folder names (apostrophe variants)
    alt_folder = primary_folder.replace("'", "").replace("'s_", "s_").replace("'S_", "S_")
    candidates = list(dict.fromkeys([primary_folder, alt_folder]))

    for game_folder in candidates:
        model_dir = os.path.join(root_dir, folder, game_folder)
        if os.path.isdir(model_dir):
            # Check for alternate model file names (e.g., "best_gru.keras" vs "best_gru_model.keras")
            model_path = os.path.join(model_dir, model_file)
            if not os.path.exists(model_path) and model_name in DEEP_MODELS:
                # Try without "_model" suffix for deep learning models
                alt_file = model_file.replace("_model.", ".")
                if os.path.exists(os.path.join(model_dir, alt_file)):
                    model_file = alt_file
            return model_dir, model_file

    # Fall back to primary even if dir doesn't exist (so callers get a clear FileNotFoundError)
    return os.path.join(root_dir, folder, primary_folder), model_file


def _load_metrics(model_dir: str, model_name: str) -> Optional[Dict[str, Any]]:
    candidates = [
        os.path.join(model_dir, "metrics.json"),
        os.path.join(model_dir, f"best_{model_name.lower()}_metrics.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    return None


def _load_deep_learning_scaler(model_dir: str) -> Optional[Any]:
    for candidate in ["scaler.pkl", "scaler.joblib", "scaler.sav"]:
        scaler_path = os.path.join(model_dir, candidate)
        if os.path.exists(scaler_path):
            try:
                return joblib.load(scaler_path)
            except Exception:
                continue
    return None


def discover_best_model(game_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    best_result = None
    first_available = None  # fallback when no metrics have RMSE

    for model_name in MODEL_ORDER:
        model_dir, model_file = build_model_path(model_name, game_name)
        # Must have a model file to be usable
        if not os.path.isdir(model_dir) or not os.path.exists(os.path.join(model_dir, model_file)):
            continue
        if first_available is None:
            first_available = (model_name, {})
        metrics = _load_metrics(model_dir, model_name)
        if not metrics:
            continue
        rmse = metrics.get("evaluation", {}).get("RMSE") or metrics.get("RMSE")
        if rmse is None:
            continue
        if best_result is None or rmse < best_result[1].get("rmse", float("inf")):
            best_result = (model_name, {"rmse": float(rmse), "metrics": metrics})

    return best_result or first_available


def get_available_models_for_game(game_name: str) -> List[str]:
    available = []
    
    # Check if TensorFlow is available for deep learning models
    try:
        import tensorflow
        tensorflow_available = True
    except ImportError:
        tensorflow_available = False
    
    for model_name in MODEL_ORDER:
        # Skip LSTM/GRU if TensorFlow not available
        if model_name in DEEP_MODELS and not tensorflow_available:
            continue
            
        model_dir, model_file = build_model_path(model_name, game_name)
        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, model_file)):
            available.append(model_name)
    return available


def load_saved_model(model_name: str, game_name: str) -> Tuple[Any, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    model_dir, model_file = build_model_path(model_name, game_name)
    model_path = os.path.join(model_dir, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Saved model file was not found.")

    metrics = _load_metrics(model_dir, model_name)

    if model_file.endswith(".keras"):
        try:
            from keras.models import load_model
            model = load_model(model_path)
            metadata = dict(metrics or {})
            scaler = _load_deep_learning_scaler(model_dir)
            if scaler is not None:
                metadata["scaler"] = scaler
            return model, metadata, metrics
        except ImportError:
            raise RuntimeError("TensorFlow/Keras not installed. Deep learning models (LSTM/GRU) are not available in this deployment.")
        except Exception as exc:
            raise RuntimeError("The deep-learning model could not be loaded.") from exc

    try:
        payload = joblib.load(model_path)
    except EOFError:
        raise RuntimeError(f"The model file for {model_name} is corrupted and cannot be loaded. Please retrain this model.")
    except Exception as exc:
        raise RuntimeError(f"Failed to unpickle the model file: {str(exc)}") from exc
    
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        metadata = payload
    else:
        model = payload
        metadata = {}

    # Merge metrics into metadata for easy access
    if metrics:
        metadata = {**metrics, **metadata}

    return model, metadata, metrics


def _get_game_data_resampled(data: pd.DataFrame, game_name: str) -> pd.DataFrame:
    """Return the full hourly-resampled numeric DataFrame for a game (mirrors training preprocessing)."""
    available_titles = [str(t).strip() for t in data["title"].dropna().unique()]
    resolved_game = _normalize_game_name(game_name, available_titles)
    game_df = data.loc[data["title"] == resolved_game].copy()
    if game_df.empty:
        return pd.DataFrame()

    if "datetime" in game_df.columns:
        game_df["_dt"] = pd.to_datetime(game_df["datetime"])
    elif "timestamp" in game_df.columns:
        game_df["_dt"] = pd.to_datetime(game_df["timestamp"])
    else:
        return pd.DataFrame()

    game_df = game_df.sort_values("_dt").set_index("_dt")
    game_df = game_df[~game_df.index.duplicated(keep="first")]
    numeric_cols = game_df.select_dtypes(include=[np.number]).columns.tolist()
    game_df = game_df[numeric_cols].resample("h").mean().ffill().bfill()
    return game_df


def _compute_tree_feature_row(
    working_history: list,
    timestamp: pd.Timestamp,
    peak_players: float,
    hours_played: float,
    rank_val: float,
    feature_names: List[str],
) -> Dict[str, float]:
    """Compute a single row of tree model features, matching training exactly."""
    h = len(working_history)
    row: Dict[str, float] = {}
    for feat in feature_names:
        if feat == "lag_1":
            row[feat] = float(working_history[-1]) if h >= 1 else 0.0
        elif feat == "lag_24":
            row[feat] = float(working_history[-24]) if h >= 24 else float(working_history[0]) if h > 0 else 0.0
        elif feat == "rolling_mean_24":
            window = working_history[-24:] if h >= 24 else working_history
            row[feat] = float(np.mean(window)) if window else 0.0
        elif feat == "rolling_std_24":
            window = working_history[-24:] if h >= 24 else working_history
            row[feat] = float(np.std(window)) if len(window) > 1 else 0.0
        elif feat == "peak_players":
            row[feat] = peak_players
        elif feat == "hours_played":
            row[feat] = hours_played
        elif feat == "rank":
            row[feat] = rank_val
        elif feat == "hour":
            row[feat] = float(timestamp.hour)
        elif feat == "day_of_week":
            row[feat] = float(timestamp.weekday())
        elif feat == "is_weekend":
            row[feat] = 1.0 if timestamp.weekday() >= 5 else 0.0
        elif feat == "month":
            row[feat] = float(timestamp.month)
        elif feat == "quarter":
            row[feat] = float(timestamp.quarter)
        elif feat == "week_of_year":
            row[feat] = float(timestamp.isocalendar().week)
        elif feat == "hour_sin":
            row[feat] = float(np.sin(2 * np.pi * timestamp.hour / 24.0))
        elif feat == "hour_cos":
            row[feat] = float(np.cos(2 * np.pi * timestamp.hour / 24.0))
        else:
            row[feat] = 0.0
    return row


def _get_tree_feature_names(model: Any) -> List[str]:
    """Extract actual feature names from the trained tree model object."""
    if hasattr(model, "feature_names_in_"):
        return [str(f) for f in model.feature_names_in_]
    elif hasattr(model, "feature_name_") and callable(model.feature_name_):
        return [str(f) for f in model.feature_name_()]
    elif hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_names"):
        fn = model.get_booster().feature_names
        if fn:
            return [str(f) for f in fn]
    return TREE_FEATURE_COLUMNS  # fallback


def _recursive_tree_forecast(
    model: Any,
    history: pd.Series,
    steps: int,
    data: pd.DataFrame,
    game_name: str,
) -> np.ndarray:
    """Autoregressive forecast for XGBoost/LightGBM with exact training features."""
    feature_names = _get_tree_feature_names(model)

    # Get latest static exog values from resampled game data
    game_df = _get_game_data_resampled(data, game_name)
    peak_players = float(game_df["peak_players"].iloc[-1]) if "peak_players" in game_df.columns and not game_df.empty else 0.0
    hours_played = float(game_df["hours_played"].iloc[-1]) if "hours_played" in game_df.columns and not game_df.empty else 0.0
    rank_val = float(game_df["rank"].iloc[-1]) if "rank" in game_df.columns and not game_df.empty else 0.0

    working_history = list(history.astype(float).tolist())
    forecast_values: List[float] = []

    for step in range(steps):
        timestamp = history.index[-1] + pd.Timedelta(hours=step + 1)
        row = _compute_tree_feature_row(working_history, timestamp, peak_players, hours_played, rank_val, feature_names)
        frame = pd.DataFrame([row], columns=feature_names)

        try:
            pred = float(model.predict(frame)[0])
        except Exception:
            pred = working_history[-1]

        pred = max(0.0, pred)
        forecast_values.append(pred)
        working_history.append(pred)

    return np.asarray(forecast_values, dtype=float)


def _looks_like_constant_or_copied_predictions(predictions: np.ndarray, history: pd.Series) -> bool:
    """Detect repeated, copied, constant, or near-zero forecasts before returning them."""
    arr = np.asarray(predictions, dtype=float).ravel()
    if arr.size == 0:
        return False
    # For single-step forecasts, skip constant detection (it's a single value)
    if arr.size == 1:
        return False
    
    # Check if all predictions are near-zero (broken model)
    if np.all(np.abs(arr) < 1e-6):
        return True
    
    # Relax: only flag if ALL values are EXACTLY identical (bit-for-bit)
    # This allows statsmodels to return slowly-changing forecasts
    if np.all(arr == arr[0]):
        return True
    
    return False


def _forecast_deep_learning(
    model: Any,
    history: pd.Series,
    steps: int,
    metadata: Dict[str, Any],
    data: pd.DataFrame,
    game_name: str,
) -> np.ndarray:
    """Autoregressive sequence forecast for LSTM/GRU with MinMaxScaler reconstruction."""
    seq_len = int(metadata.get("best_config", {}).get("seq", 24) or 24)

    # Reconstruct the full numeric feature matrix (same as training)
    game_df = _get_game_data_resampled(data, game_name)
    if game_df.empty or "current_players" not in game_df.columns:
        # Fallback: univariate on history only
        game_df = pd.DataFrame({"current_players": history.values}, index=history.index)

    # Put target first (matches training: current_players first col)
    cols = ["current_players"] + [c for c in game_df.columns if c != "current_players"]
    game_df = game_df[cols]

    n_features = int(model.input_shape[-1]) if getattr(model, "input_shape", None) else len(cols)
    # Use only as many columns as the model was trained on, in the same order used at training time.
    if len(cols) >= n_features:
        game_df = game_df.iloc[:, :n_features]
    else:
        # Pad with zeros if we have fewer columns
        for i in range(n_features - len(cols)):
            game_df[f"_pad_{i}"] = 0.0

    values = game_df.values.astype(float)

    scaler = metadata.get("scaler")
    if scaler is None:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(values)

    scaled = scaler.transform(values)
    target_idx = 0  # current_players is always col 0

    # Seed the rolling window from the last seq_len rows; never use a single row.
    seed_window = scaled[-seq_len:].copy()  # shape (seq_len, n_features)
    if len(seed_window) < seq_len:
        pad = np.zeros((seq_len - len(seed_window), n_features))
        seed_window = np.vstack([pad, seed_window])
    elif len(seed_window) > seq_len:
        seed_window = seed_window[-seq_len:]

    forecast_values: List[float] = []
    current_window = seed_window.copy()
    
    # Track the last timestamp to generate time features
    last_timestamp = history.index[-1] if hasattr(history.index, '__getitem__') else pd.Timestamp.now()

    for step in range(steps):
        x_input = current_window.reshape(1, seq_len, n_features)
        try:
            pred_scaled = float(model.predict(x_input, verbose=0)[0, 0])
        except Exception:
            pred_scaled = float(current_window[-1, target_idx])

        pred_scaled = np.clip(pred_scaled, 0.0, 1.0)

        # Build next row: update target and regenerate time-based features
        next_row = current_window[-1].copy()
        next_row[target_idx] = pred_scaled
        
        # Update time-based features if they exist in the data
        future_timestamp = last_timestamp + pd.Timedelta(hours=step + 1)
        if n_features > 1:
            # Try to update hour, day_of_week, hour_sin, hour_cos if present
            # This is a best-effort update based on common feature positions
            # Most training sets have: [current_players, hour, lag_1, lag_24, hour_cos, day_of_week, ...]
            try:
                if n_features >= 2:  # hour feature
                    hour_val = float(future_timestamp.hour) / 23.0  # normalize
                    next_row[1] = hour_val
                if n_features >= 5:  # hour_cos feature
                    hour_cos = float(np.cos(2 * np.pi * future_timestamp.hour / 24.0))
                    # Assume hour_cos is around index 4 (common pattern)
                    next_row[4] = (hour_cos + 1.0) / 2.0  # normalize to [0,1]
                if n_features >= 6:  # day_of_week feature
                    dow = float(future_timestamp.weekday()) / 6.0  # normalize
                    next_row[5] = dow
            except Exception:
                pass  # Keep original features if update fails
        
        current_window = np.vstack([current_window[1:], next_row])

        # Inverse-scale only the target column
        dummy = np.zeros((1, n_features))
        dummy[0, target_idx] = pred_scaled
        pred_real = float(scaler.inverse_transform(dummy)[0, target_idx])
        forecast_values.append(max(0.0, pred_real))

    return np.asarray(forecast_values, dtype=float)


def _build_exog_for_forecast(
    steps: int,
    base_timestamp: pd.Timestamp,
    feature_names: List[str],
    history: pd.Series,
    data: pd.DataFrame,
    game_name: str,
    scale_factor: float = 1.0,
) -> Optional[np.ndarray]:
    """Build the exog matrix for ARIMAX/SARIMAX inference, matching training feature computation."""
    if not feature_names:
        return None

    game_df = _get_game_data_resampled(data, game_name)
    peak_players = float(game_df["peak_players"].iloc[-1]) if "peak_players" in game_df.columns and not game_df.empty else 0.0
    hours_played = float(game_df["hours_played"].iloc[-1]) if "hours_played" in game_df.columns and not game_df.empty else 0.0
    rank_val = float(game_df["rank"].iloc[-1]) if "rank" in game_df.columns and not game_df.empty else 0.0

    # For ARIMAX/SARIMAX the endogenous (target) series was scaled by `scale_factor` during training.
    # Lag values of the scaled target must also be in scaled space.
    working_history = [v / scale_factor for v in history.values]

    rows = []
    for step in range(steps):
        ts = base_timestamp + pd.Timedelta(hours=step + 1)
        row: Dict[str, float] = {}
        for feat in feature_names:
            if feat == "lag_1":
                row[feat] = float(working_history[-1]) if working_history else 0.0
            elif feat == "lag_24":
                row[feat] = float(working_history[-24]) if len(working_history) >= 24 else (float(working_history[0]) if working_history else 0.0)
            elif feat == "rolling_mean_24":
                w = working_history[-24:] if len(working_history) >= 24 else working_history
                row[feat] = float(np.mean(w)) if w else 0.0
            elif feat == "rolling_std_24":
                w = working_history[-24:] if len(working_history) >= 24 else working_history
                row[feat] = float(np.std(w)) if len(w) > 1 else 0.0
            elif feat == "peak_players":
                row[feat] = peak_players / scale_factor if scale_factor else peak_players
            elif feat == "hours_played":
                row[feat] = hours_played
            elif feat == "rank":
                row[feat] = rank_val
            elif feat == "hour":
                row[feat] = float(ts.hour)
            elif feat == "day_of_week":
                row[feat] = float(ts.weekday())
            elif feat == "is_weekend":
                row[feat] = 1.0 if ts.weekday() >= 5 else 0.0
            elif feat == "quarter":
                row[feat] = float(ts.quarter)
            elif feat == "hour_sin":
                row[feat] = float(np.sin(2 * np.pi * ts.hour / 24.0))
            elif feat == "hour_cos":
                row[feat] = float(np.cos(2 * np.pi * ts.hour / 24.0))
            else:
                row[feat] = 0.0
        rows.append([row[f] for f in feature_names])
        # Extend working_history with a copy of last value (we don't know future target)
        working_history.append(working_history[-1] if working_history else 0.0)

    return np.asarray(rows, dtype=float)


def _forecast_statsmodels(
    model: Any,
    history: pd.Series,
    steps: int,
    metadata: Dict[str, Any],
    data: pd.DataFrame,
    game_name: str,
    model_name: str,
) -> np.ndarray:
    """Produce a forecast from any statsmodels fitted result object using proper multi-step forecasting."""
    needs_exog = model_name in {"ARIMAX", "SARIMAX"}
    base_timestamp = history.index[-1]

    scale_factor = float(metadata.get("scale", 1.0) or 1.0)
    selected_exog_features: List[str] = metadata.get("selected_exog_features", [])

    # Fix deprecated freq attribute if present
    for attr_path in [("data", "freq"), ("model", "data", "freq"), ("model", "model", "data", "freq")]:
        obj = model
        try:
            for attr in attr_path[:-1]:
                obj = getattr(obj, attr)
            if getattr(obj, attr_path[-1], None) == "H":
                setattr(obj, attr_path[-1], "h")
        except Exception:
            pass

    preds_unscaled: List[float] = []

    # For ARIMAX/SARIMAX, build all exog rows at once
    exog_data = None
    if needs_exog:
        feature_names = [str(f) for f in selected_exog_features] if selected_exog_features else EXOG_CANDIDATES
        exog_data = _build_exog_for_forecast(steps, base_timestamp, feature_names, history, data, game_name, scale_factor)

    # Try using get_forecast() which is the proper method for statsmodels
    if hasattr(model, "get_forecast"):
        try:
            if needs_exog and exog_data is not None:
                fc = model.get_forecast(steps=steps, exog=exog_data)
            else:
                fc = model.get_forecast(steps=steps)
            
            forecasts = fc.predicted_mean.values
            for f in forecasts:
                pred_scaled = float(f)
                pred_unscaled = pred_scaled * scale_factor
                preds_unscaled.append(max(0.0, pred_unscaled))
            
            if len(preds_unscaled) == steps:
                return np.asarray(preds_unscaled, dtype=float)
            preds_unscaled = []
        except Exception:
            pass

    # Fallback: use forecast() method
    if hasattr(model, "forecast"):
        try:
            if needs_exog and exog_data is not None:
                forecasts = model.forecast(steps=steps, exog=exog_data)
            else:
                forecasts = model.forecast(steps=steps)
            
            for f in forecasts:
                pred_scaled = float(f)
                pred_unscaled = pred_scaled * scale_factor
                preds_unscaled.append(max(0.0, pred_unscaled))
            
            if len(preds_unscaled) == steps:
                return np.asarray(preds_unscaled, dtype=float)
        except Exception:
            pass

    # If we still have no predictions, raise an error
    if not preds_unscaled:
        raise ValueError("The statsmodels model could not produce a forward-looking forecast.")


def predict_with_saved_model(
    data: pd.DataFrame,
    game_name: str,
    model_name: str,
    horizon_hours: int,
    horizon_label: str,
) -> Dict[str, Any]:
    # Resolve "Best Model" shortcut before any loading
    if model_name == "⭐ Best Model":
        best = discover_best_model(game_name)
        if not best:
            return {"success": False, "message": "Could not determine the best model for this game."}
        resolved_model_name = best[0]
    else:
        resolved_model_name = model_name

    try:
        history = pd.Series(prepare_game_history(data, game_name))
    except Exception as exc:
        return {"success": False, "message": f"The selected game could not be loaded from the dataset: {exc}"}

    if history.empty:
        return {"success": False, "message": "There is not enough historical data for forecasting."}

    try:
        # Load the actual model using the resolved name
        model, metadata, metrics = load_saved_model(resolved_model_name, game_name)
    except FileNotFoundError:
        return {"success": False, "message": f"The selected model file is missing for this game ({resolved_model_name})."}
    except RuntimeError as exc:
        return {"success": False, "message": str(exc)}
    except Exception as exc:
        import traceback
        error_detail = traceback.format_exc()
        return {"success": False, "message": f"Failed to load the model: {str(exc)}\n{error_detail}"}

    metadata = metadata or {}

    try:
        if resolved_model_name in TREE_MODELS:
            forecast = _recursive_tree_forecast(model, history, horizon_hours, data, game_name)
        elif resolved_model_name in DEEP_MODELS:
            forecast = _forecast_deep_learning(model, history, horizon_hours, metadata, data, game_name)
        else:
            forecast = _forecast_statsmodels(model, history, horizon_hours, metadata, data, game_name, resolved_model_name)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"Prediction failed for the selected model: {str(exc)}"}

    # Removed constant-forecast validation as requested by user
    # Some statsmodels ARIMAX/ARIMA models produce slowly-changing multi-step forecasts
    # which are valid but were being incorrectly flagged

    future_index = pd.date_range(
        history.index[-1] + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="h",
    )

    # Build insights (unchanged)
    insights = []
    latest = float(history.iloc[-1])
    last_forecast = float(forecast[-1]) if len(forecast) > 0 else latest
    delta = last_forecast - latest

    if delta > 0.05 * latest:
        insights.append({"title": "Traffic Trend", "body": "A strong increase is expected over the forecast window."})
    elif delta < -0.05 * latest:
        insights.append({"title": "Traffic Trend", "body": "A moderate decline is expected over the forecast window."})
    else:
        insights.append({"title": "Traffic Trend", "body": "Traffic is expected to remain relatively stable."})

    if last_forecast > latest:
        insights.append({"title": "Expected Change", "body": f"Peak activity is expected to rise to about {int(last_forecast):,} players."})
    else:
        insights.append({"title": "Expected Change", "body": f"Activity is expected to soften to about {int(last_forecast):,} players."})

    insights.append({"title": "Forecast Confidence", "body": "The forecast is generated from the selected pretrained model."})

    return {
        "success": True,
        "model_name": resolved_model_name,
        "history": history,
        "forecast": forecast,
        "future_index": future_index,
        "prediction_time": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "horizon_label": horizon_label,
        "insights": insights,
    }

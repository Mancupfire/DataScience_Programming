import pandas as pd
import numpy as np
import lightgbm as lgb
import os

def load_lightgbm_model(model_path):
    """Loads the trained LightGBM model."""
    if not os.path.exists(model_path):
        return None
    model = lgb.Booster(model_file=model_path)
    return model

def create_temporal_features(df, target_col='PM2.5'):
    """
    Creates temporal features for a single row or dataframe.
    Must match the feature engineering in the training notebook.
    """
    df_features = df.copy()
    
    # Ensure datetime index
    if not isinstance(df_features.index, pd.DatetimeIndex):
        try:
            df_features.index = pd.to_datetime(df_features.index)
        except:
            pass # Handle cases where index isn't datetime if needed, but for forecasting it should be

    # Time-based features
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)
    
    # Note: Lag and rolling features are created dynamically during recursive forecasting
    return df_features

def recursive_forecast(model, initial_data, horizon_hours, target_col='PM2.5', base_features=None):
    """
    Performs recursive forecasting for the specified horizon.
    
    Args:
        model: Trained LightGBM Booster model.
        initial_data: DataFrame containing historical data (enough for lags).
        horizon_hours: Number of hours to forecast.
        target_col: Name of the target column.
        base_features: List of base feature names (excluding lags/rolling).
        
    Returns:
        List of forecasted values.
    """
    if base_features is None:
        # Default base features from notebook
        base_features = ['CO2', 'TVOC', 'PM10', 'Air Quality', 'CO', 'O3', 'Temp', 'Hum', 'LDR']

    # We need the last portion of data to calculate lags for the first prediction
    # Max lag is 24 hours, max rolling window is 24 hours.
    # We need at least 24 hours of history.
    history = initial_data.copy()
    
    forecasts = []
    current_time = history.index[-1]
    
    # Calculate number of steps based on 5-minute intervals
    # Training data has 5-minute frequency, so we MUST forecast at 5-minute steps
    steps_per_hour = 12 # 60 / 5
    total_steps = int(horizon_hours * steps_per_hour)
    
    for i in range(total_steps):
        next_time = current_time + pd.Timedelta(minutes=5)
        
        # Create a new row for the next timestamp
        # For exogenous variables (like Temp, Hum), we apply a simple linear trend based on recent history
        # to make the forecast more dynamic.
        next_row = history.iloc[[-1]].copy()
        next_row.index = [next_time]
        
        # Calculate trends for exogenous variables
        # We use the last 6 hours to estimate a slope
        trend_window = 6 * steps_per_hour # 6 hours * 12 steps/hour = 72 steps
        if len(history) >= trend_window:
            recent_history = history.iloc[-trend_window:]
            
            # Identify exogenous columns (everything except target and time features)
            # We exclude lag/rolling features as they are re-calculated
            exclude_cols = [target_col, 'hour', 'day_of_week', 'month', 'is_weekend']
            exo_cols = [c for c in history.columns if c not in exclude_cols and not c.endswith(('_lag_', '_rolling_'))]
            
            for col in exo_cols:
                if pd.api.types.is_numeric_dtype(recent_history[col]):
                    # Simple linear slope: (y_end - y_start) / steps
                    # We add a small damping factor to prevent runaway trends over long horizons
                    y = recent_history[col].values
                    slope = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0
                    
                    # Apply trend with damping (e.g., 0.9 decay per step for slope)
                    # For this iteration i (0 to horizon-1), we project from the last known value
                    # However, next_row is initialized with the last value, so we just add the slope
                    
                    # To keep it simple and stable: just add the slope
                    # But maybe dampen it if it's too steep? 
                    # Let's just add the slope for now, but clamp it if needed?
                    # For simplicity: new_val = last_val + slope
                    
                    new_val = next_row[col].values[0] + slope
                    next_row[col] = new_val

        
        # Update time-based features
        next_row['hour'] = next_time.hour
        next_row['day_of_week'] = next_time.dayofweek
        next_row['month'] = next_time.month
        next_row['is_weekend'] = int(next_time.dayofweek >= 5)
        
        # Append to history temporarily to calculate lags
        history_extended = pd.concat([history, next_row])
        
        # Re-calculate lag and rolling features for this new row
        # This is computationally expensive but ensures correctness with the recursive target
        
        # Lag features for target
        for lag in [1, 2, 3, 6, 12, 24]:
            history_extended[f'{target_col}_lag_{lag}'] = history_extended[target_col].shift(lag)
            
        # Lag features for key variables
        key_vars = ['CO2', 'TVOC', 'PM10']
        for var in key_vars:
            if var in history_extended.columns:
                for lag in [1, 3, 6]:
                    history_extended[f'{var}_lag_{lag}'] = history_extended[var].shift(lag)
        
        # Rolling statistics for target
        for window in [3, 6, 12, 24]:
            history_extended[f'{target_col}_rolling_mean_{window}'] = history_extended[target_col].rolling(window=window).mean()
        
        for window in [6, 12, 24]:
            history_extended[f'{target_col}_rolling_std_{window}'] = history_extended[target_col].rolling(window=window).std()
            
        for window in [12, 24]:
            history_extended[f'{target_col}_rolling_min_{window}'] = history_extended[target_col].rolling(window=window).min()
            history_extended[f'{target_col}_rolling_max_{window}'] = history_extended[target_col].rolling(window=window).max()
            
        # Get the feature vector for the prediction time
        # LightGBM Booster expects specific column order matching training
        
        # Extract the row to predict
        pred_row = history_extended.iloc[[-1]]
        
        # Drop target if present (it shouldn't be used as feature, but lags should)
        if target_col in pred_row.columns:
            pred_row = pred_row.drop(columns=[target_col])
            
        # Make prediction
        # Note: We need to ensure we only pass the features the model expects.
        # We explicitly select only the temporal features (lags, rolling, time)
        # to match the training data which excluded concurrent sensor values.
        model_features = [c for c in pred_row.columns if '_lag_' in c or '_rolling_' in c or c in ['hour', 'day_of_week', 'month', 'is_weekend']]
        pred_row = pred_row[model_features]
        
        # LightGBM Booster uses predict() directly on the data
        prediction = model.predict(pred_row)[0]
        forecasts.append(prediction)
        
        # Update the history with the predicted value for the next iteration
        history_extended.loc[next_time, target_col] = prediction
        history = history_extended
        current_time = next_time
        
    return forecasts

"""
LSTM-based next-day price prediction helper (module4)

This module provides simple utilities to train a Keras LSTM on historical
stock prices (OHLCV) and predict the next-day 'best' price (Close by default).

Functions:
- prepare_data(df, feature='Close', window=60): build sliding windows
- build_lstm_model(input_shape): returns compiled Keras model
- train_model(...): trains model with EarlyStopping and optional checkpointing
- predict_next_day_from_df(df, model_path=None, window=60, retrain=False, **train_kwargs)

Notes:
- This is a lightweight, single-file implementation intended for experimentation
  and small datasets. For production use, consider more robust preprocessing,
  hyperparameter tuning, and model management.

Dependencies: pandas, numpy, tensorflow (>=2.x), scikit-learn

"""

from typing import Tuple, Optional
import os
import numpy as np
import pandas as pd
import logging

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except Exception as e:
    raise ImportError(
        "TensorFlow is required for prediction_module4. Install tensorflow (e.g. pip install tensorflow)."
    )

from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def prepare_data(
    df: pd.DataFrame,
    feature: str = "Close",
    window: int = 60,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Prepare sliding-window sequences for LSTM.

    Args:
        df: DataFrame containing OHLCV with a 'Close' column (or other feature).
        feature: column to predict.
        window: number of past timesteps to use.

    Returns:
        X: shape (n_samples, window, 1)
        y: shape (n_samples,)
        scaler: fitted MinMaxScaler used to scale the feature
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in dataframe columns")

    series = df[feature].astype('float32').copy()
    # Ensure chronological order
    series = series.sort_index()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    # reshape for LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler


def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Construct a small LSTM model.

    Args:
        input_shape: (timesteps, features)

    Returns:
        compiled Keras model
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 50,
    batch_size: int = 32,
    model_path: Optional[str] = None,
    patience: int = 8,
    verbose: int = 1,
) -> tf.keras.Model:
    """Train the model with early stopping and optional checkpointing.

    Returns the trained model (and writes checkpoint if model_path provided).
    """
    callbacks = [EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', patience=patience, restore_best_weights=True)]
    if model_path:
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        callbacks.append(ModelCheckpoint(model_path, monitor='val_loss' if X_val is not None else 'loss', save_best_only=True))

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    return model


def predict_next_day(
    model: tf.keras.Model,
    recent_sequence: np.ndarray,
    scaler: MinMaxScaler,
) -> float:
    """Predict the next value (in original scale) from a recent scaled sequence.

    Args:
        model: trained Keras model
        recent_sequence: array shape (window, 1) or (1, window, 1)
        scaler: fitted MinMaxScaler to inverse-transform prediction

    Returns:
        predicted value (unscaled float)
    """
    seq = np.array(recent_sequence)
    if seq.ndim == 2:
        seq = seq.reshape((1, seq.shape[0], seq.shape[1]))
    pred_scaled = model.predict(seq)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
    return float(pred)


def save_model(model: tf.keras.Model, path: str) -> None:
    model.save(path)


def load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)


def predict_next_day_from_df(
    df: pd.DataFrame,
    feature: str = 'Close',
    window: int = 60,
    model_path: Optional[str] = None,
    retrain: bool = False,
    epochs: int = 20,
    batch_size: int = 32,
    val_split: float = 0.1,
) -> Tuple[float, dict]:
    """End-to-end helper: given historical OHLCV DataFrame, return predicted next-day price.

    Args:
        df: DataFrame indexed by date with at least `feature` column
        model_path: optional path to load/save a model
        retrain: if True, train a new model even if model_path exists
        epochs, batch_size: training hyperparams

    Returns:
        predicted_price, metadata dict
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in df")

    X, y, scaler = prepare_data(df, feature=feature, window=window)

    if len(X) == 0:
        raise ValueError("Not enough data to create any training samples - increase data length or reduce window")

    # split into train/val
    n_val = max(1, int(len(X) * val_split))
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    model = None
    if model_path and os.path.exists(model_path) and not retrain:
        try:
            model = load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception:
            logger.info("Failed to load model from path; will train a new one")

    if model is None:
        model = build_lstm_model(input_shape=(window, 1))
        model = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_path=model_path,
        )
        if model_path:
            try:
                save_model(model, model_path)
                logger.info(f"Saved model to {model_path}")
            except Exception as e:
                logger.warning(f"Could not save model to {model_path}: {e}")

    # prepare recent sequence (last `window` rows)
    recent = df[feature].sort_index().values[-window:]
    recent_scaled = scaler.transform(recent.reshape(-1, 1))
    predicted = predict_next_day(model, recent_scaled.reshape(window, 1), scaler)

    meta = {
        'window': window,
        'feature': feature,
        'model_path': model_path,
        'trained_samples': len(X_train),
    }

    return predicted, meta


if __name__ == '__main__':
    print("prediction_module4: helpers for LSTM prediction. Import and call functions from your app.")

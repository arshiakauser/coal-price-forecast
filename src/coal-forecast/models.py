"""Forecasting models behind one interface.

Every model is a function ``f(history, horizon, seed) -> np.ndarray`` that
takes the observed prices (USD/t, unscaled, 1-D array) and returns
``horizon`` multi-step-ahead forecasts in USD/t. Any scaling happens inside
the function and is fitted on ``history`` only, so nothing from the
evaluation period can leak into training.

statsmodels / TensorFlow are imported lazily: if a library is missing, only
the models that need it are skipped.
"""
from __future__ import annotations

import numpy as np

N_LAGS = 10    # lagged daily changes used by the ML models
WINDOW = 20    # sequence length used by the neural networks
RECENT = 250   # observations used by drift / exponential smoothing / ARIMA


# --------------------------------------------------------------------- baselines
def naive(history, horizon, seed=42):
    """Tomorrow = today. Any real model must beat this."""
    return np.full(horizon, float(history[-1]))


def drift(history, horizon, seed=42):
    """Last value plus the average daily change of the recent past."""
    h = np.asarray(history, float)[-RECENT:]
    slope = (h[-1] - h[0]) / max(len(h) - 1, 1)
    return h[-1] + slope * np.arange(1, horizon + 1)


# ------------------------------------------------------------ statistical models
def ses(history, horizon, seed=42):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    h = np.asarray(history, float)[-RECENT:]
    fit = SimpleExpSmoothing(h, initialization_method="estimated").fit()
    return np.asarray(fit.forecast(horizon))


def holt_damped(history, horizon, seed=42):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    h = np.asarray(history, float)[-RECENT:]
    fit = ExponentialSmoothing(
        h, trend="add", damped_trend=True, initialization_method="estimated"
    ).fit()
    return np.asarray(fit.forecast(horizon))


def arima(history, horizon, seed=42):
    from statsmodels.tsa.arima.model import ARIMA
    h = np.asarray(history, float)[-RECENT:]
    fit = ARIMA(h, order=(5, 1, 0)).fit()
    return np.asarray(fit.forecast(horizon))


# ------------------------------------------------- ML models on lagged changes
def _recursive_diff_forecast(estimator, history, horizon):
    """Fit on lagged daily changes, then roll forward one step at a time.

    Modelling *changes* (not the level or a time index) lets tree models and
    SVR work on stationary inputs instead of flat-lining outside the training
    range.
    """
    h = np.asarray(history, float)
    d = np.diff(h)
    sd = d.std() or 1.0
    d = d / sd
    X = np.array([d[i:i + N_LAGS] for i in range(len(d) - N_LAGS)])
    y = d[N_LAGS:]
    estimator.fit(X, y)

    window = list(d[-N_LAGS:])
    level, out = h[-1], []
    for _ in range(horizon):
        step = float(estimator.predict(np.asarray(window)[None, :])[0])
        level += step * sd
        out.append(level)
        window = window[1:] + [step]
    return np.array(out)


def linear_regression(history, horizon, seed=42):
    from sklearn.linear_model import LinearRegression
    return _recursive_diff_forecast(LinearRegression(), history, horizon)


def random_forest(history, horizon, seed=42):
    from sklearn.ensemble import RandomForestRegressor
    est = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1
    )
    return _recursive_diff_forecast(est, history, horizon)


def svr(history, horizon, seed=42):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    est = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1))
    return _recursive_diff_forecast(est, history, horizon)


# ------------------------------------------------------------ neural networks
def _build(kind: str, window: int):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        GRU, LSTM, Bidirectional, Conv1D, Dense, Dropout, Input, MaxPooling1D,
    )
    layers = [Input(shape=(window, 1))]
    if kind == "lstm":
        layers += [LSTM(64)]
    elif kind == "bilstm":
        layers += [Bidirectional(LSTM(64))]
    elif kind == "gru":
        layers += [GRU(64)]
    elif kind == "cnn_lstm":
        layers += [Conv1D(64, 3, activation="relu"), MaxPooling1D(2), LSTM(64), Dropout(0.2)]
    else:
        raise ValueError(kind)
    layers += [Dense(1)]
    return Sequential(layers)


def _nn_forecast(kind, history, horizon, seed):
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler

    tf.keras.utils.set_random_seed(seed)
    h = np.asarray(history, float).reshape(-1, 1)
    scaler = MinMaxScaler().fit(h)                 # fitted on history only
    s = scaler.transform(h).ravel()

    X = np.array([s[i:i + WINDOW] for i in range(len(s) - WINDOW)])[..., None]
    y = s[WINDOW:]
    model = _build(kind, WINDOW)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1,
              callbacks=[stop], verbose=0)

    window, out = list(s[-WINDOW:]), []
    for _ in range(horizon):
        x = np.asarray(window, dtype="float32").reshape(1, WINDOW, 1)
        nxt = float(model(x, training=False).numpy()[0, 0])
        out.append(nxt)
        window = window[1:] + [nxt]
    return scaler.inverse_transform(np.array(out).reshape(-1, 1)).ravel()


def lstm(history, horizon, seed=42):
    return _nn_forecast("lstm", history, horizon, seed)


def bilstm(history, horizon, seed=42):
    return _nn_forecast("bilstm", history, horizon, seed)


def gru(history, horizon, seed=42):
    return _nn_forecast("gru", history, horizon, seed)


def cnn_lstm(history, horizon, seed=42):
    return _nn_forecast("cnn_lstm", history, horizon, seed)


# -------------------------------------------------------------------- registry
MODELS = {
    "Naive": naive,
    "Drift": drift,
    "SES": ses,
    "Holt (damped)": holt_damped,
    "ARIMA(5,1,0)": arima,
    "Linear Regression (lags)": linear_regression,
    "Random Forest (lags)": random_forest,
    "SVR (lags)": svr,
    "LSTM": lstm,
    "BiLSTM": bilstm,
    "GRU": gru,
    "CNN-LSTM": cnn_lstm,
}
BASELINES = ("Naive", "Drift")
DEEP = ("LSTM", "BiLSTM", "GRU", "CNN-LSTM")

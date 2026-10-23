"""
create_dummy_models.py
----------------------
Standalone script to generate and export dummy Scikit-Learn LinearRegression models
for Solar and Wind power prediction.

Models generated:
  1. models/solar_model.pkl
     Inputs: temperature (°C), irradiance (W/m²), humidity (%)
     Output: power (kW)

  2. models/wind_model.pkl
     Inputs: wind_speed (m/s), wind_direction (degrees), air_pressure (hPa)
     Output: power (kW)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def generate_solar_model(output_path: Path) -> LinearRegression:
    """
    Trains a LinearRegression model for solar power prediction based on:
      - temperature (°C)
      - irradiance (W/m²)
      - humidity (%)
    """
    np.random.seed(42)
    n_samples = 2500

    # Realistic synthetic operational ranges
    temperature = np.random.uniform(5.0, 48.0, n_samples)
    irradiance = np.random.uniform(50.0, 1100.0, n_samples)
    humidity = np.random.uniform(15.0, 95.0, n_samples)

    # Simplified physical correlation:
    # - Solar power scales primarily with irradiance
    # - High temperatures cause slight thermal panel degradation (-0.08 kW per °C above 25°C)
    # - High humidity correlates with atmospheric scattering / haze (-0.03 kW per %)
    noise = np.random.normal(0, 1.2, n_samples)
    power_kw = (
        0.045 * irradiance
        - 0.08 * (temperature - 25.0)
        - 0.035 * humidity
        + noise
    )
    power_kw = np.clip(power_kw, 0.0, None)  # Power cannot be negative

    X = pd.DataFrame({
        "temperature": temperature,
        "irradiance": irradiance,
        "humidity": humidity,
    })
    y = power_kw

    model = LinearRegression()
    model.fit(X, y)

    # Export model artifact
    joblib.dump(model, output_path)
    print(f"[OK] Solar model saved to: {output_path.resolve()}")
    print(f"    Coefficients: {dict(zip(X.columns, np.round(model.coef_, 4)))}")
    print(f"    Intercept:    {round(model.intercept_, 4)}")

    # Verification sample
    sample = pd.DataFrame([{"temperature": 28.5, "irradiance": 800.0, "humidity": 45.0}])
    pred = max(0.0, float(model.predict(sample)[0]))
    print(f"    Test prediction for {sample.to_dict(orient='records')[0]}: {pred:.2f} kW\n")
    return model


def generate_wind_model(output_path: Path) -> LinearRegression:
    """
    Trains a LinearRegression model for wind turbine power prediction based on:
      - wind_speed (m/s)
      - wind_direction (degrees 0-360)
      - air_pressure (hPa)
    """
    np.random.seed(42)
    n_samples = 2500

    # Realistic synthetic operational ranges
    wind_speed = np.random.uniform(1.0, 25.0, n_samples)
    wind_direction = np.random.uniform(0.0, 360.0, n_samples)
    air_pressure = np.random.uniform(970.0, 1040.0, n_samples)

    # Simplified physical correlation:
    # - Wind power strongly scales with wind speed (approx. linear proxy in LinearRegression)
    # - Higher air pressure (density) increases energy transfer
    # - Minor directional bias based on turbine prevailing alignment
    noise = np.random.normal(0, 2.5, n_samples)
    effective_speed = np.maximum(wind_speed - 2.5, 0.0)  # Cut-in wind speed threshold ~2.5 m/s
    power_kw = (
        3.2 * effective_speed
        + 0.012 * (air_pressure - 1013.25)
        + 0.008 * np.cos(np.radians(wind_direction))
        + noise
    )
    power_kw = np.clip(power_kw, 0.0, None)  # Power cannot be negative

    X = pd.DataFrame({
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "air_pressure": air_pressure,
    })
    y = power_kw

    model = LinearRegression()
    model.fit(X, y)

    # Export model artifact
    joblib.dump(model, output_path)
    print(f"[OK] Wind model saved to:  {output_path.resolve()}")
    print(f"    Coefficients: {dict(zip(X.columns, np.round(model.coef_, 4)))}")
    print(f"    Intercept:    {round(model.intercept_, 4)}")

    # Verification sample
    sample = pd.DataFrame([{"wind_speed": 12.0, "wind_direction": 180.0, "air_pressure": 1013.25}])
    pred = max(0.0, float(model.predict(sample)[0]))
    print(f"    Test prediction for {sample.to_dict(orient='records')[0]}: {pred:.2f} kW\n")
    return model


def generate_solar_xgboost(output_path: Path):
    """Trains an XGBoost regressor for high-accuracy solar forecasting."""
    if not XGBOOST_AVAILABLE:
        print("[SKIP] XGBoost not installed; skipping XGBoost solar model.")
        return None
    np.random.seed(42)
    n_samples = 3000
    temperature = np.random.uniform(5.0, 48.0, n_samples)
    irradiance = np.random.uniform(50.0, 1100.0, n_samples)
    humidity = np.random.uniform(15.0, 95.0, n_samples)
    noise = np.random.normal(0, 0.8, n_samples)
    power_kw = np.clip(
        0.045 * irradiance - 0.08 * (temperature - 25.0) - 0.035 * humidity + noise,
        0.0, None
    )
    X = pd.DataFrame({"temperature": temperature, "irradiance": irradiance, "humidity": humidity})
    y = power_kw
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
    model.fit(X, y)
    joblib.dump(model, output_path)
    print(f"[OK] Solar XGBoost model saved to: {output_path.resolve()}")
    return model


def generate_wind_xgboost(output_path: Path):
    """Trains an XGBoost regressor for high-accuracy wind forecasting."""
    if not XGBOOST_AVAILABLE:
        print("[SKIP] XGBoost not installed; skipping XGBoost wind model.")
        return None
    np.random.seed(42)
    n_samples = 3000
    wind_speed = np.random.uniform(1.0, 25.0, n_samples)
    wind_direction = np.random.uniform(0.0, 360.0, n_samples)
    air_pressure = np.random.uniform(970.0, 1040.0, n_samples)
    noise = np.random.normal(0, 1.5, n_samples)
    effective_speed = np.maximum(wind_speed - 2.5, 0.0)
    power_kw = np.clip(
        3.2 * effective_speed + 0.012 * (air_pressure - 1013.25) + 0.008 * np.cos(np.radians(wind_direction)) + noise,
        0.0, None
    )
    X = pd.DataFrame({"wind_speed": wind_speed, "wind_direction": wind_direction, "air_pressure": air_pressure})
    y = power_kw
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
    model.fit(X, y)
    joblib.dump(model, output_path)
    print(f"[OK] Wind XGBoost model saved to: {output_path.resolve()}")
    return model


def main():
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("Generating ML Models for RenewEase (Linear & XGBoost)")
    print("==================================================")

    solar_path = models_dir / "solar_model.pkl"
    wind_path = models_dir / "wind_model.pkl"
    solar_xgb_path = models_dir / "solar_xgboost.pkl"
    wind_xgb_path = models_dir / "wind_xgboost.pkl"

    generate_solar_model(solar_path)
    generate_wind_model(wind_path)

    generate_solar_xgboost(solar_xgb_path)
    generate_wind_xgboost(wind_xgb_path)

    print("==================================================")
    print("All models successfully trained and exported!")
    print("==================================================")


if __name__ == "__main__":
    main()


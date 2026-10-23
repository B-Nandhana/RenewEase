"""
model_loader.py
---------------
ModelManager class responsible for locating, safely loading, and serving
predictions from trained Scikit-Learn .pkl models using Joblib and Pandas.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging
import joblib
import pandas as pd

from app.schemas import SolarInput, WindInput, CombinedInput

logger = logging.getLogger("renewable_energy_app.model_loader")


class ModelManager:
    """
    Manages loading and inference for Solar and Wind ML models.
    """

    SOLAR_FEATURES = ["temperature", "irradiance", "humidity"]
    WIND_FEATURES = ["wind_speed", "wind_direction", "air_pressure"]

    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        self.models_dir = self._resolve_models_dir(models_dir)
        self.solar_model: Any = None
        self.wind_model: Any = None
        self.solar_xgb_model: Any = None
        self.wind_xgb_model: Any = None
        self._is_solar_loaded: bool = False
        self._is_wind_loaded: bool = False
        self._is_solar_xgb_loaded: bool = False
        self._is_wind_xgb_loaded: bool = False

    def _resolve_models_dir(self, custom_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Locates the models directory dynamically across common working directory contexts.
        """
        if custom_path:
            p = Path(custom_path).resolve()
            if p.exists():
                return p

        candidates = [
            Path(__file__).resolve().parent.parent / "models",
            Path.cwd() / "models",
            Path.cwd() / "renewable_energy_app" / "models",
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate.resolve()

        # Default fallback to parent/models
        return (Path(__file__).resolve().parent.parent / "models").resolve()

    def load_models(self) -> None:
        """
        Safely loads linear models and XGBoost models from the models directory.
        """
        solar_path = self.models_dir / "solar_model.pkl"
        wind_path = self.models_dir / "wind_model.pkl"
        solar_xgb_path = self.models_dir / "solar_xgboost.pkl"
        wind_xgb_path = self.models_dir / "wind_xgboost.pkl"

        if not solar_path.exists() or not wind_path.exists():
            missing = []
            if not solar_path.exists():
                missing.append(str(solar_path))
            if not wind_path.exists():
                missing.append(str(wind_path))

            error_msg = f"Missing core model file(s): {', '.join(missing)}."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info("Loading solar model from %s", solar_path)
            self.solar_model = joblib.load(solar_path)
            self._is_solar_loaded = True

            logger.info("Loading wind model from %s", wind_path)
            self.wind_model = joblib.load(wind_path)
            self._is_wind_loaded = True

            # Load XGBoost models if present
            if solar_xgb_path.exists():
                try:
                    self.solar_xgb_model = joblib.load(solar_xgb_path)
                    self._is_solar_xgb_loaded = True
                    logger.info("Loaded solar XGBoost model from %s", solar_xgb_path)
                except Exception as xe:
                    logger.warning("Could not load solar XGBoost model: %s", xe)

            if wind_xgb_path.exists():
                try:
                    self.wind_xgb_model = joblib.load(wind_xgb_path)
                    self._is_wind_xgb_loaded = True
                    logger.info("Loaded wind XGBoost model from %s", wind_xgb_path)
                except Exception as xe:
                    logger.warning("Could not load wind XGBoost model: %s", xe)

            logger.info("Predictive models loaded successfully.")
        except Exception as e:
            logger.error("Failed to load models with joblib: %s", str(e), exc_info=True)
            raise RuntimeError(f"Error deserializing model files: {e}") from e

    @property
    def is_ready(self) -> bool:
        """Checks if models are loaded and ready for inference."""
        return self._is_solar_loaded and self._is_wind_loaded

    def predict_solar(self, data: Union[SolarInput, Dict[str, Any]], use_xgboost: bool = True) -> float:
        """
        Predicts solar power output in kW given temperature, irradiance, and humidity.
        """
        if not self._is_solar_loaded or self.solar_model is None:
            raise RuntimeError("Solar model is not loaded. System is calibrating.")

        if isinstance(data, SolarInput):
            features_dict = {
                "temperature": float(data.temperature),
                "irradiance": float(data.irradiance),
                "humidity": float(data.humidity)
            }
        else:
            features_dict = {col: float(data[col]) for col in self.SOLAR_FEATURES}

        df = pd.DataFrame([features_dict], columns=self.SOLAR_FEATURES)

        if use_xgboost and self._is_solar_xgb_loaded and self.solar_xgb_model is not None:
            xgb_pred = float(self.solar_xgb_model.predict(df)[0])
            lin_pred = float(self.solar_model.predict(df)[0])
            # High-accuracy weighted ensemble (70% XGBoost, 30% baseline)
            pred = 0.7 * xgb_pred + 0.3 * lin_pred
        else:
            pred = float(self.solar_model.predict(df)[0])

        return round(max(0.0, pred), 2)

    def predict_wind(self, data: Union[WindInput, Dict[str, Any]], use_xgboost: bool = True) -> float:
        """
        Predicts wind power output in kW given wind_speed, wind_direction, and air_pressure.
        """
        if not self._is_wind_loaded or self.wind_model is None:
            raise RuntimeError("Wind model is not loaded. System is calibrating.")

        if isinstance(data, WindInput):
            features_dict = {
                "wind_speed": float(data.wind_speed),
                "wind_direction": float(data.wind_direction),
                "air_pressure": float(data.air_pressure)
            }
        else:
            features_dict = {col: float(data[col]) for col in self.WIND_FEATURES}

        df = pd.DataFrame([features_dict], columns=self.WIND_FEATURES)

        if use_xgboost and self._is_wind_xgb_loaded and self.wind_xgb_model is not None:
            xgb_pred = float(self.wind_xgb_model.predict(df)[0])
            lin_pred = float(self.wind_model.predict(df)[0])
            pred = 0.7 * xgb_pred + 0.3 * lin_pred
        else:
            pred = float(self.wind_model.predict(df)[0])

        return round(max(0.0, pred), 2)

    def predict_combined(self, data: CombinedInput) -> Dict[str, float]:
        """
        Predicts both solar and wind power, and calculates total generation.
        """
        solar_input = data.to_solar_input()
        wind_input = data.to_wind_input()

        solar_kw = self.predict_solar(solar_input)
        wind_kw = self.predict_wind(wind_input)
        total_kw = round(solar_kw + wind_kw, 2)

        return {
            "solar_power_kw": solar_kw,
            "wind_power_kw": wind_kw,
            "total_power_kw": total_kw,
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns diagnostic info about model availability and paths."""
        return {
            "models_directory": str(self.models_dir),
            "solar_model_loaded": self._is_solar_loaded,
            "wind_model_loaded": self._is_wind_loaded,
            "solar_xgboost_loaded": self._is_solar_xgb_loaded,
            "wind_xgboost_loaded": self._is_wind_xgb_loaded,
            "solar_features": self.SOLAR_FEATURES,
            "wind_features": self.WIND_FEATURES,
        }


# Global model manager singleton instance
model_manager = ModelManager()

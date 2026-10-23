"""
schemas.py
----------
Pydantic schemas for request validation and response models for
Solar, Wind, and Combined renewable energy power predictions.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator


class SolarInput(BaseModel):
    """
    Input features required for Solar power prediction.
    """
    temperature: float = Field(
        ...,
        ge=-50.0,
        le=70.0,
        description="Ambient air temperature in degrees Celsius (°C)",
        json_schema_extra={"example": 28.5}
    )
    irradiance: float = Field(
        ...,
        ge=0.0,
        le=1500.0,
        description="Global Horizontal Solar Irradiance in Watts per square meter (W/m²)",
        json_schema_extra={"example": 850.0}
    )
    humidity: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relative ambient humidity percentage (%)",
        json_schema_extra={"example": 42.0}
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "temperature": 28.5,
                "irradiance": 850.0,
                "humidity": 42.0
            }
        }
    }


class WindInput(BaseModel):
    """
    Input features required for Wind power prediction.
    """
    wind_speed: float = Field(
        ...,
        ge=0.0,
        le=60.0,
        description="Wind speed measured at hub height in meters per second (m/s)",
        json_schema_extra={"example": 12.5}
    )
    wind_direction: float = Field(
        ...,
        ge=0.0,
        le=360.0,
        description="Wind compass direction in degrees (0° - 360°)",
        json_schema_extra={"example": 180.0}
    )
    air_pressure: float = Field(
        ...,
        ge=800.0,
        le=1150.0,
        description="Barometric atmospheric air pressure in hectopascals (hPa)",
        json_schema_extra={"example": 1013.25}
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "wind_speed": 12.5,
                "wind_direction": 180.0,
                "air_pressure": 1013.25
            }
        }
    }


class CombinedInput(BaseModel):
    """
    Input features for combined Solar and Wind power prediction.
    Supports either flat dictionary or nested {solar: ..., wind: ...} structure.
    """
    temperature: Optional[float] = Field(
        None, ge=-50.0, le=70.0, description="Ambient air temperature in °C",
        json_schema_extra={"example": 28.5}
    )
    irradiance: Optional[float] = Field(
        None, ge=0.0, le=1500.0, description="Solar Irradiance in W/m²",
        json_schema_extra={"example": 850.0}
    )
    humidity: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Relative humidity in %",
        json_schema_extra={"example": 42.0}
    )

    wind_speed: Optional[float] = Field(
        None, ge=0.0, le=60.0, description="Wind speed in m/s",
        json_schema_extra={"example": 12.5}
    )
    wind_direction: Optional[float] = Field(
        None, ge=0.0, le=360.0, description="Wind direction in degrees",
        json_schema_extra={"example": 180.0}
    )
    air_pressure: Optional[float] = Field(
        None, ge=800.0, le=1150.0, description="Air pressure in hPa",
        json_schema_extra={"example": 1013.25}
    )

    solar: Optional[SolarInput] = None
    wind: Optional[WindInput] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_inputs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # If passed with nested keys 'solar' and 'wind'
        if "solar" in data and isinstance(data["solar"], dict):
            for k, v in data["solar"].items():
                data.setdefault(k, v)
        if "wind" in data and isinstance(data["wind"], dict):
            for k, v in data["wind"].items():
                data.setdefault(k, v)

        # Validate that all required solar and wind fields are present either way
        required_fields = ["temperature", "irradiance", "humidity", "wind_speed", "wind_direction", "air_pressure"]
        missing = [f for f in required_fields if data.get(f) is None]
        if missing:
            raise ValueError(f"Missing required parameters for combined prediction: {', '.join(missing)}")

        return data

    def to_solar_input(self) -> SolarInput:
        return SolarInput(
            temperature=self.temperature,
            irradiance=self.irradiance,
            humidity=self.humidity
        )

    def to_wind_input(self) -> WindInput:
        return WindInput(
            wind_speed=self.wind_speed,
            wind_direction=self.wind_direction,
            air_pressure=self.air_pressure
        )

    model_config = {
        "json_schema_extra": {
            "example": {
                "temperature": 28.5,
                "irradiance": 850.0,
                "humidity": 42.0,
                "wind_speed": 12.5,
                "wind_direction": 180.0,
                "air_pressure": 1013.25
            }
        }
    }


# Response Schemas

class SolarPredictionResponse(BaseModel):
    status: str = "success"
    power_kw: float = Field(..., description="Predicted solar power output rounded to 2 decimal places")
    unit: str = "kW"
    input_parameters: Dict[str, Any]


class WindPredictionResponse(BaseModel):
    status: str = "success"
    power_kw: float = Field(..., description="Predicted wind power output rounded to 2 decimal places")
    unit: str = "kW"
    input_parameters: Dict[str, Any]


class CombinedPredictionResponse(BaseModel):
    status: str = "success"
    solar_power_kw: float = Field(..., description="Predicted solar power output in kW")
    wind_power_kw: float = Field(..., description="Predicted wind power output in kW")
    total_power_kw: float = Field(..., description="Total combined power output (solar + wind) in kW")
    unit: str = "kW"
    input_parameters: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    solar_model_loaded: bool
    wind_model_loaded: bool
    version: str

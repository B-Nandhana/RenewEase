# RenewEase — Enterprise Microgrid Management

An enterprise-grade **FastAPI** web application and digital twin platform for autonomous industrial microgrid orchestration and predictive renewable energy forecasting (Solar Photovoltaic, Industrial Wind Turbines, and BESS Battery Storage). Featuring an interactive 3D digital twin of customer facilities with dynamic visual synchronization.

---

## 📁 Project Architecture & Structure

```
renewable_energy_app/
│
├── models/
│   ├── solar_model.pkl          # Serialized Solar LinearRegression model artifact
│   └── wind_model.pkl           # Serialized Wind LinearRegression model artifact
│
├── app/
│   ├── __init__.py              # Application package initializer
│   ├── main.py                  # FastAPI application, routing, and Jinja2 template rendering
│   ├── model_loader.py          # ModelManager singleton for safe deserialization & inference
│   └── schemas.py               # Pydantic input validation and API response models
│
├── templates/
│   └── index.html               # Modern Tailwind CSS dark-theme dashboard with live fetch()
│
├── create_dummy_models.py       # Standalone training script to generate and export dummy .pkl models
├── requirements.txt             # Project dependencies
└── README.md                    # System documentation and setup guide
```

---

## ⚡ Machine Learning Model Specifications

| System | Input Features | Feature Descriptions | Valid Ranges | Output Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Solar PV** | `temperature`<br>`irradiance`<br>`humidity` | Ambient air temperature (°C)<br>Global Horizontal Irradiance (W/m²)<br>Relative humidity (%) | -50°C to 70°C<br>0 to 1500 W/m²<br>0% to 100% | Power Output in **kW** (rounded to 2 decimal places, non-negative) |
| **Wind Turbine** | `wind_speed`<br>`wind_direction`<br>`air_pressure` | Wind velocity at hub height (m/s)<br>Wind compass azimuth (degrees)<br>Atmospheric air pressure (hPa) | 0 to 60 m/s<br>0° to 360°<br>800 to 1150 hPa | Power Output in **kW** (rounded to 2 decimal places, non-negative) |

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- **Python 3.10+** (tested on Python 3.12)
- `pip` package manager

### 2. Navigate to Project Directory
```bash
cd renewable_energy_app
```

### 3. Create & Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(Or use `py -m venv venv` if Python launcher is standard)*

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Generate ML Model Artifacts
Train and export the dummy Scikit-Learn models into the `models/` directory:
```bash
python create_dummy_models.py
```
*Expected output:*
```
[✓] Solar model saved to: .../models/solar_model.pkl
[✓] Wind model saved to:  .../models/wind_model.pkl
[✓] All models successfully trained and exported!
```

### 6. Launch the FastAPI Server
Run the Uvicorn ASGI server with live reloading enabled:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Access the Application
- **Interactive UI Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Specification**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🔌 API Reference & Usage

### 1. Solar Power Prediction
- **Endpoint**: `POST /predict/solar`
- **Content-Type**: `application/json`

**Request Body:**
```json
{
  "temperature": 28.5,
  "irradiance": 850.0,
  "humidity": 42.0
}
```

**Response (`200 OK`):**
```json
{
  "status": "success",
  "power_kw": 36.52,
  "unit": "kW",
  "input_parameters": {
    "temperature": 28.5,
    "irradiance": 850.0,
    "humidity": 42.0
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict/solar" \
  -H "Content-Type: application/json" \
  -d '{"temperature": 28.5, "irradiance": 850.0, "humidity": 42.0}'
```

---

### 2. Wind Power Prediction
- **Endpoint**: `POST /predict/wind`
- **Content-Type**: `application/json`

**Request Body:**
```json
{
  "wind_speed": 12.5,
  "wind_direction": 180.0,
  "air_pressure": 1013.25
}
```

**Response (`200 OK`):**
```json
{
  "status": "success",
  "power_kw": 31.98,
  "unit": "kW",
  "input_parameters": {
    "wind_speed": 12.5,
    "wind_direction": 180.0,
    "air_pressure": 1013.25
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict/wind" \
  -H "Content-Type: application/json" \
  -d '{"wind_speed": 12.5, "wind_direction": 180.0, "air_pressure": 1013.25}'
```

---

### 3. Combined Solar + Wind Prediction
- **Endpoint**: `POST /predict/combined`
- **Content-Type**: `application/json`

**Request Body:**
```json
{
  "temperature": 28.5,
  "irradiance": 850.0,
  "humidity": 42.0,
  "wind_speed": 12.5,
  "wind_direction": 180.0,
  "air_pressure": 1013.25
}
```

**Response (`200 OK`):**
```json
{
  "status": "success",
  "solar_power_kw": 36.52,
  "wind_power_kw": 31.98,
  "total_power_kw": 68.50,
  "unit": "kW",
  "input_parameters": {
    "temperature": 28.5,
    "irradiance": 850.0,
    "humidity": 42.0,
    "wind_speed": 12.5,
    "wind_direction": 180.0,
    "air_pressure": 1013.25
  }
}
```

---

## 🎨 Frontend Dashboard Features
1. **Live Summary Cards**: Real-time KPI cards for Solar, Wind, and Total Power output with dynamic numeric count-up animations.
2. **Two-Way Synchronized Inputs**: Sliders and numeric text boxes are synchronized with instant validation bounds.
3. **Meteorological Presets**: 1-click test configurations (Sunny Midday, Overcast Sky, Desert Heat, Gentle Breeze, Optimal Production, Storm High-Wind).
4. **Dynamic Fetch Integration**: Zero page reloads using asynchronous JavaScript `fetch()` calls.
5. **Glassmorphic Dark UI**: Built with Tailwind CSS and glowing ambient radial backlights.

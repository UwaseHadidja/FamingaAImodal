## Smart Irrigation API
A REST API that provides intelligent irrigation recommendations based on real-time soil sensor data and weather conditions. The system analyzes moisture levels, temperature, humidity, and forecast data to deliver actionable decisions: **Irrigate, Hold, or Alert**.

## Features

- **Real-time Decision Making:** Get instant irrigation recommendations based on current conditions
  
- **Multi-factor Analysis:** Combines soil moisture, temperature, humidity, and weather forecasts
  
- **Smart Alerts:** Notifies users of critical conditions requiring immediate attention
  
- **Weather Integration:** Incorporates forecast data to optimize watering schedules
  
- **RESTful Design:** Easy to integrate with existing systems and IoT devices

## Getting Started

**Prerequisites**

- Python
- Database (Firebase)
- API key for weather service

## API Documentation

**Base URL**
https://famingaaimodal.onrender.com/

**Endpoints**

**Get Irrigation Decision**
```http
POST api/v1/irrigation/advice
```
**Request Body**
```json

{
        "soil_data": {
            "soil_moisture": 45.5,
            "ph": 6.2,
            "nitrogen": 120,
            "phosphorus": 50,
            "potassium": 180,
            "temperature": 22.5
        },
        "weather_data": {
            "temperature": 28,
            "humidity": 65,
            "rain_probability": 20,
            "rain_amount_mm": 0,
            "wind_speed": 5.5
        },
        "crop_type": "tomato",
        "growth_stage": "flowering",
        "field_capacity": 100.0
    }
```

**Response**
```json

}
    "confidence": 100,
    "crop": "Tomato",
    "decision": "IRRIGATE",
    "growth_stage": "flowering",
    "reasons": [
"Low moisture: 45.5% (optimal for flowering: 72.0-96.0%)",
"Recommended irrigation: 50.5mm for 303 minutes"],
    "recommendations": {
        "action_summary": "Irrigate now - soil moisture below optimal levels",
        "irrigation_amount_mm": 50.5,
        "irrigation_duration_minutes": 303.0,
        "next_check_hours": 12
    }
    "timestamp": "2025-12-03T08:38:33.981791"
}
```
## Decision Types

**IRRIGATE:** Conditions require immediate watering

**HOLD:** Sufficient moisture or rain expected soon

**ALERT:** Critical conditions detected (sensor malfunction, extreme weather, etc.)

## Decision Logic

The API uses the following criteria to make irrigation decisions:

- **Soil Moisture Analysis:** Compares current levels against crop-specific thresholds
- **Weather Forecast:** Checks for upcoming precipitation
- **Temperature & Evapotranspiration:** Calculates water loss rate
- **Historical Patterns:** Learns from past irrigation cycles
- **Crop Requirements:** Adjusts recommendations based on plant type and growth stage

## Contact

Uwase Hadija - uwasedj@gmail.com

Project Link: https://github.com/UwaseHadidja/FamingaAImodal

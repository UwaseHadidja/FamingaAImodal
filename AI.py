
from flask import Flask, request, jsonify
from pyngrok import ngrok
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

app = Flask(__name__)

public_url = ngrok.connect(5000)
print(" * ngrok tunnel:", public_url)

class IrrigationAdvice(Enum):
    """Irrigation decision types"""
    IRRIGATE = "IRRIGATE"
    HOLD = "HOLD"
    ALERT = "ALERT"


class AlertType(Enum):
    """Types of alerts"""
    LOW_MOISTURE = "LOW_MOISTURE"
    HIGH_MOISTURE = "HIGH_MOISTURE"
    NUTRIENT_DEFICIENCY = "NUTRIENT_DEFICIENCY"
    PH_IMBALANCE = "PH_IMBALANCE"
    EXTREME_TEMPERATURE = "EXTREME_TEMPERATURE"
    RAIN_EXPECTED = "RAIN_EXPECTED"
    SENSOR_MALFUNCTION = "SENSOR_MALFUNCTION"


@dataclass
class CropProfile:
    """Crop-specific water and nutrient requirements"""
    name: str
    moisture_optimal_min: float  # %
    moisture_optimal_max: float  # %
    moisture_critical_min: float  # %
    moisture_critical_max: float  # %
    ph_optimal_min: float
    ph_optimal_max: float
    nitrogen_min: float  # mg/kg
    phosphorus_min: float  # mg/kg
    potassium_min: float  # mg/kg
    temp_optimal_min: float  # °C
    temp_optimal_max: float  # °C
    growth_stage_factor: Dict[str, float]  # Water needs by growth stage


# Predefined crop profiles
CROP_PROFILES = {
    'tomato': CropProfile(
        name='Tomato',
        moisture_optimal_min=60, moisture_optimal_max=80,
        moisture_critical_min=40, moisture_critical_max=90,
        ph_optimal_min=6.0, ph_optimal_max=6.8,
        nitrogen_min=100, phosphorus_min=40, potassium_min=150,
        temp_optimal_min=18, temp_optimal_max=28,
        growth_stage_factor={'seedling': 0.7, 'vegetative': 1.0, 'flowering': 1.2, 'fruiting': 1.3}
    ),
    'maize': CropProfile(
        name='Maize',
        moisture_optimal_min=55, moisture_optimal_max=75,
        moisture_critical_min=35, moisture_critical_max=85,
        ph_optimal_min=5.8, ph_optimal_max=7.0,
        nitrogen_min=120, phosphorus_min=50, potassium_min=100,
        temp_optimal_min=20, temp_optimal_max=30,
        growth_stage_factor={'seedling': 0.6, 'vegetative': 1.0, 'tasseling': 1.4, 'grain_fill': 1.2}
    ),
    'potato': CropProfile(
        name='Potato',
        moisture_optimal_min=65, moisture_optimal_max=85,
        moisture_critical_min=45, moisture_critical_max=90,
        ph_optimal_min=5.0, ph_optimal_max=6.5,
        nitrogen_min=110, phosphorus_min=45, potassium_min=180,
        temp_optimal_min=15, temp_optimal_max=24,
        growth_stage_factor={'planting': 0.5, 'vegetative': 0.8, 'tuber_init': 1.2, 'bulking': 1.4}
    ),
    'lettuce': CropProfile(
        name='Lettuce',
        moisture_optimal_min=70, moisture_optimal_max=85,
        moisture_critical_min=50, moisture_critical_max=90,
        ph_optimal_min=6.0, ph_optimal_max=7.0,
        nitrogen_min=80, phosphorus_min=30, potassium_min=120,
        temp_optimal_min=15, temp_optimal_max=20,
        growth_stage_factor={'seedling': 0.6, 'vegetative': 1.0, 'heading': 1.1}
    ),
    'default': CropProfile(
        name='Generic Crop',
        moisture_optimal_min=60, moisture_optimal_max=80,
        moisture_critical_min=40, moisture_critical_max=90,
        ph_optimal_min=6.0, ph_optimal_max=7.0,
        nitrogen_min=100, phosphorus_min=40, potassium_min=150,
        temp_optimal_min=18, temp_optimal_max=28,
        growth_stage_factor={'default': 1.0}
    )
}


class IrrigationDecisionEngine:
    """Core decision engine for irrigation advice"""

    def __init__(self):
        self.decision_log = []

    def analyze(self,
                soil_data: Dict,
                weather_data: Dict,
                crop_type: str = 'default',
                growth_stage: str = 'vegetative',
                field_capacity: float = 100.0) -> Dict:
        """
        Analyze soil and weather data to provide irrigation advice

        Args:
            soil_data: Current soil sensor readings
            weather_data: Weather forecast data
            crop_type: Type of crop being grown
            growth_stage: Current growth stage
            field_capacity: Field's water holding capacity (%)

        Returns:
            Decision dictionary with advice and reasoning
        """
        # Get crop profile
        crop = CROP_PROFILES.get(crop_type.lower(), CROP_PROFILES['default'])

        # Extract data
        moisture = soil_data.get('soil_moisture', 50)
        ph = soil_data.get('ph', 6.5)
        nitrogen = soil_data.get('nitrogen', 100)
        phosphorus = soil_data.get('phosphorus', 40)
        potassium = soil_data.get('potassium', 150)
        soil_temp = soil_data.get('temperature', 22)

        # Weather data
        rain_probability = weather_data.get('rain_probability', 0)
        rain_amount = weather_data.get('rain_amount_mm', 0)
        temperature = weather_data.get('temperature', 25)
        humidity = weather_data.get('humidity', 60)
        wind_speed = weather_data.get('wind_speed', 5)

        # Calculate scores and flags
        alerts = []
        reasons = []
        confidence_score = 100

        # 1. SOIL MOISTURE ANALYSIS
        moisture_status = self._analyze_moisture(moisture, crop, growth_stage)

        if moisture < crop.moisture_critical_min:
            alerts.append(AlertType.LOW_MOISTURE.value)
            reasons.append(f"Critical: Moisture at {moisture:.1f}% (min: {crop.moisture_critical_min}%)")
        elif moisture > crop.moisture_critical_max:
            alerts.append(AlertType.HIGH_MOISTURE.value)
            reasons.append(f"Critical: Moisture at {moisture:.1f}% (max: {crop.moisture_critical_max}%)")

        # 2. WEATHER ANALYSIS
        weather_status = self._analyze_weather(weather_data)

        if rain_probability > 60:
            alerts.append(AlertType.RAIN_EXPECTED.value)
            reasons.append(f"Rain expected: {rain_probability}% chance, {rain_amount:.1f}mm")

        # 3. NUTRIENT ANALYSIS
        nutrient_status = self._analyze_nutrients(nitrogen, phosphorus, potassium, crop)

        if nitrogen < crop.nitrogen_min:
            alerts.append(AlertType.NUTRIENT_DEFICIENCY.value)
            reasons.append(f"Low nitrogen: {nitrogen:.1f} mg/kg (min: {crop.nitrogen_min})")
        if phosphorus < crop.phosphorus_min:
            alerts.append(AlertType.NUTRIENT_DEFICIENCY.value)
            reasons.append(f"Low phosphorus: {phosphorus:.1f} mg/kg (min: {crop.phosphorus_min})")
        if potassium < crop.potassium_min:
            alerts.append(AlertType.NUTRIENT_DEFICIENCY.value)
            reasons.append(f"Low potassium: {potassium:.1f} mg/kg (min: {crop.potassium_min})")

        # 4. pH ANALYSIS
        if not (crop.ph_optimal_min <= ph <= crop.ph_optimal_max):
            alerts.append(AlertType.PH_IMBALANCE.value)
            reasons.append(f"pH imbalance: {ph:.1f} (optimal: {crop.ph_optimal_min}-{crop.ph_optimal_max})")

        # 5. TEMPERATURE ANALYSIS
        if soil_temp < crop.temp_optimal_min or soil_temp > crop.temp_optimal_max:
            alerts.append(AlertType.EXTREME_TEMPERATURE.value)
            reasons.append(f"Soil temp: {soil_temp:.1f}°C (optimal: {crop.temp_optimal_min}-{crop.temp_optimal_max}°C)")

        # 6. MAKE DECISION
        decision = self._make_decision(
            moisture, moisture_status, weather_status,
            rain_probability, crop, alerts
        )

        # 7. CALCULATE EVAPOTRANSPIRATION (ET)
        et_rate = self._calculate_et(temperature, humidity, wind_speed, growth_stage, crop)

        # 8. IRRIGATION RECOMMENDATION
        irrigation_amount = 0
        irrigation_duration = 0

        if decision == IrrigationAdvice.IRRIGATE:
            # Calculate how much water needed
            moisture_deficit = crop.moisture_optimal_max - moisture
            irrigation_amount = (moisture_deficit / 100) * field_capacity
            irrigation_duration = self._calculate_duration(irrigation_amount)

            reasons.append(f"Recommended irrigation: {irrigation_amount:.1f}mm for {irrigation_duration:.0f} minutes")

        # Build response
        response = {
            'timestamp': datetime.now().isoformat(),
            'decision': decision.value,
            'confidence': confidence_score,
            'crop': crop.name,
            'growth_stage': growth_stage,
            'analysis': {
                'soil': {
                    'moisture': moisture,
                    'moisture_status': moisture_status,
                    'ph': ph,
                    'temperature': soil_temp,
                    'nutrients': {
                        'nitrogen': nitrogen,
                        'phosphorus': phosphorus,
                        'potassium': potassium,
                        'status': nutrient_status
                    }
                },
                'weather': {
                    'temperature': temperature,
                    'humidity': humidity,
                    'rain_probability': rain_probability,
                    'rain_amount': rain_amount,
                    'wind_speed': wind_speed,
                    'status': weather_status
                },
                'evapotranspiration': {
                    'daily_et_mm': et_rate,
                    'water_loss_rate': 'moderate' if 3 <= et_rate <= 7 else ('low' if et_rate < 3 else 'high')
                }
            },
            'alerts': list(set(alerts)),
            'reasons': reasons,
            'recommendations': {
                'irrigation_amount_mm': irrigation_amount,
                'irrigation_duration_minutes': irrigation_duration,
                'next_check_hours': self._next_check_interval(decision, weather_status)
            }
        }

        # Log decision
        self.decision_log.append(response)

        return response

    def _analyze_moisture(self, moisture: float, crop: CropProfile, growth_stage: str) -> str:
        """Analyze moisture level"""
        # Adjust thresholds based on growth stage
        stage_factor = crop.growth_stage_factor.get(growth_stage, 1.0)
        optimal_min = crop.moisture_optimal_min * stage_factor
        optimal_max = crop.moisture_optimal_max * stage_factor

        if moisture < crop.moisture_critical_min:
            return 'CRITICALLY_LOW'
        elif moisture < optimal_min:
            return 'LOW'
        elif moisture <= optimal_max:
            return 'OPTIMAL'
        elif moisture <= crop.moisture_critical_max:
            return 'HIGH'
        else:
            return 'CRITICALLY_HIGH'

    def _analyze_weather(self, weather: Dict) -> str:
        """Analyze weather conditions"""
        rain_prob = weather.get('rain_probability', 0)
        temp = weather.get('temperature', 25)

        if rain_prob > 70:
            return 'HEAVY_RAIN_EXPECTED'
        elif rain_prob > 40:
            return 'RAIN_LIKELY'
        elif temp > 35:
            return 'EXTREME_HEAT'
        elif temp < 5:
            return 'EXTREME_COLD'
        else:
            return 'NORMAL'

    def _analyze_nutrients(self, n: float, p: float, k: float, crop: CropProfile) -> str:
        """Analyze nutrient levels"""
        n_ok = n >= crop.nitrogen_min
        p_ok = p >= crop.phosphorus_min
        k_ok = k >= crop.potassium_min

        if n_ok and p_ok and k_ok:
            return 'ADEQUATE'
        elif not n_ok or not p_ok or not k_ok:
            return 'DEFICIENT'
        else:
            return 'BORDERLINE'

    def _make_decision(self, moisture: float, moisture_status: str,
                      weather_status: str, rain_prob: float,
                      crop: CropProfile, alerts: List) -> IrrigationAdvice:
        """Make final irrigation decision"""

        # ALERT conditions (critical issues)
        if moisture_status in ['CRITICALLY_LOW', 'CRITICALLY_HIGH']:
            return IrrigationAdvice.ALERT

        if AlertType.SENSOR_MALFUNCTION.value in alerts:
            return IrrigationAdvice.ALERT

        # HOLD conditions (don't irrigate)
        if weather_status in ['HEAVY_RAIN_EXPECTED', 'RAIN_LIKELY']:
            return IrrigationAdvice.HOLD

        if moisture_status in ['OPTIMAL', 'HIGH']:
            return IrrigationAdvice.HOLD

        if rain_prob > 60:
            return IrrigationAdvice.HOLD

        # IRRIGATE conditions
        if moisture_status == 'LOW' and rain_prob < 30:
            return IrrigationAdvice.IRRIGATE

        # Default to HOLD if uncertain
        return IrrigationAdvice.HOLD

    def _calculate_et(self, temp: float, humidity: float,
                     wind: float, growth_stage: str, crop: CropProfile) -> float:
        """Calculate daily evapotranspiration (simplified Penman-Monteith)"""
        # Simplified ET calculation (mm/day)
        base_et = 0.0023 * (temp + 17.8) * (temp - humidity/10) * wind**0.5

        # Adjust for crop and growth stage
        stage_factor = crop.growth_stage_factor.get(growth_stage, 1.0)
        et_daily = base_et * stage_factor * 1.2  # Crop coefficient

        return max(0, et_daily)

    def _calculate_duration(self, amount_mm: float, flow_rate: float = 10.0) -> float:
        """Calculate irrigation duration based on water amount needed"""
        # flow_rate in mm/hour
        return (amount_mm / flow_rate) * 60  # Convert to minutes

    def _next_check_interval(self, decision: IrrigationAdvice, weather: str) -> int:
        """Determine when to check again (hours)"""
        if decision == IrrigationAdvice.ALERT:
            return 2  # Check again soon
        elif decision == IrrigationAdvice.IRRIGATE:
            return 12  # Check after irrigation
        elif weather in ['HEAVY_RAIN_EXPECTED', 'RAIN_LIKELY']:
            return 6  # Check after rain
        else:
            return 24  # Daily check


# Initialize decision engine
engine = IrrigationDecisionEngine()


# ==================== API ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/v1/irrigation/advice', methods=['POST'])
def get_irrigation_advice():
    """
    Main endpoint for irrigation advice

    Request body:
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
    """
    try:
        data = request.get_json()

        # Validate required fields
        if not data.get('soil_data') or not data.get('weather_data'):
            return jsonify({
                'error': 'Missing required fields: soil_data and weather_data'
            }), 400

        # Get advice
        advice = engine.analyze(
            soil_data=data['soil_data'],
            weather_data=data['weather_data'],
            crop_type=data.get('crop_type', 'default'),
            growth_stage=data.get('growth_stage', 'vegetative'),
            field_capacity=data.get('field_capacity', 100.0)
        )

        return jsonify(advice), 200

    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/v1/crops', methods=['GET'])
def get_crop_profiles():
    """Get available crop profiles"""
    profiles = {}
    for key, crop in CROP_PROFILES.items():
        profiles[key] = {
            'name': crop.name,
            'moisture_range': f"{crop.moisture_optimal_min}-{crop.moisture_optimal_max}%",
            'ph_range': f"{crop.ph_optimal_min}-{crop.ph_optimal_max}",
            'temp_range': f"{crop.temp_optimal_min}-{crop.temp_optimal_max}°C"
        }

    return jsonify({'crops': profiles})


@app.route('/api/v1/history', methods=['GET'])
def get_decision_history():
    """Get recent decision history"""
    limit = request.args.get('limit', 10, type=int)
    return jsonify({
        'decisions': engine.decision_log[-limit:],
        'total': len(engine.decision_log)
    })



from flask_ngrok import run_with_ngrok

run_with_ngrok(app)
app.run()

from fastapi import APIRouter, HTTPException, Depends
from api.schemas import PredictionRequest, PredictionResponse, ModelInfo, TrainingResponse
from ml.model import PricePredictor
from core.logger import LoggerFactory
from utils.cache import prediction_cache as cache
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from functools import lru_cache

router = APIRouter()
logger = LoggerFactory.get_logger()


@lru_cache()
def get_predictor() -> PricePredictor:
    """Create and cache a singleton instance of PricePredictor."""
    try:
        predictor = PricePredictor()
        return predictor
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise HTTPException(status_code=500, detail="Model not available")

class WeatherCorrelations(BaseModel):
    temperature: float
    rain: float
    humidity: float
    wind: float

class PriceDistributionStats(BaseModel):
    count: float
    mean: float
    std: float
    min: float
    percentile_25: float = Field(alias="25%")
    percentile_50: float = Field(alias="50%")
    percentile_75: float = Field(alias="75%")
    max: float

    class Config:
        allow_population_by_field_name = True

class WeatherPriceDistribution(BaseModel):
    rainy: PriceDistributionStats
    normal: PriceDistributionStats
    extreme_temp: PriceDistributionStats
    normal_temp: PriceDistributionStats

class WeatherAnalysisResponse(BaseModel):
    correlations: WeatherCorrelations
    price_distributions: WeatherPriceDistribution
    hourly_prices: Dict[str, float]

class ModelPerformanceResponse(BaseModel):
    r2_scores: Dict[str, float]
    rmse_scores: Dict[str, float]
    feature_importance: Dict[str, float]

@router.post("/predict", response_model=PredictionResponse)
async def predict_price(
    request: PredictionRequest,
    predictor: PricePredictor = Depends(get_predictor)
) -> Dict:
    try:
        # Create a DataFrame with a single row for prediction
        data = pd.DataFrame([{
            'time': request.time,
            'distance': float(request.distance),
            'temperature': float(request.temperature),
            'rain_amount': float(request.rain_amount),
            'wind_speed': float(request.wind_speed),
            'humidity': float(request.humidity)
        }])
        
        # Make prediction
        predicted_price = float(predictor.predict(data)[0])
        
        # Extract time to check peak hours
        hour = pd.to_datetime(request.time).hour
        is_peak = (7 <= hour <= 9) or (16 <= hour <= 18)
        
        # Process weather conditions
        weather_conditions = {
            'temp_extreme': float(request.temperature) > 35 or float(request.temperature) < 15,
            'has_rain': float(request.rain_amount) > 0,
            'heavy_rain': float(request.rain_amount) > 10
        }
        
        return {
            "predicted_price": predicted_price,
            "peak_hour": is_peak,
            "weather_conditions": weather_conditions
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.get("/model/metrics")
async def get_model_metrics(
    predictor: PricePredictor = Depends(get_predictor)
) -> Dict:
    """Get comprehensive model metrics and analysis."""
    metrics = predictor.get_model_metrics()
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics available. Model needs training.")
    return metrics




@router.get("/api/weather/correlations")
async def get_weather_correlations() -> Dict[str, Any]:
    """
    Get correlations between different weather features
    Returns:
        Dict containing correlation coefficients and their interpretations
    """
    predictor = PricePredictor()
    try:
        correlations = predictor.calculate_weather_correlations()
        return {
            "status": "success",
            "data": correlations,
            "message": "Weather correlations calculated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/model/metrics")
async def get_model_metrics() -> Dict[str, Any]:
    """
    Get various model performance metrics including MAE, RMSE, MAPE, and R2
    Returns:
        Dict containing model metrics and their interpretations
    """
    predictor = PricePredictor()
    try:
        metrics = predictor.calculate_model_metrics()
        return {
            "status": "success",
            "data": metrics,
            "message": "Model metrics calculated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

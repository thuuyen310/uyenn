from pydantic import BaseModel, Field
from datetime import time

class PredictionRequest(BaseModel):
    time: str = Field(..., example="00:00:00", description="Time of the ride booking")
    distance: float = Field(..., example=5.0, description="Distance of the ride in kilometers")
    temperature: float = Field(..., example=25.0, description="Temperature in Celsius")
    rain_amount: float = Field(..., ge=0, example=0.0, description="Rain amount in mm")
    wind_speed: float = Field(..., ge=0, example=10.0, description="Wind speed in km/h")
    humidity: float = Field(..., ge=0, le=100, example=65.0, description="Humidity percentage")

    class Config:
        schema_extra = {
            "example": {
                "time": "08:30:00",
                "distance": 5.0,
                "temperature": 25.0,
                "rain_amount": 0.0,
                "wind_speed": 10.0,
                "humidity": 65.0
            }
        }

class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted ride price")
    peak_hour: bool = Field(..., description="Whether the ride is during peak hours")
    weather_conditions: dict = Field(..., description="Processed weather conditions")

class ModelInfo(BaseModel):
    feature_count: int = Field(..., description="Number of features used in the model")
    model_type: str = Field(..., description="Type of the model")
    features: list[str] = Field(..., description="List of features used in the model")
    peak_hours: dict = Field(..., description="Peak hours configuration")
    weather_thresholds: dict = Field(..., description="Weather thresholds used in the model")

class TrainingResponse(BaseModel):
    status: str = Field(..., description="Training status")
    train_score: float = Field(..., description="R² score on training data")
    test_score: float = Field(..., description="R² score on test data")
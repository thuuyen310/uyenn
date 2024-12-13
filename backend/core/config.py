from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "Car Price Prediction API"
    DEBUG_MODE: bool = False
    MODEL_PATH: str = "models/price_model.joblib"
    SCALER_PATH: str = "models/scaler.joblib"
    CACHE_TTL: int = 3600
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

# core/logger.py
import logging
import sys

def setup_logger():
    logger = logging.getLogger("price_prediction")
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(handler)
    return logger

logger = setup_logger()
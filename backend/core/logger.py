import logging
import sys
from typing import Optional

class LoggerFactory:
    _instance: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._instance is None:
            logger = logging.getLogger("price_prediction")
            logger.setLevel(logging.INFO)
            
            # Avoid adding handlers if they already exist
            if not logger.hasHandlers():
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d')
                )
                logger.addHandler(handler)
            
            cls._instance = logger
        
        return cls._instance
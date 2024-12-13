from typing import Dict, Any, Optional, TypeVar, Callable
from functools import lru_cache
from datetime import datetime, timedelta
from core.logger import LoggerFactory
from core.config import get_settings
import hashlib
import json

logger = LoggerFactory.get_logger()
settings = get_settings()

T = TypeVar('T')  # Generic type for cache value

class PredictionCache:
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self.ttl_seconds = ttl_seconds
        logger.info(f"Initialized prediction cache with TTL: {ttl_seconds} seconds")

    def _generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Generate a unique cache key from request data."""
        # Sort dictionary to ensure consistent key generation
        sorted_data = json.dumps(request_data, sort_keys=True)
        return hashlib.md5(sorted_data.encode()).hexdigest()

    def get(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve item from cache if it exists and is not expired."""
        try:
            cache_key = self._generate_cache_key(request_data)
            if cache_key in self._cache:
                value, timestamp = self._cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return value
                else:
                    # Remove expired entry
                    del self._cache[cache_key]
                    logger.debug(f"Removed expired cache entry for key: {cache_key}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None

    def set(self, request_data: Dict[str, Any], value: Dict[str, Any]) -> None:
        """Store item in cache with current timestamp."""
        try:
            cache_key = self._generate_cache_key(request_data)
            self._cache[cache_key] = (value, datetime.now())
            logger.debug(f"Cached prediction for key: {cache_key}")
            
            # Cleanup old entries if cache is too large (optional)
            if len(self._cache) > 1000:  # Configurable maximum cache size
                self._cleanup_old_entries()
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")

    def _cleanup_old_entries(self) -> None:
        """Remove expired entries from cache."""
        current_time = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp > timedelta(seconds=self.ttl_seconds)
        ]
        for key in expired_keys:
            del self._cache[key]
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    def clear(self) -> None:
        """Clear all entries from cache."""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get current cache statistics."""
        current_time = datetime.now()
        active_entries = sum(
            1 for _, timestamp in self._cache.values()
            if current_time - timestamp < timedelta(seconds=self.ttl_seconds)
        )
        return {
            "total_entries": len(self._cache),
            "active_entries": active_entries,
            "expired_entries": len(self._cache) - active_entries
        }

# Create global cache instance
prediction_cache = PredictionCache(ttl_seconds=settings.CACHE_TTL)
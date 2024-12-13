from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from core.config import get_settings
from core.logger import LoggerFactory
import warnings

logger = LoggerFactory.get_logger()
settings = get_settings()

warnings.filterwarnings("ignore")
app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG_MODE
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, you can specify a list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
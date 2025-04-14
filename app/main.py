from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.dependencies import get_or_initialize_data_loader
from app.logger import logger
from app.routes import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for FastAPI application.

    This pre-loads the data_loader singleton during application startup,
    ensuring it's ready when the first request comes in.
    """
    # Pre-load the data_loader at startup
    logger.info("Initializing data loader...")
    get_or_initialize_data_loader()
    logger.info(f"Data loader initialized with FAISS")

    # Yield control back to FastAPI
    logger.info("Fast API starting up ...")
    yield

    # Cleanup logic (if needed in the future)
    logger.info("Shutting down application...")


# Initialize application with lifespan manager
app = FastAPI(
    title="GL Code Predictor",
    description="API for predicting General Ledger codes for financial transactions",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
logger.info("Include Router")
settings = get_settings()
app.include_router(api_router, prefix=f"/{settings.API_PREFIX}")


@app.get("/test")
async def test_route():
    return {"status": "test works"}


# if __name__ == "__main__":
#     uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

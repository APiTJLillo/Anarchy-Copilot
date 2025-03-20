"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .proxy.endpoints import router as proxy_router
from .health import router as health_router
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure health logger
health_logger = logging.getLogger("health")
health_logger.setLevel(logging.DEBUG)

app = FastAPI(
    title="Anarchy Copilot API",
    description="API for Anarchy Copilot proxy and development tools",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(proxy_router, prefix="/api/proxy", tags=["proxy"])
app.include_router(health_router, prefix="/api/health", tags=["health"])

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to Anarchy Copilot API"} 